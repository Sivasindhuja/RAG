"""EXP-002: structure-aware chunking vs the accepted fixed-character baseline.

Only the chunking strategy changes.  The source PDF, approved 15-question
dataset, verified EXP-001 evidence units, embedding model, dense retrieval
algorithm, top-K and metric definitions are held constant.

Run:
    python -m experiments.exp002
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from sentence_transformers import SentenceTransformer

from src.rag_core import (
    DEFAULT_EMBEDDING_MODEL,
    DenseIndex,
    DenseIndexConfig,
    StructureAwareChunker,
    aggregate_metrics,
    audit_source_coverage,
    compute_retrieval_metrics,
    evidence_coverage_summary,
    evidence_for_question,
    extract_source_document,
    load_exp001_evidence,
    load_questions,
)

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_ID = "EXP-002"
PDF_PATH = ROOT / "data" / "satcom-ngp.pdf"
QUESTIONS_CSV = ROOT / "experiments" / "exp001_questions.csv"
GROUND_TRUTH_PATH = ROOT / "experiments" / "exp001_ground_truth.json"
BASELINE_RESULTS = ROOT / "experiments" / "results" / "exp001_v3" / "retrieval_results.json"
RESULTS_DIR = ROOT / "experiments" / "results" / "exp002"
PERSIST_DIR = ROOT / "chroma_db_exp002"
COLLECTION_NAME = "satcom_ngp_exp002_structure_aware"
EMBEDDING_MODEL = DEFAULT_EMBEDDING_MODEL
TOP_K = 5
CHUNK_SIZE = 1000
SMALL_CHUNK_THRESHOLD = 100


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    fraction = rank - low
    return round(ordered[low] * (1 - fraction) + ordered[high] * fraction, 2)


def _distribution(values: Sequence[int]) -> Dict[str, float]:
    if not values:
        return {"min": 0, "median": 0, "mean": 0, "p90": 0, "max": 0}
    return {
        "min": int(min(values)),
        "median": round(float(statistics.median(values)), 2),
        "mean": round(float(statistics.mean(values)), 2),
        "p90": _percentile(values, 0.90),
        "max": int(max(values)),
    }


def _token_lengths(chunks, tokenizer) -> List[int]:
    if tokenizer is None:
        return []
    lengths: List[int] = []
    for chunk in chunks:
        encoded = tokenizer.encode(chunk.text, add_special_tokens=False)
        lengths.append(len(encoded))
    return lengths


def _chunk_stats(chunks, token_lengths: Sequence[int], chunk_size: int) -> Dict[str, Any]:
    char_lengths = [len(chunk.text) for chunk in chunks]
    by_article: Dict[str, int] = {}
    for chunk in chunks:
        key = chunk.article or "PREAMBLE"
        by_article[key] = by_article.get(key, 0) + 1
    multi_page = [chunk.chunk_id for chunk in chunks if chunk.start_page != chunk.end_page]
    split_clauses = sorted({chunk.section for chunk in chunks if chunk.part > 0 and chunk.section})
    no_section = [chunk.chunk_id for chunk in chunks if chunk.section is None]
    return {
        "chunk_count": len(chunks),
        "characters": _distribution(char_lengths),
        "tokens": _distribution(token_lengths) if token_lengths else None,
        "tokenizer": "embedding_model_tokenizer" if token_lengths else None,
        "oversized_chunks": [chunk.chunk_id for chunk in chunks if len(chunk.text) > chunk_size],
        "small_chunks": [chunk.chunk_id for chunk in chunks if len(chunk.text) < SMALL_CHUNK_THRESHOLD],
        "small_chunk_threshold_chars": SMALL_CHUNK_THRESHOLD,
        "multi_page_chunks": multi_page,
        "split_clauses": split_clauses,
        "headless_chunks": no_section,
        "chunks_by_article": dict(sorted(by_article.items())),
    }


def _baseline() -> Dict[str, Any]:
    if not BASELINE_RESULTS.exists():
        return {}
    payload = json.loads(BASELINE_RESULTS.read_text(encoding="utf-8"))
    return {
        "source": str(BASELINE_RESULTS.relative_to(ROOT)),
        "aggregate_metrics": payload.get("aggregate_metrics", {}),
        "per_question_metrics": payload.get("per_question_metrics", []),
        "retrieval_results": payload.get("retrieval_results", []),
    }


def _comparison(
    baseline: Dict[str, Any],
    aggregate: Dict[str, float],
    per_question: List[Dict[str, Any]],
    top_k: int = TOP_K,
) -> Dict[str, Any]:
    recall_key, precision_key, mrr_key = f"recall@{top_k}", f"precision@{top_k}", f"mrr@{top_k}"
    if not baseline:
        return {"baseline_available": False, "reason": "baseline results not found"}
    baseline_aggregate = baseline.get("aggregate_metrics", {})
    if recall_key not in baseline_aggregate:
        return {
            "baseline_available": False,
            "reason": f"baseline has no {recall_key}; comparison only valid at the accepted top-K",
        }
    baseline_by_id = {row["question_id"]: row for row in baseline.get("per_question_metrics", [])}
    changes: List[Dict[str, Any]] = []
    improved = regressed = unchanged = 0
    for row in per_question:
        qid = row["question_id"]
        before = baseline_by_id.get(qid)
        if not before:
            continue
        delta_recall = round(row[recall_key] - before[recall_key], 4)
        delta_mrr = round(row[mrr_key] - before[mrr_key], 4)
        if delta_recall > 0 or delta_mrr > 0:
            improved += 1
        elif delta_recall < 0 or delta_mrr < 0:
            regressed += 1
        else:
            unchanged += 1
        changes.append(
            {
                "question_id": qid,
                "baseline": before,
                "exp002": {
                    recall_key: row[recall_key],
                    precision_key: row[precision_key],
                    mrr_key: row[mrr_key],
                },
                "delta": {
                    recall_key: delta_recall,
                    precision_key: round(row[precision_key] - before[precision_key], 4),
                    mrr_key: delta_mrr,
                },
            }
        )
    return {
        "baseline_available": True,
        "baseline_source": baseline["source"],
        "baseline_aggregate_metrics": baseline_aggregate,
        "delta_aggregate": {
            metric: round(aggregate.get(metric, 0.0) - baseline_aggregate.get(metric, 0.0), 4)
            for metric in (recall_key, precision_key, mrr_key)
        },
        "improved_questions": improved,
        "regressed_questions": regressed,
        "unchanged_questions": unchanged,
        "per_question": changes,
    }


def _failure_analysis(
    questions,
    per_question: List[Dict[str, Any]],
    retrieval_rows: List[Dict[str, Any]],
    top_k: int = TOP_K,
) -> List[Dict[str, Any]]:
    recall_key, precision_key, mrr_key = f"recall@{top_k}", f"precision@{top_k}", f"mrr@{top_k}"
    question_by_id = {question.question_id: question for question in questions}
    rows: List[Dict[str, Any]] = []
    for metrics in per_question:
        qid = metrics["question_id"]
        if metrics[recall_key] >= 1.0 and qid not in {"EXP001-Q05", "EXP001-Q08", "EXP001-Q13"}:
            continue
        question = question_by_id[qid]
        retrieval = next(row for row in retrieval_rows if row["question_id"] == qid)
        rows.append(
            {
                "question_id": qid,
                "question": question.question,
                "required_clause_ids": list(question.relevant_clause_ids),
                "recall@5": metrics[recall_key],
                "precision@5": metrics[precision_key],
                "mrr@5": metrics[mrr_key],
                "missing_evidence_ids": metrics["missing_evidence_ids"],
                "covered_evidence_ids": metrics["covered_evidence_ids"],
                "evidence_coverage": metrics["evidence_coverage"],
                "top_results": [
                    {
                        "rank": hit["rank"],
                        "chunk_id": hit["chunk_id"],
                        "section": hit["section"],
                        "pages": f"{hit['start_page']}-{hit['end_page']}",
                        "relevant": hit["relevant"],
                        "evidence_ids": hit["evidence_ids"],
                        "overlap_chars": hit["overlap_chars"],
                        "distance": hit["distance"],
                        "preview": hit["text"][:160].replace("\n", " "),
                    }
                    for hit in retrieval["retrieved_chunks"]
                ],
            }
        )
    return rows


def _summary_markdown(
    aggregate: Dict[str, float],
    comparison: Dict[str, Any],
    stats: Dict[str, Any],
    audit: Dict[str, Any],
    failures: List[Dict[str, Any]],
    config: Dict[str, Any],
    top_k: int = TOP_K,
) -> str:
    recall_key, precision_key, mrr_key = f"recall@{top_k}", f"precision@{top_k}", f"mrr@{top_k}"
    lines = [
        "# EXP-002 — Structure-aware chunking",
        "",
        f"- Experiment ID: `{EXPERIMENT_ID}`",
        f"- Embedding: `{config['embedding_model']}`",
        f"- Chunking: structure-aware, max {config['chunking']['max_chars']} characters",
        f"- Retrieval: dense-only, top-K = {config['retrieval']['top_k']}",
        f"- Dataset: 15 approved questions / 20 verified evidence units",
        "",
        "## Aggregate metrics",
        "",
        "| Metric | EXP-002 | EXP-001 v3 | Delta |",
        "|---|---:|---:|---:|",
    ]
    baseline = comparison.get("baseline_aggregate_metrics", {})
    delta = comparison.get("delta_aggregate", {})
    for metric in (recall_key, precision_key, mrr_key):
        lines.append(
            f"| {metric} | {aggregate.get(metric, 0.0):.4f} | "
            f"{baseline.get(metric, 0.0):.4f} | {delta.get(metric, 0.0):+.4f} |"
        )
    lines += [
        "",
        "## Chunk statistics",
        "",
        f"- Chunks: {stats['chunk_count']}",
        f"- Characters: min {stats['characters']['min']}, median {stats['characters']['median']}, "
        f"p90 {stats['characters']['p90']}, max {stats['characters']['max']}",
        f"- Oversized chunks: {len(stats['oversized_chunks'])}",
        f"- Small chunks (<{stats['small_chunk_threshold_chars']} chars): {len(stats['small_chunks'])}",
        f"- Multi-page chunks: {len(stats['multi_page_chunks'])}",
        f"- Clauses split into multiple parts: {len(stats['split_clauses'])}",
        f"- Source coverage: {audit['coverage_percent']:.4f}% non-whitespace, "
        f"overlaps={len(audit['overlaps'])}, integrity ok={audit['ok']}",
        "",
        "## Per-question comparison",
        "",
        f"| Question | Recall@{top_k} | Precision@{top_k} | MRR@{top_k} | ΔRecall | ΔMRR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for change in comparison.get("per_question", []):
        row = change["exp002"]
        d = change["delta"]
        lines.append(
            f"| {change['question_id']} | {row[recall_key]:.4f} | {row[precision_key]:.4f} | "
            f"{row[mrr_key]:.4f} | {d[recall_key]:+.4f} | {d[mrr_key]:+.4f} |"
        )
    lines += ["", "## Failure inspection", ""]
    if not failures:
        lines.append("No question below full recall.")
    for failure in failures:
        lines.append(
            f"### {failure['question_id']} — Recall@{top_k} {failure[f'recall@{top_k}']:.4f}, "
            f"MRR@{top_k} {failure[f'mrr@{top_k}']:.4f}"
        )
        lines.append("")
        lines.append(f"- Question: {failure['question']}")
        lines.append(f"- Required clauses: {', '.join(failure['required_clause_ids']) or 'n/a'}")
        lines.append(f"- Missing evidence: {', '.join(failure['missing_evidence_ids']) or 'none'}")
        lines.append("")
        for hit in failure["top_results"]:
            lines.append(
                f"  {hit['rank']}. `{hit['section'] or '-'}` p{hit['pages']} "
                f"relevant={hit['relevant']} overlap={hit['overlap_chars']} — {hit['preview'][:120]}"
            )
        lines.append("")
    lines += [
        "## Decision",
        "",
        "See `docs/ENGINEERING_LOG.md` for the accepted/rejected/inconclusive decision and the",
        "manual inspection that accompanies these metrics. The aggregate numbers alone are not",
        "sufficient evidence because the dataset has only 15 questions.",
        "",
    ]
    return "\n".join(lines)


def run(
    pdf_path: Path = PDF_PATH,
    questions_csv: Path = QUESTIONS_CSV,
    ground_truth_path: Path = GROUND_TRUTH_PATH,
    results_dir: Path = RESULTS_DIR,
    persist_dir: Path = PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
    chunk_size: int = CHUNK_SIZE,
    top_k: int = TOP_K,
    build_index: bool = True,
) -> Dict[str, Any]:
    document = extract_source_document(pdf_path)
    chunks = StructureAwareChunker(max_chars=chunk_size).chunk(document)
    audit = audit_source_coverage(document, chunks, max_chars=chunk_size)
    if not audit["ok"]:
        raise RuntimeError(f"Chunker integrity audit failed: {audit['issues']}")

    questions = load_questions(questions_csv)
    units, problems = load_exp001_evidence(ground_truth_path, document)
    if problems:
        raise RuntimeError(f"Ground-truth mapping failed for {len(problems)} evidence unit(s): {problems[0]}")
    question_ids = {question.question_id for question in questions}
    if {unit.question_id for unit in units} != question_ids:
        raise RuntimeError("Evidence manifest question ids do not match the approved question set")

    model = SentenceTransformer(EMBEDDING_MODEL)
    token_lengths = _token_lengths(chunks, getattr(model, "tokenizer", None))
    stats = _chunk_stats(chunks, token_lengths, chunk_size)

    config = {
        "experiment_id": EXPERIMENT_ID,
        "pdf_path": str(pdf_path),
        "questions_csv": str(questions_csv),
        "ground_truth": str(ground_truth_path),
        "embedding_model": EMBEDDING_MODEL,
        "chunking": {
            "strategy": "structure_aware",
            "max_chars": chunk_size,
            "paragraph_aware_splitting": True,
            "page_aware_heading_scan": True,
            "overlap": 0,
        },
        "retrieval": {"method": "dense_vector_similarity", "top_k": top_k},
        "vector_db": {
            "backend": "chroma",
            "collection_name": collection_name,
            "persist_directory": str(persist_dir),
        },
        "metrics": [f"Recall@{top_k}", f"Precision@{top_k}", f"MRR@{top_k}"],
        "fixed_variables": [
            "source_pdf",
            "evaluation_questions",
            "ground_truth_evidence_units",
            "embedding_model",
            "dense_retrieval",
            "top_k",
            "metric_definitions",
        ],
        "changed_variables": ["chunking_strategy"],
    }

    index = DenseIndex(
        DenseIndexConfig(
            collection_name=collection_name,
            persist_dir=persist_dir,
            embedding_model=EMBEDDING_MODEL,
        ),
        model=model,
    )
    build_stats = index.build(chunks, reset=True) if build_index else {"chunks": index.count()}

    retrieval_rows: List[Dict[str, Any]] = []
    per_question: List[Dict[str, Any]] = []
    latencies: List[float] = []
    for question in questions:
        evidence = evidence_for_question(units, question.question_id)
        if not evidence:
            raise RuntimeError(f"No verified evidence for {question.question_id}")

        started = time.perf_counter()
        hits = index.search(question.question, k=top_k)
        latencies.append((time.perf_counter() - started) * 1000.0)
        metrics = compute_retrieval_metrics(hits, evidence, k=top_k)

        retrieval_rows.append(
            {
                "question_id": question.question_id,
                "question": question.question,
                "expected_answer": question.expected_answer,
                "relevant_clause_ids": list(question.relevant_clause_ids),
                "relevant_source_pages": list(question.relevant_source_pages),
                "retrieved_chunks": [
                    {
                        "rank": hit.rank,
                        "chunk_id": hit.chunk_id,
                        "text": hit.text,
                        "section": hit.section,
                        "article": hit.article,
                        "heading": hit.heading,
                        "start_page": hit.start_page,
                        "end_page": hit.end_page,
                        "start_offset": hit.start_offset,
                        "end_offset": hit.end_offset,
                        "distance": hit.distance,
                        "relevant": any(
                            hit.chunk_id == overlap["chunk_id"] for overlap in metrics["evidence_overlaps"]
                        ),
                        "evidence_ids": next(
                            (
                                overlap["evidence_ids"]
                                for overlap in metrics["evidence_overlaps"]
                                if overlap["chunk_id"] == hit.chunk_id
                            ),
                            [],
                        ),
                        "overlap_chars": next(
                            (
                                overlap["overlap_chars"]
                                for overlap in metrics["evidence_overlaps"]
                                if overlap["chunk_id"] == hit.chunk_id
                            ),
                            0,
                        ),
                    }
                    for hit in hits
                ],
                **metrics,
            }
        )
        per_question.append(
            {
                "question_id": question.question_id,
                f"recall@{top_k}": metrics[f"recall@{top_k}"],
                f"precision@{top_k}": metrics[f"precision@{top_k}"],
                f"mrr@{top_k}": metrics[f"mrr@{top_k}"],
                "required_evidence_ids": metrics["required_evidence_ids"],
                "missing_evidence_ids": metrics["missing_evidence_ids"],
                "covered_evidence_ids": metrics["covered_evidence_ids"],
                "evidence_coverage": metrics["evidence_coverage"],
            }
        )

    aggregate = aggregate_metrics(per_question, k=top_k)
    baseline = _baseline()
    comparison = _comparison(baseline, aggregate, per_question, top_k=top_k)
    failures = _failure_analysis(questions, per_question, retrieval_rows, top_k=top_k)
    coverage_rows = evidence_coverage_summary(per_question, questions)

    latency_summary = {
        "mean_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        "median_ms": round(float(statistics.median(latencies)), 2) if latencies else 0.0,
        "p90_ms": _percentile(latencies, 0.90),
        "max_ms": round(max(latencies), 2) if latencies else 0.0,
    }

    results_dir = Path(results_dir)
    _write_json(results_dir / "config.json", config)
    _write_json(results_dir / "integrity_audit.json", audit)
    _write_json(results_dir / "chunk_stats.json", stats)
    _write_json(
        results_dir / "chunks.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "config": config,
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.source,
                    "article": chunk.article,
                    "section": chunk.section,
                    "section_hierarchy": list(chunk.section_hierarchy) if chunk.section_hierarchy else None,
                    "heading": chunk.heading,
                    "start_page": chunk.start_page,
                    "end_page": chunk.end_page,
                    "start_offset": chunk.start_offset,
                    "end_offset": chunk.end_offset,
                    "start_page_char": chunk.start_page_char,
                    "end_page_char": chunk.end_page_char,
                    "chunk_sequence": chunk.chunk_sequence,
                    "part": chunk.part,
                    "chunking_strategy": chunk.chunking_strategy,
                    "characters": len(chunk.text),
                    "tokens": token_lengths[index] if index < len(token_lengths) else None,
                    "text": chunk.text,
                }
                for index, chunk in enumerate(chunks)
            ],
        },
    )
    _write_json(
        results_dir / "evidence.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "question_count": len(questions),
            "evidence_unit_count": len(units),
            "units": [unit.to_dict() for unit in units],
        },
    )
    _write_json(
        results_dir / "retrieval_results.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "config": config,
            "build_stats": build_stats,
            "latency": latency_summary,
            "aggregate_metrics": aggregate,
            "per_question_metrics": per_question,
            "retrieval_results": retrieval_rows,
        },
    )
    _write_json(results_dir / "evidence_coverage.json", coverage_rows)
    _write_json(results_dir / "comparison.json", comparison)
    _write_json(results_dir / "failure_analysis.json", failures)

    with open(results_dir / "per_question_metrics.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "question_id",
                f"recall@{top_k}",
                f"precision@{top_k}",
                f"mrr@{top_k}",
                "required_units",
                "covered_units",
                "missing_units",
            ]
        )
        for row in coverage_rows:
            metrics = next(item for item in per_question if item["question_id"] == row["question_id"])
            writer.writerow(
                [
                    row["question_id"],
                    metrics[f"recall@{top_k}"],
                    metrics[f"precision@{top_k}"],
                    metrics[f"mrr@{top_k}"],
                    row["required_units"],
                    row["covered_units"],
                    ";".join(row["missing_units"]),
                ]
            )

    (results_dir / "summary.md").write_text(
        _summary_markdown(aggregate, comparison, stats, audit, failures, config, top_k=top_k),
        encoding="utf-8",
    )

    return {
        "experiment_id": EXPERIMENT_ID,
        "config": config,
        "aggregate_metrics": aggregate,
        "per_question_metrics": per_question,
        "chunk_stats": stats,
        "integrity_audit": audit,
        "comparison": comparison,
        "failure_analysis": failures,
        "latency": latency_summary,
        "build_stats": build_stats,
        "results_dir": str(results_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP-002 structure-aware chunking retrieval experiment")
    parser.add_argument("--pdf", type=Path, default=PDF_PATH)
    parser.add_argument("--questions-csv", type=Path, default=QUESTIONS_CSV)
    parser.add_argument("--ground-truth", type=Path, default=GROUND_TRUTH_PATH)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--persist-dir", type=Path, default=PERSIST_DIR)
    parser.add_argument("--collection", default=COLLECTION_NAME)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()

    result = run(
        pdf_path=args.pdf,
        questions_csv=args.questions_csv,
        ground_truth_path=args.ground_truth,
        results_dir=args.results_dir,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        chunk_size=args.chunk_size,
        top_k=args.top_k,
    )
    print(json.dumps(result["aggregate_metrics"], indent=2))
    print(f"Results: {result['results_dir']}")


if __name__ == "__main__":
    main()
