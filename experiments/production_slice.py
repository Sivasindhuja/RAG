"""Production slice: accepted fixed-character retrieval + grounded answers.

This is not a new retrieval experiment.  It productionises the **accepted
EXP-001 baseline** in the shared core, then evaluates the grounded answer layer:

- every answer sentence is verbatim source text with exact global offsets;
- every citation points at a chunk, page and source span;
- unsupported questions must abstain.

Run:
    python -m experiments.production_slice
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from sentence_transformers import SentenceTransformer

from src.rag_core import (
    ABSTAINED,
    ANSWERED,
    DEFAULT_EMBEDDING_MODEL,
    AbstentionPolicy,
    DenseIndex,
    DenseIndexConfig,
    FixedCharacterChunker,
    GroundedPipeline,
    audit_source_coverage,
    compute_retrieval_metrics,
    evidence_for_question,
    extract_source_document,
    load_exp001_evidence,
    load_questions,
)

ROOT = Path(__file__).resolve().parent.parent
SLICE_ID = "PROD-001"
PDF_PATH = ROOT / "data" / "satcom-ngp.pdf"
QUESTIONS_CSV = ROOT / "experiments" / "exp001_questions.csv"
GROUND_TRUTH_PATH = ROOT / "experiments" / "exp001_ground_truth.json"
UNANSWERABLE_CSV = ROOT / "experiments" / "unanswerable_questions.csv"
BASELINE_RESULTS = ROOT / "experiments" / "results" / "exp001_v3" / "retrieval_results.json"
RESULTS_DIR = ROOT / "experiments" / "results" / "production_slice"
PERSIST_DIR = ROOT / "chroma_db_production"
COLLECTION_NAME = "satcom_ngp_production_fixed_char"
EMBEDDING_MODEL = DEFAULT_EMBEDDING_MODEL
CHUNK_SIZE = 1000
OVERLAP = 200
TOP_K = 5


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_unanswerable(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return [
            {
                "question_id": (row.get("question_id") or "").strip(),
                "question": (row.get("question") or "").strip(),
                "split": (row.get("split") or "calibration").strip(),
                "category": (row.get("category") or "").strip(),
            }
            for row in csv.DictReader(handle)
        ]


def _overlapping_evidence(start: int, end: int, units: Sequence[Any]) -> List[str]:
    return [unit.evidence_id for unit in units if unit.start_offset < end and start < unit.end_offset]


def _evaluate_answer(answer, units) -> Dict[str, Any]:
    statement_rows = []
    supported_statements = 0
    for statement in answer.statements:
        evidence_ids = _overlapping_evidence(statement.start_offset, statement.end_offset, units)
        if evidence_ids:
            supported_statements += 1
        statement_rows.append(
            {
                **statement.to_dict(),
                "evidence_ids": evidence_ids,
                "supported": bool(evidence_ids),
            }
        )

    citation_rows = []
    supported_citations = 0
    for citation in answer.citations:
        evidence_ids = _overlapping_evidence(citation.start_offset, citation.end_offset, units)
        if evidence_ids:
            supported_citations += 1
        citation_rows.append({**citation.to_dict(), "evidence_ids": evidence_ids, "supported": bool(evidence_ids)})

    return {
        "status": answer.status,
        "reason": answer.reason,
        "text": answer.text,
        "token_support": round(answer.token_support, 4),
        "best_distance": answer.best_distance,
        "statements": statement_rows,
        "citations": citation_rows,
        "statement_count": len(statement_rows),
        "supported_statement_count": supported_statements,
        "citation_count": len(citation_rows),
        "supported_citation_count": supported_citations,
        "answer_supported": supported_statements > 0,
    }


def _baseline_aggregate() -> Dict[str, float]:
    if not BASELINE_RESULTS.exists():
        return {}
    payload = json.loads(BASELINE_RESULTS.read_text(encoding="utf-8"))
    return payload.get("aggregate_metrics", {})


def _summary(
    config: Dict[str, Any],
    aggregate: Dict[str, float],
    baseline: Dict[str, float],
    audit: Dict[str, Any],
    answer_summary: Dict[str, Any],
    abstention_summary: Dict[str, Any],
) -> str:
    top_k = config["retrieval"]["top_k"]
    lines = [
        "# PROD-001 — Accepted baseline production slice",
        "",
        f"- Chunking: fixed-character {config['chunking']['chunk_size']}/"
        f"{config['chunking']['overlap']} (accepted EXP-001 strategy)",
        f"- Embedding: `{config['embedding_model']}`",
        f"- Retrieval: dense-only, top-{top_k}",
        f"- Answer mode: {config['answering']['mode']}",
        f"- Abstention policy: `{json.dumps(config['answering']['policy'])}`",
        "",
        "## Retrieval (must match the accepted baseline)",
        "",
        f"| Metric | Slice | EXP-001 v3 | Delta |",
        "|---|---:|---:|---:|",
    ]
    for metric in (f"recall@{top_k}", f"precision@{top_k}", f"mrr@{top_k}"):
        lines.append(
            f"| {metric} | {aggregate.get(metric, 0.0):.4f} | {baseline.get(metric, 0.0):.4f} | "
            f"{aggregate.get(metric, 0.0) - baseline.get(metric, 0.0):+.4f} |"
        )
    lines += [
        "",
        "## Document integrity",
        "",
        f"- Chunks: {config['chunking']['chunk_count']}",
        f"- Non-whitespace coverage: {audit['coverage_percent']:.4f}%",
        f"- Intentional overlap characters: {audit['overlap_chars']}",
        f"- Fidelity errors: {len(audit['fidelity_errors'])}; duplicate IDs: {len(audit['duplicate_ids'])}; "
        f"page errors: {len(audit['page_mapping_errors'])}",
        f"- Audit ok (overlap allowed): {audit['ok']}",
        "",
        "## Answers and citations (15 in-corpus questions)",
        "",
        "| Measure | Value |",
        "|---|---:|",
        f"| Answered | {answer_summary['answered']} / {answer_summary['question_count']} |",
        f"| Abstained | {answer_summary['abstained']} / {answer_summary['question_count']} |",
        f"| Answers with >=1 evidence-supported statement | {answer_summary['supported_answers']} |",
        f"| Statement support rate | {answer_summary['statement_support_rate']:.4f} |",
        f"| Citation support rate | {answer_summary['citation_support_rate']:.4f} |",
        f"| Mean statements per answered question | {answer_summary['mean_statements']:.2f} |",
        f"| Mean citations per answered question | {answer_summary['mean_citations']:.2f} |",
        "",
        "## Abstention smoke test",
        "",
        "| Split | Questions | Abstained | Abstention rate |",
        "|---|---:|---:|---:|",
    ]
    for split, row in abstention_summary["by_split"].items():
        lines.append(
            f"| {split} | {row['count']} | {row['abstained']} | {row['abstention_rate']:.4f} |"
        )
    lines += [
        "",
        f"Overall abstention rate: **{abstention_summary['overall']['abstention_rate']:.4f}** "
        f"({abstention_summary['overall']['abstained']}/{abstention_summary['overall']['count']}).",
        "",
        "## Limitations",
        "",
        "- The abstention thresholds were calibrated on the 8 `calibration` questions; the",
        "  4 `heldout` questions are a small, non-representative check, not a validation set.",
        "- Answers are extractive (verbatim sentences). Fluency and synthesis are not claimed.",
        "- Sentence-level support is measured against the 20 verified evidence units, not",
        "  against an independent entailment model.",
        "- The generator hook is not evaluated; only the extractive path is validated.",
        "",
        "See `docs/PRODUCTION_SLICE.md` and `docs/ENGINEERING_LOG.md` for the decision record.",
        "",
    ]
    return "\n".join(lines)


def run(
    pdf_path: Path = PDF_PATH,
    questions_csv: Path = QUESTIONS_CSV,
    ground_truth_path: Path = GROUND_TRUTH_PATH,
    unanswerable_csv: Path = UNANSWERABLE_CSV,
    results_dir: Path = RESULTS_DIR,
    persist_dir: Path = PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP,
    top_k: int = TOP_K,
    build_index: bool = True,
    policy: Optional[AbstentionPolicy] = None,
) -> Dict[str, Any]:
    policy = policy or AbstentionPolicy()
    document = extract_source_document(pdf_path)
    chunks = FixedCharacterChunker(chunk_size=chunk_size, overlap=overlap).chunk(document)
    audit = audit_source_coverage(document, chunks, max_chars=chunk_size, allow_overlap=True)
    if not audit["ok"]:
        raise RuntimeError(f"Fixed-character chunk audit failed: {audit['issues']}")

    questions = load_questions(questions_csv)
    units, problems = load_exp001_evidence(ground_truth_path, document)
    if problems:
        raise RuntimeError(f"Ground-truth mapping failed: {problems[0]}")
    if {unit.question_id for unit in units} != {question.question_id for question in questions}:
        raise RuntimeError("Evidence manifest question ids do not match the approved question set")

    model = SentenceTransformer(EMBEDDING_MODEL)
    index = DenseIndex(
        DenseIndexConfig(
            collection_name=collection_name,
            persist_dir=persist_dir,
            embedding_model=EMBEDDING_MODEL,
        ),
        model=model,
    )
    build_stats = index.build(chunks, reset=True) if build_index else {"chunks": index.count()}

    pipeline = GroundedPipeline(index=index, top_k=top_k, policy=policy)

    retrieval_rows: List[Dict[str, Any]] = []
    per_question_metrics: List[Dict[str, Any]] = []
    answer_rows: List[Dict[str, Any]] = []
    latencies: List[float] = []

    for question in questions:
        evidence = evidence_for_question(units, question.question_id)
        if not evidence:
            raise RuntimeError(f"No verified evidence for {question.question_id}")

        hits = index.search(question.question, k=top_k)
        metrics = compute_retrieval_metrics(hits, evidence, k=top_k)
        per_question_metrics.append(
            {
                "question_id": question.question_id,
                f"recall@{top_k}": metrics[f"recall@{top_k}"],
                f"precision@{top_k}": metrics[f"precision@{top_k}"],
                f"mrr@{top_k}": metrics[f"mrr@{top_k}"],
                "required_evidence_ids": metrics["required_evidence_ids"],
                "covered_evidence_ids": metrics["covered_evidence_ids"],
                "missing_evidence_ids": metrics["missing_evidence_ids"],
            }
        )
        retrieval_rows.append(
            {
                "question_id": question.question_id,
                "question": question.question,
                "relevant_clause_ids": list(question.relevant_clause_ids),
                "retrieved_chunks": [
                    {
                        "rank": hit.rank,
                        "chunk_id": hit.chunk_id,
                        "section": hit.section,
                        "start_page": hit.start_page,
                        "end_page": hit.end_page,
                        "start_offset": hit.start_offset,
                        "end_offset": hit.end_offset,
                        "distance": hit.distance,
                        "evidence_ids": _overlapping_evidence(hit.start_offset, hit.end_offset, evidence),
                        "preview": hit.text[:200],
                    }
                    for hit in hits
                ],
                **{key: metrics[key] for key in metrics if key.startswith(("recall@", "precision@", "mrr@"))},
            }
        )

        started = time.perf_counter()
        answer = pipeline.answer(question.question)
        latencies.append((time.perf_counter() - started) * 1000.0)
        answer_rows.append(
            {
                "question_id": question.question_id,
                "question": question.question,
                "expected_answer": question.expected_answer,
                "relevant_clause_ids": list(question.relevant_clause_ids),
                **_evaluate_answer(answer, evidence),
            }
        )

    count = len(per_question_metrics) or 1
    aggregate = {
        f"recall@{top_k}": round(sum(row[f"recall@{top_k}"] for row in per_question_metrics) / count, 4),
        f"precision@{top_k}": round(sum(row[f"precision@{top_k}"] for row in per_question_metrics) / count, 4),
        f"mrr@{top_k}": round(sum(row[f"mrr@{top_k}"] for row in per_question_metrics) / count, 4),
    }
    baseline = _baseline_aggregate()

    answered = [row for row in answer_rows if row["status"] == ANSWERED]
    statement_count = sum(row["statement_count"] for row in answered)
    supported_statement_count = sum(row["supported_statement_count"] for row in answered)
    citation_count = sum(row["citation_count"] for row in answered)
    supported_citation_count = sum(row["supported_citation_count"] for row in answered)
    answer_summary = {
        "question_count": len(answer_rows),
        "answered": len(answered),
        "abstained": len(answer_rows) - len(answered),
        "supported_answers": sum(1 for row in answered if row["answer_supported"]),
        "statement_count": statement_count,
        "supported_statement_count": supported_statement_count,
        "statement_support_rate": (
            supported_statement_count / statement_count if statement_count else 0.0
        ),
        "citation_count": citation_count,
        "supported_citation_count": supported_citation_count,
        "citation_support_rate": (
            supported_citation_count / citation_count if citation_count else 0.0
        ),
        "mean_statements": (statement_count / len(answered)) if answered else 0.0,
        "mean_citations": (citation_count / len(answered)) if answered else 0.0,
    }

    unanswerable = _load_unanswerable(unanswerable_csv)
    abstention_rows = []
    for item in unanswerable:
        answer = pipeline.answer(item["question"])
        abstention_rows.append(
            {
                **item,
                "status": answer.status,
                "abstained": answer.status == ABSTAINED,
                "reason": answer.reason,
                "best_distance": answer.best_distance,
                "token_support": round(answer.token_support, 4),
                "top_sections": [hit.section for hit in answer.retrieved[:3]],
            }
        )

    by_split: Dict[str, Dict[str, Any]] = {}
    for split in sorted({row["split"] for row in abstention_rows}):
        rows = [row for row in abstention_rows if row["split"] == split]
        abstained = sum(1 for row in rows if row["abstained"])
        by_split[split] = {
            "count": len(rows),
            "abstained": abstained,
            "abstention_rate": round(abstained / len(rows), 4) if rows else 0.0,
        }
    total_abstained = sum(1 for row in abstention_rows if row["abstained"])
    abstention_summary = {
        "by_split": by_split,
        "overall": {
            "count": len(abstention_rows),
            "abstained": total_abstained,
            "abstention_rate": round(total_abstained / len(abstention_rows), 4) if abstention_rows else 0.0,
        },
    }

    config = {
        "slice_id": SLICE_ID,
        "pdf_path": str(pdf_path),
        "questions_csv": str(questions_csv),
        "ground_truth": str(ground_truth_path),
        "unanswerable_csv": str(unanswerable_csv),
        "embedding_model": EMBEDDING_MODEL,
        "chunking": {
            "strategy": "fixed_character",
            "chunk_size": chunk_size,
            "overlap": overlap,
            "step": chunk_size - overlap,
            "per_page": True,
            "chunk_count": len(chunks),
            "reproduces": "EXP-001 v3 accepted baseline",
        },
        "retrieval": {"method": "dense_vector_similarity", "top_k": top_k},
        "answering": {
            "mode": "extractive_verbatim",
            "max_answer_sentences": pipeline.max_answer_sentences,
            "max_answer_chunks": pipeline.max_answer_chunks,
            "max_quote_chars": pipeline.max_quote_chars,
            "policy": asdict(policy),
        },
        "vector_db": {
            "backend": "chroma",
            "collection_name": collection_name,
            "persist_directory": str(persist_dir),
        },
    }

    results_dir = Path(results_dir)
    _write_json(results_dir / "config.json", config)
    _write_json(results_dir / "integrity_audit.json", audit)
    _write_json(
        results_dir / "chunks.json",
        {
            "slice_id": SLICE_ID,
            "config": config,
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "article": chunk.article,
                    "section": chunk.section,
                    "heading": chunk.heading,
                    "start_page": chunk.start_page,
                    "end_page": chunk.end_page,
                    "start_offset": chunk.start_offset,
                    "end_offset": chunk.end_offset,
                    "characters": len(chunk.text),
                    "text": chunk.text,
                }
                for chunk in chunks
            ],
        },
    )
    _write_json(
        results_dir / "retrieval_results.json",
        {
            "slice_id": SLICE_ID,
            "config": config,
            "build_stats": build_stats,
            "aggregate_metrics": aggregate,
            "baseline_aggregate_metrics": baseline,
            "per_question_metrics": per_question_metrics,
            "retrieval_results": retrieval_rows,
        },
    )
    _write_json(
        results_dir / "answers.json",
        {
            "slice_id": SLICE_ID,
            "summary": answer_summary,
            "answers": answer_rows,
        },
    )
    _write_json(
        results_dir / "abstention_results.json",
        {
            "slice_id": SLICE_ID,
            "summary": abstention_summary,
            "results": abstention_rows,
        },
    )

    with open(results_dir / "per_question_metrics.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["question_id", f"recall@{top_k}", f"precision@{top_k}", f"mrr@{top_k}", "answered", "answer_supported"]
        )
        answers_by_id = {row["question_id"]: row for row in answer_rows}
        for row in per_question_metrics:
            answer = answers_by_id[row["question_id"]]
            writer.writerow(
                [
                    row["question_id"],
                    row[f"recall@{top_k}"],
                    row[f"precision@{top_k}"],
                    row[f"mrr@{top_k}"],
                    answer["status"] == ANSWERED,
                    answer["answer_supported"],
                ]
            )

    (results_dir / "summary.md").write_text(
        _summary(config, aggregate, baseline, audit, answer_summary, abstention_summary),
        encoding="utf-8",
    )

    latency_summary = {
        "mean_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        "median_ms": round(float(statistics.median(latencies)), 2) if latencies else 0.0,
        "max_ms": round(max(latencies), 2) if latencies else 0.0,
    }

    return {
        "slice_id": SLICE_ID,
        "aggregate_metrics": aggregate,
        "baseline_aggregate_metrics": baseline,
        "answer_summary": answer_summary,
        "abstention_summary": abstention_summary,
        "integrity_audit": audit,
        "latency": latency_summary,
        "build_stats": build_stats,
        "results_dir": str(results_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Production slice evaluation (accepted baseline + grounded answers)")
    parser.add_argument("--pdf", type=Path, default=PDF_PATH)
    parser.add_argument("--questions-csv", type=Path, default=QUESTIONS_CSV)
    parser.add_argument("--ground-truth", type=Path, default=GROUND_TRUTH_PATH)
    parser.add_argument("--unanswerable-csv", type=Path, default=UNANSWERABLE_CSV)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--persist-dir", type=Path, default=PERSIST_DIR)
    parser.add_argument("--collection", default=COLLECTION_NAME)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=OVERLAP)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()

    result = run(
        pdf_path=args.pdf,
        questions_csv=args.questions_csv,
        ground_truth_path=args.ground_truth,
        unanswerable_csv=args.unanswerable_csv,
        results_dir=args.results_dir,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        top_k=args.top_k,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
