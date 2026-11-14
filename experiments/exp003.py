"""EXP-003: controlled answer-layer selection experiment.

Hypothesis
----------
When the strongest lexical sentence is a numbered clause, selecting sibling
clauses under the same parent (within 50% of the strongest score) improves
multi-clause evidence coverage.  This targets the observed Q13 failure without
changing retrieval.

Controlled variable
-------------------
Answer sentence selection only: ``sibling_score_ratio=None`` (PROD-001 lexical
baseline) versus ``sibling_score_ratio=0.5`` (sibling-clause arm).

Held fixed
----------
PDF, 15 questions, 20 evidence units, fixed-character 1000/200 chunks, MiniLM
embeddings, dense top-5 retrieval, answer length cap, citation construction,
abstention policy and support metrics.  Both arms receive the *same* precomputed
hits via ``GroundedPipeline.answer_from_hits``.

Pre-registered decision rule
----------------------------
Accept only if all hold:

1. statement support rate improves (absolute delta >= +0.02);
2. supported answers do not decrease;
3. citation support rate does not decrease by more than 0.02;
4. abstention on the 12 out-of-corpus questions is unchanged (12/12).

Otherwise reject.  Results are reported even when unfavourable.

Run:
    python -m experiments.exp003
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

from experiments.production_slice import (
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    GROUND_TRUTH_PATH,
    OVERLAP,
    PDF_PATH,
    QUESTIONS_CSV,
    TOP_K,
    UNANSWERABLE_CSV,
    _evaluate_answer,
    _load_unanswerable,
    _overlapping_evidence,
    _write_json,
)
from src.rag_core import (
    ABSTAINED,
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
EXPERIMENT_ID = "EXP-003"
RESULTS_DIR = ROOT / "experiments" / "results" / "exp003"
PERSIST_DIR = ROOT / "chroma_db_exp003"
COLLECTION_NAME = "satcom_ngp_exp003_answer_selection"
BASELINE_RATIO: Optional[float] = None
SIBLING_RATIO = 0.5


def _answer_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    answered = [row for row in rows if row["status"] == "answered"]
    statement_count = sum(row["statement_count"] for row in answered)
    supported_statement_count = sum(row["supported_statement_count"] for row in answered)
    citation_count = sum(row["citation_count"] for row in answered)
    supported_citation_count = sum(row["supported_citation_count"] for row in answered)
    return {
        "question_count": len(rows),
        "answered": len(answered),
        "abstained": len(rows) - len(answered),
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


def _covered_evidence(row: Dict[str, Any]) -> List[str]:
    covered: set[str] = set()
    for statement in row["statements"]:
        covered.update(statement["evidence_ids"])
    return sorted(covered)


def _abstention_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_split: Dict[str, Dict[str, Any]] = {}
    for split in sorted({row["split"] for row in rows}):
        split_rows = [row for row in rows if row["split"] == split]
        abstained = sum(1 for row in split_rows if row["abstained"])
        by_split[split] = {
            "count": len(split_rows),
            "abstained": abstained,
            "abstention_rate": round(abstained / len(split_rows), 4) if split_rows else 0.0,
        }
    total = sum(1 for row in rows if row["abstained"])
    return {
        "by_split": by_split,
        "overall": {
            "count": len(rows),
            "abstained": total,
            "abstention_rate": round(total / len(rows), 4) if rows else 0.0,
        },
    }


def _decision(
    baseline: Dict[str, Any],
    candidate: Dict[str, Any],
    baseline_abstention: Dict[str, Any],
    candidate_abstention: Dict[str, Any],
) -> Dict[str, Any]:
    statement_delta = candidate["statement_support_rate"] - baseline["statement_support_rate"]
    citation_delta = candidate["citation_support_rate"] - baseline["citation_support_rate"]
    checks = {
        "statement_support_delta_ge_0.02": statement_delta >= 0.02,
        "supported_answers_not_lower": candidate["supported_answers"] >= baseline["supported_answers"],
        "citation_support_delta_ge_minus_0.02": citation_delta >= -0.02,
        "abstention_unchanged": (
            candidate_abstention["overall"]["abstained"]
            == baseline_abstention["overall"]["abstained"]
            == baseline_abstention["overall"]["count"]
        ),
    }
    return {
        "accepted": all(checks.values()),
        "checks": checks,
        "statement_support_delta": round(statement_delta, 4),
        "citation_support_delta": round(citation_delta, 4),
        "supported_answer_delta": candidate["supported_answers"] - baseline["supported_answers"],
    }


def _summary(
    config: Dict[str, Any],
    baseline: Dict[str, Any],
    candidate: Dict[str, Any],
    comparison: Sequence[Dict[str, Any]],
    decision: Dict[str, Any],
    baseline_abstention: Dict[str, Any],
    candidate_abstention: Dict[str, Any],
) -> str:
    lines = [
        "# EXP-003 — Sibling-clause answer selection",
        "",
        f"- Experiment id: `{EXPERIMENT_ID}`",
        f"- Changed variable: `sibling_score_ratio` (baseline "
        f"`{config['arms']['baseline']['sibling_score_ratio']}` → candidate "
        f"`{config['arms']['candidate']['sibling_score_ratio']}`)",
        "- Retrieval: fixed-character 1000/200, MiniLM, dense top-5 (unchanged)",
        "- Decision: **" + ("ACCEPTED" if decision["accepted"] else "REJECTED") + "**",
        "",
        "## Primary metrics",
        "",
        "| Measure | Baseline | Sibling-clause | Delta |",
        "|---|---:|---:|---:|",
        f"| Supported answers | {baseline['supported_answers']} | {candidate['supported_answers']} | "
        f"{candidate['supported_answers'] - baseline['supported_answers']:+d} |",
        f"| Supported statements | {baseline['supported_statement_count']}/{baseline['statement_count']} | "
        f"{candidate['supported_statement_count']}/{candidate['statement_count']} | "
        f"{candidate['supported_statement_count'] - baseline['supported_statement_count']:+d} |",
        f"| Statement support rate | {baseline['statement_support_rate']:.4f} | "
        f"{candidate['statement_support_rate']:.4f} | {decision['statement_support_delta']:+.4f} |",
        f"| Supported citations | {baseline['supported_citation_count']}/{baseline['citation_count']} | "
        f"{candidate['supported_citation_count']}/{candidate['citation_count']} | "
        f"{candidate['supported_citation_count'] - baseline['supported_citation_count']:+d} |",
        f"| Citation support rate | {baseline['citation_support_rate']:.4f} | "
        f"{candidate['citation_support_rate']:.4f} | {decision['citation_support_delta']:+.4f} |",
        f"| Mean statements | {baseline['mean_statements']:.2f} | {candidate['mean_statements']:.2f} | "
        f"{candidate['mean_statements'] - baseline['mean_statements']:+.2f} |",
        f"| Mean citations | {baseline['mean_citations']:.2f} | {candidate['mean_citations']:.2f} | "
        f"{candidate['mean_citations'] - baseline['mean_citations']:+.2f} |",
        "",
        "## Decision checks",
        "",
        "| Check | Result |",
        "|---|---|",
    ]
    for name, passed in decision["checks"].items():
        lines.append(f"| `{name}` | {'PASS' if passed else 'FAIL'} |")
    lines += [
        "",
        "## Per-question statement support",
        "",
        "| ID | Recall@5 | Baseline | Sibling | Delta | Missing evidence (sibling) |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in comparison:
        missing = ", ".join(row["missing_evidence_ids"]) or "—"
        lines.append(
            f"| {row['question_id']} | {row['recall@5']:.4f} | "
            f"{row['baseline_supported_statements']}/{row['baseline_statements']} | "
            f"{row['candidate_supported_statements']}/{row['candidate_statements']} | "
            f"{row['supported_statement_delta']:+d} | {missing} |"
        )
    lines += [
        "",
        "## Abstention smoke test",
        "",
        "| Split | Questions | Baseline abstained | Sibling abstained |",
        "|---|---:|---:|---:|",
    ]
    for split in baseline_abstention["by_split"]:
        baseline_row = baseline_abstention["by_split"][split]
        candidate_row = candidate_abstention["by_split"][split]
        lines.append(
            f"| {split} | {baseline_row['count']} | {baseline_row['abstained']} | "
            f"{candidate_row['abstained']} |"
        )
    lines += [
        "",
        f"Overall: baseline {baseline_abstention['overall']['abstained']}/"
        f"{baseline_abstention['overall']['count']}; sibling "
        f"{candidate_abstention['overall']['abstained']}/"
        f"{candidate_abstention['overall']['count']}.",
        "",
        "## Interpretation",
        "",
        "- The rule only activates when the top-ranked sentence starts with a numbered",
        "  clause.  It promotes same-parent clauses scoring at least 50% of the top",
        "  sentence, then fills any remaining slots from the global ranking.",
        "- Q13 improves because `4.3.2` and `4.3.3` are retrieved and now selected",
        "  alongside `4.3.4`.",
        "- Q05 and Q08 are retrieval failures; answer-layer selection cannot recover",
        "  evidence that is not in the top-5.",
        "- Q10 remains unsolved: its required evidence sits inside a `3.6.8` body",
        "  sentence that does not begin with the clause ID, so sibling detection does",
        "  not find it.",
        "- The 15-question set is small.  The 0.5 ratio is a natural threshold but was",
        "  chosen during exploratory sweeps; this experiment is a controlled check, not",
        "  independent confirmation.",
        "",
        "See `docs/EXP-003.md` and `docs/ENGINEERING_LOG.md` for the decision record.",
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

    baseline_pipeline = GroundedPipeline(
        index=index,
        top_k=top_k,
        policy=policy,
        sibling_score_ratio=BASELINE_RATIO,
    )
    candidate_pipeline = GroundedPipeline(
        index=index,
        top_k=top_k,
        policy=policy,
        sibling_score_ratio=SIBLING_RATIO,
    )

    retrieval_rows: List[Dict[str, Any]] = []
    baseline_rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    comparison: List[Dict[str, Any]] = []
    baseline_latencies: List[float] = []
    candidate_latencies: List[float] = []

    for question in questions:
        evidence = evidence_for_question(units, question.question_id)
        if not evidence:
            raise RuntimeError(f"No verified evidence for {question.question_id}")

        hits = index.search(question.question, k=top_k)
        metrics = compute_retrieval_metrics(hits, evidence, k=top_k)
        retrieval_rows.append(
            {
                "question_id": question.question_id,
                "question": question.question,
                "retrieved_chunks": [
                    {
                        "rank": hit.rank,
                        "chunk_id": hit.chunk_id,
                        "section": hit.section,
                        "start_offset": hit.start_offset,
                        "end_offset": hit.end_offset,
                        "distance": hit.distance,
                        "evidence_ids": _overlapping_evidence(hit.start_offset, hit.end_offset, evidence),
                    }
                    for hit in hits
                ],
                **{key: metrics[key] for key in metrics if key.startswith(("recall@", "precision@", "mrr@"))},
            }
        )

        started = time.perf_counter()
        baseline_answer = baseline_pipeline.answer_from_hits(question.question, hits)
        baseline_latencies.append((time.perf_counter() - started) * 1000.0)
        baseline_row = {
            "question_id": question.question_id,
            "question": question.question,
            "expected_answer": question.expected_answer,
            "relevant_clause_ids": list(question.relevant_clause_ids),
            **_evaluate_answer(baseline_answer, evidence),
        }
        baseline_rows.append(baseline_row)

        started = time.perf_counter()
        candidate_answer = candidate_pipeline.answer_from_hits(question.question, hits)
        candidate_latencies.append((time.perf_counter() - started) * 1000.0)
        candidate_row = {
            "question_id": question.question_id,
            "question": question.question,
            "expected_answer": question.expected_answer,
            "relevant_clause_ids": list(question.relevant_clause_ids),
            **_evaluate_answer(candidate_answer, evidence),
        }
        candidate_rows.append(candidate_row)

        comparison.append(
            {
                "question_id": question.question_id,
                f"recall@{top_k}": metrics[f"recall@{top_k}"],
                "baseline_statements": baseline_row["statement_count"],
                "baseline_supported_statements": baseline_row["supported_statement_count"],
                "candidate_statements": candidate_row["statement_count"],
                "candidate_supported_statements": candidate_row["supported_statement_count"],
                "supported_statement_delta": (
                    candidate_row["supported_statement_count"] - baseline_row["supported_statement_count"]
                ),
                "baseline_supported_citations": baseline_row["supported_citation_count"],
                "candidate_supported_citations": candidate_row["supported_citation_count"],
                "baseline_covered_evidence_ids": _covered_evidence(baseline_row),
                "candidate_covered_evidence_ids": _covered_evidence(candidate_row),
                "missing_evidence_ids": list(metrics["missing_evidence_ids"]),
            }
        )

    baseline_summary = _answer_summary(baseline_rows)
    candidate_summary = _answer_summary(candidate_rows)

    unanswerable = _load_unanswerable(unanswerable_csv)
    abstention_rows: Dict[str, List[Dict[str, Any]]] = {"baseline": [], "candidate": []}
    for arm, pipeline in (("baseline", baseline_pipeline), ("candidate", candidate_pipeline)):
        for item in unanswerable:
            answer = pipeline.answer(item["question"])
            abstention_rows[arm].append(
                {
                    **item,
                    "status": answer.status,
                    "abstained": answer.status == ABSTAINED,
                    "reason": answer.reason,
                    "best_distance": answer.best_distance,
                    "token_support": round(answer.token_support, 4),
                }
            )
    baseline_abstention = _abstention_summary(abstention_rows["baseline"])
    candidate_abstention = _abstention_summary(abstention_rows["candidate"])
    decision = _decision(baseline_summary, candidate_summary, baseline_abstention, candidate_abstention)

    config = {
        "experiment_id": EXPERIMENT_ID,
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
        },
        "retrieval": {"method": "dense_vector_similarity", "top_k": top_k, "changed": False},
        "answering": {
            "mode": "extractive_verbatim",
            "max_answer_sentences": baseline_pipeline.max_answer_sentences,
            "max_answer_chunks": baseline_pipeline.max_answer_chunks,
            "max_quote_chars": baseline_pipeline.max_quote_chars,
            "policy": asdict(policy),
        },
        "arms": {
            "baseline": {"name": "lexical_top_n", "sibling_score_ratio": BASELINE_RATIO},
            "candidate": {"name": "sibling_clause", "sibling_score_ratio": SIBLING_RATIO},
        },
        "decision_rule": {
            "statement_support_delta_min": 0.02,
            "supported_answers_not_lower": True,
            "citation_support_delta_min": -0.02,
            "abstention_unchanged": True,
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
        results_dir / "retrieval_results.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "build_stats": build_stats,
            "retrieval_results": retrieval_rows,
        },
    )
    _write_json(
        results_dir / "answers_baseline.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "arm": config["arms"]["baseline"],
            "summary": baseline_summary,
            "answers": baseline_rows,
        },
    )
    _write_json(
        results_dir / "answers_sibling.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "arm": config["arms"]["candidate"],
            "summary": candidate_summary,
            "answers": candidate_rows,
        },
    )
    _write_json(
        results_dir / "abstention_results.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "baseline": {"summary": baseline_abstention, "results": abstention_rows["baseline"]},
            "candidate": {"summary": candidate_abstention, "results": abstention_rows["candidate"]},
        },
    )
    _write_json(
        results_dir / "comparison.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "baseline_summary": baseline_summary,
            "candidate_summary": candidate_summary,
            "decision": decision,
            "per_question": comparison,
        },
    )

    with open(results_dir / "per_question_metrics.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "question_id",
                f"recall@{top_k}",
                "baseline_supported_statements",
                "baseline_statements",
                "candidate_supported_statements",
                "candidate_statements",
                "supported_statement_delta",
                "baseline_supported_citations",
                "baseline_citations",
                "candidate_supported_citations",
                "candidate_citations",
            ]
        )
        baseline_by_id = {row["question_id"]: row for row in baseline_rows}
        candidate_by_id = {row["question_id"]: row for row in candidate_rows}
        for row in comparison:
            question_id = row["question_id"]
            baseline_row = baseline_by_id[question_id]
            candidate_row = candidate_by_id[question_id]
            writer.writerow(
                [
                    question_id,
                    row[f"recall@{top_k}"],
                    row["baseline_supported_statements"],
                    row["baseline_statements"],
                    row["candidate_supported_statements"],
                    row["candidate_statements"],
                    row["supported_statement_delta"],
                    row["baseline_supported_citations"],
                    baseline_row["citation_count"],
                    row["candidate_supported_citations"],
                    candidate_row["citation_count"],
                ]
            )

    (results_dir / "summary.md").write_text(
        _summary(
            config,
            baseline_summary,
            candidate_summary,
            comparison,
            decision,
            baseline_abstention,
            candidate_abstention,
        ),
        encoding="utf-8",
    )

    latency_summary = {
        arm: {
            "mean_ms": round(sum(values) / len(values), 2) if values else 0.0,
            "median_ms": round(float(statistics.median(values)), 2) if values else 0.0,
            "max_ms": round(max(values), 2) if values else 0.0,
        }
        for arm, values in (
            ("baseline", baseline_latencies),
            ("candidate", candidate_latencies),
        )
    }

    return {
        "experiment_id": EXPERIMENT_ID,
        "decision": decision,
        "baseline_summary": baseline_summary,
        "candidate_summary": candidate_summary,
        "abstention": {
            "baseline": baseline_abstention,
            "candidate": candidate_abstention,
        },
        "latency": latency_summary,
        "build_stats": build_stats,
        "results_dir": str(results_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP-003 sibling-clause answer selection experiment")
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
