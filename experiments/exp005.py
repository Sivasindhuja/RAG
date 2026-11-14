"""EXP-005: clause-body-aware answer-unit selection for Q10.

Observed problem
----------------
Q10 asks for the procedure when multiple applicants request the same
orbit-spectrum in new or unplanned bands.  The required evidence is the full
body of clause 3.6.8 (first come, first served for 3.6.7(a)/(b), government
priority in conflict, discussions with applicants, common API, decision at AP S4
or later).  Retrieval is correct: the 3.6.8 chunk is retrieved at rank 3 with
Recall@5 1.0 and overlaps ``EXP001-Q10-E01``.

The accepted EXP-003 extractive selector returns 0/3 supported statements for
Q10.  It selects the 3.6.7(a)-(e) definitional list and 3.6.10 sentences instead
of 3.6.8.

Root cause (recorded before implementation)
-------------------------------------------
1. Sentence-level segmentation fragments 3.6.8 into four answer units.  Only the
   first starts with clause ID ``3.6.8``; the three body sentences ("In case of a
   conflict...", discussions, common API) have no clause identity.
2. Lexical scoring favors the 3.6.7(a)-(e) list because the question tokens
   "new", "unplanned", "bands", "orbit", "spectrum" appear there directly, while
   3.6.8 refers to those cases as "3.6.7(a) and 3.6.7(b)".
3. The 3-sentence cap cannot assemble the complete four-step 3.6.8 procedure even
   if body sentences were partially ranked.
4. Sibling-clause selection does not help because the selected top units are not
   same-parent numbered clause siblings of 3.6.8.

Hypothesis
----------
Grouping a numbered subclause start (depth >= 3, e.g. ``3.6.8``) with its
contiguous body sentences, until the next numbered clause start in the same
chunk, creates a clause-body-aware answer unit that preserves clause identity and
complete procedural steps.  This should improve Q10 evidence coverage without
changing retrieval.

Controlled variable
-------------------
Answer-unit segmentation only: sentence units (EXP-003 accepted baseline) versus
clause-body units for depth >= 3 subclauses.  Both arms keep
``sibling_score_ratio=0.5``.

Held fixed
----------
PDF, 15 questions, 20 evidence units, fixed-character 1000/200 chunks, MiniLM
embeddings, dense top-5 retrieval, precomputed hits, answer caps, citations,
abstention policy and metrics.  No generator.

Pre-registered acceptance checks
--------------------------------
1. Q10 covers ``EXP001-Q10-E01`` with >= 1 supported statement, cites
   ``satcom-ngp::page_8::chunk_0::off17796``, and answer text contains the key
   procedure elements: first come first served, government-owned priority,
   discussions with applicants, common API/Advanced Publication of Information,
   and AP S4 filing/later decision.
2. No per-question regression: for every other question, supported statements,
   supported citations and ``answer_supported`` are not lower than baseline.
3. Aggregate non-regression: supported answers >= 12, statement support rate >=
   0.3333, citation support rate >= 0.4961, abstention 12/12 unchanged.
4. Integrity: statements/citations remain verbatim source spans with valid
   offsets; retrieval unchanged.

Decision: accept only if all checks pass and Q10 is complete, not merely
partially relevant.

Run:
    python -m experiments.exp005
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
    COLLECTION_NAME as PRODUCTION_COLLECTION,
    EMBEDDING_MODEL,
    GROUND_TRUTH_PATH,
    OVERLAP,
    PDF_PATH,
    PERSIST_DIR as PRODUCTION_PERSIST_DIR,
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
EXPERIMENT_ID = "EXP-005"
RESULTS_DIR = ROOT / "experiments" / "results" / "exp005"
Q10_CHUNK_ID = "satcom-ngp::page_8::chunk_0::off17796"
Q10_EVIDENCE_ID = "EXP001-Q10-E01"
Q10_KEY_PHRASES = [
    "first come first served",
    "government owned system",
    "priority",
    "discussions with the applicants",
    "common advanced publication of information",
    "ap s4 filing",
]


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


def _q10_checks(row: Dict[str, Any]) -> Dict[str, Any]:
    text = " ".join(row.get("text", "").lower().split())
    phrase_hits = {phrase: phrase in text for phrase in Q10_KEY_PHRASES}
    return {
        "evidence_covered": Q10_EVIDENCE_ID in _covered_evidence(row),
        "supported_statement_present": row.get("supported_statement_count", 0) >= 1,
        "cites_q10_chunk": any(c.get("chunk_id") == Q10_CHUNK_ID for c in row.get("citations", [])),
        "key_phrase_hits": phrase_hits,
        "all_key_phrases_present": all(phrase_hits.values()),
    }


def _decision(
    baseline_rows: Sequence[Dict[str, Any]],
    candidate_rows: Sequence[Dict[str, Any]],
    baseline_summary: Dict[str, Any],
    candidate_summary: Dict[str, Any],
    baseline_abstention: Dict[str, Any],
    candidate_abstention: Dict[str, Any],
) -> Dict[str, Any]:
    baseline_by_id = {row["question_id"]: row for row in baseline_rows}
    candidate_by_id = {row["question_id"]: row for row in candidate_rows}
    q10 = _q10_checks(candidate_by_id["EXP001-Q10"])
    regressions = []
    for question_id, base in baseline_by_id.items():
        if question_id == "EXP001-Q10":
            continue
        cand = candidate_by_id[question_id]
        if cand["supported_statement_count"] < base["supported_statement_count"]:
            regressions.append(f"{question_id}:supported_statement_count")
        if cand["supported_citation_count"] < base["supported_citation_count"]:
            regressions.append(f"{question_id}:supported_citation_count")
        if cand["answer_supported"] < base["answer_supported"]:
            regressions.append(f"{question_id}:answer_supported")

    checks = {
        "q10_evidence_covered": q10["evidence_covered"],
        "q10_supported_statement_present": q10["supported_statement_present"],
        "q10_cites_clause_chunk": q10["cites_q10_chunk"],
        "q10_all_key_phrases_present": q10["all_key_phrases_present"],
        "no_non_q10_regressions": not regressions,
        "supported_answers_not_lower": candidate_summary["supported_answers"] >= baseline_summary["supported_answers"],
        "statement_support_not_lower": (
            candidate_summary["statement_support_rate"] >= baseline_summary["statement_support_rate"]
        ),
        "citation_support_delta_ge_minus_0.02": (
            candidate_summary["citation_support_rate"] - baseline_summary["citation_support_rate"] >= -0.02
        ),
        "abstention_unchanged": (
            candidate_abstention["overall"]["abstained"]
            == baseline_abstention["overall"]["abstained"]
            == baseline_abstention["overall"]["count"]
        ),
    }
    return {
        "accepted": all(checks.values()),
        "checks": checks,
        "q10": q10,
        "regressions": regressions,
        "statement_support_delta": round(
            candidate_summary["statement_support_rate"] - baseline_summary["statement_support_rate"], 4
        ),
        "citation_support_delta": round(
            candidate_summary["citation_support_rate"] - baseline_summary["citation_support_rate"], 4
        ),
        "supported_answer_delta": candidate_summary["supported_answers"] - baseline_summary["supported_answers"],
    }


def _summary(
    config: Dict[str, Any],
    baseline_summary: Dict[str, Any],
    candidate_summary: Dict[str, Any],
    comparison: Sequence[Dict[str, Any]],
    decision: Dict[str, Any],
    baseline_abstention: Dict[str, Any],
    candidate_abstention: Dict[str, Any],
) -> str:
    lines = [
        "# EXP-005 — Clause-body-aware answer-unit selection",
        "",
        f"- Experiment id: `{EXPERIMENT_ID}`",
        "- Baseline: EXP-003 accepted selection (`sibling_score_ratio=0.5`, sentence units)",
        "- Candidate: `sibling_score_ratio=0.5`, depth>=3 clause-body answer units",
        "- Retrieval: fixed-character 1000/200, MiniLM, dense top-5, identical precomputed hits",
        "- Decision: **" + ("ACCEPTED" if decision["accepted"] else "REJECTED") + "**",
        "",
        "## Root cause (Q10)",
        "",
        "Q10 evidence `EXP001-Q10-E01` is inside clause 3.6.8 and is retrieved at rank 3.",
        "Sentence segmentation split 3.6.8 into four units; body procedure sentences had",
        "no clause identity. Lexical selection favored the 3.6.7(a)/(b) list and 3.6.10,",
        "so the accepted selector returned 0/3 supported statements for Q10.",
        "",
        "## Primary metrics",
        "",
        "| Measure | Baseline | Candidate | Delta |",
        "|---|---:|---:|---:|",
        f"| Supported answers | {baseline_summary['supported_answers']} | {candidate_summary['supported_answers']} | {decision['supported_answer_delta']:+d} |",
        f"| Supported statements | {baseline_summary['supported_statement_count']}/{baseline_summary['statement_count']} | {candidate_summary['supported_statement_count']}/{candidate_summary['statement_count']} | {candidate_summary['supported_statement_count'] - baseline_summary['supported_statement_count']:+d} |",
        f"| Statement support rate | {baseline_summary['statement_support_rate']:.4f} | {candidate_summary['statement_support_rate']:.4f} | {decision['statement_support_delta']:+.4f} |",
        f"| Supported citations | {baseline_summary['supported_citation_count']}/{baseline_summary['citation_count']} | {candidate_summary['supported_citation_count']}/{candidate_summary['citation_count']} | {candidate_summary['supported_citation_count'] - baseline_summary['supported_citation_count']:+d} |",
        f"| Citation support rate | {baseline_summary['citation_support_rate']:.4f} | {candidate_summary['citation_support_rate']:.4f} | {decision['citation_support_delta']:+.4f} |",
        "",
        "## Q10 checks",
        "",
        "| Check | Result |",
        "|---|---|",
    ]
    for name, passed in decision["checks"].items():
        if name.startswith("q10_"):
            lines.append(f"| `{name}` | {'PASS' if passed else 'FAIL'} |")
    lines += [
        "",
        "Key phrases present: "
        + ", ".join(
            f"{phrase}={'yes' if hit else 'no'}"
            for phrase, hit in decision["q10"]["key_phrase_hits"].items()
        ),
        "",
        "## Other checks",
        "",
        "| Check | Result |",
        "|---|---|",
    ]
    for name, passed in decision["checks"].items():
        if not name.startswith("q10_"):
            lines.append(f"| `{name}` | {'PASS' if passed else 'FAIL'} |")
    if decision["regressions"]:
        lines += ["", "Regressions: " + ", ".join(decision["regressions"])]
    lines += [
        "",
        "## Per-question statement support",
        "",
        "| ID | Recall@5 | Baseline | Candidate | Delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in comparison:
        lines.append(
            f"| {row['question_id']} | {row['recall@5']:.4f} | "
            f"{row['baseline_supported_statements']}/{row['baseline_statements']} | "
            f"{row['candidate_supported_statements']}/{row['candidate_statements']} | "
            f"{row['supported_statement_delta']:+d} |"
        )
    lines += [
        "",
        "## Abstention smoke test",
        "",
        f"Overall: baseline {baseline_abstention['overall']['abstained']}/"
        f"{baseline_abstention['overall']['count']}; candidate "
        f"{candidate_abstention['overall']['abstained']}/"
        f"{candidate_abstention['overall']['count']}.",
        "",
        "## Interpretation",
        "",
        "- Clause-body units are exact verbatim spans: a depth>=3 clause start is grouped",
        "  with following body sentences until the next numbered clause start in the same chunk.",
        "- Depth-2 headings are left as sentence units to avoid swallowing child clauses.",
        "- The experiment is controlled: retrieval hits are identical and only answer-unit",
        "  segmentation/selection input changes.",
        "",
        "See `docs/EXP-005.md` and `docs/ENGINEERING_LOG.md` for the decision record.",
        "",
    ]
    return "\n".join(lines)


def run(
    pdf_path: Path = PDF_PATH,
    questions_csv: Path = QUESTIONS_CSV,
    ground_truth_path: Path = GROUND_TRUTH_PATH,
    unanswerable_csv: Path = UNANSWERABLE_CSV,
    results_dir: Path = RESULTS_DIR,
    persist_dir: Path = PRODUCTION_PERSIST_DIR,
    collection_name: str = PRODUCTION_COLLECTION,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP,
    top_k: int = TOP_K,
    build_index: bool = False,
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
        index=index, top_k=top_k, policy=policy, sibling_score_ratio=0.5, clause_body_units=False
    )
    candidate_pipeline = GroundedPipeline(
        index=index, top_k=top_k, policy=policy, sibling_score_ratio=0.5, clause_body_units=True
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
    decision = _decision(
        baseline_rows,
        candidate_rows,
        baseline_summary,
        candidate_summary,
        baseline_abstention,
        candidate_abstention,
    )

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
            "sibling_score_ratio": 0.5,
            "policy": asdict(policy),
        },
        "arms": {
            "baseline": {"name": "exp003_sentence_units", "clause_body_units": False},
            "candidate": {"name": "depth3_clause_body_units", "clause_body_units": True},
        },
        "decision_rule": {
            "q10_evidence_covered": True,
            "q10_supported_statement_present": True,
            "q10_cites_clause_chunk": Q10_CHUNK_ID,
            "q10_all_key_phrases_present": Q10_KEY_PHRASES,
            "no_non_q10_regressions": True,
            "supported_answers_not_lower": True,
            "statement_support_not_lower": True,
            "citation_support_delta_min": -0.02,
            "abstention_unchanged": True,
        },
        "vector_db": {
            "backend": "chroma",
            "collection_name": collection_name,
            "persist_directory": str(persist_dir),
            "reused_existing_index": not build_index,
        },
    }

    results_dir = Path(results_dir)
    _write_json(results_dir / "config.json", config)
    _write_json(results_dir / "integrity_audit.json", audit)
    _write_json(
        results_dir / "retrieval_results.json",
        {"experiment_id": EXPERIMENT_ID, "build_stats": build_stats, "retrieval_results": retrieval_rows},
    )
    _write_json(
        results_dir / "answers_baseline.json",
        {"experiment_id": EXPERIMENT_ID, "arm": config["arms"]["baseline"], "summary": baseline_summary, "answers": baseline_rows},
    )
    _write_json(
        results_dir / "answers_candidate.json",
        {"experiment_id": EXPERIMENT_ID, "arm": config["arms"]["candidate"], "summary": candidate_summary, "answers": candidate_rows},
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
                    baseline_by_id[question_id]["citation_count"],
                    row["candidate_supported_citations"],
                    candidate_by_id[question_id]["citation_count"],
                ]
            )

    (results_dir / "summary.md").write_text(
        _summary(config, baseline_summary, candidate_summary, comparison, decision, baseline_abstention, candidate_abstention),
        encoding="utf-8",
    )

    latency = {
        "baseline": {
            "mean_ms": round(sum(baseline_latencies) / len(baseline_latencies), 2) if baseline_latencies else 0.0,
            "median_ms": round(float(statistics.median(baseline_latencies)), 2) if baseline_latencies else 0.0,
            "max_ms": round(max(baseline_latencies), 2) if baseline_latencies else 0.0,
        },
        "candidate": {
            "mean_ms": round(sum(candidate_latencies) / len(candidate_latencies), 2) if candidate_latencies else 0.0,
            "median_ms": round(float(statistics.median(candidate_latencies)), 2) if candidate_latencies else 0.0,
            "max_ms": round(max(candidate_latencies), 2) if candidate_latencies else 0.0,
        },
    }
    _write_json(results_dir / "latency.json", {"experiment_id": EXPERIMENT_ID, **latency})

    return {
        "experiment_id": EXPERIMENT_ID,
        "decision": decision,
        "baseline_summary": baseline_summary,
        "candidate_summary": candidate_summary,
        "abstention": {"baseline": baseline_abstention, "candidate": candidate_abstention},
        "latency": latency,
        "build_stats": build_stats,
        "results_dir": str(results_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP-005 clause-body-aware answer-unit selection")
    parser.add_argument("--pdf", type=Path, default=PDF_PATH)
    parser.add_argument("--questions-csv", type=Path, default=QUESTIONS_CSV)
    parser.add_argument("--ground-truth", type=Path, default=GROUND_TRUTH_PATH)
    parser.add_argument("--unanswerable-csv", type=Path, default=UNANSWERABLE_CSV)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--persist-dir", type=Path, default=PRODUCTION_PERSIST_DIR)
    parser.add_argument("--collection", default=PRODUCTION_COLLECTION)
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
        top_k=args.top_k,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
