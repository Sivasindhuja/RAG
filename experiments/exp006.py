"""EXP-006: cross-reference applicability selection for Q10 (controlled).

Observed problem
----------------
Q10 asks for the procedure when multiple applicants request the same
orbit-spectrum in new or unplanned bands. The required answer is the full
procedure in clause 3.6.8. Retrieval is correct (Recall@5=1.0, 3.6.8 chunk at
rank 3, overlaps EXP001-Q10-E01), but the accepted EXP-003 extractive selector
returns 0 supported statements.

Root cause (recorded before implementation)
-------------------------------------------
1. The question's key conditions ("new or unplanned bands", "orbit-spectrum")
   appear directly in the 3.6.7(a)/(b) list (rank-1 chunk, score 11.323), not in
   3.6.8.
2. Clause 3.6.8 states "For 3.6.7(a) and 3.6.7(b)..." and contains the procedure
   (first come, first served; government priority; discussions; common API; AP S4
   decision) but references those conditions by clause/subclause IDs rather than
   repeating the terms.
3. Sentence-level units split 3.6.8 into fragments; its body procedure sentences
   have low direct lexical scores (best 3.414) and no clause identity.
4. 3.6.10/3.6.9/3.6.11 and unrelated text outrank 3.6.8 procedure content.

Hypothesis
----------
Applicability clauses that declare "For <clause/subclause refs>" provide a
document-grounded cross-reference signal. If referenced submarkers (e.g. (a),
(b)) correspond to list items/segments in retrieved candidates that are
lexically relevant to the question, promoting the applicability clause together
with its contiguous body (as one verbatim answer unit) can select the correct
procedure clause without Q10-specific rules.

Controlled variable
-------------------
Selection only: baseline EXP-003 (`cross_reference_selection=False`) vs candidate
(`cross_reference_selection=True`). Candidate groups promoted applicability
clauses ("<id> For ...") with contiguous body sentences into one unit and boosts
them based on query-relevant referenced submarkers.

Held fixed
----------
PDF, 15 questions, 20 evidence units, fixed-character 1000/200 chunks, MiniLM
embeddings, dense top-5 retrieval, precomputed hits, answer caps, citations,
abstention policy, metrics, EXP-003 sibling ratio (0.5). No generator.

Pre-registered acceptance checks
--------------------------------
1. Q10: covers EXP001-Q10-E01, >=1 supported statement, cites
   satcom-ngp::page_8::chunk_0::off17796, and answer contains key procedure
   elements (first come first served, government-owned priority, discussions,
   common API/Advanced Publication of Information, AP S4 filing/later decision).
2. No non-Q10 regression in supported statements, supported citations, or
   answer_supported for any question.
3. Aggregate: supported answers >= baseline, statement support rate >= baseline,
   citation support rate delta >= -0.02.
4. Abstention unchanged (12/12).
5. Statements/citations remain verbatim source spans with valid offsets.

Accept only if all checks pass.

Run:
    python -m experiments.exp006
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
EXPERIMENT_ID = "EXP-006"
RESULTS_DIR = ROOT / "experiments" / "results" / "exp006"
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
        "statement_support_rate": (supported_statement_count / statement_count) if statement_count else 0.0,
        "citation_count": citation_count,
        "supported_citation_count": supported_citation_count,
        "citation_support_rate": (supported_citation_count / citation_count) if citation_count else 0.0,
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
    raw_text = row.get("text", "")
    # Preserve original text semantics; normalize only for matching to tolerate
    # PDF extraction whitespace artifacts (e.g. "prio rity") without changing
    # source/ground-truth content.
    space_norm = " ".join(raw_text.lower().split())
    compact_norm = "".join(raw_text.lower().split())
    phrase_hits = {}
    for phrase in Q10_KEY_PHRASES:
        phrase_space = " ".join(phrase.lower().split())
        phrase_compact = "".join(phrase.lower().split())
        phrase_hits[phrase] = phrase_space in space_norm or phrase_compact in compact_norm
    return {
        "evidence_covered": Q10_EVIDENCE_ID in _covered_evidence(row),
        "supported_statement_present": row.get("supported_statement_count", 0) >= 1,
        "cites_q10_chunk": any(c.get("chunk_id") == Q10_CHUNK_ID for c in row.get("citations", [])),
        "key_phrase_hits": phrase_hits,
        "all_key_phrases_present": all(phrase_hits.values()),
    }


def _decision(baseline_rows, candidate_rows, baseline_summary, candidate_summary, baseline_abstention, candidate_abstention):
    baseline_by_id = {row["question_id"]: row for row in baseline_rows}
    candidate_by_id = {row["question_id"]: row for row in candidate_rows}
    q10 = _q10_checks(candidate_by_id["EXP001-Q10"])
    regressions = []
    for qid, base in baseline_by_id.items():
        if qid == "EXP001-Q10":
            continue
        cand = candidate_by_id[qid]
        if cand["supported_statement_count"] < base["supported_statement_count"]:
            regressions.append(f"{qid}:supported_statement_count")
        if cand["supported_citation_count"] < base["supported_citation_count"]:
            regressions.append(f"{qid}:supported_citation_count")
        if cand["answer_supported"] < base["answer_supported"]:
            regressions.append(f"{qid}:answer_supported")
    checks = {
        "q10_evidence_covered": q10["evidence_covered"],
        "q10_supported_statement_present": q10["supported_statement_present"],
        "q10_cites_clause_chunk": q10["cites_q10_chunk"],
        "q10_all_key_phrases_present": q10["all_key_phrases_present"],
        "no_non_q10_regressions": not regressions,
        "supported_answers_not_lower": candidate_summary["supported_answers"] >= baseline_summary["supported_answers"],
        "statement_support_not_lower": candidate_summary["statement_support_rate"] >= baseline_summary["statement_support_rate"],
        "citation_support_delta_ge_minus_0_02": candidate_summary["citation_support_rate"] - baseline_summary["citation_support_rate"] >= -0.02,
        "abstention_unchanged": (
            candidate_abstention["overall"]["abstained"] == baseline_abstention["overall"]["abstained"] == baseline_abstention["overall"]["count"]
        ),
    }
    return {
        "accepted": all(checks.values()),
        "checks": checks,
        "q10": q10,
        "regressions": regressions,
        "statement_support_delta": round(candidate_summary["statement_support_rate"] - baseline_summary["statement_support_rate"], 4),
        "citation_support_delta": round(candidate_summary["citation_support_rate"] - baseline_summary["citation_support_rate"], 4),
        "supported_answer_delta": candidate_summary["supported_answers"] - baseline_summary["supported_answers"],
    }


def run(
    pdf_path: Path = PDF_PATH,
    questions_csv: Path = QUESTIONS_CSV,
    ground_truth_path: Path = GROUND_TRUTH_PATH,
    unanswerable_csv: Path = UNANSWERABLE_CSV,
    results_dir: Path = RESULTS_DIR,
    persist_dir: Path = PRODUCTION_PERSIST_DIR,
    collection_name: str = PRODUCTION_COLLECTION,
    top_k: int = TOP_K,
    build_index: bool = False,
) -> Dict[str, Any]:
    policy = AbstentionPolicy()
    document = extract_source_document(pdf_path)
    chunks = FixedCharacterChunker(chunk_size=CHUNK_SIZE, overlap=OVERLAP).chunk(document)
    audit = audit_source_coverage(document, chunks, max_chars=CHUNK_SIZE, allow_overlap=True)
    if not audit["ok"]:
        raise RuntimeError(f"Fixed-character chunk audit failed: {audit['issues']}")

    questions = load_questions(questions_csv)
    units, problems = load_exp001_evidence(ground_truth_path, document)
    if problems:
        raise RuntimeError(f"Ground-truth mapping failed: {problems[0]}")

    model = SentenceTransformer(EMBEDDING_MODEL)
    index = DenseIndex(
        DenseIndexConfig(collection_name=collection_name, persist_dir=persist_dir, embedding_model=EMBEDDING_MODEL),
        model=model,
    )
    build_stats = index.build(chunks, reset=True) if build_index else {"chunks": index.count()}

    baseline_pipeline = GroundedPipeline(index=index, top_k=top_k, policy=policy, sibling_score_ratio=0.5, cross_reference_selection=False)
    candidate_pipeline = GroundedPipeline(index=index, top_k=top_k, policy=policy, sibling_score_ratio=0.5, cross_reference_selection=True)

    retrieval_rows = []
    baseline_rows = []
    candidate_rows = []
    comparison = []
    for question in questions:
        evidence = evidence_for_question(units, question.question_id)
        hits = index.search(question.question, k=top_k)
        metrics = compute_retrieval_metrics(hits, evidence, k=top_k)
        retrieval_rows.append({
            "question_id": question.question_id,
            "question": question.question,
            "retrieved_chunks": [
                {"rank": h.rank, "chunk_id": h.chunk_id, "section": h.section, "start_offset": h.start_offset, "end_offset": h.end_offset,
                 "distance": h.distance, "evidence_ids": _overlapping_evidence(h.start_offset, h.end_offset, evidence)} for h in hits
            ],
            **{k: metrics[k] for k in metrics if k.startswith(("recall@", "precision@", "mrr@"))},
        })

        b = baseline_pipeline.answer_from_hits(question.question, hits)
        c = candidate_pipeline.answer_from_hits(question.question, hits)
        b_row = {"question_id": question.question_id, "question": question.question, "expected_answer": question.expected_answer,
                 "relevant_clause_ids": list(question.relevant_clause_ids), **_evaluate_answer(b, evidence)}
        c_row = {"question_id": question.question_id, "question": question.question, "expected_answer": question.expected_answer,
                 "relevant_clause_ids": list(question.relevant_clause_ids), **_evaluate_answer(c, evidence)}
        baseline_rows.append(b_row)
        candidate_rows.append(c_row)
        comparison.append({
            "question_id": question.question_id,
            f"recall@{top_k}": metrics[f"recall@{top_k}"],
            "baseline_statements": b_row["statement_count"], "baseline_supported_statements": b_row["supported_statement_count"],
            "candidate_statements": c_row["statement_count"], "candidate_supported_statements": c_row["supported_statement_count"],
            "supported_statement_delta": c_row["supported_statement_count"] - b_row["supported_statement_count"],
            "baseline_supported_citations": b_row["supported_citation_count"], "candidate_supported_citations": c_row["supported_citation_count"],
            "baseline_answer_supported": b_row["answer_supported"], "candidate_answer_supported": c_row["answer_supported"],
            "baseline_covered_evidence_ids": _covered_evidence(b_row), "candidate_covered_evidence_ids": _covered_evidence(c_row),
            "missing_evidence_ids": list(metrics["missing_evidence_ids"]),
        })

    baseline_summary = _answer_summary(baseline_rows)
    candidate_summary = _answer_summary(candidate_rows)

    unanswerable = _load_unanswerable(unanswerable_csv)
    abstention_rows = {"baseline": [], "candidate": []}
    for arm, pipeline in (("baseline", baseline_pipeline), ("candidate", candidate_pipeline)):
        for item in unanswerable:
            answer = pipeline.answer(item["question"])
            abstention_rows[arm].append({**item, "status": answer.status, "abstained": answer.status == ABSTAINED, "reason": answer.reason,
                                         "best_distance": answer.best_distance, "token_support": round(answer.token_support, 4)})
    baseline_abstention = _abstention_summary(abstention_rows["baseline"])
    candidate_abstention = _abstention_summary(abstention_rows["candidate"])
    decision = _decision(baseline_rows, candidate_rows, baseline_summary, candidate_summary, baseline_abstention, candidate_abstention)

    config = {
        "experiment_id": EXPERIMENT_ID,
        "embedding_model": EMBEDDING_MODEL,
        "chunking": {"strategy": "fixed_character", "chunk_size": CHUNK_SIZE, "overlap": OVERLAP, "chunk_count": len(chunks)},
        "retrieval": {"method": "dense_vector_similarity", "top_k": top_k, "changed": False},
        "selection": {"sibling_score_ratio": 0.5, "policy": asdict(policy)},
        "arms": {"baseline": {"cross_reference_selection": False}, "candidate": {"cross_reference_selection": True}},
        "frozen": ["pdf", "questions", "ground_truth", "chunking", "embeddings", "retrieval_hits", "top_k", "citations", "abstention", "metrics"],
        "decision_rule": decision["checks"],
    }

    results_dir = Path(results_dir)
    _write_json(results_dir / "config.json", config)
    _write_json(results_dir / "integrity_audit.json", audit)
    _write_json(results_dir / "retrieval_results.json", {"experiment_id": EXPERIMENT_ID, "build_stats": build_stats, "retrieval_results": retrieval_rows})
    _write_json(results_dir / "answers_baseline.json", {"experiment_id": EXPERIMENT_ID, "arm": "baseline", "summary": baseline_summary, "answers": baseline_rows})
    _write_json(results_dir / "answers_candidate.json", {"experiment_id": EXPERIMENT_ID, "arm": "candidate", "summary": candidate_summary, "answers": candidate_rows})
    _write_json(results_dir / "abstention_results.json", {"baseline": {"summary": baseline_abstention, "results": abstention_rows["baseline"]}, "candidate": {"summary": candidate_abstention, "results": abstention_rows["candidate"]}})
    _write_json(results_dir / "comparison.json", {"experiment_id": EXPERIMENT_ID, "baseline_summary": baseline_summary, "candidate_summary": candidate_summary, "decision": decision, "per_question": comparison})

    with open(results_dir / "per_question_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["question_id", f"recall@{top_k}", "baseline_supported_statements", "candidate_supported_statements", "delta", "baseline_supported_citations", "candidate_supported_citations", "baseline_answer_supported", "candidate_answer_supported"])
        for row in comparison:
            w.writerow([row["question_id"], row[f"recall@{top_k}"], row["baseline_supported_statements"], row["candidate_supported_statements"], row["supported_statement_delta"], row["baseline_supported_citations"], row["candidate_supported_citations"], row["baseline_answer_supported"], row["candidate_answer_supported"]])

    # summary
    lines = [
        f"# {EXPERIMENT_ID} — Cross-reference applicability selection",
        "",
        f"- Decision: **{'ACCEPTED' if decision['accepted'] else 'REJECTED'}**",
        "- Changed variable: selection only (`cross_reference_selection`)",
        "- Frozen: PDF, questions, ground truth, chunking, embeddings, retrieval hits, top-k, citations, abstention, metrics",
        "",
        "## Aggregate",
        f"- Supported answers: {baseline_summary['supported_answers']} -> {candidate_summary['supported_answers']} ({decision['supported_answer_delta']:+d})",
        f"- Statement support: {baseline_summary['supported_statement_count']}/{baseline_summary['statement_count']} ({baseline_summary['statement_support_rate']:.4f}) -> {candidate_summary['supported_statement_count']}/{candidate_summary['statement_count']} ({candidate_summary['statement_support_rate']:.4f}) ({decision['statement_support_delta']:+.4f})",
        f"- Citation support: {baseline_summary['supported_citation_count']}/{baseline_summary['citation_count']} ({baseline_summary['citation_support_rate']:.4f}) -> {candidate_summary['supported_citation_count']}/{candidate_summary['citation_count']} ({candidate_summary['citation_support_rate']:.4f}) ({decision['citation_support_delta']:+.4f})",
        "",
        "## Q10 checks",
    ]
    for k, v in decision['q10'].items():
        if k == 'key_phrase_hits':
            continue
        lines.append(f"- {k}: {v}")
    lines.append("- key phrases: " + ", ".join(f"{k}={v}" for k, v in decision['q10']['key_phrase_hits'].items()))
    lines += ["", "## Checks", *[f"- {k}: {v}" for k, v in decision['checks'].items()], ""]
    if decision['regressions']:
        lines.append("## Regressions\n- " + "\n- ".join(decision['regressions']) + "\n")
    (results_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    result = {"experiment_id": EXPERIMENT_ID, "decision": decision, "baseline_summary": baseline_summary, "candidate_summary": candidate_summary}
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    args = parser.parse_args()
    run(results_dir=Path(args.results_dir))


if __name__ == "__main__":
    main()
