"""EXP-007: heading-introduced associated answer-unit investigation (Q04/Q12).

Observed problems
-----------------
Q04 asks for broad classifications of INSAT user sectors. Required evidence is
2.4.1 including the four classifications. Baseline (EXP-003) selects the heading
"The user sectors can be broadly classified as:" but omits the immediately
following list items (a)-(d), so the answer is incomplete.

Q12 asks how international satellite communication services are defined. Required
evidence is 4.2 including PSTN/GMPCS/Value Added Services/Programme Exchange and
the foreign-country termination/origination condition, plus MMSS/AMSS treatment.
Baseline fails to select the key definition sentences/content.

Source structure / selection diagnosis (recorded before implementation)
----------------------------------------------------------------------
Q04 (Recall@5 1.0, rank 1 hit satcom-ngp::page_3::chunk_0::off4699): sentence
spans split the clause into heading "2.4.1 ... classified as:" (span 35-87,
score 5.251, selected) and a separate list span "(a)...(d)" (span 91-238, low
query overlap, not selected). Evidence EXP001-Q04-E01 (4740-4938) is covered by
the heading overlap but answer omits the four classifications.

Q12 (Recall@5 1.0, rank 1/2 hits satcom-ngp::page_13::chunk_1::off30978 and
::chunk_0::off30178): key definition sentences are split across spans/chunks
(4.2 heading/body in chunk 0, "International Communications Services will
include..." and MMSS/AMSS treatment in chunk 1). Baseline selects generic
classification/4.2 introductory and unrelated 4.3.4 units; the actual definition
elements are present in retrieved evidence but not selected.

Hypothesis
----------
A heading/introductory span that ends with a colon signals an immediately
associated enumeration/definition block in the same chunk. Grouping such an
introductory span with contiguous following spans until the next numbered clause
start preserves heading-associated lists/definitions as one verbatim answer unit
without question-specific rules.

Controlled variable
-------------------
Answer-unit construction/selection only: baseline EXP-003 sentence units
(`associated_heading_units=False`) vs candidate (`associated_heading_units=True`).
Both use `sibling_score_ratio=0.5`, `cross_reference_selection=False`.

Held fixed
----------
PDF, 15 questions, 20 evidence units, 12 unanswerable questions, fixed-character
1000/200 chunks, MiniLM embeddings, dense top-5 retrieval, precomputed hits,
top-k, answer caps, citations, abstention policy, metrics. EXP-006 remains a
separate opt-in comparison on Q10 only (not combined).

Run:
    HF_HUB_OFFLINE=1 python -m experiments.exp007
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

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
EXPERIMENT_ID = "EXP-007"
RESULTS_DIR = ROOT / "experiments" / "results" / "exp007"
Q04_ID = "EXP001-Q04"
Q12_ID = "EXP001-Q12"
Q10_ID = "EXP001-Q10"
Q04_REQUIRED_TERMS = [
    "telecommunications",
    "broadcasting",
    "education and developmental communications",
    "security communications",
]
Q12_REQUIRED_TERMS = [
    "pstn",
    "gmpcs",
    "value added services",
    "programme exchange",
    "foreign country",
    "maritime mobile",
    "aeronautical mobile",
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
    for statement in row.get("statements", []):
        covered.update(statement.get("evidence_ids", []))
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
    return {"by_split": by_split, "overall": {"count": len(rows), "abstained": total, "abstention_rate": round(total / len(rows), 4) if rows else 0.0}}


def _term_checks(text: str, terms: Sequence[str]) -> Dict[str, Any]:
    norm = " ".join(text.lower().split())
    compact = "".join(norm.split())
    hits = {}
    for term in terms:
        t = " ".join(term.lower().split())
        hits[term] = t in norm or "".join(t.split()) in compact
    return {"term_hits": hits, "all_terms_present": all(hits.values()), "present_count": sum(hits.values()), "required_count": len(terms)}


def _target_checks(row: Dict[str, Any]) -> Dict[str, Any]:
    text = row.get("text", "")
    qid = row["question_id"]
    checks = {
        "evidence_covered": len(_covered_evidence(row)) > 0,
        "answer_supported": row.get("answer_supported", False),
        "supported_statement_count": row.get("supported_statement_count", 0),
        "statement_count": row.get("statement_count", 0),
        "supported_citation_count": row.get("supported_citation_count", 0),
    }
    if qid == Q04_ID:
        checks.update({"q04": _term_checks(text, Q04_REQUIRED_TERMS)})
    if qid == Q12_ID:
        checks.update({"q12": _term_checks(text, Q12_REQUIRED_TERMS)})
    return checks


def _valid_offsets(row: Dict[str, Any]) -> bool:
    for s in row.get("statements", []):
        if s.get("start_offset") is None or s.get("end_offset") is None:
            return False
        if s["end_offset"] < s["start_offset"]:
            return False
    for c in row.get("citations", []):
        if c.get("start_offset") is None or c.get("end_offset") is None:
            return False
        if c["end_offset"] < c["start_offset"]:
            return False
    return True


def run(results_dir: Path = RESULTS_DIR, build_index: bool = False) -> Dict[str, Any]:
    policy = AbstentionPolicy()
    document = extract_source_document(PDF_PATH)
    chunks = FixedCharacterChunker(chunk_size=CHUNK_SIZE, overlap=OVERLAP).chunk(document)
    audit = audit_source_coverage(document, chunks, max_chars=CHUNK_SIZE, allow_overlap=True)
    if not audit["ok"]:
        raise RuntimeError(f"Audit failed: {audit['issues']}")

    questions = load_questions(QUESTIONS_CSV)
    units, problems = load_exp001_evidence(GROUND_TRUTH_PATH, document)
    if problems:
        raise RuntimeError(f"Ground-truth mapping failed: {problems[0]}")

    model = SentenceTransformer(EMBEDDING_MODEL)
    index = DenseIndex(
        DenseIndexConfig(collection_name=PRODUCTION_COLLECTION, persist_dir=PRODUCTION_PERSIST_DIR, embedding_model=EMBEDDING_MODEL),
        model=model,
    )
    build_stats = index.build(chunks, reset=True) if build_index else {"chunks": index.count()}

    pipelines = {
        "baseline": GroundedPipeline(index=index, top_k=TOP_K, policy=policy, sibling_score_ratio=0.5, cross_reference_selection=False, associated_heading_units=False),
        "exp007": GroundedPipeline(index=index, top_k=TOP_K, policy=policy, sibling_score_ratio=0.5, cross_reference_selection=False, associated_heading_units=True),
        "exp006_only": GroundedPipeline(index=index, top_k=TOP_K, policy=policy, sibling_score_ratio=0.5, cross_reference_selection=True, associated_heading_units=False),
    }

    retrieval_rows = []
    answers = {k: [] for k in pipelines}
    comparison = []
    for question in questions:
        evidence = evidence_for_question(units, question.question_id)
        hits = index.search(question.question, k=TOP_K)
        metrics = compute_retrieval_metrics(hits, evidence, k=TOP_K)
        retrieval_rows.append({
            "question_id": question.question_id,
            "question": question.question,
            "retrieval": {k: metrics[k] for k in ("recall@5", "precision@5", "mrr@5")},
            "retrieved_chunks": [{"rank": h.rank, "chunk_id": h.chunk_id, "section": h.section, "start_offset": h.start_offset, "end_offset": h.end_offset, "distance": h.distance, "evidence_ids": _overlapping_evidence(h.start_offset, h.end_offset, evidence)} for h in hits],
        })
        row_by_arm = {}
        for arm, pipeline in pipelines.items():
            ans = pipeline.answer_from_hits(question.question, hits)
            row = {"question_id": question.question_id, "question": question.question, "expected_answer": question.expected_answer,
                   "relevant_clause_ids": list(question.relevant_clause_ids), **_evaluate_answer(ans, evidence)}
            row["valid_offsets_and_traceability"] = _valid_offsets(row)
            row["target_checks"] = _target_checks(row) if question.question_id in {Q04_ID, Q12_ID} else None
            answers[arm].append(row)
            row_by_arm[arm] = row
        comparison.append({
            "question_id": question.question_id,
            "recall@5": metrics["recall@5"],
            "baseline_supported_statements": row_by_arm["baseline"]["supported_statement_count"],
            "exp007_supported_statements": row_by_arm["exp007"]["supported_statement_count"],
            "exp007_supported_statement_delta": row_by_arm["exp007"]["supported_statement_count"] - row_by_arm["baseline"]["supported_statement_count"],
            "baseline_supported_citations": row_by_arm["baseline"]["supported_citation_count"],
            "exp007_supported_citations": row_by_arm["exp007"]["supported_citation_count"],
            "baseline_answer_supported": row_by_arm["baseline"]["answer_supported"],
            "exp007_answer_supported": row_by_arm["exp007"]["answer_supported"],
            "baseline_covered_evidence_ids": _covered_evidence(row_by_arm["baseline"]),
            "exp007_covered_evidence_ids": _covered_evidence(row_by_arm["exp007"]),
            "exp006_only_supported_statements": row_by_arm["exp006_only"]["supported_statement_count"],
            "exp006_only_answer_supported": row_by_arm["exp006_only"]["answer_supported"],
            "exp006_only_covered_evidence_ids": _covered_evidence(row_by_arm["exp006_only"]),
            "offsets_ok_baseline": row_by_arm["baseline"]["valid_offsets_and_traceability"],
            "offsets_ok_exp007": row_by_arm["exp007"]["valid_offsets_and_traceability"],
        })

    summaries = {arm: _answer_summary(rows) for arm, rows in answers.items()}

    unanswerable = _load_unanswerable(UNANSWERABLE_CSV)
    abstention_rows = {k: [] for k in pipelines}
    for arm, pipeline in pipelines.items():
        for item in unanswerable:
            ans = pipeline.answer(item["question"])
            abstention_rows[arm].append({**item, "status": ans.status, "abstained": ans.status == ABSTAINED, "reason": ans.reason, "best_distance": ans.best_distance, "token_support": round(ans.token_support, 4)})
    abstention_summaries = {arm: _abstention_summary(rows) for arm, rows in abstention_rows.items()}

    # decision / checks
    base_by = {r["question_id"]: r for r in answers["baseline"]}
    exp7_by = {r["question_id"]: r for r in answers["exp007"]}
    exp6_by = {r["question_id"]: r for r in answers["exp006_only"]}
    regressions = []
    for qid, b in base_by.items():
        c = exp7_by[qid]
        if c["supported_statement_count"] < b["supported_statement_count"]:
            regressions.append(f"{qid}:supported_statement_count")
        if c["supported_citation_count"] < b["supported_citation_count"]:
            regressions.append(f"{qid}:supported_citation_count")
        if c["answer_supported"] < b["answer_supported"]:
            regressions.append(f"{qid}:answer_supported")
    q04 = _target_checks(exp7_by[Q04_ID])
    q12 = _target_checks(exp7_by[Q12_ID])
    q10_exp006 = {"answer_supported": exp6_by[Q10_ID]["answer_supported"], "supported_statement_count": exp6_by[Q10_ID]["supported_statement_count"], "covered_evidence_ids": _covered_evidence(exp6_by[Q10_ID]), "cites_q10_chunk": any(c.get("chunk_id") == "satcom-ngp::page_8::chunk_0::off17796" for c in exp6_by[Q10_ID].get("citations", []))}
    q10_exp007 = {"answer_supported": exp7_by[Q10_ID]["answer_supported"], "supported_statement_count": exp7_by[Q10_ID]["supported_statement_count"], "covered_evidence_ids": _covered_evidence(exp7_by[Q10_ID])}
    checks = {
        "q04_all_four_classifications_present": q04["q04"]["all_terms_present"],
        "q04_evidence_covered": q04["evidence_covered"],
        "q12_all_required_definition_elements_present": q12["q12"]["all_terms_present"],
        "q12_evidence_covered": q12["evidence_covered"],
        "no_non_target_regressions_vs_baseline": not [r for r in regressions if not r.startswith(Q04_ID) and not r.startswith(Q12_ID)],
        "no_regressions_vs_baseline_any": not regressions,
        "supported_answers_not_lower": summaries["exp007"]["supported_answers"] >= summaries["baseline"]["supported_answers"],
        "statement_support_not_lower": summaries["exp007"]["statement_support_rate"] >= summaries["baseline"]["statement_support_rate"],
        "citation_support_delta_ge_minus_0_02": summaries["exp007"]["citation_support_rate"] - summaries["baseline"]["citation_support_rate"] >= -0.02,
        "abstention_unchanged": abstention_summaries["exp007"]["overall"]["abstained"] == abstention_summaries["baseline"]["overall"]["abstained"] == 12,
        "all_offsets_valid_exp007": all(r["valid_offsets_and_traceability"] for r in answers["exp007"]),
    }
    accepted = all(checks.values())
    decision = {"accepted_for_promotion_consideration": accepted, "checks": checks, "regressions": regressions, "q04": q04, "q12": q12, "q10_exp006_only_comparison": q10_exp006, "q10_exp007_comparison": q10_exp007,
                "statement_support_delta": round(summaries["exp007"]["statement_support_rate"] - summaries["baseline"]["statement_support_rate"], 4),
                "citation_support_delta": round(summaries["exp007"]["citation_support_rate"] - summaries["baseline"]["citation_support_rate"], 4),
                "supported_answer_delta": summaries["exp007"]["supported_answers"] - summaries["baseline"]["supported_answers"]}

    config = {
        "experiment_id": EXPERIMENT_ID,
        "hypothesis": "Introductory spans ending with ':' grouped with contiguous associated content until next numbered clause start preserve heading-associated lists/definitions.",
        "changed_variable": "answer-unit construction/selection only (associated_heading_units)",
        "arms": {"baseline": {"sibling_score_ratio": 0.5, "cross_reference_selection": False, "associated_heading_units": False},
                 "exp007": {"sibling_score_ratio": 0.5, "cross_reference_selection": False, "associated_heading_units": True},
                 "exp006_only": {"sibling_score_ratio": 0.5, "cross_reference_selection": True, "associated_heading_units": False, "note": "separate opt-in comparison, not combined"}},
        "frozen": ["pdf", "questions", "ground_truth", "chunking", "embeddings", "retrieval_hits", "top_k", "answer_caps", "citations", "abstention", "metrics", "exp003_sibling_ratio"],
        "build_stats": build_stats,
    }

    results_dir = Path(results_dir)
    _write_json(results_dir / "config.json", config)
    _write_json(results_dir / "integrity_audit.json", audit)
    _write_json(results_dir / "retrieval_results.json", {"experiment_id": EXPERIMENT_ID, "retrieval_results": retrieval_rows})
    _write_json(results_dir / "answers_baseline.json", {"experiment_id": EXPERIMENT_ID, "summary": summaries["baseline"], "answers": answers["baseline"]})
    _write_json(results_dir / "answers_exp007.json", {"experiment_id": EXPERIMENT_ID, "summary": summaries["exp007"], "answers": answers["exp007"]})
    _write_json(results_dir / "answers_exp006_only.json", {"experiment_id": EXPERIMENT_ID, "summary": summaries["exp006_only"], "answers": answers["exp006_only"]})
    _write_json(results_dir / "abstention_results.json", {arm: {"summary": abstention_summaries[arm], "results": abstention_rows[arm]} for arm in pipelines})
    _write_json(results_dir / "comparison.json", {"experiment_id": EXPERIMENT_ID, "summaries": summaries, "abstention_summaries": abstention_summaries, "decision": decision, "per_question": comparison})

    with open(results_dir / "per_question_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["question_id", "recall@5", "baseline_supported_statements", "exp007_supported_statements", "delta", "baseline_supported_citations", "exp007_supported_citations", "baseline_answer_supported", "exp007_answer_supported", "exp006_only_supported_statements", "exp006_only_answer_supported"])
        for r in comparison:
            w.writerow([r["question_id"], r["recall@5"], r["baseline_supported_statements"], r["exp007_supported_statements"], r["exp007_supported_statement_delta"], r["baseline_supported_citations"], r["exp007_supported_citations"], r["baseline_answer_supported"], r["exp007_answer_supported"], r["exp006_only_supported_statements"], r["exp006_only_answer_supported"]])

    lines = [
        f"# {EXPERIMENT_ID} — Heading-introduced associated answer units (Q04/Q12)",
        "",
        f"- Decision (promotion consideration): **{'ACCEPTED' if accepted else 'REJECTED'}**",
        "- Changed variable: answer-unit construction/selection only (`associated_heading_units`)",
        "- EXP-006 kept separate (opt-in comparison, not combined). EXP-003 remains default.",
        "",
        "## Aggregate (baseline -> EXP-007)",
        f"- Supported answers: {summaries['baseline']['supported_answers']} -> {summaries['exp007']['supported_answers']} ({decision['supported_answer_delta']:+d})",
        f"- Statement support: {summaries['baseline']['supported_statement_count']}/{summaries['baseline']['statement_count']} ({summaries['baseline']['statement_support_rate']:.4f}) -> {summaries['exp007']['supported_statement_count']}/{summaries['exp007']['statement_count']} ({summaries['exp007']['statement_support_rate']:.4f}) ({decision['statement_support_delta']:+.4f})",
        f"- Citation support: {summaries['baseline']['supported_citation_count']}/{summaries['baseline']['citation_count']} ({summaries['baseline']['citation_support_rate']:.4f}) -> {summaries['exp007']['supported_citation_count']}/{summaries['exp007']['citation_count']} ({summaries['exp007']['citation_support_rate']:.4f}) ({decision['citation_support_delta']:+.4f})",
        "",
        "## Target checks",
        f"- Q04 all four classifications present: {q04['q04']['all_terms_present']} ({q04['q04']['present_count']}/{q04['q04']['required_count']}) {q04['q04']['term_hits']}",
        f"- Q04 evidence covered: {q04['evidence_covered']}, answer supported: {q04['answer_supported']}, supported stmts: {q04['supported_statement_count']}/{q04['statement_count']}",
        f"- Q12 all required definition elements present: {q12['q12']['all_terms_present']} ({q12['q12']['present_count']}/{q12['q12']['required_count']}) {q12['q12']['term_hits']}",
        f"- Q12 evidence covered: {q12['evidence_covered']}, answer supported: {q12['answer_supported']}, supported stmts: {q12['supported_statement_count']}/{q12['statement_count']}",
        "",
        "## Q10 (separate comparison)",
        f"- EXP-006 only: supported={q10_exp006['answer_supported']}, supported_statements={q10_exp006['supported_statement_count']}, covered={q10_exp006['covered_evidence_ids']}, cites_q10_chunk={q10_exp006['cites_q10_chunk']}",
        f"- EXP-007 only: supported={q10_exp007['answer_supported']}, supported_statements={q10_exp007['supported_statement_count']}, covered={q10_exp007['covered_evidence_ids']}",
        "",
        "## Checks",
        *[f"- {k}: {v}" for k, v in checks.items()],
        "",
    ]
    if regressions:
        lines += ["## Regressions (EXP-007 vs baseline)", *[f"- {r}" for r in regressions], ""]
    (results_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({"experiment_id": EXPERIMENT_ID, "accepted": accepted, "checks": checks, "q04": q04["q04"], "q12": q12["q12"], "summaries": summaries, "regressions": regressions}, indent=2))
    return decision


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    args = parser.parse_args()
    run(results_dir=Path(args.results_dir))
