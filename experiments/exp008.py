"""EXP-008: Q12 definitional evidence selection (controlled).

Observed problem
----------------
Q12 asks how international satellite communication services are defined.
Retrieval is correct (Recall@5=1.0, key evidence in ranks 1-2) but baseline
EXP-003 selection does not include required definition elements:
PSTN, GMPCS, Value Added Services, Programme Exchange, foreign-country
origination/termination, and MMSS/AMSS treatment.

Diagnosis (recorded before implementation)
-----------------------------------------
Query tokens: communication, defined, international, satellite, services.
Baseline lexical scores (sibling_ratio=0.5, default units):
- 4.049: "The Satellite Communication Services could be broadly classified as" (4.1)
- 3.751: "In so far as International communication is concerned..." (practice, supported but non-definitional)
- 3.674: "4.3.4 Land Mobile Satellite Services..." (irrelevant)
Key retrieved definitional sentences scored 2.962 despite containing required elements:
- chunk1 span 28-276: "The International Communications Services will include
  PSTN, GMPCS, Value Added Services, Programme Exchange etc., where one end...
  originates or terminates in a foreign country." (E01)
- chunk1 span 277-528: "The Maritime Mobile Satellite Service (MMSS) and
  Aeronautical Mobile Satellite Servie (AMSS) may be treated as International
  communications..." (E01)
- chunk0 span 828-1000: partial definition start (E01, truncated at chunk boundary)

Root cause: definitional-intent query token "defined" has weight but does not
appear in the actual definition text; definition is expressed via patterns
("will include...", "treated as...", "where...") with only partial query-term
overlap, so key sentences are outranked by generic/classification/practice units.

Hypothesis
----------
For questions expressing definitional intent (define/defined/definition/meaning/
means/what is/are/how is/are/describe), candidate sentences that contain
document-grounded definitional/explanatory markers (include(s)/including, means,
defined as, treated as, refers to, consist of, comprises, where, i.e.) and
overlap query content tokens should receive a modest selection-score boost.
This is a general ranking/selection signal, not Q12-specific.

Controlled variable
-------------------
Selection/ranking only: baseline EXP-003 (`definition_intent_boost=False`) vs
candidate (`definition_intent_boost=True`). One variable changed.

Held fixed
----------
PDF, 15 questions, 20 evidence units, 12 unanswerable questions, fixed-character
1000/200 chunks, MiniLM embeddings, dense top-5 retrieval, precomputed hits,
top-k, answer caps, citations, abstention policy, metrics, EXP-003 sibling ratio.
EXP-006 and EXP-007 remain separate opt-in comparisons (not combined).

Run:
    HF_HUB_OFFLINE=1 PYTHONPATH=. python -m experiments.exp008
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
EXPERIMENT_ID = "EXP-008"
RESULTS_DIR = ROOT / "experiments" / "results" / "exp008"
Q12_ID = "EXP001-Q12"
Q04_ID = "EXP001-Q04"
Q10_ID = "EXP001-Q10"
Q12_REQUIRED_TERMS = [
    "pstn",
    "gmpcs",
    "value added services",
    "programme exchange",
    "foreign country",
    "maritime mobile",
    "aeronautical mobile",
]
Q04_REQUIRED_TERMS = [
    "telecommunications",
    "broadcasting",
    "education and developmental communications",
    "security communications",
]
Q10_CHUNK_ID = "satcom-ngp::page_8::chunk_0::off17796"


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


def _valid_offsets(row: Dict[str, Any]) -> bool:
    for s in row.get("statements", []):
        if s.get("start_offset") is None or s.get("end_offset") is None or s["end_offset"] < s["start_offset"]:
            return False
    for c in row.get("citations", []):
        if c.get("start_offset") is None or c.get("end_offset") is None or c["end_offset"] < c["start_offset"]:
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
        "baseline": GroundedPipeline(index=index, top_k=TOP_K, policy=policy, sibling_score_ratio=0.5),
        "exp008": GroundedPipeline(index=index, top_k=TOP_K, policy=policy, sibling_score_ratio=0.5, definition_intent_boost=True),
        "exp006_only": GroundedPipeline(index=index, top_k=TOP_K, policy=policy, sibling_score_ratio=0.5, cross_reference_selection=True),
        "exp007_only": GroundedPipeline(index=index, top_k=TOP_K, policy=policy, sibling_score_ratio=0.5, associated_heading_units=True),
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
            answers[arm].append(row)
            row_by_arm[arm] = row
        comparison.append({
            "question_id": question.question_id,
            "recall@5": metrics["recall@5"],
            "baseline_supported_statements": row_by_arm["baseline"]["supported_statement_count"],
            "exp008_supported_statements": row_by_arm["exp008"]["supported_statement_count"],
            "delta_supported_statements": row_by_arm["exp008"]["supported_statement_count"] - row_by_arm["baseline"]["supported_statement_count"],
            "baseline_supported_citations": row_by_arm["baseline"]["supported_citation_count"],
            "exp008_supported_citations": row_by_arm["exp008"]["supported_citation_count"],
            "baseline_answer_supported": row_by_arm["baseline"]["answer_supported"],
            "exp008_answer_supported": row_by_arm["exp008"]["answer_supported"],
            "baseline_covered": _covered_evidence(row_by_arm["baseline"]),
            "exp008_covered": _covered_evidence(row_by_arm["exp008"]),
            "exp006_only_answer_supported": row_by_arm["exp006_only"]["answer_supported"],
            "exp006_only_supported_statements": row_by_arm["exp006_only"]["supported_statement_count"],
            "exp006_only_covered": _covered_evidence(row_by_arm["exp006_only"]),
            "exp007_only_answer_supported": row_by_arm["exp007_only"]["answer_supported"],
            "exp007_only_supported_statements": row_by_arm["exp007_only"]["supported_statement_count"],
            "exp007_only_covered": _covered_evidence(row_by_arm["exp007_only"]),
        })

    summaries = {arm: _answer_summary(rows) for arm, rows in answers.items()}
    unanswerable = _load_unanswerable(UNANSWERABLE_CSV)
    abstention_rows = {k: [] for k in pipelines}
    for arm, pipeline in pipelines.items():
        for item in unanswerable:
            ans = pipeline.answer(item["question"])
            abstention_rows[arm].append({**item, "status": ans.status, "abstained": ans.status == ABSTAINED, "reason": ans.reason, "best_distance": ans.best_distance, "token_support": round(ans.token_support, 4)})
    abstention_summaries = {arm: _abstention_summary(rows) for arm, rows in abstention_rows.items()}

    base_by = {r["question_id"]: r for r in answers["baseline"]}
    exp8_by = {r["question_id"]: r for r in answers["exp008"]}
    exp6_by = {r["question_id"]: r for r in answers["exp006_only"]}
    exp7_by = {r["question_id"]: r for r in answers["exp007_only"]}

    q12 = {
        "terms": _term_checks(exp8_by[Q12_ID]["text"], Q12_REQUIRED_TERMS),
        "evidence_covered": len(_covered_evidence(exp8_by[Q12_ID])) > 0,
        "answer_supported": exp8_by[Q12_ID]["answer_supported"],
        "supported_statement_count": exp8_by[Q12_ID]["supported_statement_count"],
        "statement_count": exp8_by[Q12_ID]["statement_count"],
        "valid_offsets": _valid_offsets(exp8_by[Q12_ID]),
        "citations": [{"chunk_id": c["chunk_id"], "section": c.get("section"), "start_offset": c["start_offset"], "end_offset": c["end_offset"]} for c in exp8_by[Q12_ID].get("citations", [])],
        "statements": [{"chunk_id": s["chunk_id"], "start_offset": s["start_offset"], "end_offset": s["end_offset"], "supported": s.get("supported"), "evidence_ids": s.get("evidence_ids")} for s in exp8_by[Q12_ID].get("statements", [])],
    }
    q04_check = {
        "exp008": {"supported": exp8_by[Q04_ID]["answer_supported"], "supported_statements": exp8_by[Q04_ID]["supported_statement_count"], "covered": _covered_evidence(exp8_by[Q04_ID]), "terms": _term_checks(exp8_by[Q04_ID]["text"], Q04_REQUIRED_TERMS)},
        "exp007_only": {"supported": exp7_by[Q04_ID]["answer_supported"], "supported_statements": exp7_by[Q04_ID]["supported_statement_count"], "covered": _covered_evidence(exp7_by[Q04_ID]), "terms": _term_checks(exp7_by[Q04_ID]["text"], Q04_REQUIRED_TERMS)},
        "baseline": {"supported": base_by[Q04_ID]["answer_supported"], "supported_statements": base_by[Q04_ID]["supported_statement_count"], "covered": _covered_evidence(base_by[Q04_ID]), "terms": _term_checks(base_by[Q04_ID]["text"], Q04_REQUIRED_TERMS)},
    }
    q10_check = {
        "exp008": {"supported": exp8_by[Q10_ID]["answer_supported"], "supported_statements": exp8_by[Q10_ID]["supported_statement_count"], "covered": _covered_evidence(exp8_by[Q10_ID]), "cites_q10_chunk": any(c.get("chunk_id") == Q10_CHUNK_ID for c in exp8_by[Q10_ID].get("citations", []))},
        "exp006_only": {"supported": exp6_by[Q10_ID]["answer_supported"], "supported_statements": exp6_by[Q10_ID]["supported_statement_count"], "covered": _covered_evidence(exp6_by[Q10_ID]), "cites_q10_chunk": any(c.get("chunk_id") == Q10_CHUNK_ID for c in exp6_by[Q10_ID].get("citations", []))},
        "baseline": {"supported": base_by[Q10_ID]["answer_supported"], "supported_statements": base_by[Q10_ID]["supported_statement_count"], "covered": _covered_evidence(base_by[Q10_ID])},
    }

    regressions = []
    for qid, b in base_by.items():
        c = exp8_by[qid]
        if c["supported_statement_count"] < b["supported_statement_count"]:
            regressions.append(f"{qid}:supported_statement_count")
        if c["supported_citation_count"] < b["supported_citation_count"]:
            regressions.append(f"{qid}:supported_citation_count")
        if c["answer_supported"] < b["answer_supported"]:
            regressions.append(f"{qid}:answer_supported")

    checks = {
        "q12_all_required_elements_present": q12["terms"]["all_terms_present"],
        "q12_evidence_covered": q12["evidence_covered"],
        "q12_answer_supported": q12["answer_supported"],
        "q12_valid_offsets_traceability": q12["valid_offsets"],
        "no_regressions_vs_baseline": not regressions,
        "supported_answers_not_lower": summaries["exp008"]["supported_answers"] >= summaries["baseline"]["supported_answers"],
        "statement_support_not_lower": summaries["exp008"]["statement_support_rate"] >= summaries["baseline"]["statement_support_rate"],
        "citation_support_delta_ge_minus_0_02": summaries["exp008"]["citation_support_rate"] - summaries["baseline"]["citation_support_rate"] >= -0.02,
        "abstention_unchanged": abstention_summaries["exp008"]["overall"]["abstained"] == 12 and abstention_summaries["baseline"]["overall"]["abstained"] == 12,
        "all_offsets_valid_exp008": all(r["valid_offsets_and_traceability"] for r in answers["exp008"]),
    }
    accepted = all(checks.values())
    decision = {
        "accepted_for_promotion_consideration": accepted,
        "checks": checks,
        "regressions": regressions,
        "q12": q12,
        "q04_separate_check": q04_check,
        "q10_separate_check": q10_check,
        "deltas": {
            "supported_answers": summaries["exp008"]["supported_answers"] - summaries["baseline"]["supported_answers"],
            "statement_support_rate": round(summaries["exp008"]["statement_support_rate"] - summaries["baseline"]["statement_support_rate"], 4),
            "citation_support_rate": round(summaries["exp008"]["citation_support_rate"] - summaries["baseline"]["citation_support_rate"], 4),
        },
    }

    config = {
        "experiment_id": EXPERIMENT_ID,
        "changed_variable": "selection/ranking only: definition_intent_boost",
        "hypothesis": "For definitional-intent questions, boost candidates containing definitional/explanatory markers with query-content overlap.",
        "arms": {
            "baseline": {"definition_intent_boost": False},
            "exp008": {"definition_intent_boost": True},
            "exp006_only": {"cross_reference_selection": True, "note": "separate, not combined"},
            "exp007_only": {"associated_heading_units": True, "note": "separate, not combined"},
        },
        "frozen": ["pdf", "questions", "ground_truth", "chunking", "embeddings", "retrieval_hits", "top_k", "answer_caps", "citations", "abstention", "metrics", "exp003_sibling_ratio"],
        "build_stats": build_stats,
    }

    results_dir = Path(results_dir)
    _write_json(results_dir / "config.json", config)
    _write_json(results_dir / "integrity_audit.json", audit)
    _write_json(results_dir / "retrieval_results.json", {"experiment_id": EXPERIMENT_ID, "retrieval_results": retrieval_rows})
    _write_json(results_dir / "answers_baseline.json", {"summary": summaries["baseline"], "answers": answers["baseline"]})
    _write_json(results_dir / "answers_exp008.json", {"summary": summaries["exp008"], "answers": answers["exp008"]})
    _write_json(results_dir / "answers_exp006_only.json", {"summary": summaries["exp006_only"], "answers": answers["exp006_only"]})
    _write_json(results_dir / "answers_exp007_only.json", {"summary": summaries["exp007_only"], "answers": answers["exp007_only"]})
    _write_json(results_dir / "abstention_results.json", {arm: {"summary": abstention_summaries[arm], "results": abstention_rows[arm]} for arm in pipelines})
    _write_json(results_dir / "comparison.json", {"experiment_id": EXPERIMENT_ID, "summaries": summaries, "abstention_summaries": abstention_summaries, "decision": decision, "per_question": comparison})

    with open(results_dir / "per_question_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["question_id", "recall@5", "base_sup_stmts", "exp008_sup_stmts", "delta", "base_sup_cites", "exp008_sup_cites", "base_supported", "exp008_supported", "exp006_sup_stmts", "exp006_supported", "exp007_sup_stmts", "exp007_supported"])
        for r in comparison:
            w.writerow([r["question_id"], r["recall@5"], r["baseline_supported_statements"], r["exp008_supported_statements"], r["delta_supported_statements"], r["baseline_supported_citations"], r["exp008_supported_citations"], r["baseline_answer_supported"], r["exp008_answer_supported"], r["exp006_only_supported_statements"], r["exp006_only_answer_supported"], r["exp007_only_supported_statements"], r["exp007_only_answer_supported"]])

    lines = [
        f"# {EXPERIMENT_ID} — Q12 definitional evidence selection",
        "",
        f"- Decision (promotion consideration): **{'ACCEPTED' if accepted else 'REJECTED'}**",
        "- Changed variable: selection/ranking only (`definition_intent_boost`)",
        "- EXP-006 and EXP-007 kept separate (not combined). EXP-003 default unchanged.",
        "",
        "## Aggregate (baseline -> EXP-008)",
        f"- Supported answers: {summaries['baseline']['supported_answers']} -> {summaries['exp008']['supported_answers']} ({decision['deltas']['supported_answers']:+d})",
        f"- Statement support: {summaries['baseline']['supported_statement_count']}/{summaries['baseline']['statement_count']} ({summaries['baseline']['statement_support_rate']:.4f}) -> {summaries['exp008']['supported_statement_count']}/{summaries['exp008']['statement_count']} ({summaries['exp008']['statement_support_rate']:.4f}) ({decision['deltas']['statement_support_rate']:+.4f})",
        f"- Citation support: {summaries['baseline']['supported_citation_count']}/{summaries['baseline']['citation_count']} ({summaries['baseline']['citation_support_rate']:.4f}) -> {summaries['exp008']['supported_citation_count']}/{summaries['exp008']['citation_count']} ({summaries['exp008']['citation_support_rate']:.4f}) ({decision['deltas']['citation_support_rate']:+.4f})",
        "",
        "## Q12",
        f"- All required elements present: {q12['terms']['all_terms_present']} ({q12['terms']['present_count']}/{q12['terms']['required_count']})",
        f"- Term hits: {q12['terms']['term_hits']}",
        f"- Evidence covered: {q12['evidence_covered']}, answer supported: {q12['answer_supported']}, supported stmts: {q12['supported_statement_count']}/{q12['statement_count']}, valid offsets: {q12['valid_offsets']}",
        "",
        "## Separate checks (not combined)",
        f"- Q04 EXP-008: supported={q04_check['exp008']['supported']}, stmts={q04_check['exp008']['supported_statements']}, all terms={q04_check['exp008']['terms']['all_terms_present']} {q04_check['exp008']['terms']['term_hits']}",
        f"- Q04 EXP-007 only: supported={q04_check['exp007_only']['supported']}, stmts={q04_check['exp007_only']['supported_statements']}, all terms={q04_check['exp007_only']['terms']['all_terms_present']}",
        f"- Q10 EXP-008: supported={q10_check['exp008']['supported']}, stmts={q10_check['exp008']['supported_statements']}, covered={q10_check['exp008']['covered']}",
        f"- Q10 EXP-006 only: supported={q10_check['exp006_only']['supported']}, stmts={q10_check['exp006_only']['supported_statements']}, covered={q10_check['exp006_only']['covered']}, cites_q10_chunk={q10_check['exp006_only']['cites_q10_chunk']}",
        "",
        "## Checks",
        *[f"- {k}: {v}" for k, v in checks.items()],
        "",
    ]
    if regressions:
        lines += ["## Regressions", *[f"- {r}" for r in regressions], ""]
    (results_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"experiment_id": EXPERIMENT_ID, "accepted": accepted, "checks": checks, "q12": q12["terms"], "summaries": summaries, "regressions": regressions, "q10_exp006_only": q10_check["exp006_only"]}, indent=2))
    return decision


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    args = parser.parse_args()
    run(results_dir=Path(args.results_dir))
