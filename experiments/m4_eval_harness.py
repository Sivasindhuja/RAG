"""M4 offline evaluation harness (foundation).

Runs reproducible evaluation against the local validated RAG service/core
(extractive default, EXP-003, corrected production index) without changing
retrieval/chunking/selection/threshold behavior.

Measures (as applicable): retrieval (recall/precision/MRR, evidence coverage),
answer support/faithfulness proxies (span overlap with verified evidence),
citation correctness/coverage proxies, abstention behavior, and latency.

Usage:
  python -m experiments.m4_eval_harness --out experiments/results/m4_baseline
  python -m experiments.m4_eval_harness --mode extractive --k 5
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.rag_app import AppConfig, RAGService
from src.rag_core import (
    evaluate_answer_support,
    evidence_for_question,
    extract_source_document,
    load_exp001_evidence,
    load_questions,
)
from src.rag_core.evaluation import compute_retrieval_metrics
from src.rag_core.index import RetrievedChunk

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS = ROOT / "experiments" / "exp001_questions.csv"
DEFAULT_GROUND_TRUTH = ROOT / "experiments" / "exp001_ground_truth.json"
DEFAULT_UNANSWERABLE = ROOT / "experiments" / "unanswerable_questions.csv"
DEFAULT_PDF = ROOT / "data" / "satcom-ngp.pdf"


def _percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    vals = sorted(values)
    if len(vals) == 1:
        return round(vals[0], 3)
    k = (len(vals) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return round(vals[int(k)], 3)
    return round(vals[f] * (c - k) + vals[c] * (k - f), 3)


def _latency_summary(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "count": len(values),
        "mean": round(statistics.mean(values), 3),
        "p50": _percentile(values, 0.5),
        "p95": _percentile(values, 0.95),
        "max": round(max(values), 3),
    }


def _load_unanswerable(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "id": r.get("question_id") or r.get("id"),
                    "question": r.get("question"),
                    "type": "unanswerable_out_of_scope",
                    "abstention_expected": True,
                    "split": r.get("split"),
                    "category": r.get("category"),
                }
            )
    return rows


def _citation_structural(resp: Any) -> Dict[str, Any]:
    cits = resp.citations or []
    stmts = resp.statements or []

    def valid(c: Any) -> bool:
        return (
            c.start_offset < c.end_offset
            and c.start_page >= 1
            and c.end_page >= c.start_page
            and bool(c.chunk_id)
            and bool(c.source)
        )

    valid_cits = sum(1 for c in cits if valid(c))
    return {
        "citation_count": len(cits),
        "valid_citation_count": valid_cits,
        "statement_count": len(stmts),
        "selected_evidence_count": len(resp.selected_evidence_ids or []),
        "citation_structural_valid_rate": round(valid_cits / len(cits), 4) if cits else None,
    }


def _to_answer_like(resp: Any) -> Any:
    class _Stmt:
        pass

    class _Cit:
        pass

    class _Ans:
        pass

    ans = _Ans()
    ans.status = resp.status
    ans.reason = resp.abstention_reason
    ans.text = resp.answer
    ans.token_support = (resp.debug or {}).get("token_support", 0.0)
    ans.best_distance = (resp.debug or {}).get("best_distance")
    ans.statements = []
    for s in (resp.statements or []):
        o = _Stmt()
        o.text = s.text
        o.marker = s.marker
        o.chunk_id = s.chunk_id
        o.start_offset = s.start_offset
        o.end_offset = s.end_offset
        o.to_dict = lambda _o=o: {
            "text": _o.text,
            "marker": _o.marker,
            "chunk_id": _o.chunk_id,
            "start_offset": _o.start_offset,
            "end_offset": _o.end_offset,
        }
        ans.statements.append(o)
    ans.citations = []
    for c in (resp.citations or []):
        o = _Cit()
        o.start_offset = c.start_offset
        o.end_offset = c.end_offset
        o.to_dict = lambda _o=c: _o.model_dump() if hasattr(_o, "model_dump") else dict(_o)
        ans.citations.append(o)
    return ans


def run(out_dir: Path, mode: str = "extractive", k: int = 5) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    config = AppConfig.from_env()
    service = RAGService(config=config)
    doc = extract_source_document(DEFAULT_PDF)
    questions = load_questions(DEFAULT_QUESTIONS)
    evidence_units, problems = load_exp001_evidence(DEFAULT_GROUND_TRUTH, doc)
    if problems:
        raise RuntimeError(f"Ground truth validation problems: {problems[:3]}")

    per_question: List[Dict[str, Any]] = []
    latencies: List[float] = []
    stage_retrieval: List[float] = []
    stage_selection: List[float] = []

    for q in questions:
        resp = service.query(q.question, mode=mode, top_k=k)
        trace = (resp.debug or {}).get("trace") or {}
        total_ms = trace.get("total_ms", sum((resp.timings_ms or {}).values()))
        latencies.append(float(total_ms))
        if (resp.timings_ms or {}).get("retrieval") is not None:
            stage_retrieval.append(float(resp.timings_ms["retrieval"]))
        if (resp.timings_ms or {}).get("selection_and_extractive") is not None:
            stage_selection.append(float(resp.timings_ms["selection_and_extractive"]))

        units = evidence_for_question(evidence_units, q.question_id)
        retrieved = [
            RetrievedChunk(
                rank=r.rank,
                chunk_id=r.chunk_id,
                text="",
                source="satcom-ngp.pdf",
                start_offset=r.start_offset,
                end_offset=r.end_offset,
                start_page=r.start_page,
                end_page=r.end_page,
                article=None,
                section=r.section,
                heading=None,
                distance=r.distance,
                metadata={},
            )
            for r in (resp.retrieved or [])
        ]
        ret_metrics = compute_retrieval_metrics(retrieved, units, k=k) if units else {}
        support = evaluate_answer_support(_to_answer_like(resp), units) if units else {}
        row = {
            "id": q.question_id,
            "question": q.question,
            "type": "answerable",
            "expected_answer": q.expected_answer,
            "relevant_clause_ids": list(q.relevant_clause_ids),
            "response_status": resp.status,
            "abstained": resp.abstained,
            "abstention_reason": resp.abstention_reason,
            "retrieval": ret_metrics,
            "answer_support": {
                "answer_supported": support.get("answer_supported"),
                "statement_count": support.get("statement_count"),
                "supported_statement_count": support.get("supported_statement_count"),
                "citation_count": support.get("citation_count"),
                "supported_citation_count": support.get("supported_citation_count"),
            }
            if support
            else {},
            "citation_structural": _citation_structural(resp),
            "timings_ms": resp.timings_ms,
            "total_ms": round(float(total_ms), 3),
            "request_id": resp.request_id,
        }
        per_question.append(row)

    unans_rows = _load_unanswerable(DEFAULT_UNANSWERABLE)
    unans_results = []
    for q in unans_rows:
        resp = service.query(q["question"], mode=mode, top_k=k)
        trace = (resp.debug or {}).get("trace") or {}
        total_ms = trace.get("total_ms", sum((resp.timings_ms or {}).values()))
        latencies.append(float(total_ms))
        unans_results.append(
            {
                "id": q["id"],
                "question": q["question"],
                "type": q["type"],
                "split": q.get("split"),
                "category": q.get("category"),
                "abstention_expected": True,
                "response_status": resp.status,
                "abstained": resp.abstained,
                "abstention_reason": resp.abstention_reason,
                "correct_abstention": bool(resp.abstained or resp.status == "abstained"),
                "timings_ms": resp.timings_ms,
                "total_ms": round(float(total_ms), 3),
                "request_id": resp.request_id,
            }
        )

    answered_support = [r for r in per_question if r["answer_support"]]
    supported_answers = sum(1 for r in answered_support if r["answer_support"].get("answer_supported"))
    stmt_total = sum(r["answer_support"].get("statement_count", 0) for r in answered_support)
    stmt_supported = sum(r["answer_support"].get("supported_statement_count", 0) for r in answered_support)
    cit_total = sum(r["answer_support"].get("citation_count", 0) for r in answered_support)
    cit_supported = sum(r["answer_support"].get("supported_citation_count", 0) for r in answered_support)
    recalls = [r["retrieval"].get(f"recall@{k}") for r in per_question if r["retrieval"].get(f"recall@{k}") is not None]
    precisions = [r["retrieval"].get(f"precision@{k}") for r in per_question if r["retrieval"].get(f"precision@{k}") is not None]
    mrrs = [r["retrieval"].get(f"mrr@{k}") for r in per_question if r["retrieval"].get(f"mrr@{k}") is not None]
    unans_correct = sum(1 for r in unans_results if r["correct_abstention"])

    summary = {
        "config": {
            "mode": mode,
            "k": k,
            "embedding_model": config.embedding_model,
            "collection": config.collection_name,
            "defaults": {
                "exp003_sibling_ratio": config.sibling_score_ratio,
                "answerability_threshold": config.gemini_sufficiency_threshold,
                "cross_reference_selection": config.cross_reference_selection,
                "associated_heading_units": config.associated_heading_units,
                "definition_intent_boost": config.definition_intent_boost,
            },
        },
        "answerable": {
            "count": len(per_question),
            "answered": sum(1 for r in per_question if r["response_status"] == "answered"),
            "abstained": sum(1 for r in per_question if r["response_status"] == "abstained"),
            "supported_answers": supported_answers,
            "supported_answer_rate": round(supported_answers / len(per_question), 4) if per_question else None,
            "statement_support_rate": round(stmt_supported / stmt_total, 4) if stmt_total else None,
            "citation_support_rate": round(cit_supported / cit_total, 4) if cit_total else None,
            "retrieval": {
                f"mean_recall@{k}": round(statistics.mean(recalls), 4) if recalls else None,
                f"mean_precision@{k}": round(statistics.mean(precisions), 4) if precisions else None,
                f"mean_mrr@{k}": round(statistics.mean(mrrs), 4) if mrrs else None,
            },
        },
        "unanswerable": {
            "count": len(unans_results),
            "correct_abstentions": unans_correct,
            "abstention_accuracy": round(unans_correct / len(unans_results), 4) if unans_results else None,
        },
        "latency_ms": {
            "total": _latency_summary(latencies),
            "retrieval": _latency_summary(stage_retrieval),
            "selection_and_extractive": _latency_summary(stage_selection),
        },
        "notes": "Baseline measured on preserved datasets; not a production-quality claim. Thresholds must be derived from measured baselines, not set arbitrarily.",
    }

    (out_dir / "answerable_results.json").write_text(
        json.dumps(per_question, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "unanswerable_results.json").write_text(
        json.dumps(unans_results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    md = [
        f"# M4 Evaluation Baseline ({mode})",
        "",
        "## Summary",
        "```json",
        json.dumps(summary, indent=2),
        "```",
        "",
        "## Answerable per-question",
        f"| id | status | recall@{k} | supported | stmt sup/total | cit sup/total | total_ms |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in per_question:
        ret = r.get("retrieval") or {}
        sup = r.get("answer_support") or {}
        md.append(
            f"| {r['id']} | {r['response_status']} | {ret.get(f'recall@{k}', '')} | "
            f"{sup.get('answer_supported', '')} | {sup.get('supported_statement_count', '')}/{sup.get('statement_count', '')} | "
            f"{sup.get('supported_citation_count', '')}/{sup.get('citation_count', '')} | {r.get('total_ms', '')} |"
        )
    md += ["", "## Unanswerable", "| id | expected abstain | abstained | correct | reason |", "|---|---|---|---|---|"]
    for r in unans_results:
        reason = (r.get("abstention_reason") or "").replace("|", "/")[:80]
        md.append(f"| {r['id']} | true | {r['abstained']} | {r['correct_abstention']} | {reason} |")
    (out_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "experiments" / "results" / "m4_baseline")
    ap.add_argument("--mode", default="extractive", choices=["extractive", "gemini"])
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    summary = run(args.out, mode=args.mode, k=args.k)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
