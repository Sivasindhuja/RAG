"""CI evaluation for the fixed 15-question SATCOM NGP golden dataset.

Runs retrieval comparisons (dense, hybrid, hybrid_rerank when configured),
deterministic answer/citation/abstention checks, and optional RAGAS metrics.
Designed for PR CI: deterministic mocks for external APIs by default in CI
unless live credentials are explicitly enabled.

Usage:
  python -m evaluation.ci_eval --dataset evaluation/ci_golden_dataset.json --out evaluation/results/ci
  python -m evaluation.ci_eval --retrieval-modes dense,hybrid --check-regression
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag_app import AppConfig, RAGService
from src.rag_core import extract_source_document
from src.rag_core.evaluation import EvidenceUnit, compute_retrieval_metrics


def _percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    vals = sorted(values)
    if len(vals) == 1:
        return round(vals[0], 4)
    k = (len(vals) - 1) * p
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return round(vals[f], 4)
    return round(vals[f] * (c - k) + vals[c] * (k - f), 4)


def _stats(values: List[float]) -> Dict[str, Any]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {"count": 0, "mean": None, "p50": None, "p95": None}
    return {
        "count": len(vals),
        "mean": round(statistics.mean(vals), 4),
        "p50": _percentile(vals, 0.5),
        "p95": _percentile(vals, 0.95),
    }


def load_ci_dataset(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def evidence_units_for_question(q: Dict[str, Any], doc) -> List[EvidenceUnit]:
    units: List[EvidenceUnit] = []
    for e in q.get("evidence_units") or []:
        page_num = int(e.get("page") or 0)
        page = next((p for p in doc.pages if p.page_number == page_num), None)
        if page is None:
            continue
        raw = page.text
        stripped = raw.strip()
        leading = len(raw) - len(raw.lstrip())
        ps = int(e.get("start_char") or 0)
        pe = int(e.get("end_char") or 0)
        gs = page.start_char + leading + ps
        ge = gs + (pe - ps)
        excerpt = str(e.get("exact_supporting_excerpt") or "")
        if ge > len(doc.text) or gs < 0 or doc.text[gs:ge] != excerpt:
            raise ValueError(f"Evidence mapping mismatch for {q.get('id')} {e.get('evidence_id')}")
        units.append(
            EvidenceUnit(
                question_id=str(q.get("id")),
                evidence_id=str(e.get("evidence_id") or ""),
                source=str(e.get("source") or doc.source),
                page=page_num,
                clause_id=str(e.get("clause_id") or ""),
                start_offset=gs,
                end_offset=ge,
                page_relative_start=ps,
                page_relative_end=pe,
                excerpt=excerpt,
            )
        )
    return units


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def evaluate_answerable(q: Dict[str, Any], resp, evidence: List[EvidenceUnit], k: int) -> Dict[str, Any]:
    # retrieval
    retrieved = []
    for r in (resp.retrieved or []):
        from src.rag_core import RetrievedChunk

        retrieved.append(
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
        )
    ret = compute_retrieval_metrics(retrieved, evidence, k=k) if evidence else {}
    # answer/citation support via evidence overlap
    supported_stmts = 0
    stmt_rows = []
    for s in (resp.statements or []):
        ev_ids = [u.evidence_id for u in evidence if _overlaps(s.start_offset, s.end_offset, u.start_offset, u.end_offset)]
        if ev_ids:
            supported_stmts += 1
        stmt_rows.append({"marker": s.marker, "chunk_id": s.chunk_id, "supported": bool(ev_ids), "evidence_ids": ev_ids})
    supported_cits = 0
    cit_rows = []
    invalid_cits = 0
    for c in (resp.citations or []):
        valid = c.start_offset < c.end_offset and c.start_page >= 1 and bool(c.chunk_id) and bool(c.source)
        if not valid:
            invalid_cits += 1
        ev_ids = [u.evidence_id for u in evidence if _overlaps(c.start_offset, c.end_offset, u.start_offset, u.end_offset)]
        if ev_ids:
            supported_cits += 1
        cit_rows.append({"marker": c.marker, "chunk_id": c.chunk_id, "valid": valid, "supported": bool(ev_ids), "evidence_ids": ev_ids})
    answered = resp.status == "answered" and not resp.abstained
    return {
        "id": q["id"],
        "type": q["type"],
        "status": resp.status,
        "abstained": bool(resp.abstained),
        "abstention_reason": resp.abstention_reason,
        "retrieval": ret,
        "answer": {
            "answered": answered,
            "answer_supported": supported_stmts > 0,
            "statement_count": len(resp.statements or []),
            "supported_statement_count": supported_stmts,
            "citation_count": len(resp.citations or []),
            "supported_citation_count": supported_cits,
            "invalid_citation_count": invalid_cits,
            "statement_support_rate": round(supported_stmts / len(resp.statements), 4) if resp.statements else None,
            "citation_support_rate": round(supported_cits / len(resp.citations), 4) if resp.citations else None,
        },
        "details": {"statements": stmt_rows, "citations": cit_rows},
        "timings_ms": dict(resp.timings_ms or {}),
        "total_ms": round(sum((resp.timings_ms or {}).values()), 4) if resp.timings_ms else None,
        "request_id": resp.request_id,
        "errors": list(resp.errors or []),
    }


def evaluate_unanswerable(q: Dict[str, Any], resp) -> Dict[str, Any]:
    abstained = bool(resp.abstained or resp.status == "abstained")
    invalid_cits = sum(1 for c in (resp.citations or []) if not (c.start_offset < c.end_offset and c.start_page >= 1 and c.chunk_id))
    return {
        "id": q["id"],
        "type": q["type"],
        "category": q.get("category"),
        "status": resp.status,
        "abstained": abstained,
        "expected_abstention": True,
        "correct_abstention": abstained,
        "abstention_reason": resp.abstention_reason,
        "citation_count": len(resp.citations or []),
        "invalid_citation_count": invalid_cits,
        "unsupported_answer": (not abstained) and len(resp.citations or []) == 0,
        "timings_ms": dict(resp.timings_ms or {}),
        "total_ms": round(sum((resp.timings_ms or {}).values()), 4) if resp.timings_ms else None,
        "request_id": resp.request_id,
        "errors": list(resp.errors or []),
    }


def aggregate_mode(results: List[Dict[str, Any]], k: int) -> Dict[str, Any]:
    ans = [r for r in results if r["type"] == "answerable"]
    unans = [r for r in results if r["type"] != "answerable"]
    recalls = [r["retrieval"].get(f"recall@{k}") for r in ans if r.get("retrieval")]
    precisions = [r["retrieval"].get(f"precision@{k}") for r in ans if r.get("retrieval")]
    mrrs = [r["retrieval"].get(f"mrr@{k}") for r in ans if r.get("retrieval")]
    evidence_covered = []
    for r in ans:
        cov = r.get("retrieval", {}).get("covered_evidence_ids") or []
        req = r.get("retrieval", {}).get("required_evidence_ids") or []
        if req:
            evidence_covered.append(len(cov) / len(req))
    answered_supported = sum(1 for r in ans if r["answer"].get("answer_supported"))
    stmt_total = sum(r["answer"].get("statement_count", 0) or 0 for r in ans)
    stmt_supported = sum(r["answer"].get("supported_statement_count", 0) or 0 for r in ans)
    cit_total = sum(r["answer"].get("citation_count", 0) or 0 for r in ans)
    cit_supported = sum(r["answer"].get("supported_citation_count", 0) or 0 for r in ans)
    invalid_cits = sum(r["answer"].get("invalid_citation_count", 0) or 0 for r in ans) + sum(r.get("invalid_citation_count", 0) or 0 for r in unans)
    correct_abst = sum(1 for r in unans if r.get("correct_abstention"))
    latencies = [r.get("total_ms") for r in results if r.get("total_ms") is not None]
    failures = sum(1 for r in results if r.get("status") == "error")
    return {
        "total_questions": len(results),
        "answerable": {
            "count": len(ans),
            "answered": sum(1 for r in ans if r["answer"].get("answered")),
            "abstained": sum(1 for r in ans if r.get("abstained")),
            "supported_answers": answered_supported,
            "supported_answer_rate": round(answered_supported / len(ans), 4) if ans else None,
            "statement_support_rate": round(stmt_supported / stmt_total, 4) if stmt_total else None,
            "citation_support_rate": round(cit_supported / cit_total, 4) if cit_total else None,
            "invalid_citation_count": sum(r["answer"].get("invalid_citation_count", 0) or 0 for r in ans),
        },
        "unanswerable": {
            "count": len(unans),
            "correct_abstentions": correct_abst,
            "abstention_accuracy": round(correct_abst / len(unans), 4) if unans else None,
            "incorrect_non_abstentions": len(unans) - correct_abst,
        },
        "retrieval": {
            f"recall@{k}": round(statistics.mean(recalls), 4) if recalls else None,
            f"precision@{k}": round(statistics.mean(precisions), 4) if precisions else None,
            f"mrr@{k}": round(statistics.mean(mrrs), 4) if mrrs else None,
            "evidence_coverage": round(statistics.mean(evidence_covered), 4) if evidence_covered else None,
        },
        "quality": {
            "invalid_citation_count_total": invalid_cits,
            "failure_count": failures,
            "failure_rate": round(failures / len(results), 4) if results else None,
        },
        "latency_ms": _stats([float(x) for x in latencies]),
    }


def run_ragas(results: List[Dict[str, Any]], enabled: bool) -> Dict[str, Any]:
    if not enabled:
        return {"enabled": False, "reason": "disabled", "metrics": {}}
    try:
        # RAGAS integration: run faithfulness/answer_relevancy/context_precision/context_recall/answer_correctness
        # where possible. This may require LLM/embeddings; in CI without live APIs, callers should disable or mock.
        # LangChain Google wrappers require GOOGLE_API_KEY; map from GEMINI_API_KEY when needed.
        if not os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY", "")
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_correctness, answer_relevancy, context_precision, context_recall, faithfulness
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
        try:
            from ragas.run_config import RunConfig
        except Exception:  # noqa: BLE001
            RunConfig = None  # type: ignore

        rows = []
        for r in results:
            if r["type"] != "answerable":
                continue
            # need contexts and answer; retrieve contexts from debug not stored in result rows except not present.
            # We expect caller to include contexts in r['_contexts'] and ground_truth in r['_ground_truth'].
            if not r.get("_contexts"):
                continue
            rows.append({
                "question": r.get("_question"),
                "answer": r.get("_answer") or "",
                "contexts": r.get("_contexts") or [],
                "ground_truth": r.get("_ground_truth") or "",
            })
        if not rows:
            return {"enabled": True, "reason": "no_rows", "metrics": {}}
        ds = Dataset.from_list(rows)
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        llm_model = os.getenv("RAGAS_LLM_MODEL", "gemini-flash-lite-latest")
        emb_model = os.getenv("RAGAS_EMBED_MODEL", "models/gemini-embedding-001")
        llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(model=llm_model, google_api_key=api_key))
        emb = LangchainEmbeddingsWrapper(GoogleGenerativeAIEmbeddings(model=emb_model, google_api_key=api_key))
        metric_name_map = {
            "context_precision": context_precision,
            "context_recall": context_recall,
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "answer_correctness": answer_correctness,
        }
        requested = [m.strip() for m in os.getenv("RAGAS_METRICS", "context_precision,context_recall,faithfulness,answer_relevancy,answer_correctness").split(",") if m.strip()]
        metrics_list = [metric_name_map[m] for m in requested if m in metric_name_map] or [context_precision, context_recall, faithfulness, answer_relevancy, answer_correctness]
        eval_kwargs: Dict[str, Any] = {
            "metrics": metrics_list,
            "llm": llm,
            "embeddings": emb,
            "raise_exceptions": False,
            "show_progress": True,
        }
        if RunConfig is not None:
            eval_kwargs["run_config"] = RunConfig(
                timeout=int(os.getenv("RAGAS_TIMEOUT", "300")),
                max_retries=int(os.getenv("RAGAS_MAX_RETRIES", "5")),
                max_wait=int(os.getenv("RAGAS_MAX_WAIT", "120")),
                max_workers=int(os.getenv("RAGAS_MAX_WORKERS", "1")),
            )
        try:
            eval_kwargs["batch_size"] = int(os.getenv("RAGAS_BATCH_SIZE", "1"))
        except Exception:
            pass
        out = evaluate(ds, **eval_kwargs)
        df = out.to_pandas()
        metrics = {}
        for k, v in df.mean(numeric_only=True).to_dict().items():
            try:
                fv = float(v)
                metrics[k] = None if math.isnan(fv) else fv
            except Exception:
                metrics[k] = None
        return {
            "enabled": True,
            "metrics": metrics,
            "rows": len(rows),
            "llm_model": llm_model,
            "embedding_model": emb_model,
            "metric_names": [getattr(m, "name", str(m)) for m in metrics_list],
        }
    except Exception as exc:  # noqa: BLE001
        return {"enabled": True, "error": f"{type(exc).__name__}: {exc}", "metrics": {}}


def run_mode(service: RAGService, dataset: Dict[str, Any], doc, retrieval_mode: str, k: int, generation_mode: str = "extractive") -> Dict[str, Any]:
    results = []
    for q in dataset["questions"]:
        resp = service.query(q["question"], mode=generation_mode, top_k=k, retrieval_mode=retrieval_mode)
        if q["type"] == "answerable":
            ev = evidence_units_for_question(q, doc)
            row = evaluate_answerable(q, resp, ev, k)
        else:
            ev = []
            row = evaluate_unanswerable(q, resp)
        row["_question"] = q["question"]
        row["_answer"] = resp.answer
        row["_ground_truth"] = q.get("expected_answer") or ""
        row["_contexts"] = [s.text for s in (resp.statements or [])] or [c.quote for c in (resp.citations or [])]
        row["_retrieval_mode"] = retrieval_mode
        results.append(row)
    summary = aggregate_mode(results, k)
    return {"retrieval_mode": retrieval_mode, "summary": summary, "results": results}


def compare_modes(mode_runs: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    rows = []
    for m in mode_runs:
        s = m["summary"]
        rows.append({
            "retrieval_mode": m["retrieval_mode"],
            f"recall@{k}": s["retrieval"].get(f"recall@{k}"),
            f"precision@{k}": s["retrieval"].get(f"precision@{k}"),
            f"mrr@{k}": s["retrieval"].get(f"mrr@{k}"),
            "evidence_coverage": s["retrieval"].get("evidence_coverage"),
            "supported_answer_rate": s["answerable"].get("supported_answer_rate"),
            "citation_support_rate": s["answerable"].get("citation_support_rate"),
            "abstention_accuracy": s["unanswerable"].get("abstention_accuracy"),
            "invalid_citations": s["quality"].get("invalid_citation_count_total"),
            "p50_latency_ms": s["latency_ms"].get("p50"),
            "p95_latency_ms": s["latency_ms"].get("p95"),
            "failure_rate": s["quality"].get("failure_rate"),
        })
    return rows


def check_regression(summary: Dict[str, Any], baseline: Dict[str, Any], k: int) -> Tuple[bool, List[str]]:
    failures = []
    # thresholds: minimums derived from baseline (no invention). Allow tiny floating tolerance.
    tol = 1e-9
    def get(d, path, default=None):
        cur = d
        for p in path:
            if not isinstance(cur, dict) or p not in cur:
                return default
            cur = cur[p]
        return cur

    checks = [
        (["retrieval", f"recall@{k}"], "min"),
        (["retrieval", f"mrr@{k}"], "min"),
        (["answerable", "supported_answer_rate"], "min"),
        (["answerable", "citation_support_rate"], "min"),
        (["unanswerable", "abstention_accuracy"], "min"),
    ]
    for path, kind in checks:
        b = get(baseline, path)
        c = get(summary, path)
        if b is None or c is None:
            continue
        if c + tol < b:
            failures.append(f"regression {'.'.join(path)}: current={c} < baseline={b}")
    # invalid citations and failures should not increase above baseline (usually 0)
    for path in [["quality", "invalid_citation_count_total"], ["quality", "failure_count"]]:
        b = get(baseline, path)
        c = get(summary, path)
        if b is None or c is None:
            continue
        if c > b:
            failures.append(f"regression {'.'.join(path)}: current={c} > baseline={b}")
    return (len(failures) == 0, failures)


def write_reports(out_dir: Path, payload: Dict[str, Any], comparison: List[Dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md = ["# CI Evaluation Report", "", f"Dataset: `{payload.get('dataset')}`", f"Generation mode: `{payload.get('generation_mode')}`", "", "## Retrieval comparison", ""]
    if comparison:
        headers = list(comparison[0].keys())
        md.append("| " + " | ".join(headers) + " |")
        md.append("|" + "|".join(["---"] * len(headers)) + "|")
        for row in comparison:
            md.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    md.append("")
    for mode_run in payload.get("mode_runs", []):
        md.append(f"## Mode: {mode_run['retrieval_mode']}")
        md.append("```json")
        md.append(json.dumps(mode_run["summary"], indent=2))
        md.append("```")
        md.append("### Failed/problematic questions")
        for r in mode_run["results"]:
            problems = []
            if r["type"] == "answerable":
                if r.get("abstained"):
                    problems.append("abstained")
                if not r["answer"].get("answer_supported"):
                    problems.append("unsupported")
                if r["answer"].get("invalid_citation_count", 0):
                    problems.append("invalid_citations")
                k_val = payload.get('k', 5)
                recall_val = r.get('retrieval', {}).get(f"recall@{k_val}", 1)
                if recall_val is not None and recall_val < 1:
                    problems.append(f"recall={recall_val}")
            else:
                if not r.get("correct_abstention"):
                    problems.append("should_have_abstained")
            if problems or r.get("errors"):
                md.append(f"- **{r['id']}**: {', '.join(problems) or 'ok'}; errors={r.get('errors')}")
        md.append("")
    if payload.get("regression"):
        md.append("## Regression gate")
        md.append(f"- passed: {payload['regression']['passed']}")
        for f in payload["regression"].get("failures", []):
            md.append(f"- {f}")
    if payload.get("ragas"):
        md.append("## RAGAS")
        md.append("```json")
        md.append(json.dumps(payload["ragas"], indent=2))
        md.append("```")
    (out_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=ROOT / "evaluation" / "ci_golden_dataset.json")
    ap.add_argument("--out", type=Path, default=ROOT / "evaluation" / "results" / "ci")
    ap.add_argument("--retrieval-modes", default="dense,hybrid")
    ap.add_argument("--generation-mode", default="extractive", choices=["extractive", "gemini"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--baseline", type=Path, default=ROOT / "evaluation" / "baseline_metrics.json")
    ap.add_argument("--check-regression", action="store_true")
    ap.add_argument("--enable-ragas", action="store_true")
    ap.add_argument("--write-baseline", action="store_true", help="Write current dense-mode summary as baseline (use only with measured results)")
    args = ap.parse_args()

    dataset = load_ci_dataset(args.dataset)
    doc = extract_source_document(ROOT / "data" / "satcom-ngp.pdf")
    modes = [m.strip() for m in args.retrieval_modes.split(",") if m.strip()]
    config = AppConfig.from_env()
    service = RAGService(config=config)

    mode_runs = []
    for mode in modes:
        mode_runs.append(run_mode(service, dataset, doc, mode, args.k, generation_mode=args.generation_mode))
    comparison = compare_modes(mode_runs, args.k)

    payload: Dict[str, Any] = {
        "dataset": str(args.dataset),
        "dataset_version": dataset.get("version"),
        "dataset_id": dataset.get("dataset_id"),
        "verification_status": dataset.get("verification_status"),
        "generation_mode": args.generation_mode,
        "k": args.k,
        "retrieval_modes": modes,
        "comparison": comparison,
        "mode_runs": mode_runs,
        "timestamp": time.time(),
    }

    # RAGAS on primary mode run (first)
    ragas_enabled = args.enable_ragas or (os.getenv("ENABLE_RAGAS", "").lower() in {"1", "true", "yes"})
    ragas = run_ragas(mode_runs[0]["results"] if mode_runs else [], ragas_enabled)
    payload["ragas"] = ragas

    regression = None
    if args.write_baseline and mode_runs:
        dense = next((m for m in mode_runs if m["retrieval_mode"] == "dense"), mode_runs[0])
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(json.dumps({"retrieval_mode": dense["retrieval_mode"], "k": args.k, "summary": dense["summary"]}, indent=2), encoding="utf-8")
        payload["baseline_written"] = str(args.baseline)
    if args.check_regression and args.baseline.exists() and mode_runs:
        base = json.loads(args.baseline.read_text(encoding="utf-8"))
        dense = next((m for m in mode_runs if m["retrieval_mode"] == "dense"), mode_runs[0])
        passed, failures = check_regression(dense["summary"], base.get("summary", base), args.k)
        regression = {"passed": passed, "failures": failures, "baseline": str(args.baseline)}
        payload["regression"] = regression

    write_reports(args.out, payload, comparison)
    print(json.dumps({"comparison": comparison, "regression": regression, "ragas_error": ragas.get("error")}, indent=2))
    if regression and not regression["passed"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
