from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional


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
    d0 = vals[f] * (c - k)
    d1 = vals[c] * (k - f)
    return round(d0 + d1, 3)


@dataclass
class RequestRecord:
    request_id: str
    ts: float
    status: str
    abstained: bool
    mode: str
    total_ms: Optional[float]
    stage_timings_ms: Dict[str, float]
    citation_count: int
    statement_count: int
    retrieved_count: int
    selected_count: int
    citation_validation_errors: int
    error_count: int
    failure: bool
    generation_attempted: bool
    generation_status: Optional[str]
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    estimated_cost_usd: Optional[float] = None


class MetricsCollector:
    """Lightweight in-memory metrics collector (exportable, no paid dependencies)."""

    def __init__(self, max_records: int = 1000):
        self._lock = threading.Lock()
        self._records: Deque[RequestRecord] = deque(maxlen=max_records)
        self._started_at = time.time()
        self._requests_total = 0
        self._failures_total = 0
        self._abstentions_total = 0
        self._answered_total = 0
        self._errors_total = 0
        self._generation_attempts = 0
        self._generation_errors = 0

    def record(self, rec: RequestRecord) -> None:
        with self._lock:
            self._records.append(rec)
            self._requests_total += 1
            if rec.failure or rec.status == "error":
                self._failures_total += 1
            if rec.abstained or rec.status == "abstained":
                self._abstentions_total += 1
            if rec.status == "answered" and not rec.abstained:
                self._answered_total += 1
            self._errors_total += rec.error_count
            if rec.generation_attempted:
                self._generation_attempts += 1
            if rec.generation_status == "generation_error":
                self._generation_errors += 1

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            records = list(self._records)
            totals = {
                "requests_total": self._requests_total,
                "failures_total": self._failures_total,
                "abstentions_total": self._abstentions_total,
                "answered_total": self._answered_total,
                "errors_total": self._errors_total,
                "generation_attempts_total": self._generation_attempts,
                "generation_errors_total": self._generation_errors,
            }
        n = len(records)
        latencies = [r.total_ms for r in records if r.total_ms is not None]
        retrieval = [r.stage_timings_ms.get("retrieval") for r in records if r.stage_timings_ms.get("retrieval") is not None]
        selection = [r.stage_timings_ms.get("selection_and_extractive") for r in records if r.stage_timings_ms.get("selection_and_extractive") is not None]
        generation = [r.stage_timings_ms.get("gemini_generation") for r in records if r.stage_timings_ms.get("gemini_generation") is not None]
        prompt_tokens = [float(r.prompt_tokens) for r in records if r.prompt_tokens is not None]
        completion_tokens = [float(r.completion_tokens) for r in records if r.completion_tokens is not None]
        total_tokens = [float(r.total_tokens) for r in records if r.total_tokens is not None]
        costs = [float(r.estimated_cost_usd) for r in records if r.estimated_cost_usd is not None]
        citation_coverages = []
        for r in records:
            # citation coverage proxy: answered requests with >=1 citation/statement
            if r.status == "answered" and not r.abstained:
                citation_coverages.append(1.0 if r.citation_count > 0 else 0.0)
        abstention_rate = (totals["abstentions_total"] / totals["requests_total"]) if totals["requests_total"] else None
        failure_rate = (totals["failures_total"] / totals["requests_total"]) if totals["requests_total"] else None
        citation_cov = (sum(citation_coverages) / len(citation_coverages)) if citation_coverages else None

        def _agg(vals: List[float]) -> Dict[str, Any]:
            if not vals:
                return {"count": 0, "total": None, "mean": None, "p50": None, "p95": None}
            return {
                "count": len(vals),
                "total": round(sum(vals), 6),
                "mean": round(sum(vals) / len(vals), 6),
                "p50": _percentile(vals, 0.5),
                "p95": _percentile(vals, 0.95),
            }

        return {
            "started_at": self._started_at,
            "uptime_s": round(time.time() - self._started_at, 3),
            "window_size": n,
            "totals": totals,
            "rates": {
                "abstention_rate": round(abstention_rate, 4) if abstention_rate is not None else None,
                "failure_rate": round(failure_rate, 4) if failure_rate is not None else None,
                "answered_citation_coverage_rate": round(citation_cov, 4) if citation_cov is not None else None,
            },
            "latency_ms": {
                "total": {"count": len(latencies), "p50": _percentile(latencies, 0.5), "p95": _percentile(latencies, 0.95), "mean": round(sum(latencies)/len(latencies), 3) if latencies else None},
                "retrieval": {"count": len(retrieval), "p50": _percentile(retrieval, 0.5), "p95": _percentile(retrieval, 0.95)},
                "selection_and_extractive": {"count": len(selection), "p50": _percentile(selection, 0.5), "p95": _percentile(selection, 0.95)},
                "gemini_generation": {"count": len(generation), "p50": _percentile(generation, 0.5), "p95": _percentile(generation, 0.95)},
            },
            "token_consumption": {
                "prompt_tokens": _agg(prompt_tokens),
                "completion_tokens": _agg(completion_tokens),
                "total_tokens": _agg(total_tokens),
                "note": "Aggregated from requests reporting usage (typically Gemini). Null totals/means indicate unavailable usage data, not zero.",
            },
            "cost_per_request_usd": {
                **_agg(costs),
                "note": "Estimated cost is null when pricing/usage is unavailable; null does not mean free.",
            },
            "recent": [r.__dict__ for r in list(records)[-20:]],
        }

    def reset(self) -> None:
        with self._lock:
            self._records.clear()
            self._started_at = time.time()
            self._requests_total = 0
            self._failures_total = 0
            self._abstentions_total = 0
            self._answered_total = 0
            self._errors_total = 0
            self._generation_attempts = 0
            self._generation_errors = 0


_COLLECTOR: Optional[MetricsCollector] = None
_COLLECTOR_LOCK = threading.Lock()


def get_metrics_collector() -> MetricsCollector:
    global _COLLECTOR
    with _COLLECTOR_LOCK:
        if _COLLECTOR is None:
            _COLLECTOR = MetricsCollector()
        return _COLLECTOR
