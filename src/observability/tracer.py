"""
Langfuse observability for every RAG query.

Each query trace includes: expansion, retrieval chunks, rerank scores, generation prompt/response, tokens, cost.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document

_ENABLED = os.getenv("LANGFUSE_ENABLED", "true").lower() in ("1", "true", "yes")


def _doc_preview(doc: Document, max_chars: int = 400) -> dict:
    return {
        "page": doc.metadata.get("page"),
        "section": doc.metadata.get("section"),
        "parent_id": doc.metadata.get("parent_id"),
        "text_preview": doc.page_content[:max_chars],
        "chars": len(doc.page_content),
    }


@dataclass
class LatencyBuffer:
    values: list[float] = field(default_factory=list)

    def add(self, ms: float):
        self.values.append(ms)

    def percentiles(self) -> dict[str, float]:
        if not self.values:
            return {"p50_ms": 0.0, "p95_ms": 0.0, "count": 0}
        s = sorted(self.values)
        n = len(s)

        def pct(p: float) -> float:
            idx = min(int(round(p * (n - 1))), n - 1)
            return round(s[idx], 2)

        return {"p50_ms": pct(0.50), "p95_ms": pct(0.95), "count": n, "mean_ms": round(sum(s) / n, 2)}


class RAGTracer:
    def __init__(self):
        self._client = None
        self._enabled = _ENABLED
        self.latency = LatencyBuffer()
        self.request_costs_usd: list[float] = []
        if self._enabled:
            try:
                from langfuse import get_client
                self._client = get_client()
            except Exception:
                self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled and self._client is not None

    def trace_query(self, *, question: str, user_id: str = "satcom-user", session_id: str | None = None):
        if not self.enabled:
            return _NoopTrace()
        return _LangfuseTrace(self._client, question, self)

    def flush(self):
        if self.enabled:
            self._client.flush()


class _NoopTrace:
    def span(self, name: str, as_type: str = "span", **kwargs):
        return _NoopSpan()

    def set_metadata(self, **kwargs):
        pass

    def add_cost(self, _usd: float):
        pass


class _NoopSpan:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def update(self, **kwargs):
        pass


class _LangfuseTrace:
    def __init__(self, client, question: str, aggregator: RAGTracer):
        self._client = client
        self._question = question
        self._agg = aggregator
        self._ctx = None
        self._obs = None
        self._total_cost = 0.0
        self._t0 = 0.0
        self._meta: dict[str, Any] = {}

    def __enter__(self):
        self._t0 = time.perf_counter()
        self._ctx = self._client.start_as_current_observation(
            as_type="span",
            name="rag-query",
            input={"question": self._question},
        )
        self._obs = self._ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        ms = (time.perf_counter() - self._t0) * 1000
        self._agg.latency.add(ms)
        self._agg.request_costs_usd.append(self._total_cost)
        out = {"total_latency_ms": round(ms, 2), "cost_usd": round(self._total_cost, 6), **self._meta}
        if exc:
            out["error"] = str(exc)
        self._obs.update(output=out, metadata=self._meta)
        return self._ctx.__exit__(exc_type, exc, tb)

    def span(self, name: str, as_type: str = "span", **kwargs):
        return _LangfuseSpan(self._client, name, as_type, self, **kwargs)

    def add_cost(self, usd: float):
        self._total_cost += usd

    def set_metadata(self, **kwargs):
        self._meta.update(kwargs)


class _LangfuseSpan:
    def __init__(self, client, name: str, as_type: str, trace: _LangfuseTrace, **kwargs):
        self._client = client
        self._name = name
        self._as_type = as_type
        self._trace = trace
        self._kwargs = kwargs
        self._ctx = None
        self._obs = None
        self._t0 = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        self._ctx = self._client.start_as_current_observation(
            as_type=self._as_type,
            name=self._name,
            **self._kwargs,
        )
        self._obs = self._ctx.__enter__()
        return self._obs

    def __exit__(self, exc_type, exc, tb):
        ms = (time.perf_counter() - self._t0) * 1000
        self._trace._meta[f"{self._name}_latency_ms"] = round(ms, 2)
        return self._ctx.__exit__(exc_type, exc, tb)

    def update(self, **kwargs):
        self._obs.update(**kwargs)


_tracer: RAGTracer | None = None


def get_tracer() -> RAGTracer:
    global _tracer
    if _tracer is None:
        _tracer = RAGTracer()
    return _tracer
