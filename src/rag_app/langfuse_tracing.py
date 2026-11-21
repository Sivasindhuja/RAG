"""Langfuse tracing integration (Langfuse Python SDK v4, OTEL-based).

Best-practice oriented:
- Creates a trace/root observation per request with meaningful name and metadata.
- Captures pipeline stage observations (retrieval, selection, sufficiency, generation, citation validation).
- Records inputs/outputs and status where useful.
- Flushes at request end; failures in tracing never break request handling.
- No-op when Langfuse is disabled/unavailable or credentials are missing.

The wrapper keeps a small interface compatible with the service layer.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional


def _truthy(val: Optional[str]) -> bool:
    return str(val or "").strip().lower() in {"1", "true", "yes", "on"}


def _safe(value: Any, max_str: int = 4000) -> Any:
    """Best-effort size limiting to avoid oversized trace payloads."""
    try:
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            return value if len(value) <= max_str else value[:max_str] + "...[truncated]"
        if isinstance(value, dict):
            out = {}
            for i, (k, v) in enumerate(value.items()):
                if i >= 50:
                    out["__truncated__"] = f"{len(value) - 50} more keys"
                    break
                out[str(k)] = _safe(v, max_str=max_str)
            return out
        if isinstance(value, (list, tuple)):
            items = [_safe(v, max_str=max_str) for v in list(value)[:50]]
            if len(value) > 50:
                items.append(f"...[{len(value) - 50} more items]")
            return items
        s = str(value)
        return s if len(s) <= max_str else s[:max_str] + "...[truncated]"
    except Exception:
        return "<unserializable>"


@dataclass
class LangfuseSpanContext:
    trace: Any = None
    span: Any = None

    @property
    def observation(self) -> Any:
        return self.span or self.trace

    def update(self, **kwargs: Any) -> None:
        obj = self.observation
        if obj is None:
            return
        payload = {}
        for k, v in kwargs.items():
            if k in {"input", "output", "metadata"}:
                payload[k] = _safe(v)
            else:
                payload[k] = v
        try:
            if hasattr(obj, "update"):
                obj.update(**payload)
        except Exception:
            return
        # Trace-level IO: avoid deprecated APIs; root observation update carries output in v4.

    def end(self, **kwargs: Any) -> None:
        obj = self.span
        if obj is None:
            return
        if kwargs:
            self.update(**kwargs)
        try:
            if hasattr(obj, "end"):
                obj.end()
        except Exception:
            return


_STAGE_AS_TYPE = {
    "retrieval": "retriever",
    "evidence_selection": "chain",
    "evidence_sufficiency": "evaluator",
    "generation": "generation",
    "gemini_generation": "generation",
    "citation_validation": "evaluator",
}


class LangfuseTracer:
    def __init__(
        self,
        enabled: bool = True,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None,
    ):
        self.enabled_flag = enabled
        self.public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        self.secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
        self.host = host or os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL")
        self._client = None
        self._available = False
        self._init_error: Optional[str] = None
        self._init()

    def _init(self) -> None:
        env_enabled = os.getenv("LANGFUSE_ENABLED")
        if env_enabled is not None and not _truthy(env_enabled):
            self.enabled_flag = False
        if not self.enabled_flag:
            return
        # Require keys for live tracing; if missing, stay no-op.
        if not self.public_key or not self.secret_key:
            self._init_error = "missing_credentials"
            return
        try:
            from langfuse import Langfuse  # type: ignore

            kwargs: Dict[str, Any] = {
                "public_key": self.public_key,
                "secret_key": self.secret_key,
            }
            if self.host:
                # v4 uses base_url/host compatibility in some environments; set both safely
                kwargs["host"] = self.host
            self._client = Langfuse(**kwargs)
            self._available = True
        except Exception as exc:  # noqa: BLE001
            self._init_error = f"{type(exc).__name__}: {exc}"
            self._client = None
            self._available = False

    @property
    def available(self) -> bool:
        return bool(self._available and self._client is not None)

    def status(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled_flag,
            "available": self.available,
            "host": self.host,
            "init_error": self._init_error,
            "has_public_key": bool(self.public_key),
            "has_secret_key": bool(self.secret_key),
            "sdk": "langfuse-v4",
        }

    def start_trace(self, name: str, **kwargs: Any) -> LangfuseSpanContext:
        if not self.available:
            return LangfuseSpanContext()
        try:
            obs_kwargs: Dict[str, Any] = {"name": name, "as_type": "span"}
            if "input" in kwargs:
                obs_kwargs["input"] = _safe(kwargs.pop("input"))
            if "metadata" in kwargs:
                obs_kwargs["metadata"] = _safe(kwargs.pop("metadata"))
            # remaining kwargs ignored safely
            trace = self._client.start_observation(**obs_kwargs)
            ctx = LangfuseSpanContext(trace=trace)
            # try to set trace-level metadata/name where supported
            try:
                if hasattr(trace, "update"):
                    trace.update(metadata=_safe(kwargs.get("metadata")) if "metadata" in kwargs else None)
            except Exception:
                pass
            return ctx
        except Exception:
            return LangfuseSpanContext()

    @contextmanager
    def span(self, trace_ctx: LangfuseSpanContext, name: str, **kwargs: Any) -> Iterator[LangfuseSpanContext]:
        if not self.available or trace_ctx.trace is None:
            yield LangfuseSpanContext(trace=trace_ctx.trace)
            return
        as_type = kwargs.pop("as_type", None) or _STAGE_AS_TYPE.get(name, "span")
        obs_kwargs: Dict[str, Any] = {"name": name, "as_type": as_type}
        for key in ("input", "output", "metadata", "model", "model_parameters", "usage_details", "level", "status_message"):
            if key in kwargs and kwargs[key] is not None:
                obs_kwargs[key] = _safe(kwargs[key]) if key in {"input", "output", "metadata"} else kwargs[key]
        span_obj = None
        try:
            parent = trace_ctx.trace
            if hasattr(parent, "start_observation"):
                span_obj = parent.start_observation(**obs_kwargs)
            elif hasattr(self._client, "start_observation"):
                span_obj = self._client.start_observation(**obs_kwargs)
        except Exception:
            span_obj = None
        ctx = LangfuseSpanContext(trace=trace_ctx.trace, span=span_obj)
        try:
            yield ctx
        except Exception as exc:  # noqa: BLE001
            try:
                ctx.update(level="ERROR", status_message=str(exc), metadata={"error_type": type(exc).__name__})
            except Exception:
                pass
            raise
        finally:
            # if output provided post-yield via update, it is already set; ensure end
            try:
                if span_obj is not None and hasattr(span_obj, "end"):
                    span_obj.end()
            except Exception:
                pass

    def flush(self) -> None:
        if not self.available:
            return
        try:
            self._client.flush()
        except Exception:
            return


_GLOBAL_TRACER: Optional[LangfuseTracer] = None


def get_langfuse_tracer(
    enabled: bool = True,
    public_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    host: Optional[str] = None,
) -> LangfuseTracer:
    global _GLOBAL_TRACER
    if _GLOBAL_TRACER is None:
        _GLOBAL_TRACER = LangfuseTracer(enabled=enabled, public_key=public_key, secret_key=secret_key, host=host)
    return _GLOBAL_TRACER
