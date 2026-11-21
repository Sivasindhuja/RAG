"""FastAPI application for the SATCOM NGP RAG system (WP0/WP1).

Uses validated rag_core components: corrected production fixed-character index,
EXP-003 extractive selection (default), app-side citation resolution/validation,
explicit abstention, and optional Gemini generation with safe fallback.
"""

from __future__ import annotations

import json
import os
import secrets
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # noqa: BLE001
    pass

from src.rag_app import AppConfig, RAGService
from src.rag_app.metrics import get_metrics_collector
from src.rag_app.schemas import HealthResponse, QueryRequest, QueryResponse

DEFAULT_DEV_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]


def _parse_cors_origins(raw: str | None) -> list[str]:
    """Parse comma-separated CORS origins from env.

    - Unset/empty: safe local-dev defaults (no wildcard production default).
    - Explicit "*": wildcard allowed only when explicitly configured.
    - Otherwise: comma-separated explicit origins; whitespace trimmed; empties ignored.
    """
    if raw is None:
        return list(DEFAULT_DEV_CORS_ORIGINS)
    items = [part.strip() for part in raw.split(",")]
    origins = [item for item in items if item]
    if not origins:
        return list(DEFAULT_DEV_CORS_ORIGINS)
    if "*" in origins:
        return ["*"]
    return origins


_cors_origins = _parse_cors_origins(os.getenv("RAG_CORS_ORIGINS"))
_cors_allow_credentials = "*" not in _cors_origins

app = FastAPI(
    title="SATCOM NGP RAG API",
    description=(
        "Grounded RAG over data/satcom-ngp.pdf using validated core "
        "(fixed-char chunking, EXP-003 selection, app-validated citations)."
    ),
    version="0.2.0-wp1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_service() -> RAGService:
    config = AppConfig.from_env()
    return RAGService(config=config)


@app.get("/", response_model=Dict[str, Any])
def root() -> Dict[str, Any]:
    return {
        "service": "satcom-ngp-rag",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    """Lightweight liveness endpoint for deploy platforms/cold starts."""
    return {"status": "ok", "service": "satcom-ngp-rag"}


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        return get_service().health()
    except Exception as exc:  # noqa: BLE001
        # On cold starts / Render Free, first request may trigger model load or
        # index recovery. Return a structured 503 rather than crashing.
        raise HTTPException(status_code=503, detail=f"Health check failed: {type(exc).__name__}: {exc}")


@app.post("/ingest")
def ingest(x_ingest_token: Optional[str] = Header(default=None, alias="X-Ingest-Token")) -> Dict[str, Any]:
    """Rebuild the vector index from the configured PDF (protected; disabled by default)."""
    try:
        config = AppConfig.from_env()
        expected = (config.ingest_token or "").strip()
        if not expected:
            # Fail closed: public ingestion is disabled unless explicitly configured.
            raise HTTPException(status_code=403, detail="Ingestion endpoint is disabled. Configure RAG_INGEST_TOKEN to enable authorized ingestion.")
        provided = x_ingest_token or ""
        if not secrets.compare_digest(provided, expected):
            raise HTTPException(status_code=403, detail="Unauthorized ingestion request.")

        from src.rag_core import FixedCharacterChunker, StructureAwareChunker, audit_source_coverage, extract_source_document

        doc = extract_source_document(config.pdf_path)
        strategy = (config.chunk_strategy or "fixed_character").lower()
        if strategy == "structure_aware":
            chunker = StructureAwareChunker(max_chars=config.chunk_size)
            chunks = chunker.chunk(doc)
            audit = audit_source_coverage(doc, chunks, max_chars=config.chunk_size, allow_overlap=False)
        else:
            chunker = FixedCharacterChunker(chunk_size=config.chunk_size, overlap=config.chunk_overlap)
            chunks = chunker.chunk(doc)
            audit = audit_source_coverage(doc, chunks, max_chars=config.chunk_size, allow_overlap=True)
        if not audit.get("ok"):
            return {"status": "error", "audit": audit, "message": "Ingestion audit failed; index not rebuilt"}
        service = get_service()
        # rebuild index
        stats = service.index.build(chunks, reset=True)
        # reset retriever cache/corpus
        service._retriever = None
        return {
            "status": "ok",
            "pdf": str(config.pdf_path),
            "strategy": strategy,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap if strategy == "fixed_character" else 0,
            "chunk_count": len(chunks),
            "audit": {k: v for k, v in audit.items() if k in {"ok", "issues", "coverage_percent", "overlap_chars", "uncovered_non_whitespace_chars", "total_chars", "covered_chars"}},
            "index": stats,
            "collection": config.collection_name,
            "persist_dir": str(config.persist_dir),
        }
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {type(exc).__name__}: {exc}")


@app.get("/metrics")
def metrics() -> Dict[str, Any]:
    """Lightweight aggregated metrics snapshot (in-memory, exportable)."""
    try:
        return get_metrics_collector().snapshot()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Metrics unavailable: {type(exc).__name__}: {exc}")


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    try:
        service = get_service()
        response = service.query(
            question=request.question,
            mode=request.mode,
            top_k=request.top_k,
            retrieval_mode=getattr(request, "retrieval_mode", None),
        )
        return response
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Query failed: {type(exc).__name__}: {exc}")


# Compatibility alias for local/manual checks; prefer /query.
@app.post("/ask", response_model=QueryResponse)
def ask(request: QueryRequest) -> QueryResponse:
    return query(request)


def _read_json_if_exists(path: Path):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> str:
    root = Path(__file__).resolve().parents[1]
    live = get_metrics_collector().snapshot()
    ci = _read_json_if_exists(root / "evaluation" / "results" / "ci" / "report.json") or _read_json_if_exists(root / "evaluation" / "results" / "ci_baseline" / "report.json")
    chunking = _read_json_if_exists(root / "experiments" / "results" / "chunking_experiment" / "results.json")
    m4 = _read_json_if_exists(root / "experiments" / "results" / "m4_baseline" / "summary.json")

    def esc(s: Any) -> str:
        import html

        return html.escape(json.dumps(s, indent=2, ensure_ascii=False) if not isinstance(s, str) else s)

    sections = []
    sections.append(f"<h2>Live operational metrics (in-process)</h2><p>Source: <code>/metrics</code> runtime collector. Resets on restart; per-instance.</p><pre>{esc(live)}</pre>")
    if ci:
        sections.append(f"<h2>Offline CI evaluation (fixed 15-question suite)</h2><p>Source: <code>evaluation/results/ci*/report.json</code>. Distinguish from live metrics.</p><pre>{esc(ci.get('comparison') or ci)}</pre>")
    else:
        sections.append("<h2>Offline CI evaluation</h2><p>No CI report artifact found yet.</p>")
    if chunking:
        sections.append(f"<h2>Chunking experiment (offline)</h2><pre>{esc({k: chunking.get(k) for k in ['selected_config','selection_rationale','results'] if k in chunking})}</pre>")
    if m4:
        sections.append(f"<h2>M4 baseline (offline, historical preserved set)</h2><pre>{esc(m4)}</pre>")
    # highlight key live metrics for quick verification
    rates = live.get('rates') or {}
    lat = (live.get('latency_ms') or {}).get('total') or {}
    tok = (live.get('token_consumption') or {}).get('total_tokens') or {}
    cost = live.get('cost_per_request_usd') or {}
    key = {
        "requests_total": (live.get('totals') or {}).get('requests_total'),
        "failure_rate": rates.get('failure_rate'),
        "answered_citation_coverage_rate": rates.get('answered_citation_coverage_rate'),
        "latency_total_p50_ms": lat.get('p50'),
        "latency_total_p95_ms": lat.get('p95'),
        "token_consumption_total_requests_with_usage": tok.get('count'),
        "token_consumption_total_sum": tok.get('total'),
        "token_consumption_total_mean": tok.get('mean'),
        "cost_per_request_mean_usd": cost.get('mean'),
        "cost_per_request_p95_usd": cost.get('p95'),
    }
    sections.insert(0, f"<h2>Key live metrics (quick view)</h2><p><strong>Null/unavailable</strong> means not measured/reported (e.g. no Gemini usage, no pricing). <strong>0</strong> means measured zero. Values are from actual in-process requests.</p><pre>{esc(key)}</pre>")
    body = "\n".join(sections)
    return f"""<!doctype html>
<html><head><meta charset='utf-8'><title>SATCOM NGP RAG Dashboard</title>
<style>body{{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:24px;max-width:1200px}}pre{{background:#f6f8fa;padding:12px;overflow:auto;border:1px solid #ddd;border-radius:6px}}code{{background:#f6f8fa;padding:2px 4px;border-radius:4px}}h1,h2{{margin-top:1.2em}}</style>
</head><body>
<h1>SATCOM NGP RAG — Metrics Dashboard</h1>
<p>Shows live operational metrics and offline evaluation artifacts. Unavailable metrics are shown as <code>null</code> (not zero). No fabricated values.</p>
{body}
</body></html>"""
