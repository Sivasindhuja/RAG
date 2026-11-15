"""FastAPI application for the SATCOM NGP RAG system (WP0/WP1).

Uses validated rag_core components: corrected production fixed-character index,
EXP-003 extractive selection (default), app-side citation resolution/validation,
explicit abstention, and optional Gemini generation with safe fallback.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # noqa: BLE001
    pass

from src.rag_app import AppConfig, RAGService
from src.rag_app.metrics import get_metrics_collector
from src.rag_app.schemas import HealthResponse, QueryRequest, QueryResponse

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
    allow_origins=os.getenv("RAG_CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
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


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        return get_service().health()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Health check failed: {type(exc).__name__}: {exc}")


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
        response = service.query(question=request.question, mode=request.mode, top_k=request.top_k)
        return response
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Query failed: {type(exc).__name__}: {exc}")


# Compatibility alias for local/manual checks; prefer /query.
@app.post("/ask", response_model=QueryResponse)
def ask(request: QueryRequest) -> QueryResponse:
    return query(request)
