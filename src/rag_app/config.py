from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.rag_core import DEFAULT_EMBEDDING_MODEL

ROOT = Path(__file__).resolve().parents[2]


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AppConfig:
    """Runtime configuration for the validated RAG application."""

    pdf_path: Path = ROOT / "data" / "satcom-ngp.pdf"
    persist_dir: Path = ROOT / "chroma_db_production"
    collection_name: str = "satcom_ngp_production_fixed_char_800_160"
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    top_k: int = 5
    # Defaults: EXP-003 production behavior, no opt-in experimental candidates.
    sibling_score_ratio: Optional[float] = 0.5
    cross_reference_selection: bool = False
    associated_heading_units: bool = False
    definition_intent_boost: bool = False
    clause_body_units: bool = False
    # Retrieval (final architecture supports dense, hybrid, hybrid_rerank)
    retrieval_mode: str = "hybrid"  # dense|hybrid|hybrid_rerank (selected by measured comparison on CI set)
    dense_k: int = 20
    bm25_k: int = 20
    rrf_k: int = 60
    rerank_model: str = "rerank-english-v3.0"
    rerank_candidate_k: int = 20
    # Generation (M1 path, not accepted for quality; optional).
    default_generation_mode: str = "extractive"  # extractive|gemini
    gemini_model: str = "gemini-flash-lite-latest"
    gemini_temperature: float = 0.0
    gemini_max_output_tokens: int = 1024
    gemini_sufficiency_threshold: float = 0.5
    allow_gemini: bool = True
    # Observability
    langfuse_enabled: bool = True
    langfuse_host: Optional[str] = None
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    # Chunking/indexing (configurable, reproducible)
    # Selected by controlled chunking experiment (experiments/results/chunking_experiment):
    # 800/160 fixed-character baseline outperformed 500/100,1000/200,1200/240 on CI answerable
    # retrieval (recall@5 0.9236, mrr@5 0.8611) with same PDF/embedding/retrieval settings.
    chunk_size: int = 800
    chunk_overlap: int = 160
    chunk_strategy: str = "fixed_character"  # fixed_character|structure_aware

    @classmethod
    def from_env(cls) -> "AppConfig":
        root = Path(os.getenv("RAG_ROOT", str(ROOT)))
        return cls(
            pdf_path=Path(os.getenv("RAG_PDF_PATH", str(root / "data/satcom-ngp.pdf"))),
            persist_dir=Path(os.getenv("RAG_PERSIST_DIR", str(root / "chroma_db_production"))),
            collection_name=os.getenv("RAG_COLLECTION_NAME", "satcom_ngp_production_fixed_char_800_160"),
            embedding_model=os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            top_k=int(os.getenv("RAG_TOP_K", "5")),
            retrieval_mode=os.getenv("RAG_RETRIEVAL_MODE", "hybrid"),
            dense_k=int(os.getenv("RAG_DENSE_K", "20")),
            bm25_k=int(os.getenv("RAG_BM25_K", "20")),
            rrf_k=int(os.getenv("RAG_RRF_K", "60")),
            rerank_model=os.getenv("RAG_RERANK_MODEL", "rerank-english-v3.0"),
            rerank_candidate_k=int(os.getenv("RAG_RERANK_CANDIDATE_K", "20")),
            default_generation_mode=os.getenv("RAG_GENERATION_MODE", "extractive"),
            gemini_model=os.getenv("RAG_GEMINI_MODEL", "gemini-flash-lite-latest"),
            gemini_temperature=float(os.getenv("RAG_GEMINI_TEMPERATURE", "0.0")),
            gemini_max_output_tokens=int(os.getenv("RAG_GEMINI_MAX_OUTPUT_TOKENS", "1024")),
            gemini_sufficiency_threshold=float(os.getenv("RAG_GEMINI_SUFFICIENCY_THRESHOLD", "0.5")),
            allow_gemini=_env_bool("RAG_ALLOW_GEMINI", True),
            langfuse_enabled=_env_bool("LANGFUSE_ENABLED", True),
            langfuse_host=os.getenv("LANGFUSE_HOST"),
            langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "800")),
            chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "160")),
            chunk_strategy=os.getenv("RAG_CHUNK_STRATEGY", "fixed_character"),
        )
