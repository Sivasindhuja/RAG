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


def _env_bool_strict(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"Invalid boolean value for {name}: {value!r}. "
        "Expected one of: true/false, 1/0, yes/no, on/off."
    )


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
    # Experimental opt-in backends (default remains existing production backend).
    retriever_backend: str = "chroma"  # chroma|weaviate (experimental, opt-in)
    weaviate_url: Optional[str] = None
    weaviate_api_key: Optional[str] = None
    weaviate_collection: str = "SatcomNgpExp800160"
    weaviate_embedding_model: str = "Snowflake/snowflake-arctic-embed-l-v2.0"
    # Experimental Weaviate connection resilience; defaults match weaviate-client v4 defaults.
    weaviate_init_timeout_s: float = 2.0
    weaviate_query_timeout_s: float = 30.0
    # Experimental Weaviate retrieval controls. Defaults preserve current hybrid behavior.
    weaviate_alpha: float = 0.5
    weaviate_search_mode: str = "hybrid"  # dense|bm25|hybrid
    # Experimental Weaviate abstention policy separation. Default off preserves production behavior.
    weaviate_abstention_experimental: bool = False
    weaviate_abstention_missing_distance_behavior: str = "skip"  # skip|reject|score
    weaviate_abstention_min_retrieval_score: Optional[float] = None
    # Generation (M1 path, not accepted for quality; optional).
    default_generation_mode: str = "gemini"  # extractive|gemini
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
    # Selected by controlled chunking experiment (removed-experiment-artifacts/chunking_experiment; Task 16 cleanup, see git history):
    # 800/160 fixed-character baseline outperformed 500/100,1000/200,1200/240 on CI answerable
    # retrieval (recall@5 0.9236, mrr@5 0.8611) with same PDF/embedding/retrieval settings.
    chunk_size: int = 800
    chunk_overlap: int = 160
    chunk_strategy: str = "fixed_character"  # fixed_character|structure_aware
    # Deployment hardening: protect /ingest (disabled by default; fail closed).
    ingest_token: Optional[str] = None

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
            retriever_backend=os.getenv("RAG_RETRIEVER_BACKEND", os.getenv("RETRIEVER_BACKEND", "chroma")),
            weaviate_url=os.getenv("WEAVIATE_URL") or os.getenv("RAG_WEAVIATE_URL"),
            weaviate_api_key=os.getenv("WEAVIATE_API_KEY") or os.getenv("RAG_WEAVIATE_API_KEY"),
            weaviate_collection=os.getenv("WEAVIATE_COLLECTION") or os.getenv("RAG_WEAVIATE_COLLECTION") or "SatcomNgpExp800160",
            weaviate_embedding_model=os.getenv("WEAVIATE_EMBEDDING_MODEL") or os.getenv("RAG_WEAVIATE_EMBEDDING_MODEL") or "Snowflake/snowflake-arctic-embed-l-v2.0",
            weaviate_init_timeout_s=float(os.getenv("WEAVIATE_INIT_TIMEOUT_S") or os.getenv("RAG_WEAVIATE_INIT_TIMEOUT_S") or "2.0"),
            weaviate_query_timeout_s=float(os.getenv("WEAVIATE_QUERY_TIMEOUT_S") or os.getenv("RAG_WEAVIATE_QUERY_TIMEOUT_S") or "30.0"),
            weaviate_alpha=float(os.getenv("WEAVIATE_ALPHA") or os.getenv("RAG_WEAVIATE_ALPHA") or "0.5"),
            weaviate_search_mode=(os.getenv("WEAVIATE_SEARCH_MODE") or os.getenv("RAG_WEAVIATE_SEARCH_MODE") or "hybrid"),
            weaviate_abstention_experimental=_env_bool("RAG_WEAVIATE_ABSTENTION_EXPERIMENTAL", False),
            weaviate_abstention_missing_distance_behavior=(os.getenv("RAG_WEAVIATE_ABSTENTION_MISSING_DISTANCE_BEHAVIOR") or "skip"),
            weaviate_abstention_min_retrieval_score=(
                float(os.getenv("RAG_WEAVIATE_ABSTENTION_MIN_RETRIEVAL_SCORE"))
                if (os.getenv("RAG_WEAVIATE_ABSTENTION_MIN_RETRIEVAL_SCORE") or "").strip() != ""
                else None
            ),
            default_generation_mode=os.getenv("RAG_GENERATION_MODE", "gemini"),
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
            ingest_token=os.getenv("RAG_INGEST_TOKEN"),
            definition_intent_boost=_env_bool_strict("RAG_DEFINITION_INTENT_BOOST", False),
        )
