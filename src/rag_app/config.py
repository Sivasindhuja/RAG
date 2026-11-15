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
    collection_name: str = "satcom_ngp_production_fixed_char"
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    top_k: int = 5
    # Defaults: EXP-003 production behavior, no opt-in experimental candidates.
    sibling_score_ratio: Optional[float] = 0.5
    cross_reference_selection: bool = False
    associated_heading_units: bool = False
    definition_intent_boost: bool = False
    clause_body_units: bool = False
    # Generation (M1 path, not accepted for quality; optional).
    default_generation_mode: str = "extractive"  # extractive|gemini
    gemini_model: str = "gemini-flash-lite-latest"
    gemini_temperature: float = 0.0
    gemini_max_output_tokens: int = 1024
    gemini_sufficiency_threshold: float = 0.5
    allow_gemini: bool = True

    @classmethod
    def from_env(cls) -> "AppConfig":
        root = Path(os.getenv("RAG_ROOT", str(ROOT)))
        return cls(
            pdf_path=Path(os.getenv("RAG_PDF_PATH", str(root / "data/satcom-ngp.pdf"))),
            persist_dir=Path(os.getenv("RAG_PERSIST_DIR", str(root / "chroma_db_production"))),
            collection_name=os.getenv("RAG_COLLECTION_NAME", "satcom_ngp_production_fixed_char"),
            embedding_model=os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            top_k=int(os.getenv("RAG_TOP_K", "5")),
            default_generation_mode=os.getenv("RAG_GENERATION_MODE", "extractive"),
            gemini_model=os.getenv("RAG_GEMINI_MODEL", "gemini-flash-lite-latest"),
            gemini_temperature=float(os.getenv("RAG_GEMINI_TEMPERATURE", "0.0")),
            gemini_max_output_tokens=int(os.getenv("RAG_GEMINI_MAX_OUTPUT_TOKENS", "1024")),
            gemini_sufficiency_threshold=float(os.getenv("RAG_GEMINI_SUFFICIENCY_THRESHOLD", "0.5")),
            allow_gemini=_env_bool("RAG_ALLOW_GEMINI", True),
        )
