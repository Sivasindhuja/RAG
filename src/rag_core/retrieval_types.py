"""Lightweight shared retrieval types.

This module must remain import-light: no chromadb, rank_bm25, sentence-transformers,
torch, numpy, or other heavy runtime dependencies. It exists so Weaviate and other
backends can share result types without pulling local index/retrieval stacks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class RetrievedChunk:
    """One ranked retrieval hit with enough metadata for evaluation and citation."""

    rank: int
    chunk_id: str
    text: str
    source: str
    start_offset: int
    end_offset: int
    start_page: int
    end_page: int
    article: Optional[str]
    section: Optional[str]
    heading: Optional[str]
    distance: Optional[float]
    metadata: dict[str, Any]


@dataclass
class RetrievalResult:
    hits: List[RetrievedChunk]
    debug: Dict[str, Any]
