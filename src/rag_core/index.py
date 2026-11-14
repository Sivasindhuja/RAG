"""Shared dense-index and retrieval interface for the SATCOM NGP RAG core.

The experiment baseline (EXP-001) and the production application must not grow
separate dense-retrieval implementations.  This module owns the small amount of
code needed to:

- convert canonical :class:`~src.rag_core.chunker.Chunk` objects into Chroma
  records;
- build or rebuild an isolated persistent collection; and
- query it with the configured sentence-transformer model.

No hybrid retrieval, query expansion or reranking lives here.  Those are
separate, experiment-driven layers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional

import chromadb

from .chunker import Chunk

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class DenseIndexConfig:
    """Configuration for one isolated dense vector collection."""

    collection_name: str
    persist_dir: Path
    embedding_model: str = DEFAULT_EMBEDDING_MODEL


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


def chunk_to_metadata(chunk: Chunk) -> dict[str, Any]:
    """Flatten a chunk into Chroma-compatible scalar metadata."""
    return {
        "chunk_id": chunk.chunk_id,
        "source": chunk.source,
        "article": chunk.article or "",
        "section": chunk.section or "",
        "section_hierarchy": ".".join(chunk.section_hierarchy) if chunk.section_hierarchy else "",
        "heading": chunk.heading or "",
        "start_page": int(chunk.start_page),
        "end_page": int(chunk.end_page),
        "start_offset": int(chunk.start_offset),
        "end_offset": int(chunk.end_offset),
        "chunk_sequence": int(chunk.chunk_sequence),
        "part": int(chunk.part),
        "chunking_strategy": chunk.chunking_strategy,
        "start_page_char": int(chunk.start_page_char),
        "end_page_char": int(chunk.end_page_char),
    }


def _encode(model: Any, texts: List[str]) -> List[List[float]]:
    vectors = model.encode(texts)
    if hasattr(vectors, "tolist"):
        vectors = vectors.tolist()
    return [list(vector) for vector in vectors]


class DenseIndex:
    """A persistent Chroma collection backed by a sentence-transformer model."""

    def __init__(self, config: DenseIndexConfig, model: Any = None):
        self.config = config
        self._model = model
        Path(config.persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(config.persist_dir))
        self._collection = self._client.get_or_create_collection(name=config.collection_name)

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.config.embedding_model)
        return self._model

    @property
    def collection(self) -> Any:
        return self._collection

    def count(self) -> int:
        return int(self._collection.count())

    def build(self, chunks: Iterable[Chunk], reset: bool = True) -> dict[str, Any]:
        """Embed and persist ``chunks``, returning build statistics."""
        chunk_list = list(chunks)
        if not chunk_list:
            raise ValueError("cannot build a dense index from zero chunks")
        if reset and self.count() > 0:
            existing = self._collection.get()
            ids = existing.get("ids") or []
            if ids:
                self._collection.delete(ids=list(ids))

        started = time.perf_counter()
        vectors = _encode(self.model, [chunk.text for chunk in chunk_list])
        self._collection.add(
            ids=[chunk.chunk_id for chunk in chunk_list],
            documents=[chunk.text for chunk in chunk_list],
            metadatas=[chunk_to_metadata(chunk) for chunk in chunk_list],
            embeddings=vectors,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return {"chunks": len(chunk_list), "build_ms": round(elapsed_ms, 2), "collection_count": self.count()}

    def search(self, query: str, k: int = 5) -> List[RetrievedChunk]:
        """Return the top-``k`` dense hits in ranked order."""
        if not query.strip():
            raise ValueError("query must not be empty")
        if self.count() == 0:
            return []
        query_vector = _encode(self.model, [query])[0]
        result = self._collection.query(
            query_embeddings=[query_vector],
            n_results=k,
        )
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0] if result.get("distances") else []

        hits: List[RetrievedChunk] = []
        for index, chunk_id in enumerate(ids):
            metadata = metadatas[index] if index < len(metadatas) else {}
            hits.append(
                RetrievedChunk(
                    rank=index + 1,
                    chunk_id=str(chunk_id),
                    text=str(documents[index]) if index < len(documents) else "",
                    source=str(metadata.get("source") or ""),
                    start_offset=int(metadata.get("start_offset") or 0),
                    end_offset=int(metadata.get("end_offset") or 0),
                    start_page=int(metadata.get("start_page") or 0),
                    end_page=int(metadata.get("end_page") or 0),
                    article=str(metadata.get("article") or "") or None,
                    section=str(metadata.get("section") or "") or None,
                    heading=str(metadata.get("heading") or "") or None,
                    distance=float(distances[index]) if index < len(distances) else None,
                    metadata=dict(metadata),
                )
            )
        return hits
