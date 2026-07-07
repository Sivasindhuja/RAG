"""Dense (Chroma) + sparse (BM25) fusion with debug payloads for observability."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.config import SETTINGS


@dataclass
class RetrievalDebug:
    vector_parents: list[Document]
    bm25_children: list[Document]
    bm25_scores: list[tuple[int, float]]
    merged: list[Document]


class HybridRetriever:
    def __init__(self, hierarchical, child_chunks: list[Document]):
        self.hier = hierarchical
        self.child_chunks = child_chunks
        self.bm25 = BM25Okapi([self._tokenize(doc.page_content) for doc in child_chunks])

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [t.lower() for t in text.split() if len(t) > 1]

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        use_hybrid: bool = True,
        *,
        return_debug: bool = False,
    ) -> list[Document] | tuple[list[Document], RetrievalDebug]:
        k = k or SETTINGS["retrieval"]["hybrid_k"]
        vector_parents = self.hier.get_relevant_documents(query, k=k)
        bm25_children: list[Document] = []
        bm25_scores: list[tuple[int, float]] = []

        if use_hybrid:
            tokens = self._tokenize(query)
            scores = self.bm25.get_scores(tokens)
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
            bm25_scores = [(i, float(scores[i])) for i in top_indices]
            bm25_children = [self.child_chunks[i] for i in top_indices]

        combined = vector_parents + bm25_children
        seen: set[str] = set()
        unique: list[Document] = []
        for doc in combined:
            key = doc.page_content[:200]
            if key not in seen:
                seen.add(key)
                unique.append(doc)

        if return_debug:
            return unique, RetrievalDebug(vector_parents, bm25_children, bm25_scores, unique)
        return unique
