"""Hybrid retrieval (dense + BM25) and optional Cohere reranking for rag_core.

This module builds on the canonical dense Chroma index and chunk corpus stored in
Chroma metadata/documents. It returns ``RetrievedChunk`` objects compatible with
the existing grounded pipeline.

Fusion: Reciprocal Rank Fusion (RRF) over dense and BM25 ranked lists.
Reranking: Cohere Rerank API over fused candidates (optional).
"""

from __future__ import annotations

import math
import os
import re
import time
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .retrieval_types import RetrievalResult, RetrievedChunk

if TYPE_CHECKING:
    from rank_bm25 import BM25Okapi

    from .index import DenseIndex

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*")

DEFAULT_RRF_K = 60
DEFAULT_DENSE_K = 20
DEFAULT_BM25_K = 20
DEFAULT_COHERE_MODEL = "rerank-english-v3.0"


def tokenize(text: str) -> List[str]:
    # Simple alphanumeric/dotted tokenization to preserve clause-like tokens (e.g. 2.5.6).
    return [t.lower() for t in _WORD_RE.findall(text or "")]


class CorpusIndex:
    """In-memory corpus loaded from a persisted Chroma collection for BM25/fusion."""

    def __init__(self, dense_index: DenseIndex):
        self.dense_index = dense_index
        self._loaded = False
        self.ids: List[str] = []
        self.documents: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self._id_to_pos: Dict[str, int] = {}
        self._bm25: Optional[BM25Okapi] = None

    def load(self) -> None:
        if self._loaded:
            return
        col = self.dense_index.collection
        # Load all records from collection. include documents/metadatas (ids always returned).
        data = col.get(include=["documents", "metadatas"])
        ids = data.get("ids") or []
        docs = data.get("documents") or []
        metas = data.get("metadatas") or []
        # Ensure aligned lengths
        n = min(len(ids), len(docs), len(metas))
        self.ids = [str(x) for x in ids[:n]]
        self.documents = [str(x or "") for x in docs[:n]]
        self.metadatas = [dict(x or {}) for x in metas[:n]]
        self._id_to_pos = {cid: i for i, cid in enumerate(self.ids)}
        tokenized = [tokenize(d) for d in self.documents]
        if tokenized:
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi(tokenized)
        else:
            self._bm25 = None
        self._loaded = True

    @property
    def bm25(self) -> Optional[BM25Okapi]:
        self.load()
        return self._bm25

    def get(self, chunk_id: str) -> Optional[Tuple[str, str, Dict[str, Any]]]:
        self.load()
        pos = self._id_to_pos.get(chunk_id)
        if pos is None:
            return None
        return self.ids[pos], self.documents[pos], self.metadatas[pos]

    def bm25_search(self, query: str, k: int) -> List[Tuple[str, float, int]]:
        self.load()
        if not self._bm25 or k <= 0 or not query.strip():
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: float(x[1]), reverse=True)
        out: List[Tuple[str, float, int]] = []
        for rank, (idx, score) in enumerate(ranked[:k], start=1):
            if idx < 0 or idx >= len(self.ids):
                continue
            out.append((self.ids[idx], float(score), rank))
        return out

    def to_retrieved_chunk(self, chunk_id: str, rank: int, distance: Optional[float] = None) -> Optional[RetrievedChunk]:
        item = self.get(chunk_id)
        if item is None:
            return None
        cid, text, meta = item
        return RetrievedChunk(
            rank=rank,
            chunk_id=cid,
            text=text,
            source=str(meta.get("source") or "satcom-ngp.pdf"),
            start_offset=int(meta.get("start_offset") or 0),
            end_offset=int(meta.get("end_offset") or 0),
            start_page=int(meta.get("start_page") or 0),
            end_page=int(meta.get("end_page") or 0),
            article=str(meta.get("article") or "") or None,
            section=str(meta.get("section") or "") or None,
            heading=str(meta.get("heading") or "") or None,
            distance=distance,
            metadata=meta,
        )


def rrf_fuse(
    dense: Sequence[RetrievedChunk],
    bm25_ranked: Sequence[Tuple[str, float, int]],
    *,
    rrf_k: int = DEFAULT_RRF_K,
) -> List[Tuple[str, float, Dict[str, Any]]]:
    scores: Dict[str, float] = {}
    details: Dict[str, Dict[str, Any]] = {}
    for hit in dense:
        scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (rrf_k + hit.rank)
        d = details.setdefault(hit.chunk_id, {"dense_rank": None, "bm25_rank": None, "bm25_score": None, "dense_distance": None})
        d["dense_rank"] = hit.rank
        d["dense_distance"] = hit.distance
    for cid, bscore, brank in bm25_ranked:
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + brank)
        d = details.setdefault(cid, {"dense_rank": None, "bm25_rank": None, "bm25_score": None, "dense_distance": None})
        d["bm25_rank"] = brank
        d["bm25_score"] = bscore
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(cid, score, details.get(cid, {})) for cid, score in fused]


def _get_cohere_client():
    api_key = os.getenv("COHERE_API_KEY") or os.getenv("CO_API_KEY")
    if not api_key:
        raise RuntimeError("Cohere API key not configured (COHERE_API_KEY/CO_API_KEY)")
    import cohere  # lazy import

    return cohere.Client(api_key)


def cohere_rerank(
    query: str,
    candidates: List[RetrievedChunk],
    *,
    top_n: int,
    model: str = DEFAULT_COHERE_MODEL,
) -> Tuple[List[RetrievedChunk], Dict[str, Any]]:
    if not candidates:
        return [], {"model": model, "applied": False, "reason": "no_candidates"}
    docs = [c.text for c in candidates]
    started = time.perf_counter()
    client = _get_cohere_client()
    resp = client.rerank(model=model, query=query, documents=docs, top_n=min(top_n, len(docs)))
    latency_ms = (time.perf_counter() - started) * 1000.0
    # Build mapping index -> candidate
    by_index = {i: c for i, c in enumerate(candidates)}
    reranked: List[RetrievedChunk] = []
    scores: List[Dict[str, Any]] = []
    for new_rank, item in enumerate(getattr(resp, "results", []) or [], start=1):
        idx = getattr(item, "index", None)
        score = getattr(item, "relevance_score", None)
        cand = by_index.get(idx)
        if cand is None:
            continue
        # preserve metadata but update rank; keep distance as-is and store rerank score in metadata
        meta = dict(cand.metadata or {})
        if score is not None:
            meta["rerank_score"] = float(score)
        reranked.append(
            RetrievedChunk(
                rank=new_rank,
                chunk_id=cand.chunk_id,
                text=cand.text,
                source=cand.source,
                start_offset=cand.start_offset,
                end_offset=cand.end_offset,
                start_page=cand.start_page,
                end_page=cand.end_page,
                article=cand.article,
                section=cand.section,
                heading=cand.heading,
                distance=cand.distance,
                metadata=meta,
            )
        )
        scores.append({"chunk_id": cand.chunk_id, "index": idx, "relevance_score": float(score) if score is not None else None})
    debug = {
        "model": model,
        "applied": True,
        "latency_ms": round(latency_ms, 3),
        "scores": scores,
    }
    # capture meta/usage if available
    meta_obj = getattr(resp, "meta", None)
    if meta_obj is not None:
        try:
            debug["meta"] = getattr(meta_obj, "billed_units", None) or getattr(meta_obj, "__dict__", None) or str(meta_obj)
        except Exception:
            debug["meta"] = str(meta_obj)
    return reranked, debug


class HybridRetriever:
    """Configurable retriever: dense, hybrid (dense+BM25 RRF), optional Cohere rerank."""

    def __init__(
        self,
        dense_index: DenseIndex,
        *,
        mode: str = "dense",  # dense|hybrid|hybrid_rerank
        dense_k: int = DEFAULT_DENSE_K,
        bm25_k: int = DEFAULT_BM25_K,
        rrf_k: int = DEFAULT_RRF_K,
        rerank_model: str = DEFAULT_COHERE_MODEL,
        rerank_candidate_k: int = 20,
    ):
        self.dense_index = dense_index
        self.mode = mode
        self.dense_k = dense_k
        self.bm25_k = bm25_k
        self.rrf_k = rrf_k
        self.rerank_model = rerank_model
        self.rerank_candidate_k = rerank_candidate_k
        self.corpus = CorpusIndex(dense_index)

    def retrieve(self, question: str, k: int = 5) -> RetrievalResult:
        q = (question or "").strip()
        if not q:
            return RetrievalResult(hits=[], debug={"mode": self.mode, "error": "empty_question"})
        mode = self.mode
        debug: Dict[str, Any] = {"mode": mode, "k": k}

        t0 = time.perf_counter()
        dense_hits = self.dense_index.search(q, k=max(k, self.dense_k if mode != "dense" else k))
        debug["dense_latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        debug["dense"] = [{"chunk_id": h.chunk_id, "rank": h.rank, "distance": h.distance} for h in dense_hits]

        if mode == "dense":
            hits = list(dense_hits[:k])
            for i, h in enumerate(hits, start=1):
                h.rank = i
            debug["final"] = [h.chunk_id for h in hits]
            debug["total_latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
            return RetrievalResult(hits=hits, debug=debug)

        # hybrid / hybrid_rerank
        t1 = time.perf_counter()
        bm25_ranked = self.corpus.bm25_search(q, k=max(k, self.bm25_k))
        debug["bm25_latency_ms"] = round((time.perf_counter() - t1) * 1000.0, 3)
        debug["bm25"] = [{"chunk_id": cid, "rank": r, "score": s} for cid, s, r in bm25_ranked]

        fused = rrf_fuse(dense_hits, bm25_ranked, rrf_k=self.rrf_k)
        debug["fusion"] = [{"chunk_id": cid, "rrf": round(score, 6), **detail} for cid, score, detail in fused[: max(k, self.rerank_candidate_k)]]

        # Build candidate RetrievedChunk objects in fused order
        # Prefer dense hit objects when available to keep distance; else build from corpus
        dense_by_id = {h.chunk_id: h for h in dense_hits}
        candidates: List[RetrievedChunk] = []
        for cid, _score, detail in fused[: max(k, self.rerank_candidate_k)]:
            base = dense_by_id.get(cid)
            if base is not None:
                cand = RetrievedChunk(
                    rank=len(candidates) + 1,
                    chunk_id=base.chunk_id,
                    text=base.text,
                    source=base.source,
                    start_offset=base.start_offset,
                    end_offset=base.end_offset,
                    start_page=base.start_page,
                    end_page=base.end_page,
                    article=base.article,
                    section=base.section,
                    heading=base.heading,
                    distance=base.distance,
                    metadata=dict(base.metadata or {}),
                )
            else:
                cand = self.corpus.to_retrieved_chunk(cid, rank=len(candidates) + 1)
            if cand is None:
                continue
            cand.metadata = dict(cand.metadata or {})
            cand.metadata["rrf_score"] = _score
            cand.metadata.update({k: v for k, v in detail.items() if v is not None})
            candidates.append(cand)

        if mode == "hybrid_rerank":
            try:
                reranked, rdebug = cohere_rerank(q, candidates, top_n=k, model=self.rerank_model)
                debug["rerank"] = rdebug
                hits = reranked
            except Exception as exc:  # noqa: BLE001
                debug["rerank"] = {"applied": False, "error": f"{type(exc).__name__}: {exc}", "model": self.rerank_model}
                hits = candidates[:k]
        else:
            hits = candidates[:k]

        for i, h in enumerate(hits, start=1):
            h.rank = i
        debug["final"] = [{"chunk_id": h.chunk_id, "rank": h.rank, "distance": h.distance, "rrf_score": (h.metadata or {}).get("rrf_score"), "rerank_score": (h.metadata or {}).get("rerank_score")} for h in hits]
        debug["total_latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        return RetrievalResult(hits=hits, debug=debug)
