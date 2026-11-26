"""Opt-in Weaviate Cloud retriever for isolated retrieval experiments.

This module must NOT be imported by the default production path in a way that
requires weaviate-client at import time. All weaviate imports are lazy.

The adapter returns the same RetrievedChunk/RetrievalResult types used by the
existing grounded pipeline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .retrieval_types import RetrievalResult, RetrievedChunk


class WeaviateConfigError(ValueError):
    pass


class WeaviateRetrievalError(RuntimeError):
    pass


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    raw = value.strip()
    if not raw:
        return default
    try:
        parsed = float(raw)
    except Exception as exc:  # noqa: BLE001
        raise WeaviateConfigError(f"Invalid float value for {name}: {value!r}") from exc
    if parsed <= 0:
        raise WeaviateConfigError(f"{name} must be > 0, got {value!r}")
    return parsed


@dataclass(frozen=True)
class WeaviateRetrieverConfig:
    url: str
    api_key: str
    collection_name: str
    embedding_model: str = "Snowflake/snowflake-arctic-embed-l-v2.0"
    alpha: float = 0.5  # Weaviate hybrid alpha; 0=bm25, 1=vector
    # Connection resilience (experimental). Defaults preserve weaviate-client v4 defaults.
    init_timeout_s: float = 2.0
    query_timeout_s: float = 30.0
    # Experimental search mode. Default preserves current hybrid behavior.
    search_mode: str = "hybrid"  # dense|bm25|hybrid

    @classmethod
    def from_env(cls) -> "WeaviateRetrieverConfig":
        url = (os.getenv("WEAVIATE_URL") or os.getenv("RAG_WEAVIATE_URL") or "").strip()
        api_key = (os.getenv("WEAVIATE_API_KEY") or os.getenv("RAG_WEAVIATE_API_KEY") or "").strip()
        collection = (os.getenv("WEAVIATE_COLLECTION") or os.getenv("RAG_WEAVIATE_COLLECTION") or "").strip()
        model = (os.getenv("WEAVIATE_EMBEDDING_MODEL") or os.getenv("RAG_WEAVIATE_EMBEDDING_MODEL") or "Snowflake/snowflake-arctic-embed-l-v2.0").strip()
        init_timeout_s = _env_float("WEAVIATE_INIT_TIMEOUT_S", _env_float("RAG_WEAVIATE_INIT_TIMEOUT_S", 2.0))
        query_timeout_s = _env_float("WEAVIATE_QUERY_TIMEOUT_S", _env_float("RAG_WEAVIATE_QUERY_TIMEOUT_S", 30.0))
        search_mode = (os.getenv("WEAVIATE_SEARCH_MODE") or os.getenv("RAG_WEAVIATE_SEARCH_MODE") or "hybrid").strip().lower()
        alpha_raw = os.getenv("WEAVIATE_ALPHA") or os.getenv("RAG_WEAVIATE_ALPHA")
        alpha = 0.5
        if alpha_raw is not None and alpha_raw.strip() != "":
            try:
                alpha = float(alpha_raw)
            except Exception as exc:  # noqa: BLE001
                raise WeaviateConfigError(f"Invalid float value for WEAVIATE_ALPHA: {alpha_raw!r}") from exc
        if search_mode not in {"dense", "bm25", "hybrid"}:
            raise WeaviateConfigError(
                f"Invalid WEAVIATE_SEARCH_MODE: {search_mode!r}. Expected one of: dense, bm25, hybrid."
            )
        if not (0.0 <= alpha <= 1.0):
            raise WeaviateConfigError(f"WEAVIATE_ALPHA must be between 0 and 1, got {alpha!r}")
        missing = []
        if not url:
            missing.append("WEAVIATE_URL")
        if not api_key:
            missing.append("WEAVIATE_API_KEY")
        if not collection:
            missing.append("WEAVIATE_COLLECTION")
        if missing:
            raise WeaviateConfigError(
                "Missing required Weaviate configuration: " + ", ".join(missing) + ". "
                "Set environment variables for the experimental Weaviate backend."
            )
        return cls(
            url=url,
            api_key=api_key,
            collection_name=collection,
            embedding_model=model,
            init_timeout_s=init_timeout_s,
            query_timeout_s=query_timeout_s,
            alpha=alpha,
            search_mode=search_mode,
        )


class WeaviateRetriever:
    """Native Weaviate hybrid retriever using hosted embeddings.

    The caller is responsible for closing the client via close() when done, or
    using the retriever as a context manager.
    """

    def __init__(self, config: WeaviateRetrieverConfig, client: Any = None):
        self.config = config
        self._client = client
        self._owns_client = client is None

    def _connect(self):
        if self._client is not None:
            return self._client
        try:
            import weaviate
            from weaviate.classes.init import Auth
        except Exception as exc:  # noqa: BLE001
            raise WeaviateRetrievalError(
                "weaviate-client is required for RETRIEVER_BACKEND=weaviate. "
                "Install it separately for the isolated experiment."
            ) from exc
        try:
            additional_config = None
            try:
                from weaviate.classes.init import AdditionalConfig, Timeout

                additional_config = AdditionalConfig(
                    timeout=Timeout(
                        init=float(self.config.init_timeout_s),
                        query=float(self.config.query_timeout_s),
                    )
                )
            except Exception:
                # Preserve connection attempt; client defaults apply if timeout config cannot be built.
                additional_config = None

            connect_kwargs: Dict[str, Any] = {
                "cluster_url": self.config.url,
                "auth_credentials": Auth.api_key(self.config.api_key),
            }
            if additional_config is not None:
                connect_kwargs["additional_config"] = additional_config
            self._client = weaviate.connect_to_weaviate_cloud(**connect_kwargs)
        except Exception as exc:  # noqa: BLE001
            raise WeaviateRetrievalError(f"Weaviate connection failed: {type(exc).__name__}: {exc}") from exc
        return self._client

    @property
    def client(self):
        return self._connect()

    def close(self) -> None:
        client = self._client
        self._client = None
        if client is not None and self._owns_client:
            try:
                client.close()
            except Exception:
                pass

    def __enter__(self):
        self._connect()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    @staticmethod
    def _normalize_search_mode(mode: Optional[str]) -> str:
        m = (mode or "hybrid").strip().lower()
        aliases = {
            "weaviate_hybrid": "hybrid",
            "weaviate_dense": "dense",
            "weaviate_bm25": "bm25",
            "vector": "dense",
            "near_text": "dense",
        }
        m = aliases.get(m, m)
        if m not in {"dense", "bm25", "hybrid"}:
            raise WeaviateRetrievalError(
                f"Invalid Weaviate search mode: {mode!r}. Expected one of: dense, bm25, hybrid."
            )
        return m

    @staticmethod
    def _normalize_alpha(alpha: Any) -> float:
        try:
            a = float(alpha)
        except Exception as exc:  # noqa: BLE001
            raise WeaviateRetrievalError(f"Invalid Weaviate alpha: {alpha!r}") from exc
        if not (0.0 <= a <= 1.0):
            raise WeaviateRetrievalError(f"Weaviate alpha must be between 0 and 1, got {alpha!r}")
        return a

    def retrieve(
        self,
        question: str,
        k: int = 5,
        *,
        search_mode: Optional[str] = None,
        alpha: Optional[float] = None,
    ) -> RetrievalResult:
        question = (question or "").strip()
        mode = self._normalize_search_mode(search_mode or self.config.search_mode)
        effective_alpha = self._normalize_alpha(self.config.alpha if alpha is None else alpha)
        debug_base: Dict[str, Any] = {
            "mode": f"weaviate_{mode}",
            "search_mode": mode,
            "k": k,
            "collection": self.config.collection_name,
            "embedding_model": self.config.embedding_model,
            "init_timeout_s": self.config.init_timeout_s,
            "query_timeout_s": self.config.query_timeout_s,
        }
        if mode == "hybrid":
            debug_base["alpha"] = effective_alpha
        if not question:
            return RetrievalResult(hits=[], debug={**debug_base, "error": "empty_question"})

        client = self._connect()
        try:
            collection = client.collections.use(self.config.collection_name)
        except Exception as exc:  # noqa: BLE001
            raise WeaviateRetrievalError(
                f"Failed to access Weaviate collection '{self.config.collection_name}': {type(exc).__name__}: {exc}"
            ) from exc

        limit = max(1, int(k))
        md_query = None
        try:
            from weaviate.classes.query import MetadataQuery

            # Request score+distance where supported; some modes may return only one.
            md_query = MetadataQuery(score=True, distance=True)
        except Exception:
            md_query = None

        try:
            if mode == "bm25":
                kwargs: Dict[str, Any] = {"query": question, "limit": limit}
                if md_query is not None:
                    kwargs["return_metadata"] = md_query
                try:
                    response = collection.query.bm25(**kwargs)
                except TypeError:
                    kwargs.pop("return_metadata", None)
                    response = collection.query.bm25(query=question, limit=limit)
            elif mode == "dense":
                # Dense-only via server-side text2vec vectorizer. This avoids local embedding model
                # loading and uses the collection's configured vectorizer semantics.
                kwargs = {"query": question, "limit": limit}
                if md_query is not None:
                    kwargs["return_metadata"] = md_query
                try:
                    response = collection.query.near_text(**kwargs)
                except TypeError:
                    kwargs.pop("return_metadata", None)
                    response = collection.query.near_text(query=question, limit=limit)
            else:
                kwargs = {"query": question, "limit": limit, "alpha": effective_alpha}
                if md_query is not None:
                    kwargs["return_metadata"] = md_query
                try:
                    response = collection.query.hybrid(**kwargs)
                except TypeError:
                    kwargs.pop("return_metadata", None)
                    response = collection.query.hybrid(query=question, limit=limit, alpha=effective_alpha)
        except Exception as exc:  # noqa: BLE001
            raise WeaviateRetrievalError(
                f"Weaviate {mode} query failed: {type(exc).__name__}: {exc}"
            ) from exc

        hits: List[RetrievedChunk] = []
        objects = getattr(response, "objects", []) or []
        for i, obj in enumerate(objects, start=1):
            props = dict(getattr(obj, "properties", {}) or {})
            md = getattr(obj, "metadata", None)
            distance = None
            score = None
            if md is not None:
                distance = getattr(md, "distance", None)
                score = getattr(md, "score", None)
                if distance is None and isinstance(md, dict):
                    distance = md.get("distance")
                    score = md.get("score")
            # Store score in metadata for debugging; RetrievedChunk.distance is optional.
            meta = dict(props)
            meta["_weaviate_score"] = score
            meta["_weaviate_distance"] = distance
            meta["_weaviate_uuid"] = str(getattr(obj, "uuid", ""))
            meta["_weaviate_search_mode"] = mode
            if mode == "hybrid":
                meta["_weaviate_alpha"] = effective_alpha
            chunk_id = str(props.get("chunk_id") or props.get("id") or meta.get("_weaviate_uuid") or f"weaviate-{i}")
            hits.append(
                RetrievedChunk(
                    rank=i,
                    chunk_id=chunk_id,
                    text=str(props.get("text") or props.get("chunk_text") or ""),
                    source=str(props.get("source") or "satcom-ngp.pdf"),
                    start_offset=int(props.get("start_offset") or 0),
                    end_offset=int(props.get("end_offset") or 0),
                    start_page=int(props.get("start_page") or 0),
                    end_page=int(props.get("end_page") or 0),
                    article=str(props.get("article") or "") or None,
                    section=str(props.get("section") or "") or None,
                    heading=str(props.get("heading") or "") or None,
                    distance=float(distance) if distance is not None else None,
                    metadata=meta,
                )
            )

        return RetrievalResult(
            hits=hits,
            debug={**debug_base, "final": [h.chunk_id for h in hits]},
        )
