from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from src.rag_core import (
    ABSTAINED,
    ANSWERED,
    DenseIndex,
    DenseIndexConfig,
    GeminiGenerator,
    GroundedPipeline,
    HybridRetriever,
    RetrievedChunk,
    RetrievalResult,
    content_tokens,
)
from src.rag_core.pipeline import ABSTENTION_MESSAGE

from .config import AppConfig
from .langfuse_tracing import get_langfuse_tracer
from .metrics import RequestRecord, get_metrics_collector
from .observability import RequestTrace
from .prompts import get_prompt_registry
from .schemas import (
    CitationModel,
    CitationValidationModel,
    GenerationModel,
    HealthResponse,
    QueryResponse,
    RetrievedChunkModel,
    StatementModel,
)


def _citation_model(c: Any) -> CitationModel:
    d = c.to_dict() if hasattr(c, "to_dict") else dict(c)
    return CitationModel(**d)


def _statement_model(s: Any) -> StatementModel:
    d = s.to_dict() if hasattr(s, "to_dict") else dict(s)
    return StatementModel(**d)


def _retrieved_model(h: RetrievedChunk) -> RetrievedChunkModel:
    meta = h.metadata or {}
    return RetrievedChunkModel(
        rank=h.rank,
        chunk_id=h.chunk_id,
        section=h.section,
        start_page=h.start_page,
        end_page=h.end_page,
        start_offset=h.start_offset,
        end_offset=h.end_offset,
        distance=h.distance,
    )


def _selected_lexical_support(question: str, statements: Sequence[Any]) -> float:
    q = set(content_tokens(question))
    if not q:
        return 1.0
    s: set[str] = set()
    for stmt in statements:
        text = stmt.text if hasattr(stmt, "text") else str(stmt)
        s.update(content_tokens(text))
    return len(q & s) / len(q) if q else 0.0


class RAGService:
    """Single application service composing validated rag_core components."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        *,
        index: Optional[DenseIndex] = None,
        pipeline: Optional[GroundedPipeline] = None,
        gemini_generator: Optional[GeminiGenerator] = None,
        retriever: Optional[HybridRetriever] = None,
    ) -> None:
        self.config = config or AppConfig.from_env()
        self._index = index
        self._pipeline = pipeline
        self._gemini_generator = gemini_generator
        self._retriever = retriever
        self._model: Any = None
        self._langfuse = get_langfuse_tracer(
            enabled=self.config.langfuse_enabled,
            public_key=self.config.langfuse_public_key,
            secret_key=self.config.langfuse_secret_key,
            host=self.config.langfuse_host,
        )

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.config.embedding_model)
        return self._model

    @property
    def index(self) -> DenseIndex:
        if self._index is None:
            self._index = DenseIndex(
                DenseIndexConfig(
                    collection_name=self.config.collection_name,
                    persist_dir=self.config.persist_dir,
                    embedding_model=self.config.embedding_model,
                ),
                model=self.model,
            )
            self._ensure_index_ready(self._index)
        return self._index

    def _ensure_index_ready(self, index: DenseIndex) -> None:
        """Ensure the configured Chroma collection is available.

        Render Free and other ephemeral environments may not have a persistent
        prebuilt index on disk. If the index is missing/empty, rebuild it from the
        source PDF using the configured chunking strategy. This preserves the
        existing RAG/indexing behavior while enabling reliable recovery.
        """
        try:
            if index.count() > 0:
                return
        except Exception:
            # Fall through to rebuild attempt.
            pass

        pdf_path = Path(self.config.pdf_path)
        if not pdf_path.exists():
            return

        try:
            from src.rag_core import (
                FixedCharacterChunker,
                StructureAwareChunker,
                audit_source_coverage,
                extract_source_document,
            )

            doc = extract_source_document(pdf_path)
            strategy = (self.config.chunk_strategy or "fixed_character").lower()
            if strategy == "structure_aware":
                chunker = StructureAwareChunker(max_chars=self.config.chunk_size)
                chunks = chunker.chunk(doc)
                audit = audit_source_coverage(doc, chunks, max_chars=self.config.chunk_size, allow_overlap=False)
            else:
                chunker = FixedCharacterChunker(chunk_size=self.config.chunk_size, overlap=self.config.chunk_overlap)
                chunks = chunker.chunk(doc)
                audit = audit_source_coverage(doc, chunks, max_chars=self.config.chunk_size, allow_overlap=True)

            if not audit.get("ok"):
                return
            index.build(chunks, reset=True)
            # Retriever may cache corpus/collection state; reset so it rebuilds.
            self._retriever = None
        except Exception:
            # Do not crash service startup; health/query will surface availability issues.
            return

    @property
    def retriever(self) -> HybridRetriever:
        if self._retriever is None:
            self._retriever = HybridRetriever(
                dense_index=self.index,
                mode=self.config.retrieval_mode,
                dense_k=self.config.dense_k,
                bm25_k=self.config.bm25_k,
                rrf_k=self.config.rrf_k,
                rerank_model=self.config.rerank_model,
                rerank_candidate_k=self.config.rerank_candidate_k,
            )
        return self._retriever

    @property
    def pipeline(self) -> GroundedPipeline:
        if self._pipeline is None:
            self._pipeline = GroundedPipeline(
                index=self.index,
                top_k=self.config.top_k,
                sibling_score_ratio=self.config.sibling_score_ratio,
                clause_body_units=self.config.clause_body_units,
                cross_reference_selection=self.config.cross_reference_selection,
                associated_heading_units=self.config.associated_heading_units,
                definition_intent_boost=self.config.definition_intent_boost,
            )
        return self._pipeline

    @property
    def gemini_generator(self) -> GeminiGenerator:
        if self._gemini_generator is None:
            self._gemini_generator = GeminiGenerator(
                model=self.config.gemini_model,
                temperature=self.config.gemini_temperature,
                max_output_tokens=self.config.gemini_max_output_tokens,
            )
        return self._gemini_generator

    def health(self) -> HealthResponse:
        index_available = False
        try:
            index_available = self.index.count() > 0
        except Exception:
            index_available = False
        gemini_configured = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
        return HealthResponse(
            status="ok",
            default_mode=self.config.default_generation_mode,
            retrieval_mode=self.config.retrieval_mode,
            collection_name=self.config.collection_name,
            top_k=self.config.top_k,
            index_available=index_available,
            gemini_configured=gemini_configured,
            langfuse=self._langfuse.status(),
        )

    def _retrieve(self, question: str, top_k: int, retrieval_mode: str) -> RetrievalResult:
        mode = (retrieval_mode or "dense").lower()
        # Fast/simple path for dense retrieval and for indexes without corpus/collection support (tests, mocks).
        if mode == "dense":
            hits = list(self.index.search(question, k=top_k))
            for i, h in enumerate(hits, start=1):
                h.rank = i
            return RetrievalResult(hits=hits, debug={"mode": "dense", "k": top_k, "final": [h.chunk_id for h in hits]})
        # Use configured retriever when mode matches, else create a lightweight one.
        try:
            if mode == self.config.retrieval_mode and self._retriever is not None:
                return self._retriever.retrieve(question, k=top_k)
            if mode == self.config.retrieval_mode and self._retriever is None:
                self._retriever = HybridRetriever(
                    dense_index=self.index,
                    mode=mode,
                    dense_k=self.config.dense_k,
                    bm25_k=self.config.bm25_k,
                    rrf_k=self.config.rrf_k,
                    rerank_model=self.config.rerank_model,
                    rerank_candidate_k=self.config.rerank_candidate_k,
                )
                return self._retriever.retrieve(question, k=top_k)
            temp = HybridRetriever(
                dense_index=self.index,
                mode=mode,
                dense_k=self.config.dense_k,
                bm25_k=self.config.bm25_k,
                rrf_k=self.config.rrf_k,
                rerank_model=self.config.rerank_model,
                rerank_candidate_k=self.config.rerank_candidate_k,
            )
            return temp.retrieve(question, k=top_k)
        except Exception as exc:  # noqa: BLE001
            # Fallback to dense retrieval to keep service robust.
            hits = list(self.index.search(question, k=top_k))
            for i, h in enumerate(hits, start=1):
                h.rank = i
            return RetrievalResult(hits=hits, debug={"mode": mode, "fallback": "dense", "error": f"{type(exc).__name__}: {exc}", "final": [h.chunk_id for h in hits]})

    def query(
        self,
        question: str,
        mode: Optional[str] = None,
        top_k: Optional[int] = None,
        retrieval_mode: Optional[str] = None,
    ) -> QueryResponse:
        trace = RequestTrace()
        question = (question or "").strip()
        requested_mode = (mode or self.config.default_generation_mode or "extractive").lower()
        if requested_mode not in {"extractive", "gemini"}:
            requested_mode = "extractive"
        req_retrieval_mode = (retrieval_mode or self.config.retrieval_mode or "dense").lower()
        if req_retrieval_mode not in {"dense", "hybrid", "hybrid_rerank"}:
            req_retrieval_mode = "dense"
        effective_top_k = int(top_k or self.config.top_k)

        prompt_version = None
        try:
            prompt_version = get_prompt_registry().generation_grounded().version
        except Exception:
            prompt_version = "grounded_json_v1"

        lf_trace = self._langfuse.start_trace(
            name="rag_request",
            input={"question": question, "mode": requested_mode, "retrieval_mode": req_retrieval_mode, "top_k": effective_top_k},
            metadata={"request_id": trace.request_id, "prompt_version": prompt_version},
        )

        if not question:
            resp = QueryResponse(
                request_id=trace.request_id,
                status="abstained",
                mode=requested_mode,
                retrieval_mode=req_retrieval_mode,
                question=question,
                answer="I cannot answer an empty question.",
                abstained=True,
                abstention_reason="empty question",
                timings_ms={},
                errors=[],
                debug={"prompt_version": prompt_version},
            )
            try:
                lf_trace.update(output={"status": resp.status, "abstained": True}, metadata={"request_id": trace.request_id})
            except Exception:
                pass
            self._langfuse.flush()
            return resp

        try:
            # 1-6 Retrieval stages (dense/bm25/fusion/rerank encapsulated but traced)
            with trace.stage("retrieval", top_k=effective_top_k, retrieval_mode=req_retrieval_mode):
                with self._langfuse.span(lf_trace, "retrieval", input={"question": question, "mode": req_retrieval_mode, "top_k": effective_top_k}) as span:
                    retrieval_result = self._retrieve(question, effective_top_k, req_retrieval_mode)
                    hits = list(retrieval_result.hits)
                    try:
                        span.update(output={"hit_count": len(hits), "chunk_ids": [h.chunk_id for h in hits], "debug": retrieval_result.debug})
                    except Exception:
                        pass
            trace.add_detail("retrieval_debug", retrieval_result.debug)
            trace.add_detail(
                "retrieval",
                [
                    {
                        "rank": h.rank,
                        "chunk_id": h.chunk_id,
                        "section": h.section,
                        "start_page": h.start_page,
                        "end_page": h.end_page,
                        "start_offset": h.start_offset,
                        "end_offset": h.end_offset,
                        "distance": h.distance,
                        "rrf_score": (h.metadata or {}).get("rrf_score"),
                        "rerank_score": (h.metadata or {}).get("rerank_score"),
                    }
                    for h in hits
                ],
            )

            # 7-8 Selection and evidence sufficiency/abstention pre-check
            with trace.stage("selection_and_extractive", hit_count=len(hits)):
                with self._langfuse.span(lf_trace, "evidence_selection", input={"hit_ids": [h.chunk_id for h in hits]}) as span:
                    grounded = self.pipeline.answer_from_hits(question, hits)
                    try:
                        span.update(output={"status": grounded.status, "statement_count": len(grounded.statements), "citation_count": len(grounded.citations)})
                    except Exception:
                        pass

            selected_ids: list[str] = []
            seen_ids: set[str] = set()
            for stmt in grounded.statements:
                if stmt.chunk_id not in seen_ids:
                    seen_ids.add(stmt.chunk_id)
                    selected_ids.append(stmt.chunk_id)

            selected_support = _selected_lexical_support(question, grounded.statements)
            citations = [_citation_model(c) for c in grounded.citations]
            statements = [_statement_model(s) for s in grounded.statements]
            trace.add_detail(
                "selection",
                {
                    "selected_evidence_ids": selected_ids,
                    "statement_count": len(statements),
                    "citation_count": len(citations),
                    "selected_lexical_support": round(selected_support, 4),
                    "pipeline_status": grounded.status,
                    "pipeline_reason": grounded.reason,
                },
            )

            # Conservative answerability guard (application-level)
            guard_abstain = False
            guard_reason = grounded.reason
            if grounded.status == ANSWERED:
                if len(grounded.statements) == 0:
                    guard_abstain = True
                    guard_reason = "insufficient_selected_evidence:no_selected_statements"
                elif selected_support < self.config.gemini_sufficiency_threshold:
                    guard_abstain = True
                    guard_reason = (
                        "insufficient_selected_evidence:"
                        f"selected_lexical_support={selected_support:.4f}"
                        f"<threshold={self.config.gemini_sufficiency_threshold}"
                    )

            with self._langfuse.span(
                lf_trace,
                "evidence_sufficiency",
                input={"pipeline_status": grounded.status, "selected_support": round(selected_support, 4)},
            ) as span:
                try:
                    span.update(output={"guard_abstain": guard_abstain, "guard_reason": guard_reason})
                except Exception:
                    pass

            authoritative_status = "abstained" if (grounded.status == ABSTAINED or guard_abstain) else "answered"
            authoritative_abstained = authoritative_status == "abstained"
            if authoritative_abstained:
                authoritative_reason = guard_reason or grounded.reason or "insufficient_evidence"
                authoritative_answer = grounded.text if grounded.status == ABSTAINED else ABSTENTION_MESSAGE
            else:
                authoritative_reason = None
                authoritative_answer = grounded.text

            response = QueryResponse(
                request_id=trace.request_id,
                status=authoritative_status,
                mode=requested_mode,
                retrieval_mode=req_retrieval_mode,
                question=question,
                answer=authoritative_answer,
                abstained=authoritative_abstained,
                abstention_reason=authoritative_reason,
                citations=citations,
                statements=statements,
                retrieved=[_retrieved_model(h) for h in grounded.retrieved],
                selected_evidence_ids=selected_ids,
                citation_validation=CitationValidationModel(
                    app_resolved=True,
                    invalid_source_ids=[],
                    uncited_claims=0,
                    errors=[],
                ),
                generation=GenerationModel(
                    attempted=False,
                    mode="extractive",
                    status=authoritative_status,
                    text=authoritative_answer,
                    citations=citations,
                    claims=[
                        {"text": s.text, "markers": [s.marker], "chunk_id": s.chunk_id}
                        for s in grounded.statements
                    ],
                    fallback_to_extractive=False,
                    metadata={"prompt_version": prompt_version},
                ),
                timings_ms=dict(trace.timings_ms),
                errors=list(trace.errors),
                debug={
                    "retrieved_chunk_ids": [h.chunk_id for h in grounded.retrieved],
                    "token_support": grounded.token_support,
                    "selected_lexical_support": round(selected_support, 4),
                    "best_distance": grounded.best_distance,
                    "answerability_guard_applied": guard_abstain,
                    "answerability_guard_reason": guard_reason if guard_abstain else None,
                    "pipeline_status": grounded.status,
                    "pipeline_reason": grounded.reason,
                    "retrieval_debug": retrieval_result.debug,
                    "prompt_version": prompt_version,
                    "langfuse": self._langfuse.status(),
                    "selection_defaults": {
                        "sibling_score_ratio": self.config.sibling_score_ratio,
                        "cross_reference_selection": self.config.cross_reference_selection,
                        "associated_heading_units": self.config.associated_heading_units,
                        "definition_intent_boost": self.config.definition_intent_boost,
                    },
                },
            )

            if requested_mode == "gemini":
                self._apply_gemini(response, question, grounded, selected_support, trace, lf_trace, prompt_version)
                response.timings_ms = dict(trace.timings_ms)
                merged = list(dict.fromkeys([*(response.errors or []), *(trace.errors or [])]))
                response.errors = merged

            # final citation validation stage trace
            with self._langfuse.span(lf_trace, "citation_validation", input={"answer_status": response.status}) as span:
                try:
                    span.update(output={
                        "app_resolved": response.citation_validation.app_resolved,
                        "invalid_source_ids": response.citation_validation.invalid_source_ids,
                        "uncited_claims": response.citation_validation.uncited_claims,
                        "errors": response.citation_validation.errors,
                        "citation_count": len(response.citations),
                    })
                except Exception:
                    pass

            response.debug["trace"] = trace.to_dict()
            self._record_metrics(response, trace)
            try:
                lf_trace.update(
                    output={
                        "status": response.status,
                        "abstained": response.abstained,
                        "abstention_reason": response.abstention_reason,
                        "answer": response.answer,
                        "citation_count": len(response.citations),
                        "retrieval_mode": response.retrieval_mode,
                    },
                    metadata={"request_id": response.request_id, "prompt_version": prompt_version},
                )
            except Exception:
                pass
            self._langfuse.flush()
            return response
        except Exception as exc:  # noqa: BLE001
            trace.errors.append(f"request: {type(exc).__name__}: {exc}")
            response = QueryResponse(
                request_id=trace.request_id,
                status="error",
                mode=requested_mode,
                retrieval_mode=req_retrieval_mode,
                question=question,
                answer="",
                abstained=False,
                abstention_reason=None,
                timings_ms=dict(trace.timings_ms),
                errors=list(trace.errors),
                debug={"error_type": type(exc).__name__, "trace": trace.to_dict(), "prompt_version": prompt_version, "langfuse": self._langfuse.status()},
            )
            self._record_metrics(response, trace)
            try:
                lf_trace.update(level="ERROR", status_message=str(exc), output={"status": "error", "errors": response.errors})
            except Exception:
                pass
            self._langfuse.flush()
            return response

    def _apply_gemini(
        self,
        response: QueryResponse,
        question: str,
        grounded: Any,
        selected_support: float,
        trace: RequestTrace,
        lf_trace: Any = None,
        prompt_version: Optional[str] = None,
    ) -> None:
        extractive_citations = [_citation_model(c) for c in grounded.citations]
        gen = GenerationModel(
            attempted=True,
            mode="gemini",
            text=response.answer,
            citations=extractive_citations,
            metadata={"prompt_version": prompt_version},
        )

        if not self.config.allow_gemini:
            gen.status = "skipped"
            gen.errors.append("Gemini generation disabled by configuration")
            gen.fallback_to_extractive = True
            response.generation = gen
            return

        no_selected = len(grounded.statements) == 0
        insufficient = (
            response.abstained
            or no_selected
            or selected_support < self.config.gemini_sufficiency_threshold
        )
        if response.abstained:
            insufficiency_reason = response.abstention_reason or grounded.reason or "abstained"
        elif no_selected:
            insufficiency_reason = "no_selected_evidence"
        else:
            insufficiency_reason = (
                f"selected_evidence_insufficient:{selected_support:.4f}"
                f"<{self.config.gemini_sufficiency_threshold}"
            )

        if insufficient:
            gen.status = "abstained_insufficient_evidence"
            gen.fallback_to_extractive = True
            gen.text = response.answer
            gen.metadata = {
                "insufficient_evidence": True,
                "reason": insufficiency_reason,
                "selected_lexical_support": round(selected_support, 4),
                "prompt_version": prompt_version,
            }
            response.generation = gen
            return

        try:
            with trace.stage("gemini_generation", model=self.config.gemini_model):
                with self._langfuse.span(
                    lf_trace,
                    "generation",
                    input={"question": question, "model": self.config.gemini_model, "prompt_version": prompt_version},
                    model=self.config.gemini_model,
                ) as span:
                    result = self.gemini_generator.generate(
                        question,
                        grounded.statements,
                        grounded.citations,
                        evidence_sufficient=True,
                    )
                    try:
                        meta = result.metadata or {}
                        usage_details = None
                        pt = meta.get("prompt_token_count") or meta.get("input_tokens")
                        ct = meta.get("candidates_token_count") or meta.get("output_tokens")
                        tt = meta.get("total_token_count")
                        if any(v is not None for v in (pt, ct, tt)):
                            usage_details = {}
                            if pt is not None:
                                usage_details["input"] = int(pt)
                            if ct is not None:
                                usage_details["output"] = int(ct)
                            if tt is not None:
                                usage_details["total"] = int(tt)
                        span.update(
                            output={"status": result.status, "claim_count": len(result.claims), "model": result.model, "metadata": meta},
                            model=result.model or self.config.gemini_model,
                            usage_details=usage_details,
                        )
                    except Exception:
                        pass
        except Exception as exc:  # noqa: BLE001
            gen.status = "generation_error"
            gen.errors.append(f"{type(exc).__name__}: {exc}")
            gen.fallback_to_extractive = True
            response.generation = gen
            response.errors.append(f"gemini_generation: {type(exc).__name__}: {exc}")
            return

        gen.status = result.status
        gen.model = result.model
        gen.prompt_version = (result.metadata or {}).get("prompt_version") or prompt_version
        gen.latency_ms = result.latency_ms
        gen.errors = list(result.errors or [])
        gen.metadata = {**(result.metadata or {}), "prompt_version": gen.prompt_version}
        gen.claims = [
            {
                "text": c.text,
                "source_ids": list(c.source_ids),
                "markers": list(c.markers),
                "invalid_source_ids": list(c.invalid_source_ids),
            }
            for c in result.claims
        ]
        invalid_ids = sorted({i for c in result.claims for i in c.invalid_source_ids})
        uncited_claims = sum(1 for c in result.claims if not c.markers)
        gen_citations = [_citation_model(c) for c in result.citations]
        gen.citations = gen_citations
        response.citation_validation = CitationValidationModel(
            app_resolved=True,
            invalid_source_ids=invalid_ids,
            uncited_claims=uncited_claims,
            errors=list(result.errors or []),
        )

        trace.add_detail(
            "citation_validation",
            {
                "invalid_source_ids": invalid_ids,
                "uncited_claims": uncited_claims,
                "error_count": len(result.errors or []),
            },
        )

        if result.status == "answered" and result.text:
            response.answer = result.text
            response.citations = gen_citations
            response.status = "answered"
            response.abstained = False
            response.abstention_reason = None
            response.generation = gen
            return

        gen.fallback_to_extractive = True
        if not gen.citations:
            gen.citations = extractive_citations
        if not gen.text:
            gen.text = grounded.text
        response.generation = gen
        response.answer = grounded.text
        response.citations = extractive_citations
        response.status = "answered" if grounded.status == ANSWERED else "abstained"
        response.abstained = grounded.status == ABSTAINED
        response.abstention_reason = grounded.reason

    def _record_metrics(self, response: QueryResponse, trace: RequestTrace) -> None:
        try:
            gen_meta = response.generation.metadata or {}
            prompt_tokens = gen_meta.get("prompt_token_count") or gen_meta.get("input_tokens")
            completion_tokens = gen_meta.get("candidates_token_count") or gen_meta.get("output_tokens")
            total_tokens = gen_meta.get("total_token_count")
            if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
                try:
                    total_tokens = int(prompt_tokens) + int(completion_tokens)
                except Exception:
                    total_tokens = None
            rec = RequestRecord(
                request_id=response.request_id,
                ts=time.time(),
                status=response.status,
                abstained=bool(response.abstained),
                mode=response.mode,
                total_ms=trace.total_ms() if trace.timings_ms else None,
                stage_timings_ms=dict(response.timings_ms or {}),
                citation_count=len(response.citations or []),
                statement_count=len(response.statements or []),
                retrieved_count=len(response.retrieved or []),
                selected_count=len(response.selected_evidence_ids or []),
                citation_validation_errors=len((response.citation_validation.errors or [])) + len(response.citation_validation.invalid_source_ids or []),
                error_count=len(response.errors or []),
                failure=response.status == "error",
                generation_attempted=bool(response.generation.attempted),
                generation_status=response.generation.status,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                estimated_cost_usd=gen_meta.get("estimated_cost_usd"),
            )
            get_metrics_collector().record(rec)
        except Exception:
            # Metrics must never break request handling.
            pass
