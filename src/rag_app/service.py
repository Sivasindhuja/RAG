from __future__ import annotations

import os
import time
from typing import Any, Optional, Sequence

from src.rag_core import (
    ABSTAINED,
    ANSWERED,
    DenseIndex,
    DenseIndexConfig,
    GeminiGenerator,
    GroundedPipeline,
    RetrievedChunk,
    content_tokens,
)
from src.rag_core.pipeline import ABSTENTION_MESSAGE

from .config import AppConfig
from .metrics import RequestRecord, get_metrics_collector
from .observability import RequestTrace
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
    return len(q & s) / len(q) if q else 1.0


class RAGService:
    """Single application service composing validated rag_core components."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        *,
        index: Optional[DenseIndex] = None,
        pipeline: Optional[GroundedPipeline] = None,
        gemini_generator: Optional[GeminiGenerator] = None,
    ) -> None:
        self.config = config or AppConfig.from_env()
        self._index = index
        self._pipeline = pipeline
        self._gemini_generator = gemini_generator
        self._model: Any = None

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
        return self._index

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
            collection_name=self.config.collection_name,
            top_k=self.config.top_k,
            index_available=index_available,
            gemini_configured=gemini_configured,
        )

    def query(self, question: str, mode: Optional[str] = None, top_k: Optional[int] = None) -> QueryResponse:
        trace = RequestTrace()
        question = (question or "").strip()
        requested_mode = (mode or self.config.default_generation_mode or "extractive").lower()
        if requested_mode not in {"extractive", "gemini"}:
            requested_mode = "extractive"
        effective_top_k = int(top_k or self.config.top_k)

        if not question:
            return QueryResponse(
                request_id=trace.request_id,
                status="abstained",
                mode=requested_mode,
                question=question,
                answer="I cannot answer an empty question.",
                abstained=True,
                abstention_reason="empty question",
                timings_ms={},
                errors=[],
            )

        try:
            with trace.stage("retrieval", top_k=effective_top_k):
                hits = list(self.index.search(question, k=effective_top_k))
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
                    }
                    for h in hits
                ],
            )
            with trace.stage("selection_and_extractive", hit_count=len(hits)):
                grounded = self.pipeline.answer_from_hits(question, hits)

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

            # Conservative, general answerability guard (application-level):
            # require selected extractive evidence to have sufficient lexical support
            # for the question. This reuses existing sufficiency signals to avoid
            # answering on traceable-but-weak evidence; it does not change retrieval,
            # chunking, selection defaults, or add experiments.
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
                    "selection_defaults": {
                        "sibling_score_ratio": self.config.sibling_score_ratio,
                        "cross_reference_selection": self.config.cross_reference_selection,
                        "associated_heading_units": self.config.associated_heading_units,
                        "definition_intent_boost": self.config.definition_intent_boost,
                    },
                },
            )

            if requested_mode == "gemini":
                self._apply_gemini(response, question, grounded, selected_support, trace)
                response.timings_ms = dict(trace.timings_ms)
                merged = list(dict.fromkeys([*(response.errors or []), *(trace.errors or [])]))
                response.errors = merged

            # include lightweight trace details for observability/export (kept under debug)
            response.debug["trace"] = trace.to_dict()
            self._record_metrics(response, trace)
            return response
        except Exception as exc:  # noqa: BLE001
            trace.errors.append(f"request: {type(exc).__name__}: {exc}")
            response = QueryResponse(
                request_id=trace.request_id,
                status="error",
                mode=requested_mode,
                question=question,
                answer="",
                abstained=False,
                abstention_reason=None,
                timings_ms=dict(trace.timings_ms),
                errors=list(trace.errors),
                debug={"error_type": type(exc).__name__, "trace": trace.to_dict()},
            )
            self._record_metrics(response, trace)
            return response

    def _apply_gemini(
        self,
        response: QueryResponse,
        question: str,
        grounded: Any,
        selected_support: float,
        trace: RequestTrace,
    ) -> None:
        extractive_citations = [_citation_model(c) for c in grounded.citations]
        gen = GenerationModel(
            attempted=True,
            mode="gemini",
            text=response.answer,
            citations=extractive_citations,
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
            }
            response.generation = gen
            return

        try:
            with trace.stage("gemini_generation", model=self.config.gemini_model):
                result = self.gemini_generator.generate(
                    question,
                    grounded.statements,
                    grounded.citations,
                    evidence_sufficient=True,
                )
        except Exception as exc:  # noqa: BLE001
            gen.status = "generation_error"
            gen.errors.append(f"{type(exc).__name__}: {exc}")
            gen.fallback_to_extractive = True
            response.generation = gen
            response.errors.append(f"gemini_generation: {type(exc).__name__}: {exc}")
            return

        gen.status = result.status
        gen.model = result.model
        gen.prompt_version = (result.metadata or {}).get("prompt_version")
        gen.latency_ms = result.latency_ms
        gen.errors = list(result.errors or [])
        gen.metadata = dict(result.metadata or {})
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
                total_tokens = prompt_tokens + completion_tokens
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
