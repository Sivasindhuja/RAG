from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question about satcom-ngp.pdf")
    mode: Literal["extractive", "gemini"] = "gemini"
    top_k: Optional[int] = Field(default=None, ge=1, le=20)
    retrieval_mode: Optional[Literal["dense", "hybrid", "hybrid_rerank"]] = None


class CitationModel(BaseModel):
    marker: str
    chunk_id: str
    source: str
    article: Optional[str] = None
    section: Optional[str] = None
    heading: Optional[str] = None
    start_page: int
    end_page: int
    start_offset: int
    end_offset: int
    quote: str
    distance: Optional[float] = None


class StatementModel(BaseModel):
    text: str
    marker: str
    chunk_id: str
    start_offset: int
    end_offset: int


class RetrievedChunkModel(BaseModel):
    rank: int
    chunk_id: str
    section: Optional[str] = None
    start_page: int
    end_page: int
    start_offset: int
    end_offset: int
    distance: Optional[float] = None
    # Optional diagnostic fields (additive; default None/empty for backward compatibility).
    score: Optional[float] = None
    retrieval_scores: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CitationValidationModel(BaseModel):
    app_resolved: bool = True
    invalid_source_ids: List[int] = Field(default_factory=list)
    uncited_claims: int = 0
    errors: List[str] = Field(default_factory=list)


class GenerationModel(BaseModel):
    attempted: bool = False
    mode: Literal["extractive", "gemini"] = "gemini"
    status: Optional[str] = None
    model: Optional[str] = None
    prompt_version: Optional[str] = None
    text: Optional[str] = None
    claims: List[Dict[str, Any]] = Field(default_factory=list)
    citations: List[CitationModel] = Field(default_factory=list)
    latency_ms: Optional[float] = None
    fallback_to_extractive: bool = False
    errors: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    request_id: str
    status: Literal["answered", "abstained", "error"]
    mode: Literal["extractive", "gemini"]
    retrieval_mode: str = "dense"  # includes experimental weaviate_* modes/alpha labels
    question: str
    answer: str
    abstained: bool
    abstention_reason: Optional[str] = None
    citations: List[CitationModel] = Field(default_factory=list)
    statements: List[StatementModel] = Field(default_factory=list)
    retrieved: List[RetrievedChunkModel] = Field(default_factory=list)
    selected_evidence_ids: List[str] = Field(default_factory=list)
    citation_validation: CitationValidationModel = Field(default_factory=CitationValidationModel)
    generation: GenerationModel = Field(default_factory=GenerationModel)
    timings_ms: Dict[str, float] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    debug: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    default_mode: str
    retrieval_mode: str
    retriever_backend: str = "chroma"
    collection_name: str
    top_k: int
    index_available: bool
    index_status: Literal["unknown", "available", "unavailable"] = "unknown"
    index_initialized: bool = False
    gemini_configured: bool
    langfuse: Dict[str, Any] = Field(default_factory=dict)
