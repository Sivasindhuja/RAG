"""Canonical document-processing and chunking core for the SATCOM NGP RAG app.

This package is the single reusable source of truth for:

- Source document representation (pages, text, character offsets).
- Structure-aware chunking driven by numbered policy clauses.
- Fixed-character chunking used by the selected production configuration.

The application ingestion pipeline and evaluation harness use this package.

Heavy submodules are lazy-loaded via __getattr__ to keep Weaviate-only runtime light.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "ABSTAINED",
    "ABSTENTION_MESSAGE",
    "ANSWERED",
    "ARTICLE_RE",
    "AbstentionPolicy",
    "CLAUSE_RE",
    "AnswerStatement",
    "Citation",
    "CorpusIndex",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_GEMINI_MODEL",
    "DEFAULT_GENERATOR_MODEL",
    "DEFAULT_GENERATOR_OPTIONS",
    "DEFAULT_SIBLING_SCORE_RATIO",
    "EVIDENCE_INSUFFICIENT",
    "FIXED_CHARACTER_STRATEGY",
    "GEMINI_PROMPT_VERSION",
    "GENERATION_ERROR",
    "INSUFFICIENT_EVIDENCE",
    "MAX_CHUNK_CHARS",
    "MODEL_ABSTAINED",
    "STRATEGY_NAME",
    "Chunk",
    "DenseIndex",
    "DenseIndexConfig",
    "EvidenceUnit",
    "FixedCharacterChunker",
    "GeminiGenerator",
    "GeneratedAnswer",
    "GeneratedClaim",
    "GenerationContractError",
    "GroundedAnswer",
    "GroundedPipeline",
    "Heading",
    "HybridRetriever",
    "OllamaGenerator",
    "Page",
    "Question",
    "RetrievalResult",
    "RetrievedChunk",
    "SourceDocument",
    "StructureAwareChunker",
    "aggregate_metrics",
    "assemble_generated_answer",
    "audit_source_coverage",
    "build_gemini_evidence_prompt",
    "build_selected_evidence_prompt",
    "chunk_to_metadata",
    "cohere_rerank",
    "compute_retrieval_metrics",
    "evaluate_answer_support",
    "evidence_coverage_summary",
    "evidence_for_question",
    "extract_source_document",
    "load_exp001_evidence",
    "load_questions",
    "overlap_chars",
    "parse_generated_claims",
    "rrf_fuse",
    "tokenize",
]

_LAZY: dict[str, tuple[str, str]] = {
    # chunker
    "ARTICLE_RE": (".chunker", "ARTICLE_RE"),
    "CLAUSE_RE": (".chunker", "CLAUSE_RE"),
    "FIXED_CHARACTER_STRATEGY": (".chunker", "FIXED_CHARACTER_STRATEGY"),
    "MAX_CHUNK_CHARS": (".chunker", "MAX_CHUNK_CHARS"),
    "STRATEGY_NAME": (".chunker", "STRATEGY_NAME"),
    "Chunk": (".chunker", "Chunk"),
    "FixedCharacterChunker": (".chunker", "FixedCharacterChunker"),
    "Heading": (".chunker", "Heading"),
    "StructureAwareChunker": (".chunker", "StructureAwareChunker"),
    "audit_source_coverage": (".chunker", "audit_source_coverage"),
    # document
    "Page": (".document", "Page"),
    "SourceDocument": (".document", "SourceDocument"),
    "extract_source_document": (".document", "extract_source_document"),
    # evaluation
    "EvidenceUnit": (".evaluation", "EvidenceUnit"),
    "Question": (".evaluation", "Question"),
    "aggregate_metrics": (".evaluation", "aggregate_metrics"),
    "compute_retrieval_metrics": (".evaluation", "compute_retrieval_metrics"),
    "evaluate_answer_support": (".evaluation", "evaluate_answer_support"),
    "evidence_coverage_summary": (".evaluation", "evidence_coverage_summary"),
    "evidence_for_question": (".evaluation", "evidence_for_question"),
    "load_exp001_evidence": (".evaluation", "load_exp001_evidence"),
    "load_questions": (".evaluation", "load_questions"),
    "overlap_chars": (".evaluation", "overlap_chars"),
    # index (heavy: chromadb) - RetrievedChunk moved to retrieval_types
    "DEFAULT_EMBEDDING_MODEL": (".index", "DEFAULT_EMBEDDING_MODEL"),
    "DenseIndex": (".index", "DenseIndex"),
    "DenseIndexConfig": (".index", "DenseIndexConfig"),
    "chunk_to_metadata": (".index", "chunk_to_metadata"),
    # retrieval (heavy: rank_bm25) - RetrievalResult moved to retrieval_types
    "CorpusIndex": (".retrieval", "CorpusIndex"),
    "HybridRetriever": (".retrieval", "HybridRetriever"),
    "cohere_rerank": (".retrieval", "cohere_rerank"),
    "rrf_fuse": (".retrieval", "rrf_fuse"),
    "tokenize": (".retrieval", "tokenize"),
    # lightweight shared types
    "RetrievedChunk": (".retrieval_types", "RetrievedChunk"),
    "RetrievalResult": (".retrieval_types", "RetrievalResult"),
    # pipeline
    "ABSTAINED": (".pipeline", "ABSTAINED"),
    "ANSWERED": (".pipeline", "ANSWERED"),
    "ABSTENTION_MESSAGE": (".pipeline", "ABSTENTION_MESSAGE"),
    "DEFAULT_SIBLING_SCORE_RATIO": (".pipeline", "DEFAULT_SIBLING_SCORE_RATIO"),
    "AbstentionPolicy": (".pipeline", "AbstentionPolicy"),
    "AnswerStatement": (".pipeline", "AnswerStatement"),
    "Citation": (".pipeline", "Citation"),
    "GroundedAnswer": (".pipeline", "GroundedAnswer"),
    "GroundedPipeline": (".pipeline", "GroundedPipeline"),
    "content_tokens": (".pipeline", "content_tokens"),
    "token_support": (".pipeline", "token_support"),
    # generation
    "DEFAULT_GEMINI_MODEL": (".generation", "DEFAULT_GEMINI_MODEL"),
    "DEFAULT_GENERATOR_MODEL": (".generation", "DEFAULT_GENERATOR_MODEL"),
    "DEFAULT_GENERATOR_OPTIONS": (".generation", "DEFAULT_GENERATOR_OPTIONS"),
    "EVIDENCE_INSUFFICIENT": (".generation", "EVIDENCE_INSUFFICIENT"),
    "GEMINI_PROMPT_VERSION": (".generation", "GEMINI_PROMPT_VERSION"),
    "GENERATION_ERROR": (".generation", "GENERATION_ERROR"),
    "INSUFFICIENT_EVIDENCE": (".generation", "INSUFFICIENT_EVIDENCE"),
    "MODEL_ABSTAINED": (".generation", "MODEL_ABSTAINED"),
    "GeminiGenerator": (".generation", "GeminiGenerator"),
    "GeneratedAnswer": (".generation", "GeneratedAnswer"),
    "GeneratedClaim": (".generation", "GeneratedClaim"),
    "GenerationContractError": (".generation", "GenerationContractError"),
    "OllamaGenerator": (".generation", "OllamaGenerator"),
    "assemble_generated_answer": (".generation", "assemble_generated_answer"),
    "build_gemini_evidence_prompt": (".generation", "build_gemini_evidence_prompt"),
    "build_selected_evidence_prompt": (".generation", "build_selected_evidence_prompt"),
    "parse_generated_claims": (".generation", "parse_generated_claims"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_name, attr = target
    mod = import_module(mod_name, __name__)
    value = getattr(mod, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
