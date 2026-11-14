"""Canonical document-processing and chunking core for the SATCOM NGP RAG app.

This package is the single reusable source of truth for:

- Source document representation (pages, text, character offsets).
- Structure-aware chunking driven by numbered policy clauses.

Experiments and the future production ingestion pipeline must both use this
package. Old, isolated implementations (for example ``src.chunking.structure``)
are historical baselines and must not be duplicated here.
"""

from .chunker import (
    ARTICLE_RE,
    CLAUSE_RE,
    FIXED_CHARACTER_STRATEGY,
    MAX_CHUNK_CHARS,
    STRATEGY_NAME,
    Chunk,
    FixedCharacterChunker,
    Heading,
    StructureAwareChunker,
    _normalise_section_id,
    _plan_parts,
    _paragraphs,
    _find_headings,
    _finalise,
    _iter_lines,
    audit_source_coverage,
)
from .document import Page, SourceDocument, extract_source_document
from .evaluation import (
    EvidenceUnit,
    Question,
    aggregate_metrics,
    compute_retrieval_metrics,
    evaluate_answer_support,
    evidence_coverage_summary,
    evidence_for_question,
    load_exp001_evidence,
    load_questions,
    overlap_chars,
)
from .index import (
    DEFAULT_EMBEDDING_MODEL,
    DenseIndex,
    DenseIndexConfig,
    RetrievedChunk,
    chunk_to_metadata,
)
from .pipeline import (
    ABSTAINED,
    ANSWERED,
    ABSTENTION_MESSAGE,
    DEFAULT_SIBLING_SCORE_RATIO,
    AbstentionPolicy,
    AnswerStatement,
    Citation,
    GroundedAnswer,
    GroundedPipeline,
    content_tokens,
    token_support,
)
from .generation import (
    DEFAULT_GENERATOR_MODEL,
    DEFAULT_GENERATOR_OPTIONS,
    GENERATION_ERROR,
    INSUFFICIENT_EVIDENCE,
    MODEL_ABSTAINED,
    GeneratedAnswer,
    GeneratedClaim,
    GenerationContractError,
    OllamaGenerator,
    assemble_generated_answer,
    build_selected_evidence_prompt,
    parse_generated_claims,
)

__all__ = [
    "ABSTAINED",
    "ABSTENTION_MESSAGE",
    "ANSWERED",
    "ARTICLE_RE",
    "AbstentionPolicy",
    "CLAUSE_RE",
    "AnswerStatement",
    "Citation",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_GENERATOR_MODEL",
    "DEFAULT_GENERATOR_OPTIONS",
    "DEFAULT_SIBLING_SCORE_RATIO",
    "FIXED_CHARACTER_STRATEGY",
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
    "GeneratedAnswer",
    "GeneratedClaim",
    "GenerationContractError",
    "GroundedAnswer",
    "GroundedPipeline",
    "Heading",
    "OllamaGenerator",
    "Page",
    "Question",
    "RetrievedChunk",
    "SourceDocument",
    "StructureAwareChunker",
    "aggregate_metrics",
    "assemble_generated_answer",
    "audit_source_coverage",
    "build_selected_evidence_prompt",
    "chunk_to_metadata",
    "compute_retrieval_metrics",
    "evaluate_answer_support",
    "evidence_coverage_summary",
    "evidence_for_question",
    "extract_source_document",
    "load_exp001_evidence",
    "load_questions",
    "overlap_chars",
    "parse_generated_claims",
]
