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
    MAX_CHUNK_CHARS,
    STRATEGY_NAME,
    Chunk,
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

__all__ = [
    "ARTICLE_RE",
    "CLAUSE_RE",
    "DEFAULT_EMBEDDING_MODEL",
    "MAX_CHUNK_CHARS",
    "STRATEGY_NAME",
    "Chunk",
    "DenseIndex",
    "DenseIndexConfig",
    "EvidenceUnit",
    "Heading",
    "Page",
    "Question",
    "RetrievedChunk",
    "SourceDocument",
    "StructureAwareChunker",
    "aggregate_metrics",
    "audit_source_coverage",
    "chunk_to_metadata",
    "compute_retrieval_metrics",
    "evidence_coverage_summary",
    "evidence_for_question",
    "extract_source_document",
    "load_exp001_evidence",
    "load_questions",
    "overlap_chars",
]
