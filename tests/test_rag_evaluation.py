"""Tests for shared evidence mapping, retrieval metrics and dense indexing."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag_core import (  # noqa: E402
    Chunk,
    DenseIndex,
    DenseIndexConfig,
    EvidenceUnit,
    RetrievedChunk,
    aggregate_metrics,
    audit_source_coverage,
    compute_retrieval_metrics,
    extract_source_document,
    load_exp001_evidence,
    load_questions,
)
from src.rag_core.index import chunk_to_metadata  # noqa: E402


PDF_PATH = ROOT / "data" / "satcom-ngp.pdf"
QUESTIONS_CSV = ROOT / "experiments" / "exp001_questions.csv"
GROUND_TRUTH = ROOT / "experiments" / "exp001_ground_truth.json"


def _evidence(evidence_id: str, start: int, end: int) -> EvidenceUnit:
    return EvidenceUnit(
        question_id="Q1",
        evidence_id=evidence_id,
        source="satcom-ngp.pdf",
        page=1,
        clause_id="1.1",
        start_offset=start,
        end_offset=end,
        page_relative_start=start,
        page_relative_end=end,
        excerpt="x" * (end - start),
    )


def _hit(chunk_id: str, start: int, end: int, rank: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        rank=rank,
        chunk_id=chunk_id,
        text="x" * (end - start),
        source="satcom-ngp.pdf",
        start_offset=start,
        end_offset=end,
        start_page=1,
        end_page=1,
        article="ARTICLE-1",
        section="1.1",
        heading="heading",
        distance=0.1,
        metadata={},
    )


def _chunk(index: int, text: str, start: int, page: int = 1) -> Chunk:
    return Chunk(
        chunk_id=f"c{index}",
        source="satcom-ngp.pdf",
        article="ARTICLE-1",
        section="1.1",
        section_hierarchy=("1", "1"),
        heading="heading",
        start_page=page,
        end_page=page,
        start_offset=start,
        end_offset=start + len(text),
        text=text,
        chunking_strategy="test",
        chunk_sequence=index,
    )


# ------------------------------------------------------------------ approved evidence


def test_approved_ground_truth_maps_to_global_offsets_and_verifies_excerpts():
    doc = extract_source_document(PDF_PATH)
    units, problems = load_exp001_evidence(GROUND_TRUTH, doc)

    assert problems == []
    assert len(units) == 20
    for unit in units:
        assert doc.text[unit.start_offset : unit.end_offset] == unit.excerpt
        assert unit.end_offset > unit.start_offset


def test_approved_question_dataset_is_preserved():
    questions = load_questions(QUESTIONS_CSV)

    assert len(questions) == 15
    assert [q.question_id for q in questions] == [f"EXP001-Q{i:02d}" for i in range(1, 16)]
    assert [q.question_id for q in questions if len(q.relevant_clause_ids) > 1] == [
        "EXP001-Q08",
        "EXP001-Q13",
    ]


# ------------------------------------------------------------------ metrics


def test_metrics_single_evidence_rank_two():
    evidence = [_evidence("E1", 100, 200)]
    hits = [_hit("c0", 0, 50, rank=1), _hit("c1", 150, 250, rank=2)]

    metrics = compute_retrieval_metrics(hits, evidence, k=5)
    assert metrics["recall@5"] == 1.0
    assert metrics["precision@5"] == 0.2
    assert metrics["mrr@5"] == 0.5
    assert metrics["relevant_flags"] == [False, True]
    assert metrics["covered_evidence_ids"] == ["E1"]
    assert metrics["missing_evidence_ids"] == []
    assert metrics["overlap_chars_by_rank"] == [0, 50]


def test_metrics_multi_evidence_requires_all_units():
    evidence = [_evidence("E1", 100, 200), _evidence("E2", 500, 600)]
    hits = [_hit("c0", 120, 180, rank=1), _hit("c1", 550, 650, rank=2)]

    metrics = compute_retrieval_metrics(hits, evidence, k=5)
    assert metrics["recall@5"] == 1.0
    assert metrics["precision@5"] == 0.4
    assert metrics["mrr@5"] == 1.0
    assert metrics["evidence_coverage"]["E1"]["best_rank"] == 1
    assert metrics["evidence_coverage"]["E2"]["best_rank"] == 2


def test_metrics_missing_one_required_evidence_unit():
    evidence = [_evidence("E1", 100, 200), _evidence("E2", 500, 600)]
    hits = [_hit("c0", 120, 180, rank=1), _hit("c1", 800, 900, rank=2)]

    metrics = compute_retrieval_metrics(hits, evidence, k=5)
    assert metrics["recall@5"] == 0.5
    assert metrics["missing_evidence_ids"] == ["E2"]


def test_metrics_no_relevant_hits_and_aggregate():
    evidence = [_evidence("E1", 100, 200)]
    hits = [_hit("c0", 300, 400, rank=1)]
    metrics = compute_retrieval_metrics(hits, evidence, k=5)
    metrics["question_id"] = "Q1"

    assert metrics["recall@5"] == 0.0
    assert metrics["precision@5"] == 0.0
    assert metrics["mrr@5"] == 0.0
    assert aggregate_metrics([metrics], k=5) == {"recall@5": 0.0, "precision@5": 0.0, "mrr@5": 0.0}


def test_metrics_reject_empty_evidence():
    with pytest.raises(ValueError):
        compute_retrieval_metrics([], [], k=5)


# ------------------------------------------------------------------ metadata / index


def test_chunk_metadata_is_scalar_and_preserves_traceability():
    chunk = _chunk(0, "hello", 10)
    metadata = chunk_to_metadata(chunk)

    assert metadata["chunk_id"] == "c0"
    assert metadata["start_offset"] == 10
    assert metadata["end_offset"] == 15
    assert metadata["section_hierarchy"] == "1.1"
    assert all(isinstance(value, (str, int, float)) for value in metadata.values())


class _FakeModel:
    """Deterministic two-dimensional encoder for index tests."""

    def encode(self, texts):
        return [[float(len(text)), float(sum(ord(char) for char in text) % 97)] for text in texts]


def test_dense_index_build_search_and_rebuild(tmp_path):
    chunks = [_chunk(0, "alpha policy clause", 0), _chunk(1, "beta satellite clause", 20), _chunk(2, "gamma", 40)]
    config = DenseIndexConfig(
        collection_name="test_collection",
        persist_dir=tmp_path / "chroma",
        embedding_model="fake-model",
    )
    index = DenseIndex(config, model=_FakeModel())
    stats = index.build(chunks)

    assert stats["chunks"] == 3
    assert index.count() == 3
    hits = index.search("alpha policy clause", k=2)
    assert len(hits) == 2
    assert hits[0].rank == 1
    assert hits[0].chunk_id == "c0"
    assert hits[0].start_offset == 0
    assert hits[0].end_offset == len("alpha policy clause")
    assert hits[0].metadata["chunking_strategy"] == "test"

    # Rebuilding must not duplicate records.
    index.build(chunks)
    assert index.count() == 3


def test_dense_index_rejects_empty_query_and_handles_empty_collection(tmp_path):
    config = DenseIndexConfig(
        collection_name="test_empty_collection",
        persist_dir=tmp_path / "chroma",
        embedding_model="fake-model",
    )
    index = DenseIndex(config, model=_FakeModel())
    assert index.search("anything", k=5) == []
    with pytest.raises(ValueError):
        index.search("   ", k=5)
    with pytest.raises(ValueError):
        index.build([])
