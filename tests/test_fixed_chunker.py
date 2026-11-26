"""Tests for the canonical fixed-character chunker (accepted EXP-001 baseline)."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag_core import (  # noqa: E402
    FIXED_CHARACTER_STRATEGY,
    FixedCharacterChunker,
    Page,
    SourceDocument,
    audit_source_coverage,
    extract_source_document,
)


PDF_PATH = ROOT / "data" / "satcom-ngp.pdf"
EXP001_CHUNKS = ROOT / "tests" / "fixtures" / "exp001" / "chunks.json"


def _doc(pages):
    built = []
    offset = 0
    for number, text in pages:
        built.append(Page(page_number=number, text=text, start_char=offset, end_char=offset + len(text)))
        offset += len(text)
    return SourceDocument(source="synthetic.pdf", pages=built)


def test_fixed_chunker_reproduces_accepted_exp001_chunks():
    doc = extract_source_document(PDF_PATH)
    chunks = FixedCharacterChunker(chunk_size=1000, overlap=200).chunk(doc)
    expected = json.loads(EXP001_CHUNKS.read_text(encoding="utf-8"))["chunks"]

    assert len(chunks) == len(expected) == 49
    for actual, reference in zip(chunks, expected):
        assert actual.text == reference["text"]
        assert actual.start_page == reference["page"]
        assert actual.end_page == reference["page"]
        assert doc.text[actual.start_offset : actual.end_offset] == actual.text
        assert actual.chunking_strategy == FIXED_CHARACTER_STRATEGY


def test_fixed_chunker_overlap_and_page_boundaries():
    doc = extract_source_document(PDF_PATH)
    chunks = FixedCharacterChunker(chunk_size=1000, overlap=200).chunk(doc)

    by_page = {}
    for chunk in chunks:
        assert chunk.start_page == chunk.end_page
        assert len(chunk.text) <= 1000
        assert doc.text[chunk.start_offset : chunk.end_offset] == chunk.text
        by_page.setdefault(chunk.start_page, []).append(chunk)

    expected_overlap = 0
    for page_chunks in by_page.values():
        for previous, following in zip(page_chunks, page_chunks[1:]):
            assert previous.end_offset - following.start_offset == 200
            expected_overlap += 200

    audit = audit_source_coverage(doc, chunks, max_chars=1000, allow_overlap=True)
    assert audit["ok"], audit["issues"]
    assert audit["coverage_percent"] == pytest.approx(100.0, abs=1e-6)
    assert audit["overlap_chars"] == expected_overlap
    assert audit["duplicate_ids"] == []
    assert audit["page_mapping_errors"] == []


def test_fixed_chunker_default_audit_rejects_overlap():
    doc = extract_source_document(PDF_PATH)
    chunks = FixedCharacterChunker(chunk_size=1000, overlap=200).chunk(doc)

    audit = audit_source_coverage(doc, chunks, max_chars=1000)
    assert not audit["ok"]
    assert any("overlap" in issue for issue in audit["issues"])


def test_fixed_chunker_ids_are_unique_and_stable():
    doc = extract_source_document(PDF_PATH)
    first = FixedCharacterChunker().chunk(doc)
    second = FixedCharacterChunker().chunk(doc)

    ids = [chunk.chunk_id for chunk in first]
    assert len(set(ids)) == len(ids)
    assert ids == [chunk.chunk_id for chunk in second]
    assert [chunk.chunk_sequence for chunk in first] == list(range(len(first)))


def test_fixed_chunker_handles_empty_pages_without_losing_content():
    doc = _doc([(1, "   \n  "), (2, "abcdefghij"), (3, "")])
    chunks = FixedCharacterChunker(chunk_size=4, overlap=1).chunk(doc)

    assert chunks
    assert all(chunk.start_page == 2 and chunk.end_page == 2 for chunk in chunks)
    assert "".join(chunk.text for chunk in chunks[:1]) == "abcd"
    audit = audit_source_coverage(doc, chunks, max_chars=4, allow_overlap=True)
    assert audit["ok"], audit["issues"]
    assert audit["coverage_percent"] == pytest.approx(100.0, abs=1e-6)


def test_fixed_chunker_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        FixedCharacterChunker(chunk_size=0)
    with pytest.raises(ValueError):
        FixedCharacterChunker(chunk_size=100, overlap=-1)
    with pytest.raises(ValueError):
        FixedCharacterChunker(chunk_size=100, overlap=100)
