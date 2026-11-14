"""Tests for the canonical document-processing and structure-aware chunking core."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag_core import (  # noqa: E402
    MAX_CHUNK_CHARS,
    STRATEGY_NAME,
    Page,
    SourceDocument,
    StructureAwareChunker,
    audit_source_coverage,
    extract_source_document,
)
from src.rag_core.chunker import _normalise_section_id  # noqa: E402


PDF_PATH = ROOT / "data" / "satcom-ngp.pdf"


def _single_page_doc(source: str, text: str) -> SourceDocument:
    return SourceDocument(source=source, pages=[Page(page_number=1, text=text, start_char=0, end_char=len(text))])


def _multi_page_doc(source: str, pages: list[tuple[int, str]]) -> SourceDocument:
    built: list[Page] = []
    offset = 0
    for number, text in pages:
        built.append(Page(page_number=number, text=text, start_char=offset, end_char=offset + len(text)))
        offset += len(text)
    return SourceDocument(source=source, pages=built)


# ------------------------------------------------------------------ document model


def test_real_pdf_document_offsets_are_consistent():
    doc = extract_source_document(PDF_PATH)
    assert doc.source == "satcom-ngp.pdf"
    assert len(doc.pages) == 15
    assert doc.total_chars > 0

    offset = 0
    for page in doc.pages:
        assert page.start_char == offset
        assert page.end_char == offset + len(page.text)
        offset = page.end_char
    assert offset == doc.total_chars

    first = doc.page_for_offset(0)
    assert first is not None and first.page_number == 1
    last = doc.page_for_offset(doc.total_chars - 1)
    assert last is not None and last.page_number == 15


def test_page_for_offset_boundaries_and_invalid_offsets():
    doc = _multi_page_doc("synthetic.pdf", [(1, "abc"), (2, "def"), (3, "ghi")])
    assert doc.page_for_offset(0).page_number == 1
    assert doc.page_for_offset(2).page_number == 1
    # Offset 3 is the first character of page 2, not the last of page 1.
    assert doc.page_for_offset(3).page_number == 2
    assert doc.page_for_offset(5).page_number == 2
    assert doc.page_for_offset(6).page_number == 3
    assert doc.page_for_offset(9).page_number == 3  # document end is attributed to last page
    assert doc.page_for_offset(-1) is None
    assert doc.page_for_offset(10) is None


# ------------------------------------------------------------------ real-document chunking


def test_real_pdf_chunks_are_source_fidelity_ordered_and_bounded():
    doc = extract_source_document(PDF_PATH)
    chunks = StructureAwareChunker().chunk(doc)
    text = doc.text

    assert chunks, "expected at least one chunk for the SATCOM NGP policy"
    previous_end = 0
    for chunk in sorted(chunks, key=lambda c: c.start_offset):
        assert text[chunk.start_offset : chunk.end_offset] == chunk.text
        assert chunk.text.strip()
        assert len(chunk.text) <= MAX_CHUNK_CHARS
        assert chunk.chunking_strategy == STRATEGY_NAME
        assert chunk.start_offset >= previous_end
        previous_end = chunk.end_offset


def test_real_pdf_detects_all_four_articles_and_normalised_clause_ids():
    doc = extract_source_document(PDF_PATH)
    chunks = StructureAwareChunker().chunk(doc)
    clause_ids = {chunk.section for chunk in chunks if chunk.section}

    assert {chunk.article for chunk in chunks if chunk.article} >= {
        "ARTICLE-1",
        "ARTICLE-2",
        "ARTICLE-3",
        "ARTICLE-4",
    }
    assert "2.4" in clause_ids
    assert "3.7.1.2" in clause_ids
    assert "4.4.6" in clause_ids
    for clause_id in clause_ids:
        parts = clause_id.split(".")
        assert all(part.isdigit() and part for part in parts), f"malformed clause id: {clause_id}"


def test_page_footer_number_is_not_fused_into_next_page_clause_id():
    """Regression: page 2 ends with '2', page 3 starts with '2.4'.

    The old page-agnostic line scanner produced ``22.4`` and attached the clause
    to page 2.  Clause ids and page ranges must reflect the real page boundary.
    """
    doc = extract_source_document(PDF_PATH)
    chunks = StructureAwareChunker().chunk(doc)
    by_section = {chunk.section: chunk for chunk in chunks if chunk.section}

    assert "22.4" not in by_section
    assert by_section["2.4"].start_page == 3
    assert by_section["2.4"].text.startswith("2.4")

    assert "32.5.7" not in by_section and by_section["2.5.7"].start_page == 4
    assert "83.7.1" not in by_section and by_section["3.7.1"].start_page == 9
    assert "103.7.3.3" not in by_section and by_section["3.7.3.3"].start_page == 11
    assert "113.7.5" not in by_section and by_section["3.7.5"].start_page == 12
    assert "124.1" not in by_section and by_section["4.1"].start_page == 13


def test_multipage_clause_chunk_references_correct_pages():
    doc = extract_source_document(PDF_PATH)
    chunks = StructureAwareChunker().chunk(doc)
    spanning = [chunk for chunk in chunks if chunk.start_page != chunk.end_page]
    assert spanning, "expected at least one clause chunk spanning multiple pages"

    sample = next(chunk for chunk in chunks if chunk.section == "3.6.1")
    assert sample.start_page < sample.end_page


def test_clause_heading_metadata_is_not_the_whole_body():
    doc = extract_source_document(PDF_PATH)
    chunks = StructureAwareChunker().chunk(doc)
    clause = next(chunk for chunk in chunks if chunk.section == "3.4.1")

    assert clause.heading is not None
    assert clause.heading.startswith("3.4.1 Authorisation from Department of Space")
    # Body prose from the following lines must not be absorbed as heading text.
    assert "UN Outer Space Treaty" not in clause.heading
    assert "UN Outer Space Treaty" in clause.text


def test_page_footer_number_is_not_part_of_heading_metadata():
    doc = extract_source_document(PDF_PATH)
    chunks = StructureAwareChunker().chunk(doc)
    clause = next(chunk for chunk in chunks if chunk.section == "4.3.3")

    assert "13" not in clause.heading.split()
    assert "Value Added Services (domestic)" in clause.heading


# ------------------------------------------------------------------ splitting


def test_oversized_clause_is_split_and_keeps_source_fidelity():
    text = (
        "ARTICLE-2: NORMS, GUIDELINES AND PROCEDURES REGARDING USE OF INSAT CAPACITY BY NON-GOVERNMENTAL PARTIES\n"
        "\n"
        "2.1 The Satellite Communications Policy Frame-work for India as approved by the Government states: \n"
        "\n"
        + ("Alpha paragraph content line one. \n \n" * 40)
        + "\n"
        + ("Beta paragraph content line two. \n \n" * 40)
    )
    doc = _single_page_doc("synthetic.pdf", text)
    chunks = StructureAwareChunker(max_chars=200).chunk(doc)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= 200
        assert doc.text[chunk.start_offset : chunk.end_offset] == chunk.text
    clause_chunks = [chunk for chunk in chunks if chunk.article == "ARTICLE-2" and chunk.section == "2.1"]
    assert sorted(chunk.part for chunk in clause_chunks) == list(range(len(clause_chunks)))
    assert all(chunk.section == "2.1" for chunk in clause_chunks)
    assert all(chunk.article == "ARTICLE-2" for chunk in clause_chunks)


def test_headingless_document_yields_single_body_chunk_without_loss():
    text = (
        "The norms, guidelines and procedures for implementation of the policy follow: \n"
        "\n"
        "Body content without any numbered clause heading. \n"
        "\n"
        "More body content that must not be discarded."
    )
    doc = _single_page_doc("synthetic.pdf", text)
    chunks = StructureAwareChunker().chunk(doc)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.article is None
    assert chunk.section is None
    assert doc.text[chunk.start_offset : chunk.end_offset] == chunk.text
    assert "Body content without any numbered clause heading" in chunk.text
    assert "More body content that must not be discarded" in chunk.text


# ------------------------------------------------------------------ normalisation


def test_normalise_section_id_removes_page_number_prefix_artefacts():
    assert _normalise_section_id("22.4", 2, "ARTICLE-2") == "2.4"
    assert _normalise_section_id("32.5.7", 3, "ARTICLE-2") == "2.5.7"
    assert _normalise_section_id("103.7.3.3", 10, "ARTICLE-3") == "3.7.3.3"
    assert _normalise_section_id("124.1", 12, "ARTICLE-4") == "4.1"


def test_normalise_section_id_preserves_legitimate_clause_ids():
    assert _normalise_section_id("2.1", 2, "ARTICLE-2") == "2.1"
    assert _normalise_section_id("4.3.1", 13, "ARTICLE-4") == "4.3.1"
    assert _normalise_section_id("3.7.1.2", 9, "ARTICLE-3") == "3.7.1.2"
    assert _normalise_section_id("4.4.6", 15, "ARTICLE-4") == "4.4.6"
    assert _normalise_section_id("3.6.7", 8, "ARTICLE-3") == "3.6.7"
    assert _normalise_section_id("2.7", 4, "ARTICLE-2") == "2.7"
    # A genuine clause in Article 12 must not be mistaken for page 12 + clause 4.1.
    assert _normalise_section_id("12.4.1", 12, "ARTICLE-12") == "12.4.1"
    # Without article context no stripping is attempted.
    assert _normalise_section_id("22.4", 2, None) == "22.4"
    # The remainder must belong to the current article.
    assert _normalise_section_id("124.1", 12, "ARTICLE-2") == "124.1"


# ------------------------------------------------------------------ source coverage / integrity


def test_real_pdf_integrity_audit_passes_every_check():
    doc = extract_source_document(PDF_PATH)
    chunks = StructureAwareChunker().chunk(doc)
    audit = audit_source_coverage(doc, chunks)

    assert audit["ok"], audit["issues"]
    assert audit["within_bounds"]
    assert audit["fidelity_errors"] == []
    assert audit["overlaps"] == []
    assert audit["uncovered_non_whitespace_regions"] == []
    assert audit["duplicate_ids"] == []
    assert audit["page_mapping_errors"] == []
    assert audit["empty_chunks"] == []
    assert audit["oversized_chunks"] == []
    assert audit["sequence_order_ok"]
    assert audit["source_order_ok"]
    assert audit["coverage_percent"] == pytest.approx(100.0, abs=1e-6)
    assert audit["uncovered_non_whitespace_chars"] == 0


def test_audit_detects_uncovered_non_whitespace_content():
    text = (
        "ARTICLE-2: NORMS, GUIDELINES AND PROCEDURES REGARDING USE OF INSAT CAPACITY BY NON-GOVERNMENTAL PARTIES\n"
        "\n"
        "2.1 The Satellite Communications Policy Frame-work for India as approved by the Government states: \n"
        "\n"
        + ("Alpha paragraph content line one. \n \n" * 40)
        + "\n"
        + ("Beta paragraph content line two. \n \n" * 40)
    )
    doc = _single_page_doc("synthetic.pdf", text)
    chunks = StructureAwareChunker(max_chars=200).chunk(doc)
    damaged = chunks[:-1]

    audit = audit_source_coverage(doc, damaged, max_chars=200)
    assert not audit["ok"]
    assert audit["uncovered_non_whitespace_chars"] > 0
    assert audit["uncovered_non_whitespace_regions"]


def test_audit_detects_chunk_text_mutation():
    doc = extract_source_document(PDF_PATH)
    chunks = StructureAwareChunker().chunk(doc)
    original = chunks[5]
    mutated = type(original)(**{**original.__dict__, "text": original.text + " injected"})
    chunks[5] = mutated

    audit = audit_source_coverage(doc, chunks)
    assert not audit["ok"]
    assert any(error["chunk_id"] == original.chunk_id for error in audit["fidelity_errors"])


def test_chunk_ids_are_unique_even_for_duplicate_section_labels():
    text = (
        "ARTICLE-2: SAMPLE ARTICLE\n"
        "\n"
        "2.1 First clause with a duplicated label.\n"
        "\n"
        "2.1 Second clause with the same label.\n"
    )
    doc = _single_page_doc("synthetic.pdf", text)
    chunks = StructureAwareChunker().chunk(doc)

    assert len(chunks) == 3
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    assert [chunk.chunk_sequence for chunk in chunks] == list(range(len(chunks)))


# ------------------------------------------------------------------ identifiers / page mapping


def test_real_pdf_chunk_ids_are_unique_and_stable():
    doc = extract_source_document(PDF_PATH)
    first = StructureAwareChunker().chunk(doc)
    second = StructureAwareChunker().chunk(doc)

    first_ids = [chunk.chunk_id for chunk in first]
    second_ids = [chunk.chunk_id for chunk in second]
    assert len(set(first_ids)) == len(first_ids)
    assert first_ids == second_ids
    assert [chunk.chunk_sequence for chunk in first] == [chunk.chunk_sequence for chunk in second]


def test_chunk_page_relative_offsets_are_consistent():
    doc = extract_source_document(PDF_PATH)
    chunks = StructureAwareChunker().chunk(doc)

    sample = next(chunk for chunk in chunks if chunk.section == "3.6.1")
    assert sample.start_page < sample.end_page
    start_page_obj = doc.page_for_offset(sample.start_offset)
    end_page_obj = doc.page_for_offset(sample.end_offset - 1)
    assert start_page_obj is not None and end_page_obj is not None
    assert sample.start_page_char == sample.start_offset - start_page_obj.start_char
    assert sample.end_page_char == sample.end_offset - end_page_obj.start_char


def test_single_page_chunk_page_relative_offsets():
    doc = extract_source_document(PDF_PATH)
    chunks = StructureAwareChunker().chunk(doc)
    for chunk in chunks:
        start_page_obj = doc.page_for_offset(chunk.start_offset)
        end_page_obj = doc.page_for_offset(max(chunk.end_offset - 1, chunk.start_offset))
        if chunk.start_page == chunk.end_page:
            assert chunk.start_page_char == chunk.start_offset - start_page_obj.start_char
            assert chunk.end_page_char == chunk.end_offset - end_page_obj.start_char


# ------------------------------------------------------------------ malformed / ambiguous headings


def test_ocr_corrupted_clause_heading_is_still_detected():
    text = (
        "3.6.9 Normatves and procedurs shall apply from time to time. \n"
        "\n"
        "Body of the corrupted heading follows."
    )
    doc = _single_page_doc("synthetic.pdf", text)
    chunks = StructureAwareChunker().chunk(doc)

    clause_chunks = [chunk for chunk in chunks if chunk.section == "3.6.9"]
    assert clause_chunks, "structurally valid OCR-corrupted heading should still be detected"
    assert any("Normatves and procedurs shall apply" in chunk.text for chunk in clause_chunks)
    assert doc.text[clause_chunks[0].start_offset : clause_chunks[0].end_offset] == clause_chunks[0].text


def test_lowercase_after_clause_number_does_not_lose_content():
    """A dot-numbered line that is not a structurally valid clause is body text.

    It may not become a heading, but it must still be present in the preceding
    chunk.  This guards against content loss when OCR damages a heading.
    """
    text = (
        "2.1 A valid clause heading starts here. \n"
        "\n"
        "2.2 lower-case continuation that is not treated as a clause heading. \n"
        "\n"
        "End of body."
    )
    doc = _single_page_doc("synthetic.pdf", text)
    chunks = StructureAwareChunker().chunk(doc)

    assert {chunk.section for chunk in chunks} == {"2.1"}
    assert any("2.2 lower-case continuation" in chunk.text for chunk in chunks)


def test_unexpected_numeric_heading_format_is_not_treated_as_clause():
    text = (
        "3-6-9 Something Something \n"
        "\n"
        "Body text that follows an unrecognised numeric heading format."
    )
    doc = _single_page_doc("synthetic.pdf", text)
    chunks = StructureAwareChunker().chunk(doc)

    assert all(chunk.section is None for chunk in chunks)
    assert any("unrecognised numeric heading format" in chunk.text for chunk in chunks)
    assert doc.text[chunks[0].start_offset : chunks[0].end_offset] == chunks[0].text


def test_clause_without_article_heading_is_chunked():
    text = (
        "2.1 Some Clause Title \n"
        "\n"
        "Body of the clause that has no article heading."
    )
    doc = _single_page_doc("synthetic.pdf", text)
    chunks = StructureAwareChunker().chunk(doc)

    clause_chunks = [chunk for chunk in chunks if chunk.section == "2.1"]
    assert clause_chunks
    assert all(chunk.article is None for chunk in clause_chunks)
    assert doc.text[clause_chunks[0].start_offset : clause_chunks[0].end_offset] == clause_chunks[0].text


def test_heading_wrapped_across_lines_is_absorbed():
    text = (
        "3.6 NORMS, GUIDELINES AND PROCEDURES TO BE FOLLOWED IN \n"
        "ISSUING THE LICENSES \n"
        "\n"
        "Body of the wrapped heading."
    )
    doc = _single_page_doc("synthetic.pdf", text)
    chunks = StructureAwareChunker().chunk(doc)

    clause_chunks = [chunk for chunk in chunks if chunk.section == "3.6"]
    assert clause_chunks
    joined = " ".join(clause_chunks[0].heading.split())
    assert "3.6 NORMS, GUIDELINES AND PROCEDURES TO BE FOLLOWED IN ISSUING THE LICENSES" in joined


def test_ordinary_numbered_body_text_is_not_a_clause():
    text = (
        "4.1 SERVICE CLASSIFICATION \n"
        "\n"
        "1. International (i.e., trans-bo rder) communications services \n"
        "2. Domestic communication services \n"
        "\n"
        "The Satellite Communication Services could be broadly classified."
    )
    doc = _single_page_doc("synthetic.pdf", text)
    chunks = StructureAwareChunker().chunk(doc)

    sections = {chunk.section for chunk in chunks}
    assert sections == {"4.1"}
    assert any("International (i.e., trans-bo rder) communications services" in chunk.text for chunk in chunks)
    assert any("The Satellite Communication Services could be broadly classified" in chunk.text for chunk in chunks)
