import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp001 import (  # noqa: E402
    QUESTIONS_CSV,
    EvidenceUnit,
    Chunk,
    PageText,
    build_clause_spans,
    chunk_overlaps,
    compute_metrics,
    generate_ground_truth_entries,
    get_relevant_evidence,
    load_ground_truth,
    load_pages,
    load_questions,
    make_chunks,
    validate_evidence_offset,
    validate_ground_truth,
)


def _chunk(cid, source, page, start, end, text, idx=0, article=None, clause=None):
    return Chunk(
        chunk_id=cid,
        source=source,
        page=page,
        start=start,
        end=end,
        text=text,
        chunk_index=idx,
        article=article,
        clause=clause,
    )


# ---------------------------------------------------------------- chunking


def test_exact_chunk_size_and_overlap():
    pages = [PageText(source="satcom-ngp.pdf", page=1, text="x" * 3000)]
    chunks = make_chunks(pages, chunk_size=1000, overlap=200, step=800)
    assert [c.start for c in chunks] == [0, 800, 1600, 2400]
    assert [c.end for c in chunks] == [1000, 1800, 2600, 3000]
    assert [len(c.text) for c in chunks] == [1000, 1000, 1000, 600]
    # overlap between consecutive windows is 200
    assert chunks[0].end - chunks[1].start == 200
    assert chunks[1].end - chunks[2].start == 200
    assert chunks[2].end - chunks[3].start == 200
    # step between window starts is 800
    assert chunks[1].start - chunks[0].start == 800
    assert chunks[2].start - chunks[1].start == 800


def test_short_page_and_final_partial_chunk():
    pages = [PageText(source="satcom-ngp.pdf", page=2, text="y" * 2500)]
    chunks = make_chunks(pages, chunk_size=1000, overlap=200, step=800)
    assert len(chunks) == 3
    assert chunks[-1].end == 2500
    assert len(chunks[-1].text) < 1000
    assert chunks[-1].text == "y" * 900


def test_page_boundary_isolation():
    pages = [
        PageText(source="satcom-ngp.pdf", page=3, text="A" * 1000),
        PageText(source="satcom-ngp.pdf", page=4, text="B" * 1000),
    ]
    chunks = make_chunks(pages, chunk_size=1000, overlap=200, step=800)
    assert len(chunks) == 2
    assert {c.page for c in chunks} == {3, 4}
    for c in chunks:
        expected = ("A" * 1000) if c.page == 3 else ("B" * 1000)
        assert c.text == expected


def test_character_offsets():
    pages = [PageText(source="satcom-ngp.pdf", page=5, text="z" * 1500)]
    chunks = make_chunks(pages, chunk_size=1000, overlap=200, step=800)
    for c in chunks:
        assert c.text == ("z" * 1500)[c.start : c.end]
    assert chunks[0].end - chunks[0].start == len(chunks[0].text)
    assert chunks[-1].end == len("z" * 1500)


# ------------------------------------------------------- ground truth mapping


def test_ground_truth_clause_mapping_overlap():
    pages = [PageText(source="satcom-ngp.pdf", page=3, text="2.4       CLASSIFICATION OF USE \n\n(a) Telecommunications\n")]
    spans = build_clause_spans(pages)
    assert len(spans) >= 1
    cl = spans[0]
    chunk = _chunk("satcom-ngp_page_3_chunk_0", "satcom-ngp.pdf", 3, 0, 20, "2.4       CL")
    assert chunk_overlaps(chunk, cl) is True

    chunk_no = _chunk("satcom-ngp_page_3_chunk_1", "satcom-ngp.pdf", 3, 500, 520, "ZZZZZZZZZZZ")
    assert chunk_overlaps(chunk_no, cl) is False


# ------------------------------------------------------------------- metrics


def test_metrics_no_relevant_results():
    retrieved = [_chunk("c1", "satcom-ngp.pdf", 1, 0, 10, "hello world")]
    relevant = [
        {
            "clause_id": "2.4",
            "source": "satcom-ngp.pdf",
            "page": 3,
            "start": 0,
            "end": 100,
            "kind": "section",
        }
    ]
    m = compute_metrics(retrieved, relevant, k=5)
    assert m["recall@5"] == 0.0
    assert m["precision@5"] == 0.0
    assert m["mrr@5"] == 0.0
    assert m["relevant_flags"] == [False]


def test_metrics_precision_and_mrr():
    retrieved = [
        _chunk("c1", "satcom-ngp.pdf", 3, 0, 20, "2.4       CLASSIFICATION"),
        _chunk("c2", "satcom-ngp.pdf", 3, 100, 120, "ZZZZZZZZZZZZZZZZZZZZ"),
        _chunk("c3", "satcom-ngp.pdf", 3, 200, 220, "ZZZZZZZZZZZZZZZZZZZZ"),
        _chunk("c4", "satcom-ngp.pdf", 3, 300, 320, "ZZZZZZZZZZZZZZZZZZZZ"),
        _chunk("c5", "satcom-ngp.pdf", 3, 400, 420, "ZZZZZZZZZZZZZZZZZZZZ"),
    ]
    relevant = [
        {
            "clause_id": "2.4",
            "source": "satcom-ngp.pdf",
            "page": 3,
            "start": 0,
            "end": 40,
            "kind": "section",
        }
    ]
    m = compute_metrics(retrieved, relevant, k=5)
    # first relevant chunk is rank 1 -> MRR = 1/1
    assert m["mrr@5"] == 1.0
    # one relevant chunk out of top-5 -> precision = 1/5
    assert m["precision@5"] == 0.2
    # one relevant clause covered -> recall = 1.0
    assert m["recall@5"] == 1.0


def test_metrics_multiple_relevant_clauses():
    retrieved = [
        _chunk("c1", "satcom-ngp.pdf", 3, 0, 20, "2.4       CLASSIFICATION"),
        _chunk("c2", "satcom-ngp.pdf", 3, 100, 120, "ZZZZZZZZZZZZZZZZZZZZ"),
    ]
    relevant = [
        {
            "clause_id": "2.4",
            "source": "satcom-ngp.pdf",
            "page": 3,
            "start": 0,
            "end": 40,
            "kind": "section",
        },
        {
            "clause_id": "2.5",
            "source": "satcom-ngp.pdf",
            "page": 3,
            "start": 200,
            "end": 240,
            "kind": "section",
        },
    ]
    m = compute_metrics(retrieved, relevant, k=5)
    # only clause 2.4 covered -> recall = 1/2
    assert m["recall@5"] == 0.5
    assert m["relevant_flags"][0] is True
    assert m["relevant_flags"][1] is False
    # distinct covered clauses recorded
    assert m["covered_clause_ids"] == ["2.4"]


# --------------------------------------------------------------- dataset load


def test_load_questions_schema():
    from experiments.exp001 import QUESTIONS_CSV, load_questions

    qs = load_questions(QUESTIONS_CSV)
    assert len(qs) == 15
    for q in qs:
        assert set(q) == {
            "question_id",
            "question",
            "expected_answer",
            "relevant_clause_ids",
            "relevant_source_pages",
        }
        assert q["question_id"].startswith("EXP001-Q")


# ------------------------------------------------- ground truth corrections (EXP-001 v3)


def test_q09_expected_answer_includes_both_scc_locations():
    qs = load_questions(QUESTIONS_CSV)
    q = next(q for q in qs if q["question_id"] == "EXP001-Q09")
    answer = q["expected_answer"].lower()
    # Both permitted SCC locations must be present in the expected answer.
    assert "within the territory of india" in answer
    assert "authorised by caiss" in answer
    assert "technical requirements" in answer


def test_q10_expected_answer_covers_complete_procedure():
    qs = load_questions(QUESTIONS_CSV)
    q = next(q for q in qs if q["question_id"] == "EXP001-Q10")
    answer = q["expected_answer"].lower()
    # The full Article 3.6.8 procedure must be captured.
    for token in (
        "first come first served",
        "government",
        "discussion",
        "common api",
        "ap s4",
        "depending on progress",
    ):
        assert token in answer, f"Expected answer for EXP001-Q10 is missing: {token}"


def test_q13_evidence_units_cover_category_headings():
    entries = generate_ground_truth_entries()
    q13 = sorted(
        [e for e in entries if e["question_id"] == "EXP001-Q13"],
        key=lambda e: e["evidence_id"],
    )
    by_clause = {e["clause_id"]: e for e in q13}
    assert set(by_clause) == {"4.3.1", "4.3.2", "4.3.3", "4.3.4"}
    # 4.3.2 evidence must start at the category heading (Public Switched
    # Telephone Network (domestic)) and therefore include the heading.
    e2 = by_clause["4.3.2"]
    assert e2["page"] == 13
    assert e2["exact_supporting_excerpt"].lstrip().startswith("4.3.2 Public Switched Telephone Network")
    # 4.3.3 heading ends page 13; the explanatory paragraph starts page 14,
    # so the heading-anchored evidence unit is used.
    e3 = by_clause["4.3.3"]
    assert e3["page"] == 13
    assert e3["exact_supporting_excerpt"].lstrip().startswith("4.3.3 Value Added Services")


def test_approved_evidence_units_all_validate():
    pages = load_pages()
    units = load_ground_truth()
    valid, problems = validate_ground_truth(units, pages)
    assert problems == []
    assert len(units) == 20
    for i in range(1, 16):
        qid = f"EXP001-Q{i:02d}"
        assert qid in valid, f"Missing valid evidence for {qid}"
        assert valid[qid], f"No valid evidence for {qid}"


# ------------------------------------------------- ground truth v2 (EXP-001)


def test_evidence_offset_must_match_exact_excerpt():
    text = "The norms, guidelines and procedures"
    pages = [PageText(source="satcom-ngp.pdf", page=1, text=text)]

    good = EvidenceUnit(
        question_id="EXP001-Q01",
        evidence_id="EXP001-Q01-E01",
        source="satcom-ngp.pdf",
        page=1,
        clause_id="ARTICLE-1",
        start_char=0,
        end_char=25,
        exact_supporting_excerpt="The norms, guidelines and",
    )
    valid, problems = validate_ground_truth([good], pages)
    assert problems == []
    assert "EXP001-Q01" in valid
    assert valid["EXP001-Q01"][0]["status"] == "valid"
    assert validate_evidence_offset(text, 0, 25, "The norms, guidelines and") is True
    assert validate_evidence_offset(text, 0, 25, "The norms, guideline and") is False
    assert validate_evidence_offset(text, 0, 999, "The norms, guidelines and") is False


def test_missing_or_invalid_evidence_mapping_raises():
    with pytest.raises(RuntimeError):
        get_relevant_evidence("EXP001-Q99", {})

    pages = [PageText(source="satcom-ngp.pdf", page=1, text="The norms, guidelines and procedures")]
    bad = EvidenceUnit(
        question_id="EXP001-Q01",
        evidence_id="EXP001-Q01-E01",
        source="satcom-ngp.pdf",
        page=1,
        clause_id="ARTICLE-1",
        start_char=0,
        end_char=25,
        exact_supporting_excerpt="The norms, guidelines and",
    )
    valid, problems = validate_ground_truth([bad], pages)
    assert not problems
    assert get_relevant_evidence("EXP001-Q01", valid)[0]["evidence_id"] == "EXP001-Q01-E01"

    invalid = EvidenceUnit(
        question_id="EXP001-Q01",
        evidence_id="EXP001-Q01-E01",
        source="satcom-ngp.pdf",
        page=1,
        clause_id="ARTICLE-1",
        start_char=0,
        end_char=25,
        exact_supporting_excerpt="not the right excerpt",
    )
    valid_invalid, problems_invalid = validate_ground_truth([invalid], pages)
    assert problems_invalid
    with pytest.raises(RuntimeError):
        get_relevant_evidence("EXP001-Q01", valid_invalid)


def test_multiple_evidence_units_for_one_question():
    retrieved = [_chunk("c1", "satcom-ngp.pdf", 13, 0, 60, "Broadcasting (moving images (TV) and associated sound")]
    relevant = [
        {
            "evidence_id": "EXP001-Q13-E01",
            "clause_id": "4.3.1",
            "source": "satcom-ngp.pdf",
            "page": 13,
            "start": 0,
            "end": 100,
            "kind": "section",
        },
        {
            "evidence_id": "EXP001-Q13-E02",
            "clause_id": "4.3.2",
            "source": "satcom-ngp.pdf",
            "page": 13,
            "start": 200,
            "end": 300,
            "kind": "section",
        },
    ]
    m = compute_metrics(retrieved, relevant, k=5)
    # Only one of the two relevant evidence units was retrieved -> recall = 1/2
    assert m["recall@5"] == 0.5
    assert m["covered_clause_ids"] == ["EXP001-Q13-E01"]
    assert m["relevant_clause_ids"] == ["EXP001-Q13-E01", "EXP001-Q13-E02"]


def test_cross_page_evidence_requires_same_page():
    relevant = [
        {"evidence_id": "EXP001-Q08-E01", "clause_id": "3.4.1", "source": "satcom-ngp.pdf", "page": 5, "start": 100, "end": 200, "kind": "section"},
        {"evidence_id": "EXP001-Q08-E02", "clause_id": "3.4.2", "source": "satcom-ngp.pdf", "page": 6, "start": 100, "end": 200, "kind": "section"},
    ]
    chunk_page5 = _chunk("c1", "satcom-ngp.pdf", 5, 100, 200, "authorisation text")
    chunk_page6 = _chunk("c2", "satcom-ngp.pdf", 6, 100, 200, "authorisation text")
    chunk_other_page = _chunk("c3", "satcom-ngp.pdf", 7, 100, 200, "authorisation text")

    assert compute_metrics([chunk_page5], relevant, k=5)["recall@5"] == 0.5
    assert compute_metrics([chunk_page6], relevant, k=5)["recall@5"] == 0.5
    assert compute_metrics([chunk_other_page], relevant, k=5)["recall@5"] == 0.0


def test_heading_only_chunk_not_relevant_when_answer_in_body():
    heading_chunk = _chunk("c1", "satcom-ngp.pdf", 5, 0, 41, "3.2 DEFINITION OF INDIAN SATELLITES")
    relevant = [
        {
            "evidence_id": "EXP001-Q07-E01",
            "clause_id": "3.2",
            "source": "satcom-ngp.pdf",
            "page": 5,
            "start": 1252,
            "end": 1699,
            "kind": "section",
        }
    ]
    m = compute_metrics([heading_chunk], relevant, k=5)
    assert m["relevant_flags"] == [False]
    assert m["recall@5"] == 0.0
    assert m["precision@5"] == 0.0
    assert m["mrr@5"] == 0.0

    body_chunk = _chunk("c2", "satcom-ngp.pdf", 5, 1252, 1300, 'Indian Satellites” shall be defined as  the satellites')
    m_body = compute_metrics([body_chunk], relevant, k=5)
    assert m_body["relevant_flags"] == [True]
    assert m_body["mrr@5"] == 1.0


def test_recall5_for_multiple_relevant_evidence_units():
    relevant = [
        {"evidence_id": "EXP001-Q13-E01", "clause_id": "4.3.1", "source": "satcom-ngp.pdf", "page": 13, "start": 0, "end": 100, "kind": "section"},
        {"evidence_id": "EXP001-Q13-E02", "clause_id": "4.3.2", "source": "satcom-ngp.pdf", "page": 13, "start": 200, "end": 300, "kind": "section"},
    ]
    # Both units covered -> recall@5 == 1.0
    both = [
        _chunk("c1", "satcom-ngp.pdf", 13, 0, 60, "broadcasting text"),
        _chunk("c2", "satcom-ngp.pdf", 13, 200, 260, "sub-divisions text"),
        _chunk("c3", "satcom-ngp.pdf", 13, 400, 460, "irrelevant"),
    ]
    m = compute_metrics(both, relevant, k=5)
    assert m["recall@5"] == 1.0
    assert m["covered_clause_ids"] == ["EXP001-Q13-E01", "EXP001-Q13-E02"]

    # Only one unit covered -> recall@5 == 0.5
    one = [_chunk("c1", "satcom-ngp.pdf", 13, 0, 60, "broadcasting text")]
    m_one = compute_metrics(one, relevant, k=5)
    assert m_one["recall@5"] == 0.5


def test_precision5_denominator_is_k():
    relevant = [
        {"evidence_id": "EXP001-Q13-E01", "clause_id": "4.3.1", "source": "satcom-ngp.pdf", "page": 13, "start": 0, "end": 300, "kind": "section"},
    ]
    retrieved = [
        _chunk("c1", "satcom-ngp.pdf", 13, 0, 60, "broadcasting text"),
        _chunk("c2", "satcom-ngp.pdf", 13, 200, 260, "sub-divisions text"),
        _chunk("c3", "satcom-ngp.pdf", 13, 400, 460, "irrelevant"),
    ]
    m = compute_metrics(retrieved, relevant, k=5)
    # 2 relevant chunks retrieved out of top-5 -> precision = 2/5, denominator is k=5
    assert m["precision@5"] == 0.4
    assert m["retrieved_count"] == 3


def test_mrr5_when_no_relevant_chunk_retrieved():
    relevant = [
        {"evidence_id": "EXP001-Q13-E01", "clause_id": "4.3.1", "source": "satcom-ngp.pdf", "page": 13, "start": 0, "end": 100, "kind": "section"},
    ]
    retrieved = [_chunk(f"c{i}", "satcom-ngp.pdf", 13, 400 + i * 50, 460 + i * 50, "irrelevant") for i in range(5)]
    m = compute_metrics(retrieved, relevant, k=5)
    assert m["mrr@5"] == 0.0
    assert m["relevant_flags"] == [False] * 5


def test_evidence_overlap_char_counts():
    retrieved = [_chunk("c1", "satcom-ngp.pdf", 1, 100, 200, "some text here")]
    relevant = [
        {"evidence_id": "EXP001-Q01-E01", "clause_id": "ARTICLE-1", "source": "satcom-ngp.pdf", "page": 1, "start": 150, "end": 300, "kind": "article"},
    ]
    m = compute_metrics(retrieved, relevant, k=5)
    # overlap = min(200, 300) - max(100, 150) = 50
    assert m["overlap_chars"] == [50]
    assert m["evidence_overlaps"] == [
        {"rank": 1, "chunk_id": "c1", "evidence_ids": ["EXP001-Q01-E01"], "overlap_chars": 50}
    ]

    no_overlap = [_chunk("c2", "satcom-ngp.pdf", 1, 0, 50, "some text here")]
    m_no = compute_metrics(no_overlap, relevant, k=5)
    assert m_no["overlap_chars"] == [0]
    assert m_no["relevant_flags"] == [False]
    assert m_no["evidence_overlaps"] == []
