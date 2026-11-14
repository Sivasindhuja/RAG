"""Tests for constrained generation over selected evidence."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag_core import (  # noqa: E402
    GENERATION_ERROR,
    MODEL_ABSTAINED,
    AnswerStatement,
    Citation,
    assemble_generated_answer,
    build_selected_evidence_prompt,
    parse_generated_claims,
)


def _statement(text: str, marker: str, chunk_id: str, start: int = 0) -> AnswerStatement:
    return AnswerStatement(
        text=text,
        marker=marker,
        chunk_id=chunk_id,
        start_offset=start,
        end_offset=start + len(text),
    )


def _citation(marker: str, chunk_id: str, start: int = 0, end: int = 20) -> Citation:
    return Citation(
        marker=marker,
        chunk_id=chunk_id,
        source="satcom-ngp.pdf",
        article="ARTICLE-1",
        section="1.1",
        heading=None,
        start_page=1,
        end_page=1,
        start_offset=start,
        end_offset=end,
        quote="quote",
        distance=0.1,
    )


def test_selected_evidence_prompt_contains_numbered_evidence_and_question():
    statements = [_statement("4.3.2 Public Switched Telephone Network.", "[1]", "c1")]
    prompt = build_selected_evidence_prompt("What are the categories?", statements)

    assert "[1] 4.3.2 Public Switched Telephone Network." in prompt
    assert "Question: What are the categories?" in prompt
    assert "1 to 1" in prompt


def test_assemble_generated_answer_maps_sources_to_original_markers():
    statements = [
        _statement("4.3.4 Land Mobile Satellite Services.", "[1]", "c1"),
        _statement("4.3.2 Public Switched Telephone Network.", "[2]", "c2"),
    ]
    citations = [_citation("[1]", "c1"), _citation("[2]", "c2")]
    raw = (
        '{"sentences": ['
        '{"text": "Land mobile services are domestic.", "sources": [1]},'
        '{"text": "PSTN is domestic.", "sources": [2, 1]}'
        "]}"
    )

    answer = assemble_generated_answer(
        "What are the categories?",
        statements,
        citations,
        raw,
        model="test-model",
        latency_ms=12.5,
    )

    assert answer.status == "answered"
    assert answer.text == (
        "Land mobile services are domestic. [1] PSTN is domestic. [2] [1]"
    )
    assert [claim.markers for claim in answer.claims] == [("[1]",), ("[2]", "[1]")]
    assert [citation.marker for citation in answer.citations] == ["[1]", "[2]"]
    assert answer.errors == []


def test_invalid_source_ids_are_recorded_and_not_cited():
    statements = [_statement("Only evidence.", "[1]", "c1")]
    raw = '{"sentences": [{"text": "Supported claim.", "sources": [1, 9, 0]}]}'

    answer = assemble_generated_answer(
        "question",
        statements,
        [_citation("[1]", "c1")],
        raw,
        model="test-model",
        latency_ms=1.0,
    )

    assert answer.status == "answered"
    assert answer.claims[0].source_ids == (1,)
    assert answer.claims[0].invalid_source_ids == (9, 0)
    assert any("invalid evidence numbers" in error for error in answer.errors)
    assert [citation.marker for citation in answer.citations] == ["[1]"]


def test_empty_sentence_list_is_model_abstention():
    statements = [_statement("Only evidence.", "[1]", "c1")]
    answer = assemble_generated_answer(
        "question",
        statements,
        [_citation("[1]", "c1")],
        '{"sentences": []}',
        model="test-model",
        latency_ms=1.0,
    )

    assert answer.status == MODEL_ABSTAINED
    assert answer.text == "INSUFFICIENT_EVIDENCE"
    assert answer.citations == []


def test_malformed_json_is_generation_error():
    statements = [_statement("Only evidence.", "[1]", "c1")]
    answer = assemble_generated_answer(
        "question",
        statements,
        [_citation("[1]", "c1")],
        "not json",
        model="test-model",
        latency_ms=1.0,
    )

    assert answer.status == GENERATION_ERROR
    assert answer.text == ""
    assert "invalid JSON" in answer.errors[0]


def test_duplicate_json_keys_are_reported():
    statements = [_statement("Only evidence.", "[1]", "c1")]
    raw = (
        '{"sentences": [{"text": "A claim.", "sources": [1]}],'
        '"sentences": []}'
    )
    claims, errors = parse_generated_claims(raw, 1)

    assert claims == []
    assert any("duplicate JSON keys" in error for error in errors)


def test_parse_rejects_missing_sentences_field():
    with pytest.raises(ValueError):
        parse_generated_claims('{"answer": "no claims"}', 1)
