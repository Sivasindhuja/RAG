"""Tests for constrained generation over selected evidence (EXP-004, M1 Gemini)."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag_core import (  # noqa: E402
    GEMINI_PROMPT_VERSION,
    GENERATION_ERROR,
    INSUFFICIENT_EVIDENCE,
    MODEL_ABSTAINED,
    AnswerStatement,
    Citation,
    GeminiGenerator,
    assemble_generated_answer,
    build_gemini_evidence_prompt,
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


def _citation(
    marker: str, chunk_id: str, start: int = 0, end: int = 20, section: str = "1.1", page: int = 1
) -> Citation:
    return Citation(
        marker=marker,
        chunk_id=chunk_id,
        source="satcom-ngp.pdf",
        article="ARTICLE-1",
        section=section,
        heading=section,
        start_page=page,
        end_page=page,
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


def test_gemini_prompt_is_versioned_evidence_only_and_excludes_source_metadata():
    statements = [_statement("Alpha fact.", "[1]", "c1"), _statement("Beta fact.", "[2]", "c2")]
    prompt = build_gemini_evidence_prompt("What is allowed?", statements)
    assert f"Prompt version: {GEMINI_PROMPT_VERSION}" in prompt
    assert "[1] Alpha fact." in prompt
    assert "[2] Beta fact." in prompt
    assert "c1" not in prompt
    assert "satcom-ngp.pdf" not in prompt
    assert "ONLY the evidence items below" in prompt
    assert "Do not use outside knowledge" in prompt
    assert "Use only evidence IDs from 1 to 2" in prompt
    assert "Do not include page numbers, section numbers" in prompt
    assert '{"sentences": []}' in prompt


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
    assert answer.text == INSUFFICIENT_EVIDENCE
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


def test_gemini_generator_abstains_when_evidence_insufficient_without_calling_api():
    statements = [_statement("Alpha fact.", "[1]", "c1")]
    citations = [_citation("[1]", "c1")]
    gen = GeminiGenerator(api_key="test-key")
    with patch("google.generativeai.GenerativeModel") as model_cls:
        answer = gen.generate(
            "q",
            statements,
            citations,
            evidence_sufficient=False,
            insufficiency_reason="selected_evidence_insufficient",
        )
    assert answer.status == MODEL_ABSTAINED
    assert answer.text == INSUFFICIENT_EVIDENCE
    assert answer.citations == []
    assert answer.metadata.get("reason") == "selected_evidence_insufficient"
    assert answer.metadata.get("prompt_version") == GEMINI_PROMPT_VERSION
    model_cls.assert_not_called()


def test_gemini_generator_abstains_when_no_statements():
    gen = GeminiGenerator(api_key="test-key")
    with patch("google.generativeai.GenerativeModel") as model_cls:
        answer = gen.generate("q", [], [])
    assert answer.status == MODEL_ABSTAINED
    assert answer.text == INSUFFICIENT_EVIDENCE
    model_cls.assert_not_called()


def test_gemini_generator_resolves_citations_from_app_metadata_not_model_text():
    statements = [_statement("Alpha fact.", "[1]", "c1"), _statement("Beta fact.", "[2]", "c2")]
    citations = [
        _citation("[1]", "c1", section="3.1", page=5),
        _citation("[2]", "c2", section="9.9", page=99),
    ]
    fake_response = MagicMock()
    # Model text must not be trusted for metadata; parser uses sources IDs only.
    fake_response.text = '{"sentences":[{"text":"Answer claim (see page 999).","sources":[1]}]}'
    fake_usage = MagicMock(prompt_token_count=10, candidates_token_count=5, total_token_count=15)
    fake_response.usage_metadata = fake_usage
    fake_model = MagicMock()
    fake_model.generate_content.return_value = fake_response

    gen = GeminiGenerator(model="gemini-test", api_key="test-key")
    with patch("google.generativeai.configure") as cfg, patch(
        "google.generativeai.GenerativeModel", return_value=fake_model
    ) as model_cls:
        answer = gen.generate("q", statements, citations)

    cfg.assert_called_once_with(api_key="test-key")
    model_cls.assert_called_once()
    args, kwargs = model_cls.call_args
    assert args[0] == "models/gemini-test"
    assert kwargs["generation_config"]["response_mime_type"] == "application/json"
    assert kwargs["generation_config"]["temperature"] == 0.0
    assert answer.status == "answered"
    assert len(answer.citations) == 1
    assert answer.citations[0].chunk_id == "c1"
    assert answer.citations[0].section == "3.1"
    assert answer.citations[0].start_page == 5
    assert "[1]" in answer.text
    assert answer.metadata.get("prompt_token_count") == 10
    assert answer.metadata.get("prompt_version") == GEMINI_PROMPT_VERSION


def test_gemini_generator_records_model_abstention_json():
    statements = [_statement("Alpha fact.", "[1]", "c1")]
    citations = [_citation("[1]", "c1")]
    fake_response = MagicMock()
    fake_response.text = '{"sentences":[]}'
    fake_response.usage_metadata = None
    fake_model = MagicMock()
    fake_model.generate_content.return_value = fake_response
    gen = GeminiGenerator(api_key="test-key")
    with patch("google.generativeai.configure"), patch(
        "google.generativeai.GenerativeModel", return_value=fake_model
    ):
        answer = gen.generate("q", statements, citations)
    assert answer.status == MODEL_ABSTAINED
    assert answer.text == INSUFFICIENT_EVIDENCE
    assert answer.citations == []


def test_gemini_generator_records_api_error_as_generation_error():
    statements = [_statement("Alpha fact.", "[1]", "c1")]
    citations = [_citation("[1]", "c1")]
    gen = GeminiGenerator(api_key="test-key")
    with patch("google.generativeai.configure"), patch(
        "google.generativeai.GenerativeModel", side_effect=RuntimeError("boom")
    ):
        answer = gen.generate("q", statements, citations)
    assert answer.status == GENERATION_ERROR
    assert any("boom" in e for e in answer.errors)
    assert answer.citations == []
