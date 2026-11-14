"""Tests for grounded answer assembly, citations and abstention."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag_core import (  # noqa: E402
    ABSTAINED,
    ANSWERED,
    AbstentionPolicy,
    Citation,
    GroundedPipeline,
    RetrievedChunk,
    content_tokens,
    token_support,
)


class _FakeIndex:
    def __init__(self, hits):
        self._hits = hits

    def search(self, query, k=5):
        return list(self._hits[:k])


def _hit(
    chunk_id: str,
    text: str,
    distance=0.4,
    rank=1,
    section="1.1",
    start_offset=0,
):
    return RetrievedChunk(
        rank=rank,
        chunk_id=chunk_id,
        text=text,
        source="satcom-ngp.pdf",
        start_offset=start_offset,
        end_offset=start_offset + len(text),
        start_page=1,
        end_page=1,
        article="ARTICLE-1",
        section=section,
        heading="heading",
        distance=distance,
        metadata={},
    )


def test_pipeline_answers_with_verbatim_sentences_and_citations():
    text = "The Wireless Planning and Coordination Wing issues the license. Other text follows."
    pipeline = GroundedPipeline(_FakeIndex([_hit("c1", text)]), top_k=5)

    answer = pipeline.answer("Who issues the license?")

    assert answer.status == ANSWERED
    assert not answer.abstained
    assert len(answer.citations) == 1
    citation = answer.citations[0]
    assert citation.chunk_id == "c1"
    assert citation.marker == "[1]"
    assert citation.start_offset == 0
    assert citation.end_offset == len(text)
    assert "[1]" in answer.text
    # The selected sentence is verbatim source text.
    assert "The Wireless Planning and Coordination Wing issues the license." in answer.text
    assert "The Wireless Planning and Coordination Wing issues the license." in text

    # Sentence-level provenance points at the exact source span.
    assert answer.statements
    statement = answer.statements[0]
    assert text[statement.start_offset : statement.end_offset] == statement.text
    assert statement.chunk_id == "c1"


def test_statement_offsets_are_global_for_later_ranked_chunks():
    first = _hit("c1", "Other material without the query.", distance=0.3, rank=1, start_offset=0)
    second = _hit(
        "c2",
        "License authority grants approval. More text.",
        distance=0.5,
        rank=2,
        start_offset=1000,
    )
    pipeline = GroundedPipeline(_FakeIndex([first, second]), max_answer_chunks=2)
    answer = pipeline.answer("license authority")

    assert answer.status == ANSWERED
    statement = next(item for item in answer.statements if item.chunk_id == "c2")
    assert second.text[statement.start_offset - second.start_offset : statement.end_offset - second.start_offset] == statement.text
    assert statement.start_offset == 1000


def test_pipeline_abstains_on_empty_question():
    answer = GroundedPipeline(_FakeIndex([_hit("c1", "text")])).answer("   ")
    assert answer.status == ABSTAINED
    assert answer.citations == []
    assert answer.reason == "empty question"


def test_pipeline_abstains_when_no_hits():
    answer = GroundedPipeline(_FakeIndex([])).answer("Who issues the license?")
    assert answer.status == ABSTAINED
    assert answer.reason == "no retrieval results"


def test_pipeline_abstains_when_hard_distance_exceeded():
    policy = AbstentionPolicy(hard_max_distance=1.30)
    answer = GroundedPipeline(_FakeIndex([_hit("c1", "license authority", distance=1.9)]), policy=policy).answer(
        "Who issues the license?"
    )
    assert answer.status == ABSTAINED
    assert "hard limit" in answer.reason


def test_pipeline_abstains_on_low_lexical_support():
    policy = AbstentionPolicy(min_token_support=0.5)
    answer = GroundedPipeline(
        _FakeIndex([_hit("c1", "satellite licensing procedure", distance=0.5)]),
        policy=policy,
    ).answer("What is the capital of France?")
    assert answer.status == ABSTAINED
    assert "lexical support" in answer.reason


def test_pipeline_soft_distance_requires_strict_support():
    policy = AbstentionPolicy(
        hard_max_distance=1.30,
        soft_max_distance=1.10,
        min_token_support=0.5,
        strict_min_token_support=0.7,
    )
    weak = _hit("c1", "alpha bravo unrelated", distance=1.2)
    answer = GroundedPipeline(_FakeIndex([weak]), policy=policy).answer("alpha bravo charlie delta")
    assert answer.status == ABSTAINED
    assert "needs support" in answer.reason

    strong = _hit("c1", "alpha bravo charlie unrelated", distance=1.2)
    answer = GroundedPipeline(_FakeIndex([strong]), policy=policy).answer("alpha bravo charlie delta")
    assert answer.status == ANSWERED


def test_pipeline_citation_quote_is_source_substring_when_truncated():
    text = "alpha bravo charlie delta epsilon zeta eta theta"
    pipeline = GroundedPipeline(_FakeIndex([_hit("c1", text)]), max_quote_chars=20)
    answer = pipeline.answer("alpha bravo charlie")

    assert answer.status == ANSWERED
    citation = answer.citations[0]
    assert text.startswith(citation.quote.replace(" …", ""))
    assert len(citation.quote) <= 22


def test_pipeline_uses_generator_hook_and_still_cites():
    def generator(question, hits, citations):
        return f"Generated answer [{citations[0].marker}]"

    pipeline = GroundedPipeline(_FakeIndex([_hit("c1", "license authority text")]), generator=generator)
    answer = pipeline.answer("license authority")

    assert answer.status == ANSWERED
    assert answer.text == "Generated answer [[1]]"
    assert answer.citations[0].chunk_id == "c1"


def test_pipeline_limits_and_dedupes_selected_sentences():
    text = "Alpha beta sentence. Alpha beta sentence. Alpha beta third."
    pipeline = GroundedPipeline(
        _FakeIndex([_hit("c1", text)]),
        max_answer_sentences=2,
        max_answer_chunks=1,
    )
    answer = pipeline.answer("alpha beta")
    assert answer.status == ANSWERED
    assert answer.text.count("[1]") == 2
    assert "Alpha beta sentence." in answer.text


def test_citations_omit_retrieved_chunks_that_do_not_contribute():
    irrelevant = _hit("c1", "Unrelated background material.", distance=0.2, rank=1)
    relevant = _hit("c2", "The licensing authority grants the license.", distance=0.5, rank=2)
    pipeline = GroundedPipeline(_FakeIndex([irrelevant, relevant]), max_answer_chunks=2)
    answer = pipeline.answer("who grants the license?")

    assert answer.status == ANSWERED
    assert [citation.chunk_id for citation in answer.citations] == ["c2"]
    assert answer.citations[0].marker == "[1]"
    assert {statement.chunk_id for statement in answer.statements} == {"c2"}
    assert "[2]" not in answer.text


def test_sibling_clause_selection_promotes_same_parent_clauses():
    text = (
        "4.3.4 Land mobile domestic communication services are described. "
        "4.3.3 Domestic communication services include value added services. "
        "4.3.2 Domestic communication services include telephone services. "
        "Land mobile domestic communication services are important."
    )
    baseline = GroundedPipeline(
        _FakeIndex([_hit("c1", text)]),
        max_answer_sentences=2,
        max_answer_chunks=1,
        sibling_score_ratio=None,
    ).answer("land mobile domestic communication services")
    candidate = GroundedPipeline(
        _FakeIndex([_hit("c1", text)]),
        max_answer_sentences=2,
        max_answer_chunks=1,
        sibling_score_ratio=0.5,
    ).answer("land mobile domestic communication services")

    baseline_text = " ".join(statement.text for statement in baseline.statements)
    candidate_text = " ".join(statement.text for statement in candidate.statements)
    assert "4.3.4" in baseline_text and "important" in baseline_text
    assert "4.3.4" in candidate_text and "4.3.3" in candidate_text
    assert "important" not in candidate_text


def test_sibling_clause_selection_ignores_weak_siblings_below_threshold():
    text = (
        "4.3.4 Land mobile domestic communication services are described. "
        "4.3.3 Domestic. "
        "Land mobile domestic communication services are important."
    )
    answer = GroundedPipeline(
        _FakeIndex([_hit("c1", text)]),
        max_answer_sentences=2,
        max_answer_chunks=1,
        sibling_score_ratio=0.5,
    ).answer("land mobile domestic communication services")

    selected = " ".join(statement.text for statement in answer.statements)
    assert "4.3.4" in selected and "important" in selected
    assert "4.3.3" not in selected


def test_answer_from_hits_reuses_supplied_retrieval_without_searching():
    class _NoSearchIndex:
        def search(self, query, k=5):
            raise AssertionError("answer_from_hits must not retrieve again")

    pipeline = GroundedPipeline(_NoSearchIndex())
    hit = _hit("c1", "The licensing authority grants the license.")
    answer = pipeline.answer_from_hits("who grants the license?", [hit])

    assert answer.status == ANSWERED
    assert answer.retrieved == [hit]
    assert answer.citations[0].chunk_id == "c1"


def test_sibling_score_ratio_is_validated():
    with pytest.raises(ValueError):
        GroundedPipeline(_FakeIndex([]), sibling_score_ratio=1.1)


def test_all_caps_heading_is_deprioritised_below_body_text():
    text = (
        "4.2 INTERNATIONAL COMMUNICATION SERVICES\n\n"
        "International communication services use inter-governmental systems."
    )
    pipeline = GroundedPipeline(
        _FakeIndex([_hit("c1", text)]),
        max_answer_sentences=1,
        max_answer_chunks=1,
    )
    answer = pipeline.answer("international communication services")

    assert answer.status == ANSWERED
    assert answer.statements
    assert answer.statements[0].text.startswith("International communication services use")


def test_pipeline_extractive_fallback_when_no_sentence_overlaps():
    policy = AbstentionPolicy(hard_max_distance=None, soft_max_distance=None, min_token_support=0.0)
    text = "Completely unrelated content without the query words."
    answer = GroundedPipeline(_FakeIndex([_hit("c1", text, distance=0.2)]), policy=policy).answer(
        "zzzz yyyy"
    )
    assert answer.status == ANSWERED
    assert text[:50] in answer.text
    assert "[1]" in answer.text


def test_abstention_policy_validation():
    with pytest.raises(ValueError):
        AbstentionPolicy(min_token_support=-0.1)
    with pytest.raises(ValueError):
        AbstentionPolicy(min_token_support=0.9, strict_min_token_support=0.5)


def test_content_tokens_and_token_support():
    assert content_tokens("The quick brown fox and a cat") == ["quick", "brown", "fox", "cat"]
    hits = [_hit("c1", "satellite licensing authority")]
    assert token_support("satellite licensing rules", hits) == pytest.approx(2 / 3)
    assert token_support("xyzabc", hits) == 0.0
