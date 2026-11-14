"""Evidence-grounded answer assembly with citations and abstention.

This module is the production-facing layer above the shared dense index.  It does
**not** invent answers:

- it retrieves evidence with :class:`~src.rag_core.index.DenseIndex`;
- it answers extractively by selecting verbatim sentences from retrieved chunks;
- every selected sentence carries a citation marker linked to exact source
  offsets and page numbers;
- it abstains when the retrieval/confidence policy says the document does not
  support an answer.

An optional ``generator`` hook can synthesise prose from the retrieved chunks, but
its faithfulness is not validated here.  The default extractive path is the only
one covered by the test suite.

Abstention thresholds are model- and index-specific heuristics.  They are
configurable and must be recalibrated whenever the embedding model or chunking
changes.  See ``docs/PRODUCTION_SLICE.md`` for the calibration set and limits.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from .index import DenseIndex, RetrievedChunk

ANSWERED = "answered"
ABSTAINED = "abstained"

ABSTENTION_MESSAGE = (
    "I cannot answer this from the SATCOM NGP document: no sufficiently "
    "supporting evidence was retrieved. Please rephrase the question or ask "
    "about the policy text itself."
)

_WORD_RE = re.compile(r"[a-z0-9]+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])|\n[ \t]*\n")
_CLAUSE_START_RE = re.compile(r"^\d+(?:\.\d+)+[ \t]+")
_LIST_ITEM_RE = re.compile(r"^\(?[a-zA-Z0-9]{1,3}[.)][ \t]+")

DEFAULT_STOPWORDS = frozenset(
    """
    a an and are as at be been being between by can could did do does for from
    had has have he her here him his how i if in into is it its itself may might
    must my no nor not of on once only or other our out over own same shall she
    should so some such than that the their them then there these they this those
    through to too under until up very was we were what when where which while who
    whom why will with would you your
    """.split()
)

# Generator contract: question, retrieved chunks, citations -> answer prose.
Generator = Callable[[str, Sequence[RetrievedChunk], Sequence["Citation"]], str]


@dataclass(frozen=True)
class Citation:
    """One citation attached to an answer sentence."""

    marker: str
    chunk_id: str
    source: str
    article: Optional[str]
    section: Optional[str]
    heading: Optional[str]
    start_page: int
    end_page: int
    start_offset: int
    end_offset: int
    quote: str
    distance: Optional[float]

    def to_dict(self) -> dict:
        return {
            "marker": self.marker,
            "chunk_id": self.chunk_id,
            "source": self.source,
            "article": self.article,
            "section": self.section,
            "heading": self.heading,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "quote": self.quote,
            "distance": self.distance,
        }


@dataclass(frozen=True)
class AbstentionPolicy:
    """Configurable retrieval-confidence policy.

    The defaults were calibrated on 8 out-of-corpus questions and checked on 4
    held-out out-of-corpus questions using the accepted fixed-character index.
    They are a heuristic, not a general answerability classifier.
    """

    hard_max_distance: Optional[float] = 1.30
    soft_max_distance: Optional[float] = 1.10
    min_token_support: float = 0.50
    strict_min_token_support: float = 0.70

    def __post_init__(self) -> None:
        if self.min_token_support < 0 or self.min_token_support > 1:
            raise ValueError("min_token_support must be within [0, 1]")
        if self.strict_min_token_support < 0 or self.strict_min_token_support > 1:
            raise ValueError("strict_min_token_support must be within [0, 1]")
        if self.strict_min_token_support < self.min_token_support:
            raise ValueError("strict_min_token_support must be >= min_token_support")


@dataclass(frozen=True)
class AnswerStatement:
    """One selected answer sentence with its exact source span."""

    text: str
    marker: str
    chunk_id: str
    start_offset: int
    end_offset: int

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "marker": self.marker,
            "chunk_id": self.chunk_id,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
        }


@dataclass
class GroundedAnswer:
    status: str
    question: str
    text: str
    citations: List[Citation]
    statements: List[AnswerStatement]
    retrieved: List[RetrievedChunk]
    reason: Optional[str]
    token_support: float
    best_distance: Optional[float]

    @property
    def abstained(self) -> bool:
        return self.status == ABSTAINED

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "question": self.question,
            "text": self.text,
            "reason": self.reason,
            "token_support": round(self.token_support, 4),
            "best_distance": self.best_distance,
            "citations": [citation.to_dict() for citation in self.citations],
            "statements": [statement.to_dict() for statement in self.statements],
            "retrieved": [
                {
                    "rank": hit.rank,
                    "chunk_id": hit.chunk_id,
                    "section": hit.section,
                    "start_page": hit.start_page,
                    "end_page": hit.end_page,
                    "distance": hit.distance,
                }
                for hit in self.retrieved
            ],
        }


def content_tokens(text: str, stopwords: frozenset[str] = DEFAULT_STOPWORDS) -> List[str]:
    """Lower-case alphanumeric content tokens (length >= 3, stopwords removed)."""
    return [token for token in _WORD_RE.findall(text.lower()) if len(token) >= 3 and token not in stopwords]


def token_support(question: str, hits: Sequence[RetrievedChunk]) -> float:
    """Fraction of distinct question content tokens present in retrieved text."""
    query_tokens = set(content_tokens(question))
    if not query_tokens:
        return 1.0
    corpus_tokens: set[str] = set()
    for hit in hits:
        corpus_tokens.update(content_tokens(hit.text))
    return len(query_tokens & corpus_tokens) / len(query_tokens)


def _best_distance(hits: Sequence[RetrievedChunk]) -> Optional[float]:
    distances = [hit.distance for hit in hits if hit.distance is not None]
    return min(distances) if distances else None


def _sentence_spans(text: str) -> List[tuple[str, int, int]]:
    """Return ``(sentence, start, end)`` spans in source order.

    Sentences are stripped for display but their offsets point at the exact
    substring in ``text`` so a statement can always be traced to the source.
    """
    spans: List[tuple[str, int, int]] = []
    cursor = 0
    for part in _SENTENCE_SPLIT.split(text):
        sentence = part.strip()
        if not sentence:
            continue
        index = text.find(sentence, cursor)
        if index < 0:
            index = text.find(sentence)
        if index < 0:
            continue
        spans.append((sentence, index, index + len(sentence)))
        cursor = index + len(sentence)
    return spans


def _idf_weights(hits: Sequence[RetrievedChunk]) -> dict[str, float]:
    """Token weights from the retrieved set, so specific terms outrank generic ones."""
    documents = [set(content_tokens(hit.text)) for hit in hits]
    total = len(documents) or 1
    document_frequency: Counter[str] = Counter()
    for document in documents:
        document_frequency.update(document)
    return {
        token: math.log((total + 1) / (frequency + 0.5)) + 1.0
        for token, frequency in document_frequency.items()
    }


def _is_heading(sentence: str) -> bool:
    """True when a sentence has no lower-case letters, i.e. is an ALL-CAPS heading."""
    letters = [char for char in sentence if char.isalpha()]
    return len(letters) >= 3 and all(char.isupper() for char in letters)


def _sentence_score(sentence: str, query_tokens: set[str], weights: dict[str, float]) -> float:
    tokens = set(content_tokens(sentence))
    matched = query_tokens & tokens
    if not matched:
        return 0.0
    score = sum(weights.get(token, 1.0) for token in matched)
    # Legal answers usually live in a numbered clause or an explicit list item.
    if _CLAUSE_START_RE.match(sentence):
        score += 1.5
    elif _LIST_ITEM_RE.match(sentence):
        score += 0.75
    # Headings share many query terms but are metadata, not answer text.
    if _is_heading(sentence):
        score *= 0.5
    return score


def _citation_for(hit: RetrievedChunk, marker: str, max_quote_chars: int) -> Citation:
    quote = hit.text[:max_quote_chars]
    if len(hit.text) > max_quote_chars:
        quote = quote.rstrip() + " …"
    return Citation(
        marker=marker,
        chunk_id=hit.chunk_id,
        source=hit.source,
        article=hit.article,
        section=hit.section,
        heading=hit.heading,
        start_page=hit.start_page,
        end_page=hit.end_page,
        start_offset=hit.start_offset,
        end_offset=hit.end_offset,
        quote=quote,
        distance=hit.distance,
    )


class GroundedPipeline:
    """Retrieve, assemble a cited extractive answer, or abstain."""

    def __init__(
        self,
        index: DenseIndex,
        top_k: int = 5,
        policy: Optional[AbstentionPolicy] = None,
        max_answer_sentences: int = 3,
        max_answer_chunks: int = 5,
        max_quote_chars: int = 500,
        generator: Optional[Generator] = None,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if max_answer_sentences <= 0:
            raise ValueError("max_answer_sentences must be positive")
        if max_answer_chunks <= 0:
            raise ValueError("max_answer_chunks must be positive")
        self.index = index
        self.top_k = top_k
        self.policy = policy or AbstentionPolicy()
        self.max_answer_sentences = max_answer_sentences
        self.max_answer_chunks = max_answer_chunks
        self.max_quote_chars = max_quote_chars
        self.generator = generator

    def answer(self, question: str) -> GroundedAnswer:
        question = question.strip()
        if not question:
            return self._abstain(question, [], "empty question", 0.0, None)

        hits = self.index.search(question, k=self.top_k)
        if not hits:
            return self._abstain(question, hits, "no retrieval results", 0.0, None)

        best_distance = _best_distance(hits)
        support = token_support(question, hits)
        policy = self.policy

        if (
            policy.hard_max_distance is not None
            and best_distance is not None
            and best_distance > policy.hard_max_distance
        ):
            return self._abstain(
                question,
                hits,
                f"best distance {best_distance:.4f} exceeds hard limit {policy.hard_max_distance}",
                support,
                best_distance,
            )

        if support < policy.min_token_support:
            return self._abstain(
                question,
                hits,
                f"lexical support {support:.4f} is below {policy.min_token_support}",
                support,
                best_distance,
            )

        if (
            policy.soft_max_distance is not None
            and best_distance is not None
            and best_distance > policy.soft_max_distance
            and support < policy.strict_min_token_support
        ):
            return self._abstain(
                question,
                hits,
                f"best distance {best_distance:.4f} needs support >= "
                f"{policy.strict_min_token_support} (got {support:.4f})",
                support,
                best_distance,
            )

        if self.generator is not None:
            # The generator receives every retrieved chunk.  A generator that
            # cites only what it actually used must emit matching markers; its
            # faithfulness is not validated by this module.
            citations = [
                _citation_for(hit, f"[{index + 1}]", self.max_quote_chars)
                for index, hit in enumerate(hits)
            ]
            text = self.generator(question, hits, citations).strip()
            if text:
                statements: List[AnswerStatement] = []
            else:
                text, statements, citations = self._extractive_answer(question, hits)
        else:
            text, statements, citations = self._extractive_answer(question, hits)

        return GroundedAnswer(
            status=ANSWERED,
            question=question,
            text=text,
            citations=citations,
            statements=statements,
            retrieved=hits,
            reason=None,
            token_support=support,
            best_distance=best_distance,
        )

    def _extractive_answer(
        self,
        question: str,
        hits: Sequence[RetrievedChunk],
    ) -> tuple[str, List[AnswerStatement], List[Citation]]:
        """Select verbatim sentences, then cite only the chunks that supplied them.

        Selecting first keeps the citation list clean: a retrieved-but-unused
        chunk is never presented as support.  Markers are renumbered ``[1]``,
        ``[2]``, ... in the order chunks first contribute to the answer.
        """
        considered = hits[: self.max_answer_chunks]
        query_tokens = set(content_tokens(question))
        weights = _idf_weights(considered)
        candidates = []
        seen: set[str] = set()
        for position, hit in enumerate(considered):
            for sentence_index, (sentence, local_start, local_end) in enumerate(_sentence_spans(hit.text)):
                normalized = " ".join(sentence.lower().split())
                if not normalized or normalized in seen:
                    continue
                score = _sentence_score(sentence, query_tokens, weights)
                if score <= 0.0:
                    continue
                seen.add(normalized)
                candidates.append(
                    (
                        score,
                        position,
                        sentence_index,
                        sentence,
                        hit,
                        local_start,
                        local_end,
                    )
                )

        if not candidates:
            top = hits[0]
            fallback = top.text[: self.max_quote_chars].strip()
            if not fallback:
                return ABSTENTION_MESSAGE, [], []
            citation = _citation_for(top, "[1]", self.max_quote_chars)
            statement = AnswerStatement(
                text=fallback,
                marker=citation.marker,
                chunk_id=top.chunk_id,
                start_offset=top.start_offset,
                end_offset=top.start_offset + len(fallback),
            )
            return f"{fallback} {citation.marker}", [statement], [citation]

        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        selected = candidates[: self.max_answer_sentences]
        # Restore source order for a readable answer.
        selected.sort(key=lambda item: (item[1], item[2]))

        # One marker per contributing chunk, numbered by first appearance.
        marker_by_chunk: dict[str, str] = {}
        for _, _, _, _, hit, _, _ in selected:
            if hit.chunk_id not in marker_by_chunk:
                marker_by_chunk[hit.chunk_id] = f"[{len(marker_by_chunk) + 1}]"

        statements = [
            AnswerStatement(
                text=sentence,
                marker=marker_by_chunk[hit.chunk_id],
                chunk_id=hit.chunk_id,
                start_offset=hit.start_offset + local_start,
                end_offset=hit.start_offset + local_end,
            )
            for _, _, _, sentence, hit, local_start, local_end in selected
        ]
        citations = [
            _citation_for(hit, marker, self.max_quote_chars)
            for hit in considered
            if (marker := marker_by_chunk.get(hit.chunk_id)) is not None
        ]
        citations.sort(key=lambda citation: citation.marker)
        text = " ".join(f"{statement.text} {statement.marker}" for statement in statements)
        return text, statements, citations

    def _abstain(
        self,
        question: str,
        hits: Sequence[RetrievedChunk],
        reason: str,
        support: float,
        best_distance: Optional[float],
    ) -> GroundedAnswer:
        return GroundedAnswer(
            status=ABSTAINED,
            question=question,
            text=ABSTENTION_MESSAGE,
            citations=[],
            statements=[],
            retrieved=list(hits),
            reason=reason,
            token_support=support,
            best_distance=best_distance,
        )
