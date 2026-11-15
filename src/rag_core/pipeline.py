"""Evidence-grounded answer assembly with citations and abstention.

This module is the production-facing layer above the shared dense index.  It does
**not** invent answers:

- it retrieves evidence with :class:`~src.rag_core.index.DenseIndex`;
- it answers extractively by selecting verbatim sentences from retrieved chunks;
- when the strongest sentence is a numbered clause, it may select sibling
  clauses under the same parent (EXP-003) so multi-clause questions are not
  truncated to one child clause;
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
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

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
_CLAUSE_START_RE = re.compile(r"^(\d+(?:\.\d+)+)[ \t]+")
_LIST_ITEM_RE = re.compile(r"^\(?[a-zA-Z0-9]{1,3}[.)][ \t]+")
_SUBMARKER_RE = re.compile(r"\([a-z0-9]+\)", re.IGNORECASE)
_CLAUSE_REF_RE = re.compile(r"\b\d+(?:\.\d+)+(?:\([a-z0-9]+\))?\b", re.IGNORECASE)
_FOR_APPLICABILITY_RE = re.compile(r"^(?:\d+(?:\.\d+)+\s+)?For\s+\d+(?:\.\d+)+", re.IGNORECASE)
_CLAUSE_FOR_START_RE = re.compile(r"^(\d+(?:\.\d+)+)\s+For\s+\d+(?:\.\d+)+", re.IGNORECASE)
_ITEM_SPLIT_RE = re.compile(r"(?=\([a-z0-9]+\))", re.IGNORECASE)

_DEFINITION_INTENT_RE = re.compile(
    r"\b(define|defined|definition|meaning|means|what is|what are|how is|how are|describe)\b",
    re.IGNORECASE,
)
_DEFINITIONAL_MARKER_RE = re.compile(
    r"\b(will include|includes?|include|including|means|defined as|treated as|refers to|"
    r"consist of|comprises|where|i\.e\.)\b",
    re.IGNORECASE,
)
_DEFAULT_DEFINITION_INTENT_BOOST = 1.5

# Sibling-clause selection: when the strongest sentence is a numbered clause,
# prefer clauses under the same parent whose score is at least this fraction of
# the strongest score.  Accepted by EXP-003 (Q13 evidence coverage 1/3 → 3/3).
DEFAULT_SIBLING_SCORE_RATIO = 0.5

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

# (score, chunk position, sentence index, sentence, hit, local start, local end)
Candidate = tuple[float, int, int, str, RetrievedChunk, int, int]


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


_ASSOC_INTRO_COLON_RE = re.compile(r":\s*$")


def _answer_unit_spans(
    text: str,
    clause_body_units: bool = False,
    associated_heading_units: bool = False,
) -> List[tuple[str, int, int]]:
    """Return answer-unit spans used for extractive selection.

    Default: sentence spans. ``clause_body_units`` (EXP-005, opt-in) groups a
    numbered subclause start (depth >= 3) with contiguous body until the next
    numbered clause start. ``associated_heading_units`` (EXP-007 candidate,
    opt-in) groups an introductory span that ends with a colon (enumeration/
    definition lead-in) with its immediately following associated content in the
    same chunk until the next numbered clause start, preserving heading-introduced
    lists/definitions as one verbatim unit.
    """
    spans = _sentence_spans(text)
    if not clause_body_units and not associated_heading_units:
        return spans

    units: List[tuple[str, int, int]] = []
    index = 0
    while index < len(spans):
        sentence, start, end = spans[index]
        stripped = sentence.strip()
        match = _CLAUSE_START_RE.match(stripped)
        group_with_body = False
        if associated_heading_units and _ASSOC_INTRO_COLON_RE.search(stripped):
            group_with_body = True
        elif clause_body_units and match and len(match.group(1).split(".")) >= 3:
            group_with_body = True

        if group_with_body:
            block_start = start
            block_end = end
            index += 1
            while index < len(spans) and not _CLAUSE_START_RE.match(spans[index][0].strip()):
                block_end = spans[index][2]
                index += 1
            raw = text[block_start:block_end]
            unit = raw.strip()
            if unit:
                left = len(raw) - len(raw.lstrip())
                right = len(raw) - len(raw.rstrip())
                units.append((unit, block_start + left, block_end - right))
            continue

        units.append((sentence, start, end))
        index += 1
    return units


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


def _clause_parent(sentence: str) -> Optional[str]:
    """Return the parent clause ID for a numbered sentence, or ``None``.

    ``4.3.4 Land Mobile ...`` has parent ``4.3``; a non-clause sentence has no
    parent.  This is deliberately lexical: it only groups clauses the source text
    already labels.
    """
    match = _CLAUSE_START_RE.match(sentence)
    if match is None:
        return None
    parts = match.group(1).split(".")
    return ".".join(parts[:-1]) if len(parts) > 1 else None


_CLAUSE_HEADING_LINE_RE = re.compile(
    r"^(?P<clause>\d+(?:\.\d+)+)[ \t]+(?P<heading>[^\n]+)", re.MULTILINE
)
_ARTICLE_HEADING_LINE_RE = re.compile(r"^(ARTICLE[-\s]*\d+[ \t]*:[^\n]*)", re.MULTILINE | re.IGNORECASE)


def _governing_metadata_for_offset(
    hit: RetrievedChunk, global_offset: Optional[int]
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (article, section, heading) governing a cited statement offset.

    Fixed-character chunks can span multiple clauses and chunk-level metadata may
    not match the specific cited statement. Prefer the nearest clause/article
    heading at or before the statement offset inside the chunk text. If the
    statement is before the first clause heading in the chunk, keep the chunk's
    start-governing metadata (do not inherit a later in-chunk clause label).
    """
    if global_offset is None:
        return hit.article, hit.section, hit.heading
    local = global_offset - hit.start_offset
    if local < 0 or local > len(hit.text):
        return hit.article, hit.section, hit.heading

    article = hit.article
    heading = hit.heading
    section = hit.section

    clause_matches = list(_CLAUSE_HEADING_LINE_RE.finditer(hit.text))
    first_clause_pos = clause_matches[0].start() if clause_matches else None
    in_chunk_clause = [m for m in clause_matches if m.start() <= local]

    article_match = None
    for m in _ARTICLE_HEADING_LINE_RE.finditer(hit.text):
        if m.start() <= local:
            article_match = m

    if in_chunk_clause:
        if article_match and article_match.start() <= in_chunk_clause[-1].start():
            article_text = " ".join(article_match.group(1).strip().split())
            article = article_text.split(":")[0].strip() or article
        m = in_chunk_clause[-1]
        clause = m.group("clause")
        section = clause
        heading = " ".join((clause + " " + m.group("heading")).strip().split())
        return article, section, heading

    # No clause heading at/before statement in this chunk.
    # If statement is before first in-chunk clause, retain chunk start-governing
    # metadata (correct for mid-clause chunk starts); avoid using a later clause.
    if first_clause_pos is not None and local < first_clause_pos:
        if article_match:
            article_text = " ".join(article_match.group(1).strip().split())
            article = article_text.split(":")[0].strip() or article
        return article, section, heading

    if article_match:
        article_text = " ".join(article_match.group(1).strip().split())
        article = article_text.split(":")[0].strip() or article
        heading = article_text
        section = None
    return article, section, heading


def _citation_for(
    hit: RetrievedChunk,
    marker: str,
    max_quote_chars: int,
    statement_start_offset: Optional[int] = None,
) -> Citation:
    quote = hit.text[:max_quote_chars]
    if len(hit.text) > max_quote_chars:
        quote = quote.rstrip() + " …"
    article, section, heading = _governing_metadata_for_offset(hit, statement_start_offset)
    return Citation(
        marker=marker,
        chunk_id=hit.chunk_id,
        source=hit.source,
        article=article,
        section=section,
        heading=heading,
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
        sibling_score_ratio: Optional[float] = DEFAULT_SIBLING_SCORE_RATIO,
        clause_body_units: bool = False,
        cross_reference_selection: bool = False,
        associated_heading_units: bool = False,
        definition_intent_boost: bool = False,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if max_answer_sentences <= 0:
            raise ValueError("max_answer_sentences must be positive")
        if max_answer_chunks <= 0:
            raise ValueError("max_answer_chunks must be positive")
        if sibling_score_ratio is not None and not 0.0 <= sibling_score_ratio <= 1.0:
            raise ValueError("sibling_score_ratio must be within [0, 1]")
        self.index = index
        self.top_k = top_k
        self.policy = policy or AbstentionPolicy()
        self.max_answer_sentences = max_answer_sentences
        self.max_answer_chunks = max_answer_chunks
        self.max_quote_chars = max_quote_chars
        self.generator = generator
        self.sibling_score_ratio = sibling_score_ratio
        self.clause_body_units = clause_body_units
        self.cross_reference_selection = cross_reference_selection
        self.associated_heading_units = associated_heading_units
        self.definition_intent_boost = definition_intent_boost

    def answer(self, question: str) -> GroundedAnswer:
        question = question.strip()
        if not question:
            return self._abstain(question, [], "empty question", 0.0, None)
        return self.answer_from_hits(question, self.index.search(question, k=self.top_k))

    def answer_from_hits(self, question: str, hits: Sequence[RetrievedChunk]) -> GroundedAnswer:
        """Answer from pre-computed retrieval results.

        Used by controlled answer-layer experiments so both arms see exactly the
        same chunks.  Production callers should normally use :meth:`answer`.
        """
        question = question.strip()
        if not question:
            return self._abstain(question, [], "empty question", 0.0, None)

        hits = list(hits)
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
        """Select verbatim answer units, then cite only the chunks that supplied them.

        Selecting first keeps the citation list clean: a retrieved-but-unused
        chunk is never presented as support. Markers are renumbered ``[1]``,
        ``[2]``, ... in the order chunks first contribute to the answer.
        """
        considered = hits[: self.max_answer_chunks]
        query_tokens = set(content_tokens(question))
        weights = _idf_weights(considered)
        definitional_intent = self.definition_intent_boost and bool(_DEFINITION_INTENT_RE.search(question))
        candidates = []
        seen: set[str] = set()
        for position, hit in enumerate(considered):
            for sentence_index, (sentence, local_start, local_end) in enumerate(
                _answer_unit_spans(hit.text, self.clause_body_units, self.associated_heading_units)
            ):
                normalized = " ".join(sentence.lower().split())
                if not normalized or normalized in seen:
                    continue
                score = _sentence_score(sentence, query_tokens, weights)
                if definitional_intent and score > 0.0 and _DEFINITIONAL_MARKER_RE.search(sentence):
                    overlap = query_tokens & set(content_tokens(sentence))
                    if overlap:
                        score += _DEFAULT_DEFINITION_INTENT_BOOST
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
            citation = _citation_for(top, "[1]", self.max_quote_chars, top.start_offset)
            statement = AnswerStatement(
                text=fallback,
                marker=citation.marker,
                chunk_id=top.chunk_id,
                start_offset=top.start_offset,
                end_offset=top.start_offset + len(fallback),
            )
            return f"{fallback} {citation.marker}", [statement], [citation]

        candidates = self._apply_cross_reference_selection(candidates, query_tokens, weights)

        # keep only positively scored candidates as before
        candidates = [c for c in candidates if c[0] > 0.0]
        if not candidates:
            top = hits[0]
            fallback = top.text[: self.max_quote_chars].strip()
            if not fallback:
                return ABSTENTION_MESSAGE, [], []
            citation = _citation_for(top, "[1]", self.max_quote_chars, top.start_offset)
            statement = AnswerStatement(
                text=fallback,
                marker=citation.marker,
                chunk_id=top.chunk_id,
                start_offset=top.start_offset,
                end_offset=top.start_offset + len(fallback),
            )
            return f"{fallback} {citation.marker}", [statement], [citation]

        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        selected = self._select_candidates(candidates)
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
        # For citation section/heading metadata, prefer the governing clause of the
        # first selected statement in each cited chunk (fixed-char chunks can span
        # multiple clauses and chunk-level metadata may be misleading).
        first_stmt_offset_by_chunk: dict[str, int] = {}
        for stmt in statements:
            if stmt.chunk_id not in first_stmt_offset_by_chunk:
                first_stmt_offset_by_chunk[stmt.chunk_id] = stmt.start_offset
        citations = [
            _citation_for(hit, marker, self.max_quote_chars, first_stmt_offset_by_chunk.get(hit.chunk_id))
            for hit in considered
            if (marker := marker_by_chunk.get(hit.chunk_id)) is not None
        ]
        citations.sort(key=lambda citation: citation.marker)
        text = " ".join(f"{statement.text} {statement.marker}" for statement in statements)
        return text, statements, citations

    def _apply_cross_reference_selection(
        self,
        candidates: List[Candidate],
        query_tokens: Set[str],
        weights: Dict[str, float],
    ) -> List[Candidate]:
        """Opt-in boost for applicability clauses linked to query-relevant subclauses.

        General document-grounded signal (EXP-006 candidate): clauses that declare
        applicability with "For <clause/subclause refs>" (e.g. "For 3.6.7(a) and
        3.6.7(b)...") are procedurally linked to those referenced conditions. If the
        referenced submarkers correspond to list items/segments in retrieved
        candidates that are lexically relevant to the question, the applicability
        clause is promoted together with its contiguous body as a single verbatim
        answer unit. This does not change retrieval and does not use question-specific
        rules.
        """
        if not self.cross_reference_selection or not candidates:
            return candidates

        positive = [c for c in candidates if c[0] > 0.0]
        if not positive:
            return candidates
        top_score = max(c[0] for c in positive)

        # 1) Identify query-relevant submarkers from retrieved candidate segments.
        query_relevant_submarkers: Set[str] = set()
        for score, _, _, sentence, _, _, _ in candidates:
            parts = _ITEM_SPLIT_RE.split(sentence)
            for part in parts:
                marker_match = _SUBMARKER_RE.search(part)
                if not marker_match:
                    continue
                marker = marker_match.group(0).lower()
                seg_tokens = set(content_tokens(part))
                overlap = query_tokens & seg_tokens
                if not overlap:
                    continue
                seg_score = sum(weights.get(token, 1.0) for token in overlap)
                # require meaningful relevance (not incidental "bands" etc. alone)
                if len(overlap) >= 2 or seg_score >= 4.0:
                    query_relevant_submarkers.add(marker)

        if not query_relevant_submarkers:
            return candidates

        # 2) Find promoted applicability clause blocks per chunk position.
        by_pos: Dict[int, List[Tuple[int, Candidate]]] = {}
        for idx, cand in enumerate(candidates):
            by_pos.setdefault(cand[1], []).append((idx, cand))

        # (pos, clause_id) -> (start_idx, end_idx_exclusive_in_items_sorted, block_indices)
        promoted_blocks: Dict[Tuple[int, str], Tuple[int, int, List[int]]] = {}
        suppressed: Set[int] = set()
        for pos, items in by_pos.items():
            items_sorted = sorted(items, key=lambda x: x[1][2])
            i = 0
            while i < len(items_sorted):
                idx, cand = items_sorted[i]
                sentence = cand[3].strip()
                for_match = _CLAUSE_FOR_START_RE.match(sentence)
                if not for_match:
                    i += 1
                    continue
                clause_id = for_match.group(1)
                refs = {m.lower() for m in _SUBMARKER_RE.findall(sentence)}
                if not (refs & query_relevant_submarkers):
                    i += 1
                    continue
                block = [idx]
                j = i + 1
                while j < len(items_sorted):
                    bidx, bcand = items_sorted[j]
                    if _CLAUSE_START_RE.match(bcand[3].strip()):
                        break
                    block.append(bidx)
                    j += 1
                promoted_blocks[(pos, clause_id)] = (i, j, block)
                suppressed.update(block)
                i = j

        if not promoted_blocks:
            return candidates

        # 3) Build merged promoted units and keep non-suppressed candidates.
        new_candidates: List[Candidate] = [c for idx, c in enumerate(candidates) if idx not in suppressed]
        promote_score = top_score * 0.95
        for pos, items in by_pos.items():
            items_sorted = sorted(items, key=lambda x: x[1][2])
            for (p, clause_id), (_, _, block) in promoted_blocks.items():
                if p != pos:
                    continue
                block_cands = [items_sorted_items for items_sorted_items in items_sorted if items_sorted_items[0] in block]
                # preserve sentence order
                block_cands = sorted([(i, c) for i, c in block_cands], key=lambda x: x[1][2])
                if not block_cands:
                    continue
                first = block_cands[0][1]
                last = block_cands[-1][1]
                hit = first[4]
                # merge exact source span from first.start to last.end
                ls = first[5]
                le = last[6]
                merged = hit.text[ls:le].strip()
                if not merged:
                    continue
                # score: strong applicability promotion, but keep below best direct match
                block_scores = [max(c[0], 0.0) for _, c in block_cands]
                new_score = max(max(block_scores), promote_score)
                # use first sentence index for stable ordering
                new_candidates.append((new_score, pos, first[2], merged, hit, ls, le))

        return new_candidates

    def _select_candidates(self, candidates: List[Candidate]) -> List[Candidate]:
        """Select at most ``max_answer_sentences`` candidates.

        The default ranking is pure score order.  When ``sibling_score_ratio`` is
        enabled and the strongest candidate starts with a numbered clause, clauses
        under the same parent are promoted first.  Legal compound questions often
        require sibling clauses (for example ``4.3.2``-``4.3.4``); selecting only
        the single highest-scoring clause produces an incomplete answer even when
        every sibling was retrieved.
        """
        maximum = self.max_answer_sentences
        if self.sibling_score_ratio is None:
            return candidates[:maximum]

        top_score = candidates[0][0]
        parent = _clause_parent(candidates[0][3])
        selected: List[Candidate] = []
        selected_keys: set[str] = set()
        if parent is not None:
            minimum = top_score * self.sibling_score_ratio
            for candidate in candidates:
                if _clause_parent(candidate[3]) != parent or candidate[0] < minimum:
                    continue
                key = " ".join(candidate[3].lower().split())
                if key in selected_keys:
                    continue
                selected.append(candidate)
                selected_keys.add(key)
                if len(selected) >= maximum:
                    return selected

        for candidate in candidates:
            if len(selected) >= maximum:
                break
            key = " ".join(candidate[3].lower().split())
            if key in selected_keys:
                continue
            selected.append(candidate)
            selected_keys.add(key)
        return selected

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
