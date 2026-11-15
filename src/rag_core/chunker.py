"""Structure-aware chunker for regulatory policy documents (canonical core).

Primary segmentation unit: the numbered policy clause (for example ``3.6.7`` or
``4.3.2``), introduced by an ``ARTICLE-N`` heading block.

Design notes:

- Heading blocks are scanned per PDF page, so a trailing page number cannot fuse
  with the next page's first clause id (``2`` + ``2.4`` must never become ``22.4``).
- Heading blocks may wrap across extracted lines (article titles and some clause
  titles are split by the PDF extractor). Upper-case continuation lines are
  absorbed; ordinary body prose is not treated as part of the heading.
- Content of one heading runs until the start of the next heading or the end of
  the document; heading context is retained in chunk metadata.
- Oversized clauses are split at paragraph boundaries; a single paragraph that
  is still too long is split at a character boundary.
- Malformed / unrecognised headings do not cause content loss: unlabelled body
  text is emitted as an unsectioned chunk instead of being dropped.
- No source text is silently discarded; chunk source ranges are exact slices of
  the extracted document text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .document import SourceDocument

ARTICLE_RE = re.compile(r"^[ \t]*ARTICLE[-\s]*\d+[ \t]*:", re.IGNORECASE)
CLAUSE_RE = re.compile(r"^[ \t]*(\d+(?:\.\d+)+)[ \t]+([A-Z])")

MAX_CHUNK_CHARS = 1000
STRATEGY_NAME = "structure_aware"
FIXED_CHARACTER_STRATEGY = "fixed_character"


@dataclass(frozen=True)
class Heading:
    """A detected heading block within the source document."""

    kind: str  # "article" | "clause"
    start: int
    article_id: Optional[str]
    section_id: Optional[str]
    hierarchy: Optional[Tuple[str, ...]]
    heading_text: str


@dataclass
class Chunk:
    """One retrievable chunk with traceable source metadata."""

    chunk_id: str
    source: str
    article: Optional[str]
    section: Optional[str]
    section_hierarchy: Optional[Tuple[str, ...]]
    heading: Optional[str]
    start_page: int
    end_page: int
    start_offset: int
    end_offset: int
    text: str
    chunking_strategy: str
    chunk_sequence: int
    part: int = 0
    start_page_char: int = 0
    end_page_char: int = 0


@dataclass
class _OpenHeading:
    heading: Heading
    lines: List[str] = field(default_factory=list)
    absorb: bool = True


def _iter_lines(text: str) -> List[Tuple[int, str]]:
    """Return ``(start_offset, line_text)`` pairs for every line in ``text``."""
    result: List[Tuple[int, str]] = []
    offset = 0
    for raw in text.split("\n"):
        result.append((offset, raw))
        offset += len(raw) + 1
    return result


def _iter_page_lines(document: SourceDocument) -> List[Tuple[int, int, str]]:
    """Return ``(page_number, start_offset, line_text)`` in document order.

    Lines are split per page rather than across the concatenated document text.
    This is essential: joining pages without a separator otherwise fuses a
    trailing page number (for example ``2`` at the bottom of page 2) with the
    first clause heading of the next page (``2.4``), producing a false clause id
    such as ``22.4`` and a chunk that starts on the wrong page.
    """
    result: List[Tuple[int, int, str]] = []
    for page in document.pages:
        offset = page.start_char
        for raw in page.text.split("\n"):
            result.append((page.page_number, offset, raw))
            offset += len(raw) + 1
    return result


def _is_page_furniture(stripped: str, page_number: int) -> bool:
    """True for a standalone page-number line belonging to the current page."""
    return stripped.isdigit() and int(stripped) == page_number


def _heading_like(text: str) -> bool:
    """Heuristic: an extracted heading line rather than ordinary body prose.

    The SATCOM NGP headings are predominantly upper-case.  Requiring a high
    upper-case ratio prevents a heading from swallowing the following body
    paragraph when the PDF extractor omitted a blank line.
    """
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return False
    upper = sum(1 for char in letters if char.isupper())
    return (upper / len(letters)) >= 0.75


def _finalise(open_heading: _OpenHeading, fallback_article: Optional[str]) -> Heading:
    heading = open_heading.heading
    resolved_article = heading.article_id if heading.article_id is not None else fallback_article
    return Heading(
        kind=heading.kind,
        start=heading.start,
        article_id=resolved_article,
        section_id=heading.section_id,
        hierarchy=heading.hierarchy,
        heading_text=" ".join(
            line.strip() for line in open_heading.lines if line and line.strip()
        ),
    )


def _article_number(article_id: Optional[str]) -> Optional[str]:
    """Extract the numeric part of an ``ARTICLE-N`` identifier."""
    if not article_id:
        return None
    match = re.search(r"(\d+)\s*$", article_id)
    return match.group(1) if match else None


def _normalise_section_id(
    section_id: str,
    page_number: Optional[int],
    article_id: Optional[str] = None,
) -> str:
    """Remove a trailing-page-number artefact from a numbered clause id.

    The extractor can place the page number immediately before the next page's
    first clause (for example ``22.4`` where ``2`` is the page footer and ``2.4``
    is the clause, or ``124.1`` where ``12`` is the footer and ``4.1`` is the
    clause).  Stripping is deliberately conservative:

    * the original identifier must not already agree with the current article;
    * the remainder must be a valid dotted numeric path; and
    * the first component of the remainder must equal the current article
      number.

    The article check is what prevents the helper from corrupting a genuine
    clause such as ``12.4.1`` inside Article 12.
    """
    if page_number is None:
        return section_id
    article_number = _article_number(article_id)
    if article_number is None:
        return section_id
    first_component = section_id.split(".", 1)[0]
    if first_component == article_number:
        return section_id

    prefix = str(page_number)
    if not section_id.startswith(prefix) or len(section_id) <= len(prefix):
        return section_id
    remainder = section_id[len(prefix) :]
    parts = remainder.split(".")
    if not remainder or not all(part.isdigit() and part for part in parts):
        return section_id
    if parts[0] != article_number:
        return section_id
    return remainder


def _find_headings(document: SourceDocument) -> List[Heading]:
    """Scan the document page by page and return heading blocks in source order."""
    headings: List[Heading] = []
    current_article: Optional[str] = None
    open_heading: Optional[_OpenHeading] = None

    def close_open() -> None:
        nonlocal open_heading
        if open_heading is not None:
            headings.append(_finalise(open_heading, current_article))
            open_heading = None

    for page_number, start, raw in _iter_page_lines(document):
        stripped = raw.strip()
        if not stripped or _is_page_furniture(stripped, page_number):
            close_open()
            continue

        if ARTICLE_RE.match(stripped) is not None:
            close_open()
            article_id = ARTICLE_RE.match(stripped).group().strip().rstrip(":").strip()
            open_heading = _OpenHeading(
                heading=Heading(
                    kind="article",
                    start=start,
                    article_id=article_id,
                    section_id=None,
                    hierarchy=None,
                    heading_text="",
                ),
                lines=[stripped],
                absorb=True,
            )
            current_article = article_id
            continue

        clause_match = CLAUSE_RE.match(stripped)
        if clause_match is not None:
            close_open()
            section_id = clause_match.group(1)
            normalised = _normalise_section_id(section_id, page_number, current_article)
            open_heading = _OpenHeading(
                heading=Heading(
                    kind="section",
                    start=start,
                    article_id=current_article,
                    section_id=normalised,
                    hierarchy=tuple(normalised.split(".")) if normalised else None,
                    heading_text="",
                ),
                lines=[stripped],
                absorb=_heading_like(stripped),
            )
            continue

        if open_heading is not None and open_heading.absorb and _heading_like(stripped):
            open_heading.lines.append(stripped)
        else:
            close_open()

    close_open()
    return headings


def _paragraphs(content: str) -> List[Tuple[str, int]]:
    """Split trimmed content into ``(text, start_offset)`` paragraphs."""
    segments: List[Tuple[str, int]] = []
    previous_end = 0
    for match in re.finditer(r"\n[ \t]*\n", content):
        segments.append((content[previous_end : match.start()], previous_end))
        previous_end = match.end()
    segments.append((content[previous_end:], previous_end))
    return [(text, start) for text, start in segments if text.strip()]


def _plan_parts(content: str, max_chars: int) -> List[Tuple[str, int]]:
    """Pack paragraphs into parts of at most ``max_chars`` characters.

    Returns ``(part_text, offset_within_content)`` pairs.
    """
    paragraphs = _paragraphs(content)
    if not paragraphs:
        return [(content, 0)] if content else []

    parts: List[Tuple[str, int]] = []
    buffer = paragraphs[0][0]
    buffer_start = paragraphs[0][1]
    for text, start in paragraphs[1:]:
        separator = content[buffer_start + len(buffer) : start]
        if len(buffer) + len(separator) + len(text) <= max_chars:
            buffer = buffer + separator + text
        else:
            parts.append((buffer, buffer_start))
            buffer = text
            buffer_start = start
    parts.append((buffer, buffer_start))

    planned: List[Tuple[str, int]] = []
    for text, start in parts:
        if len(text) <= max_chars:
            planned.append((text, start))
            continue
        position = start
        offset = 0
        while offset < len(text):
            planned.append((text[offset : offset + max_chars], position))
            position += max_chars
            offset += max_chars
    return planned


def audit_source_coverage(
    document: SourceDocument,
    chunks: List[Chunk],
    max_chars: int = MAX_CHUNK_CHARS,
    allow_overlap: bool = False,
) -> Dict[str, Any]:
    """Audit source fidelity, complete coverage, ordering and traceability.

    The audit is intentionally stricter than checking that every chunk is a
    slice of the source.  It also verifies that the union of chunk intervals
    covers every non-whitespace source character, that chunks are in source
    order, that ids are unique, that page metadata is derivable from the
    offsets, and that no chunk exceeds the configured size.

    ``allow_overlap`` exists for the accepted fixed-character sliding-window
    chunker, whose 200-character overlap is intentional.  It suppresses only the
    overlap *issue*; overlap counts are still reported.
    """
    text = document.text
    total_chars = len(text)

    fidelity_errors: List[Dict[str, Any]] = []
    bound_errors: List[Dict[str, Any]] = []
    for chunk in chunks:
        if not (0 <= chunk.start_offset <= chunk.end_offset <= total_chars):
            bound_errors.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "start_offset": chunk.start_offset,
                    "end_offset": chunk.end_offset,
                }
            )
            continue
        if text[chunk.start_offset : chunk.end_offset] != chunk.text:
            fidelity_errors.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "start_offset": chunk.start_offset,
                    "end_offset": chunk.end_offset,
                }
            )

    intervals = sorted((chunk.start_offset, chunk.end_offset) for chunk in chunks)
    within_bounds = not bound_errors

    overlaps: List[Dict[str, int]] = []
    uncovered_regions: List[Dict[str, Any]] = []
    covered_chars = 0
    overlap_chars = 0
    merged_end = 0
    for start, end in intervals:
        if start < merged_end:
            overlaps.append({"start": start, "end": min(end, merged_end)})
            overlap_chars += max(0, min(end, merged_end) - start)
        if start > merged_end:
            uncovered_regions.append({"start": merged_end, "end": start, "text": text[merged_end:start]})
        merged_end = max(merged_end, end)
        covered_chars += end - start
    if merged_end < total_chars:
        uncovered_regions.append({"start": merged_end, "end": total_chars, "text": text[merged_end:total_chars]})

    uncovered_non_whitespace = [
        {"start": region["start"], "end": region["end"], "text": region["text"]}
        for region in uncovered_regions
        if not region["text"].isspace()
    ]
    uncovered_non_whitespace_chars = sum(
        1 for region in uncovered_non_whitespace for char in region["text"] if not char.isspace()
    )
    total_non_whitespace_chars = sum(1 for char in text if not char.isspace())
    covered_non_whitespace_chars = total_non_whitespace_chars - uncovered_non_whitespace_chars
    coverage_percent = (
        (covered_non_whitespace_chars / total_non_whitespace_chars) * 100.0
        if total_non_whitespace_chars
        else 100.0
    )
    uncovered_whitespace_chars = sum(
        1
        for region in uncovered_regions
        if region["text"].isspace()
        for char in region["text"]
        if char.isspace()
    )

    ids = [chunk.chunk_id for chunk in chunks]
    duplicate_ids = sorted({chunk_id for chunk_id in ids if ids.count(chunk_id) > 1})
    sequence_order_ok = [chunk.chunk_sequence for chunk in chunks] == list(range(len(chunks)))
    source_order_ok = all(
        chunks[i].start_offset <= chunks[i + 1].start_offset for i in range(len(chunks) - 1)
    )

    page_mapping_errors: List[Dict[str, Any]] = []
    for chunk in chunks:
        if not (0 <= chunk.start_offset <= chunk.end_offset <= total_chars):
            continue
        start_page = document.page_for_offset(chunk.start_offset)
        end_page = document.page_for_offset(max(chunk.end_offset - 1, chunk.start_offset))
        expected_start = start_page.page_number if start_page else 0
        expected_end = end_page.page_number if end_page else 0
        expected_start_char = chunk.start_offset - start_page.start_char if start_page else 0
        expected_end_char = chunk.end_offset - end_page.start_char if end_page else 0
        if (
            chunk.start_page != expected_start
            or chunk.end_page != expected_end
            or chunk.start_page_char != expected_start_char
            or chunk.end_page_char != expected_end_char
        ):
            page_mapping_errors.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "expected": {
                        "start_page": expected_start,
                        "end_page": expected_end,
                        "start_page_char": expected_start_char,
                        "end_page_char": expected_end_char,
                    },
                    "actual": {
                        "start_page": chunk.start_page,
                        "end_page": chunk.end_page,
                        "start_page_char": chunk.start_page_char,
                        "end_page_char": chunk.end_page_char,
                    },
                }
            )

    empty_chunks = [chunk.chunk_id for chunk in chunks if not chunk.text.strip()]
    oversized_chunks = [chunk.chunk_id for chunk in chunks if len(chunk.text) > max_chars]

    issues: List[str] = []
    if not within_bounds:
        issues.append("chunk offsets are outside the source bounds")
    if fidelity_errors:
        issues.append("one or more chunks are not exact source slices")
    if overlaps and not allow_overlap:
        issues.append("one or more chunk source ranges overlap")
    if uncovered_non_whitespace:
        issues.append("non-whitespace source characters are not covered")
    if duplicate_ids:
        issues.append("chunk ids are not unique")
    if not sequence_order_ok:
        issues.append("chunk_sequence is not a contiguous document-order sequence")
    if not source_order_ok:
        issues.append("chunks are not emitted in source order")
    if page_mapping_errors:
        issues.append("chunk page metadata does not match the source offsets")
    if empty_chunks:
        issues.append("one or more chunks are empty")
    if oversized_chunks:
        issues.append("one or more chunks exceed the configured maximum size")

    return {
        "source": document.source,
        "ok": not issues,
        "issues": issues,
        "total_chars": total_chars,
        "covered_chars": covered_chars,
        "overlap_chars": overlap_chars,
        "overlaps": overlaps,
        "within_bounds": within_bounds,
        "bound_errors": bound_errors,
        "fidelity_errors": fidelity_errors,
        "duplicate_ids": duplicate_ids,
        "sequence_order_ok": sequence_order_ok,
        "source_order_ok": source_order_ok,
        "page_mapping_errors": page_mapping_errors,
        "empty_chunks": empty_chunks,
        "oversized_chunks": oversized_chunks,
        "uncovered_regions": uncovered_regions,
        "uncovered_non_whitespace_regions": uncovered_non_whitespace,
        "uncovered_non_whitespace_chars": uncovered_non_whitespace_chars,
        "uncovered_whitespace_chars": uncovered_whitespace_chars,
        "total_non_whitespace_chars": total_non_whitespace_chars,
        "covered_non_whitespace_chars": covered_non_whitespace_chars,
        "coverage_percent": coverage_percent,
    }


@dataclass
class StructureAwareChunker:
    """Structure-aware chunker producing clause-anchored chunks."""

    max_chars: int = MAX_CHUNK_CHARS
    strategy: str = STRATEGY_NAME

    def chunk(self, document: SourceDocument) -> List[Chunk]:
        text = document.text
        headings = _find_headings(document)
        boundaries = [heading.start for heading in headings] + [len(text)]

        chunks: List[Chunk] = []
        sequence = 0

        def emit(content_start: int, content_end: int, context: Tuple) -> None:
            nonlocal sequence
            article, section, hierarchy, heading_text = context
            raw = text[content_start:content_end]
            stripped = raw.strip()
            if not stripped:
                return
            start = content_start + (len(raw) - len(raw.lstrip()))
            content = text[start : start + len(stripped)]
            if len(content) > self.max_chars:
                plans = [(part, start + offset) for part, offset in _plan_parts(content, self.max_chars)]
            else:
                plans = [(content, start)]
            for part_index, (part_text, part_start) in enumerate(plans):
                part_end = part_start + len(part_text)
                page_start = document.page_for_offset(part_start)
                page_end = document.page_for_offset(max(part_end - 1, part_start))
                start_page_number = page_start.page_number if page_start else 0
                end_page_number = page_end.page_number if page_end else 0
                base_id = (
                    f"{document.source.replace('.pdf', '')}::"
                    f"{article or 'PREAMBLE'}::"
                    f"{section or 'BODY'}"
                    + (f"::part_{part_index}" if part_index else "")
                )
                chunks.append(
                    Chunk(
                        # The source offset is part of the id so that duplicate or
                        # malformed section labels can never produce colliding ids.
                        chunk_id=f"{base_id}::off{part_start}",
                        source=document.source,
                        article=article,
                        section=section,
                        section_hierarchy=hierarchy,
                        heading=heading_text,
                        start_page=start_page_number,
                        end_page=end_page_number,
                        start_offset=part_start,
                        end_offset=part_end,
                        text=part_text,
                        chunking_strategy=self.strategy,
                        chunk_sequence=sequence,
                        part=part_index,
                        start_page_char=(part_start - page_start.start_char) if page_start else 0,
                        end_page_char=(part_end - page_end.start_char) if page_end else 0,
                    )
                )
                sequence += 1

        if not headings or headings[0].start > 0:
            first_start = headings[0].start if headings else len(text)
            emit(0, first_start, (None, None, None, None))

        for index, heading in enumerate(headings):
            context = (
                heading.article_id,
                heading.section_id,
                heading.hierarchy,
                heading.heading_text,
            )
            emit(heading.start, boundaries[index + 1], context)

        return chunks


@dataclass
class FixedCharacterChunker:
    """Canonical fixed-character sliding-window chunker.

    Reproduces the accepted EXP-001 configuration:

    - 1000-character windows with 200-character overlap (step 800);
    - chunks never cross a PDF page boundary;
    - each page is stripped before windowing, exactly as the frozen EXP-001
      runner did, and offsets are mapped back to global document coordinates;
    - empty windows are skipped without losing coverage.

    This class exists so production ingestion can use the *accepted* retrieval
    strategy without depending on the frozen experiment script.
    """

    chunk_size: int = MAX_CHUNK_CHARS
    overlap: int = 200
    strategy: str = FIXED_CHARACTER_STRATEGY

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.overlap < 0:
            raise ValueError("overlap must not be negative")
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

    @staticmethod
    def _metadata(
        headings: List[Heading],
        start: int,
        end: int,
    ) -> Tuple[Optional[str], Optional[str], Optional[Tuple[str, ...]], Optional[str]]:
        """Return chunk-level context metadata.

        Fixed-character windows may start mid-clause or span multiple clauses. Use
        the governing heading at the chunk start (last heading with start <= chunk
        start) when available; fall back to the first heading inside the chunk.
        This avoids labeling a chunk by a later clause that appears only at its end.
        """
        inside = [heading for heading in headings if start <= heading.start < end]
        before_or_at = [heading for heading in headings if heading.start <= start]
        governing = before_or_at[-1] if before_or_at else (inside[0] if inside else None)
        if governing is None:
            return (None, None, None, None)

        # Prefer most specific governing section/clause; article heading provides
        # article context when chunk starts at/past article and before any clause.
        if governing.kind == "section":
            return (
                governing.article_id,
                governing.section_id,
                governing.hierarchy,
                governing.heading_text or None,
            )

        # Article governing: keep article, but if a clause starts inside the chunk,
        # expose that clause as chunk metadata (common for chunks beginning at
        # article boundary) only when chunk start is at/after article and before
        # clause? No - clause inside is fine to mention if chunk starts at article.
        first_section_inside = next((h for h in inside if h.kind == "section"), None)
        if first_section_inside is not None and start <= first_section_inside.start:
            return (
                first_section_inside.article_id or governing.article_id,
                first_section_inside.section_id,
                first_section_inside.hierarchy,
                first_section_inside.heading_text or None,
            )

        return (
            governing.article_id,
            None,
            None,
            governing.heading_text or None,
        )

    def chunk(self, document: SourceDocument) -> List[Chunk]:
        headings = _find_headings(document)
        chunks: List[Chunk] = []
        sequence = 0
        step = self.chunk_size - self.overlap

        for page in document.pages:
            raw = page.text
            stripped = raw.strip()
            if not stripped:
                continue
            leading = len(raw) - len(raw.lstrip())
            length = len(stripped)
            start = 0
            index = 0
            while start < length:
                end = min(start + self.chunk_size, length)
                text = stripped[start:end]
                if text.strip():
                    global_start = page.start_char + leading + start
                    global_end = page.start_char + leading + end
                    article, section, hierarchy, heading_text = self._metadata(
                        headings, global_start, global_end
                    )
                    chunks.append(
                        Chunk(
                            chunk_id=(
                                f"{document.source.replace('.pdf', '')}::"
                                f"page_{page.page_number}::chunk_{index}::off{global_start}"
                            ),
                            source=document.source,
                            article=article,
                            section=section,
                            section_hierarchy=hierarchy,
                            heading=heading_text,
                            start_page=page.page_number,
                            end_page=page.page_number,
                            start_offset=global_start,
                            end_offset=global_end,
                            text=text,
                            chunking_strategy=self.strategy,
                            chunk_sequence=sequence,
                            part=0,
                            start_page_char=global_start - page.start_char,
                            end_page_char=global_end - page.start_char,
                        )
                    )
                    sequence += 1
                if end >= length:
                    break
                start += step
                index += 1

        return chunks
