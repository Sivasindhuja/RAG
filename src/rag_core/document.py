"""Canonical source-document model for the SATCOM NGP RAG document core.

The model keeps the extracted source text intact (no silent normalisation or
discard) and records page-level character offset bookkeeping so that any chunk
can be traced back to its position in the original PDF.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from pypdf import PdfReader


@dataclass
class Page:
    """A single extracted PDF page plus source character offset bookkeeping."""

    page_number: int
    text: str
    start_char: int = 0
    end_char: int = 0


@dataclass
class SourceDocument:
    """Ordered, multi-page source document with traceable character offsets."""

    source: str
    pages: List[Page] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "".join(page.text for page in self.pages)

    @property
    def total_chars(self) -> int:
        return sum(len(page.text) for page in self.pages)

    def page_for_offset(self, offset: int) -> Optional[Page]:
        """Return the page containing ``offset``.

        Offsets are half-open (``start_char <= offset < end_char``).  An offset
        equal to the end of the document is attributed to the last page so that
        callers can map a chunk end without special-casing the final boundary.
        Negative offsets and offsets past the end of the document return
        ``None``.
        """
        if not self.pages or offset < 0:
            return None
        for page in self.pages:
            if page.start_char <= offset < page.end_char:
                return page
        if offset == self.pages[-1].end_char:
            return self.pages[-1]
        return None


def extract_source_document(pdf_path: Path, source: Optional[str] = None) -> SourceDocument:
    """Extract all pages of a PDF into a :class:`SourceDocument`.

    Extraction is intentionally conservative: the text returned by the PDF
    extractor is stored as-is (no stripping/normalisation happens here) and page
    character offsets are accumulated so chunks can be mapped back to the source.
    """
    reader = PdfReader(str(pdf_path))
    pages: List[Page] = []
    offset = 0
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        start_char = offset
        end_char = offset + len(text)
        pages.append(Page(page_number=index, text=text, start_char=start_char, end_char=end_char))
        offset = end_char
    return SourceDocument(source=source or Path(pdf_path).name, pages=pages)
