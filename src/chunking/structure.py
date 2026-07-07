"""
Structure-aware chunking for regulatory PDFs (SatCom-NGP).

Why not recursive-only?
- The policy is organized by ARTICLE-N and numbered clauses (e.g. 3.4.2).
- Recursive splitting cuts across clause boundaries and scatters acronyms (GMPCS, CAISS, INSAT).

Strategy: section-boundary parents + small children (parent-document retrieval).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

ARTICLE_RE = re.compile(r"\bARTICLE[-\s]*\d+\s*:", re.IGNORECASE)
SECTION_RE = re.compile(r"(?:^|\n)\s*(\d+\.\d+(?:\.\d+)?)\s+([A-Z])", re.MULTILINE)
# Disabled by default: PDF extraction breaks headers across lines and over-splits.
ALLCAPS_HEADER_RE = re.compile(
    r"(?:^|\n)\s*([A-Z][A-Z0-9\s,\-/]{20,}?)\s*\n",
    re.MULTILINE,
)
USE_ALLCAPS_SPLITS = False


@dataclass
class SectionSpan:
    start: int
    label: str
    kind: str  # article | section | header


def _find_section_spans(text: str) -> list[SectionSpan]:
    spans: list[SectionSpan] = []
    for m in ARTICLE_RE.finditer(text):
        spans.append(SectionSpan(m.start(), m.group().strip(), "article"))
    for m in SECTION_RE.finditer(text):
        spans.append(SectionSpan(m.start(), m.group(1), "section"))
    if USE_ALLCAPS_SPLITS:
        for m in ALLCAPS_HEADER_RE.finditer(text):
            header = m.group(1).strip()
            if len(header.split()) >= 3:
                spans.append(SectionSpan(m.start(), header[:80], "header"))
    spans.sort(key=lambda s: s.start)
    # Deduplicate nearby splits (within 80 chars keep the earliest)
    deduped: list[SectionSpan] = []
    for span in spans:
        if deduped and span.start - deduped[-1].start < 80:
            continue
        deduped.append(span)
    return deduped


def _segment_page(text: str, page: int, source: str, parent_min: int, parent_max: int) -> list[Document]:
    spans = _find_section_spans(text)
    if not spans:
        return [
            Document(
                page_content=text.strip(),
                metadata={"source": source, "page": page, "section": "body"},
            )
        ]

    boundaries = [s.start for s in spans] + [len(text)]
    parents: list[Document] = []
    for i, span in enumerate(spans):
        start = span.start
        end = boundaries[i + 1]
        chunk_text = text[start:end].strip()
        if len(chunk_text) < parent_min and parents:
            # Merge tiny tail into previous parent
            prev = parents[-1]
            merged = prev.page_content + "\n\n" + chunk_text
            if len(merged) <= parent_max:
                parents[-1] = Document(
                    page_content=merged,
                    metadata={**prev.metadata, "section": f"{prev.metadata.get('section')}+{span.label}"},
                )
                continue
        if len(chunk_text) > parent_max:
            # Oversized section: sub-split on paragraph boundaries
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", chunk_text) if p.strip()]
            buf = ""
            sub_idx = 0
            for para in paragraphs:
                candidate = f"{buf}\n\n{para}".strip() if buf else para
                if len(candidate) > parent_max and buf:
                    parents.append(
                        Document(
                            page_content=buf,
                            metadata={
                                "source": source,
                                "page": page,
                                "section": f"{span.label}.{sub_idx}",
                                "section_kind": span.kind,
                            },
                        )
                    )
                    sub_idx += 1
                    buf = para
                else:
                    buf = candidate
            if buf:
                parents.append(
                    Document(
                        page_content=buf,
                        metadata={
                            "source": source,
                            "page": page,
                            "section": f"{span.label}.{sub_idx}" if sub_idx else span.label,
                            "section_kind": span.kind,
                        },
                    )
                )
        else:
            parents.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        "source": source,
                        "page": page,
                        "section": span.label,
                        "section_kind": span.kind,
                    },
                )
            )
    return parents


def structure_aware_parent_child(
    page_documents: list[Document],
    child_chunk_size: int = 280,
    child_overlap: int = 50,
    parent_min_chars: int = 350,
    parent_max_chars: int = 2000,
) -> tuple[list[Document], list[Document]]:
    """
    Returns (parents, children) for hierarchical retrieval.
    Parents follow policy structure; children are fine-grained search units.
    """
    parents: list[Document] = []
    for doc in page_documents:
        page = doc.metadata.get("page", 0)
        source = doc.metadata.get("source", "")
        parents.extend(
            _segment_page(
                doc.page_content,
                page=page,
                source=source,
                parent_min=parent_min_chars,
                parent_max=parent_max_chars,
            )
        )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )
    children: list[Document] = []
    for i, parent in enumerate(parents):
        parent_id = f"parent_{i}"
        for child in child_splitter.split_documents([parent]):
            child.metadata = {**parent.metadata, "parent_id": parent_id}
            children.append(child)
    return parents, children


def recursive_parent_child(
    page_documents: list[Document],
    parent_chunk_size: int = 1500,
    parent_overlap: int = 200,
    child_chunk_size: int = 300,
    child_overlap: int = 50,
) -> tuple[list[Document], list[Document]]:
    """Baseline: recursive splitting (previous approach)."""
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size, chunk_overlap=parent_overlap
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size, chunk_overlap=child_overlap
    )
    parents = parent_splitter.split_documents(page_documents)
    children: list[Document] = []
    for i, parent in enumerate(parents):
        parent_id = f"parent_{i}"
        for child in child_splitter.split_documents([parent]):
            child.metadata = {**parent.metadata, "parent_id": parent_id}
            children.append(child)
    return parents, children
