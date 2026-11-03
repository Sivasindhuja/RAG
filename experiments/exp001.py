"""EXP-001: Fixed-character chunking baseline for the ISRO SATCOM NGP policy RAG.

Experiment configuration (approved):
    - Chunking:      naive fixed-character sliding window
    - Chunk size:    1000 characters
    - Overlap:       200 characters
    - Step size:     800 characters
    - Embedding:     sentence-transformers/all-MiniLM-L6-v2
    - Vector DB:      Chroma (fresh, isolated collection)
    - Retrieval:      dense vector similarity only, top-K = 5
    - Metrics:        Recall@5, Precision@5, MRR@5

This module is isolated from the production backend and from the other
experimental pipelines (experiments/rag.py, src/restart.py). It only implements
the EXP-001 baseline and does not run generation.

Implementation notes:
    - Embeddings use SentenceTransformer directly for both documents and queries.
    - Chroma persistence + retrieval use the raw chromadb client (isolated
      persist directory and collection name) so EXP-001 never touches other
      project collections.

Ground truth correction status (EXP-001 v3):
    - The original EXP-001 run (experiments/results/exp001/) is INVALIDATED: the
      evaluation dataset did not match the approved question set and the ground
      truth used heading-only spans instead of the actual supporting passages.
    - The corrected run (experiments/results/exp001_v2/) used the approved
      15-question dataset (experiments/exp001_questions.csv) and the verified
      evidence manifest (experiments/exp001_ground_truth.json).
    - The final corrected run (experiments/results/exp001_v3/) applies the last
      round of ground-truth corrections:
        * Q13 (domestic satellite communication service categories): the evidence
          spans for clauses 4.3.2 and 4.3.3 now start at the actual category
          heading so that the category names are supported by the source text.
        * Q09 (SCC location): expected answer updated to include both the
          permitted locations (within India / authorised by CAISS).
        * Q10 (orbit-spectrum conflicts): expected answer updated to capture the
          complete first-come-first-served procedure from Article 3.6.8.
    - A retrieved chunk is relevant only when it overlaps an actual supporting
      evidence span (character-level overlap on the same source page).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = ROOT / "data" / "satcom-ngp.pdf"
CHUNK_SIZE = 1000
OVERLAP = 200
STEP = 800
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "satcom_ngp_exp001_fixed_char"
PERSIST_DIR = ROOT / "chroma_db_exp001"
TOP_K = 5
RESULTS_DIR = ROOT / "experiments" / "results" / "exp001"
RESULTS_DIR_V2 = ROOT / "experiments" / "results" / "exp001_v2"
RESULTS_DIR_V3 = ROOT / "experiments" / "results" / "exp001_v3"
GROUND_TRUTH_VERIFICATION_MD = RESULTS_DIR_V3 / "ground_truth_verification.md"
QUESTIONS_CSV = ROOT / "experiments" / "exp001_questions.csv"
GOLDEN_TRUTH_PATH = ROOT / "experiments" / "exp001_ground_truth.json"

EXPERIMENT_ID = "EXP-001"

ARTICLE_RE = re.compile(r"ARTICLE[-\s]*\d+\s*:", re.IGNORECASE)
SECTION_RE = re.compile(r"(?:^|\n)\s*(\d+\.\d+(?:\.\d+)?)\s+([A-Z])", re.MULTILINE)


class _SentenceEmbeddingFunction:
    """chromadb EmbeddingFunction wrapper around SentenceTransformer."""

    def __init__(self, model: SentenceTransformer):
        self.model = model

    def __call__(self, input: Any) -> list:  # noqa: A002
        return self.model.encode(list(input))


@dataclass
class PageText:
    source: str
    page: int
    text: str


@dataclass
class Chunk:
    chunk_id: str
    source: str
    page: int
    start: int
    end: int
    text: str
    chunk_index: int
    article: Optional[str] = None
    clause: Optional[str] = None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "page": int(self.page),
            "start": int(self.start),
            "end": int(self.end),
            "chunk_index": int(self.chunk_index),
            "article": self.article or "",
            "clause": self.clause or "",
            "chunk_id": self.chunk_id,
        }


@dataclass
class EvidenceUnit:
    """A verified character-level ground-truth span for one question."""

    question_id: str
    evidence_id: str
    source: str
    page: int
    clause_id: str
    start_char: int
    end_char: int
    exact_supporting_excerpt: str
    status: str = "valid"

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "evidence_id": self.evidence_id,
            "source": self.source,
            "page": int(self.page),
            "clause_id": self.clause_id,
            "start_char": int(self.start_char),
            "end_char": int(self.end_char),
            "exact_supporting_excerpt": self.exact_supporting_excerpt,
            "status": self.status,
        }


def load_pages(pdf_path: Path = PDF_PATH) -> list[PageText]:
    reader = PdfReader(str(pdf_path))
    pages: list[PageText] = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        pages.append(PageText(source=pdf_path.name, page=i, text=text))
    return pages


def _heading_spans(text: str) -> list[tuple[str, str, int, int]]:
    spans: list[tuple[str, str, int, int]] = []
    for m in ARTICLE_RE.finditer(text):
        spans.append(("article", m.group().strip().rstrip(":").strip(), m.start(), m.end()))
    for m in SECTION_RE.finditer(text):
        spans.append(("section", m.group(1), m.start(), m.end()))
    return spans


def _heading_in(text: str, start: int, end: int) -> tuple[Optional[str], Optional[str]]:
    article: Optional[str] = None
    clause: Optional[str] = None
    for kind, ident, s, e in _heading_spans(text):
        if start <= s < end:
            if kind == "article" and article is None:
                article = ident
            elif kind == "section" and clause is None:
                clause = ident
    return article, clause


def make_chunks(
    pages: list[PageText],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP,
    step: int = STEP,
) -> list[Chunk]:
    """Naive fixed-character sliding-window splitter (per page, no cross-page chunks)."""
    chunks: list[Chunk] = []
    for page in pages:
        text = page.text
        n = len(text)
        if n <= 0:
            continue
        start = 0
        idx = 0
        while start < n:
            end = min(start + chunk_size, n)
            chunk_text = text[start:end]
            if chunk_text.strip():
                article, clause = _heading_in(text, start, end)
                chunks.append(
                    Chunk(
                        chunk_id=f"{Path(page.source).stem}_page_{page.page}_chunk_{idx}",
                        source=page.source,
                        page=page.page,
                        start=start,
                        end=end,
                        text=chunk_text,
                        chunk_index=idx,
                        article=article,
                        clause=clause,
                    )
                )
            if end >= n:
                break
            start = end - overlap
            idx += 1
    return chunks


def build_clause_spans(pages: list[PageText]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for page in pages:
        for kind, ident, s, e in _heading_spans(page.text):
            spans.append(
                {
                    "clause_id": ident,
                    "source": page.source,
                    "page": page.page,
                    "start": int(s),
                    "end": int(e),
                    "kind": kind,
                }
            )
    return spans


def chunk_overlaps(chunk: Chunk, clause: dict[str, Any]) -> bool:
    """True when the chunk's source span overlaps the ground-truth clause span."""
    return (
        chunk.source == clause.get("source")
        and chunk.page == int(clause.get("page") or 0)
        and int(chunk.end) > int(clause.get("start") or 0)
        and int(chunk.start) < int(clause.get("end") or 0)
    )


# --------------------------------------------------------------------------
# Ground-truth evidence handling (EXP-001 v2)
# --------------------------------------------------------------------------


def validate_evidence_offset(
    page_text: str,
    start_char: int,
    end_char: int,
    exact_supporting_excerpt: str,
) -> bool:
    """Exact source-offset validation: page_text[start_char:end_char] must equal the excerpt."""
    if start_char < 0 or end_char > len(page_text) or start_char > end_char:
        return False
    return page_text[start_char:end_char] == exact_supporting_excerpt


def _unit_value(unit: Any, *keys: str, default: Any = None) -> Any:
    """Read a field from a dict-like or attribute-carrying unit."""
    if isinstance(unit, dict):
        for k in keys:
            if k in unit and unit[k] is not None:
                return unit[k]
        return default
    for k in keys:
        v = getattr(unit, k, None)
        if v is not None:
            return v
    return default


def _unit_key(unit: Any) -> str:
    return str(_unit_value(unit, "evidence_id", "clause_id") or "")


def _unit_source(unit: Any) -> str:
    return str(_unit_value(unit, "source") or "")


def _unit_page(unit: Any) -> int:
    return int(_unit_value(unit, "page") or 0)


def _unit_start(unit: Any) -> int:
    return int(_unit_value(unit, "start_char", "start") or 0)


def _unit_end(unit: Any) -> int:
    return int(_unit_value(unit, "end_char", "end") or 0)


def _unit_overlaps(unit: Any, chunk: Chunk) -> bool:
    """True when a retrieved chunk overlaps an actual supporting evidence span."""
    return (
        _unit_source(unit) == chunk.source
        and _unit_page(unit) == int(chunk.page)
        and int(chunk.end) > _unit_start(unit)
        and int(chunk.start) < _unit_end(unit)
    )


def _unit_overlap_chars(unit: Any, chunk: Chunk) -> int:
    """Number of overlapping characters between a retrieved chunk and an evidence span."""
    if not _unit_overlaps(unit, chunk):
        return 0
    return min(int(chunk.end), _unit_end(unit)) - max(int(chunk.start), _unit_start(unit))


def load_ground_truth(path: Path = GOLDEN_TRUTH_PATH) -> list[EvidenceUnit]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    raw_entries = payload.get("entries", payload) if isinstance(payload, dict) else payload
    units: list[EvidenceUnit] = []
    for e in raw_entries:
        units.append(
            EvidenceUnit(
                question_id=str(e.get("question_id") or ""),
                evidence_id=str(e.get("evidence_id") or ""),
                source=str(e.get("source") or ""),
                page=int(e.get("page") or 0),
                clause_id=str(e.get("clause_id") or ""),
                start_char=int(e.get("start_char") or 0),
                end_char=int(e.get("end_char") or 0),
                exact_supporting_excerpt=str(e.get("exact_supporting_excerpt") or ""),
                status=str(e.get("status") or "valid"),
            )
        )
    return units


def validate_ground_truth(
    units: list[EvidenceUnit],
    pages: list[PageText],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Validate every evidence unit against the exact extracted page text.

    Returns (valid_units_by_question, problems). A problem is recorded when an
    offset/excerpt pair does not verify or when a page cannot be resolved.
    """
    page_texts = {(p.source, p.page): p.text for p in pages}
    valid_by_question: dict[str, list[dict[str, Any]]] = {}
    problems: list[dict[str, Any]] = []
    for u in units:
        text = page_texts.get((u.source, int(u.page)))
        ok = text is not None and validate_evidence_offset(
            text, int(u.start_char), int(u.end_char), u.exact_supporting_excerpt
        )
        status = "valid" if ok else "invalid"
        entry = u.to_dict()
        entry["status"] = status
        if not ok:
            problems.append(
                {
                    "question_id": u.question_id,
                    "evidence_id": u.evidence_id,
                    "clause_id": u.clause_id,
                    "source": u.source,
                    "page": u.page,
                    "start_char": u.start_char,
                    "end_char": u.end_char,
                    "exact_supporting_excerpt": u.exact_supporting_excerpt,
                    "status": "invalid",
                    "reason": "source_offset_mismatch",
                }
            )
        else:
            valid_by_question.setdefault(u.question_id, []).append(entry)
    return valid_by_question, problems


# --------------------------------------------------------------------------
# Verified ground-truth specification for the approved EXP-001 question set
# --------------------------------------------------------------------------

GROUND_TRUTH_SPEC: dict[str, dict[str, Any]] = {
    # Q01 -> Article 1 opening paragraph (unnumbered paragraph).
    "EXP001-Q01": {
        "clause_id": "ARTICLE-1",
        "page": 1,
        "start_anchor": "Government of India has decided that the Satellite Communications",
        "end_anchor": "be accorded preferential treatment.",
    },
    # Q02 -> Article 2.3.1.
    "EXP001-Q02": {
        "clause_id": "2.3.1",
        "page": 2,
        "start_anchor": "As a baseline making the INSAT capacity",
        "end_anchor": "sectors.",
    },
    # Q03 -> Article 2.3.2.
    "EXP001-Q03": {
        "clause_id": "2.3.2",
        "page": 2,
        "start_anchor": "All the policies regarding INSAT syst",
        "end_anchor": "Communications in India. ",
    },
    # Q04 -> Article 2.4.1.
    "EXP001-Q04": {
        "clause_id": "2.4.1",
        "page": 3,
        "start_anchor": "The user sectors can be broadly classified as",
        "end_anchor": "Defence Ministry/Services ",
    },
    # Q05 -> Article 2.5.6.
    "EXP001-Q05": {
        "clause_id": "2.5.6",
        "page": 3,
        "start_anchor": "The responsibility for obtaining the necessary licenses",
        "end_anchor": "taken the capacity on lease. ",
    },
    # Q06 -> Article 2.6.5.
    "EXP001-Q06": {
        "clause_id": "2.6.5",
        "page": 4,
        "start_anchor": "The use of INSAT capacity by non-govern",
        "end_anchor": "management clauses. ",
    },
    # Q07 -> Article 3.2.
    "EXP001-Q07": {
        "clause_id": "3.2",
        "page": 5,
        "start_anchor": "Indian Satellites” shall be defined as",
        "end_anchor": "Indian Space Objects.  ",
    },
    # Q08 -> Articles 3.4.1, 3.4.2 and 3.4.3 (one evidence unit each).
    "EXP001-Q08": {
        "clauses": [
            {"clause_id": "3.4.1", "page": 5, "start_anchor": "Authorisation from Department of Space to own and operate an Indian registered", "end_anchor": "spacecraft control centre. "},
            {"clause_id": "3.4.2", "page": 6, "start_anchor": "Authorisation by the Wi reless Planning and Coordination Wing (WPC) of the", "end_anchor": "extant ITU Radio Regulations. "},
            {"clause_id": "3.4.3", "page": 6, "start_anchor": "Operating licenses for the se rvices to be provided by the system/network.", "end_anchor": "Ministry of Information and Broadcasting. "},
        ],
    },
    # Q09 -> Article 3.6.4.
    "EXP001-Q09": {
        "clause_id": "3.6.4",
        "page": 7,
        "start_anchor": "The Satellite Control Centre/s (SCC), i. e., the facility that monitors correct",
        "end_anchor": "technical requirements. ",
    },
    # Q10 -> Article 3.6.8 (references 3.6.7(a) and 3.6.7(b) as part of the procedure).
    "EXP001-Q10": {
        "clause_id": "3.6.8",
        "page": 8,
        "start_anchor": "For 3.6.7(a) and 3.6.7(b) other conditions  being fulfilled, authorisation and",
        "end_anchor": "coordination. ",
    },
    # Q11 -> Article 3.7.1.5.
    "EXP001-Q11": {
        "clause_id": "3.7.1.5",
        "page": 9,
        "start_anchor": "Evidence that the Applicant has current financial ability to meet the cost of",
        "end_anchor": "estimated revenues shall also be provided. ",
    },
    # Q12 -> Article 4.2.
    "EXP001-Q12": {
        "clause_id": "4.2",
        "page": 13,
        "start_anchor": "In so far as International communication is concerned the existing practice of",
        "end_anchor": "operations with foreign satellites. ",
    },
    # Q13 -> Articles 4.3.1, 4.3.2, 4.3.3 and 4.3.4 (one evidence unit each).
    # Evidence spans start at the category heading so that the category names
    # that answer the question are supported by the source text itself. For
    # 4.3.3 the heading ("4.3.3 Value Added Services (domestic)") ends page 13
    # while its explanatory paragraph starts page 14; the heading-anchored span
    # is used for that evidence unit.
    "EXP001-Q13": {
        "clauses": [
            {"clause_id": "4.3.1", "page": 13, "start_anchor": "Broadcasting (moving images (TV) a nd associated sound; and audio (radio))", "end_anchor": "of this category. "},
            {"clause_id": "4.3.2", "page": 13, "start_anchor": "4.3.2 Public Switched Telephone Network (domestic) ", "end_anchor": "ISDN. "},
            {"clause_id": "4.3.3", "page": 13, "start_anchor": "4.3.3 Value Added Services (domestic) ", "end_anchor": "4.3.3 Value Added Services (domestic) "},
            {"clause_id": "4.3.4", "page": 14, "start_anchor": "Land Mobile Satellite Services (Domestic and Coastal Waters): ", "end_anchor": "small terminals. "},
        ],
    },
    # Q14 -> Article 4.4.3.
    "EXP001-Q14": {
        "clause_id": "4.4.3",
        "page": 14,
        "start_anchor": "For services listed from 4.4.2 to 4.4.4 wherever capacity and capability exists in",
        "end_anchor": "before renewal. ",
    },
    # Q15 -> Article 4.4.6.
    "EXP001-Q15": {
        "clause_id": "4.4.6",
        "page": 15,
        "start_anchor": "The process of giving operating license  must ensure that the system is",
        "end_anchor": "concerns are fully addressed. ",
    },
}


def get_relevant_evidence(
    question_id: str,
    valid_by_question: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Return the validated evidence units for a question; raise when missing/invalid."""
    relevant = sorted(
        valid_by_question.get(question_id, []),
        key=lambda u: str(u.get("evidence_id") or ""),
    )
    if not relevant:
        raise RuntimeError(
            f"Missing or invalid ground-truth evidence mapping for question '{question_id}'. "
            "REFUSING to calculate retrieval metrics."
        )
    return relevant


def generate_ground_truth_entries(pdf_path: Path = PDF_PATH) -> list[dict[str, Any]]:
    """Build verified ground-truth evidence entries from the approved question set."""
    pages = load_pages(pdf_path)
    page_texts = {(p.source, p.page): p.text for p in pages}
    entries: list[dict[str, Any]] = []
    for qid in sorted(GROUND_TRUTH_SPEC, key=lambda q: int(re.sub(r"\D", "", q) or 0)):
        spec = GROUND_TRUTH_SPEC[qid]
        if "clauses" in spec:
            clause_specs = spec["clauses"]
        else:
            clause_specs = [spec]
        for idx, cspec in enumerate(clause_specs, start=1):
            text = page_texts.get((pdf_path.name, int(cspec["page"])))
            if text is None:
                raise RuntimeError(
                    f"Ground-truth generation failed: no extracted page for {pdf_path.name} page {cspec['page']} ({qid}, {cspec['clause_id']})."
                )
            start = text.find(str(cspec["start_anchor"]))
            end = text.find(str(cspec["end_anchor"]), start)
            if start < 0 or end < 0:
                raise RuntimeError(
                    f"Ground-truth generation failed: anchor not found for {qid} {cspec['clause_id']}: "
                    f"{str(cspec['start_anchor'])!r} / {str(cspec['end_anchor'])!r}"
                )
            end += len(str(cspec["end_anchor"]))
            entries.append(
                {
                    "question_id": qid,
                    "evidence_id": f"{qid}-E{idx:02d}",
                    "source": pdf_path.name,
                    "page": int(cspec["page"]),
                    "clause_id": str(cspec["clause_id"]),
                    "start_char": int(start),
                    "end_char": int(end),
                    "exact_supporting_excerpt": text[start:end],
                    "status": "valid",
                }
            )
    return entries


def generate_ground_truth_manifest(entries: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "version": "3",
        "source": "data/satcom-ngp.pdf",
        "disclaimer": (
            "Verified character-level evidence spans for EXP-001. Offsets refer to the "
            "pypdf-extracted page text used by the experiment. Headings are included as "
            "evidence where they identify categories or obligations; only source-extracted "
            "text is covered."
        ),
        "entries": entries,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


_GROUND_TRUTH_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with", "as", "by",
    "is", "are", "was", "were", "be", "been", "being", "that", "this", "these", "those",
    "it", "its", "at", "from", "may", "will", "shall", "not", "only", "such", "under",
}


def _answer_keywords(answer: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", str(answer or "").lower())
    return [t for t in tokens if t not in _GROUND_TRUTH_STOPWORDS]


def _excerpt_supports_answer(unit: dict[str, Any], question: dict[str, Any]) -> bool:
    """Heuristic check: does the supporting excerpt contain answer key terms?"""
    excerpt = str(unit.get("exact_supporting_excerpt") or "").lower()
    if not excerpt.strip():
        return False
    keywords = _answer_keywords(str(question.get("expected_answer") or ""))
    return any(k in excerpt for k in keywords)


def generate_ground_truth_verification_markdown(
    questions: list[dict[str, Any]],
    valid_by_question: dict[str, list[dict[str, Any]]],
    problems: list[dict[str, Any]],
    version: str = "3",
) -> str:
    lines: list[str] = []
    lines.append(f"# EXP-001 Ground-Truth Verification Report (v{version})")
    lines.append("")
    lines.append("## 0. Scope and status")
    lines.append("")
    lines.append("- Experiment: **EXP-001** (fixed-character chunking baseline; corrected run).")
    lines.append("- Source document: `data/satcom-ngp.pdf`.")
    lines.append("- Verified dataset: `experiments/exp001_questions.csv` (approved 15-question set, `EXP001-Q01`-`EXP001-Q15`).")
    lines.append("- Ground-truth manifest: `experiments/exp001_ground_truth.json`.")
    lines.append("- Verification method: exact character-offset check `page_text[start_char:end_char] == exact_supporting_excerpt`.")
    lines.append("- Headings are included as evidence where they identify categories or obligations; supporting passage content is verified where applicable.")
    lines.append("")
    if problems:
        lines.append("## 0.1 Verification outcome")
        lines.append("")
        lines.append(f"**FAIL**: {len(problems)} ground-truth mapping entr(ies) could not be verified against the extracted PDF text.")
        lines.append("")
        lines.append("The corrected retrieval evaluation MUST NOT be run until the issues below are resolved.")
        lines.append("")
        lines.append("### Problems")
        lines.append("")
        for p in problems:
            lines.append(
                f"- `{p.get('evidence_id')}` (question `{p.get('question_id')}`, clause `{p.get('clause_id')}`) on page `{p.get('page')}`: "
                f"source offset verification failed (`{p.get('reason')}`)."
            )
        lines.append("")
    else:
        lines.append("**PASS**: all generated ground-truth evidence entries verified exactly against the extracted page text.")
        lines.append("")

    lines.append("## 0.2 Ground-truth corrections applied in this verification round")
    lines.append("")
    lines.append("The following corrections were applied to the EXP-001 ground truth before this verification round:")
    lines.append("")
    lines.append("- **Q09 (Satellite Control Centre location):** the expected answer was corrected to include both permitted locations: within the territory of India, or at another location authorised by CAISS based on technical requirements. The verified evidence unit (Article 3.6.4) already covered both possibilities.")
    lines.append("- **Q10 (multiple applicants / same orbit-spectrum):** the expected answer was corrected to capture the complete procedure from Article 3.6.8: first-come, first-served on fulfilment of the other conditions, priority to government-owned systems in case of conflict, discussions between non-governmental applicants and the Administration, and, if unresolved, filing of a common API with the decision taken at the AP S4 filing stage or later depending on progress.")
    lines.append("- **Q13 (domestic satellite communication service categories):** the evidence spans for clauses 4.3.2 and 4.3.3 were corrected to start at the actual category heading (`4.3.2 Public Switched Telephone Network (domestic)` and `4.3.3 Value Added Services (domestic)`) so that the category names are supported by the source. For 4.3.3 the heading ends page 13 while the explanatory paragraph starts page 14; the heading-anchored span is used.")
    lines.append("")
    lines.append("## 1. Summary")
    lines.append("")
    lines.append("| Question | Expected answer | Evidence clause(s) | Source page(s) | Status | Excerpt supports expected answer |")
    lines.append("|---|---|---|---|---|---|")
    for q in questions:
        qid = q["question_id"]
        units = sorted(valid_by_question.get(qid, []), key=lambda u: str(u.get("evidence_id") or ""))
        clauses = [u.get("clause_id") for u in units]
        pages = sorted({int(u.get("page") or 0) for u in units})
        status = "VALID" if not units or all(str(u.get("status")) == "valid" for u in units) else "INVALID"
        supported_flags = [_excerpt_supports_answer(u, q) for u in units]
        supported = all(supported_flags) if supported_flags else False
        lines.append(
            f"| {qid} | {q.get('question')} | {', '.join(clauses)} | {', '.join(str(p) for p in pages)} | {status} | {str(supported)} |"
        )
    lines.append("")

    for q in questions:
        qid = q["question_id"]
        lines.append(f"## Question {qid}")
        lines.append("")
        lines.append(f"**Question:** {q.get('question')}")
        lines.append("")
        lines.append(f"**Expected answer:** {q.get('expected_answer')}")
        lines.append("")
        units = sorted(valid_by_question.get(qid, []), key=lambda u: str(u.get("evidence_id") or ""))
        if not units:
            lines.append("**Verification status:** NOT VERIFIED (no valid evidence unit).")
            lines.append("")
            continue
        for u in units:
            excerpt = str(u.get("exact_supporting_excerpt") or "")
            lines.append(f"### Evidence unit {u.get('evidence_id')}")
            lines.append("")
            lines.append(f"- **Question ID:** {u.get('question_id')}")
            lines.append(f"- **Evidence ID:** {u.get('evidence_id')}")
            lines.append(f"- **Source:** {u.get('source')}")
            lines.append(f"- **Source page:** {u.get('page')}")
            lines.append(f"- **Clause ID:** {u.get('clause_id')}")
            lines.append(f"- **Start char:** {u.get('start_char')}")
            lines.append(f"- **End char:** {u.get('end_char')}")
            lines.append(f"- **Verification status:** {u.get('status')}")
            lines.append("- **Exact supporting excerpt:**")
            lines.append("")
            lines.append("```text")
            lines.append(excerpt)
            lines.append("```")
            lines.append("")
            lines.append(f"- **Excerpt supports the expected answer:** {'Yes' if _excerpt_supports_answer(u, q) else 'No'}")
            lines.append("")

    lines.append("## 2. Conclusion")
    lines.append("")
    if not problems:
        lines.append("All approved questions for EXP-001 passed exact source-offset verification against `data/satcom-ngp.pdf`.")
        lines.append("The corrected retrieval evaluation may proceed. Final corrected metrics will be reported only after the corrected run completes and is inspected.")
    else:
        lines.append("Ground-truth verification FAILED. The corrected retrieval evaluation must not be run.")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def compute_metrics(
    retrieved: list[Chunk],
    relevant: list[Any],
    k: int = TOP_K,
) -> dict[str, Any]:
    """Recall@5, Precision@5, MRR@5 for the fixed-character baseline.

    Relevance is evidence-span based: a retrieved chunk is relevant only when it
    overlaps an actual supporting evidence span for the question.

    - Recall@5: distinct relevant evidence units retrieved in the top `k`
      divided by the total relevant evidence units for the question.
    - Precision@5: relevant retrieved chunks in the top `k` divided by `k`.
    - MRR@5: reciprocal rank of the first retrieved chunk overlapping any valid
      evidence span; zero when none is retrieved.
    """
    top = retrieved[:k]
    flags: list[bool] = []
    overlap_chars: list[int] = []
    evidence_overlaps: list[dict[str, Any]] = []
    covered: set[str] = set()
    for i, c in enumerate(top):
        overlapping = [u for u in relevant if _unit_overlaps(u, c)]
        chars = sum(_unit_overlap_chars(u, c) for u in overlapping)
        is_relevant = len(overlapping) > 0
        flags.append(is_relevant)
        overlap_chars.append(int(chars))
        if is_relevant:
            evidence_overlaps.append(
                {
                    "rank": i + 1,
                    "chunk_id": c.chunk_id,
                    "evidence_ids": [_unit_key(u) for u in overlapping],
                    "overlap_chars": int(chars),
                }
            )
        for u in overlapping:
            covered.add(_unit_key(u))
    recall = len(covered) / len(relevant) if relevant else 0.0
    precision = sum(1 for f in flags if f) / k if k > 0 else 0.0
    mrr = 0.0
    for i, f in enumerate(flags):
        if f:
            mrr = 1.0 / (i + 1)
            break
    return {
        "recall@5": round(recall, 4),
        "precision@5": round(precision, 4),
        "mrr@5": round(mrr, 4),
        "retrieved_count": len(top),
        "relevant_flags": flags,
        "covered_clause_ids": sorted(covered),
        "relevant_clause_ids": [_unit_key(u) for u in relevant],
        "overlap_chars": overlap_chars,
        "evidence_overlaps": evidence_overlaps,
    }


# --------------------------------------------------------------------------
# Vector store / retrieval
# --------------------------------------------------------------------------


def build_vectorstore(chunks: list[Chunk], model: Optional[SentenceTransformer] = None):
    if model is None:
        model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_SentenceEmbeddingFunction(model),
    )
    if collection.count() > 0:
        existing = collection.get()
        if existing and existing.get("ids"):
            collection.delete(ids=list(existing["ids"]))
    embeddings = [model.encode(c.text) for c in chunks]
    collection.add(
        ids=[c.chunk_id for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[c.to_metadata() for c in chunks],
        embeddings=embeddings,
    )
    return collection


def dense_retrieve(
    collection,
    query: str,
    k: int = TOP_K,
    chunks: Optional[list[Chunk]] = None,
    model: Optional[SentenceTransformer] = None,
) -> list[dict[str, Any]]:
    if model is None:
        model = SentenceTransformer(EMBEDDING_MODEL)
    emb = model.encode([query])
    emb = emb[0] if getattr(emb, "shape", None) is not None and len(emb.shape) > 1 else emb
    emb_list = emb.tolist() if hasattr(emb, "tolist") else list(emb)
    res = collection.query(
        query_embeddings=[emb_list],
        n_results=k,
    )
    retrieved: list[dict[str, Any]] = []
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    chunk_by_id: dict[str, Chunk] = {c.chunk_id: c for c in (chunks or [])}
    for i in range(len(ids)):
        cid = ids[i]
        retrieved.append(
            {
                "rank": i + 1,
                "chunk_id": cid,
                "text": docs[i],
                "metadata": metas[i] if metas else {},
                "chunk": chunk_by_id.get(cid),
            }
        )
    return retrieved


# --------------------------------------------------------------------------
# Corrected experiment run
# --------------------------------------------------------------------------


def load_questions(csv_path: Path = QUESTIONS_CSV) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            clause_ids_raw = (r.get("relevant_clause_ids") or "").strip()
            pages_raw = (r.get("relevant_source_pages") or "").strip()
            rows.append(
                {
                    "question_id": (r.get("question_id") or "").strip(),
                    "question": (r.get("question") or "").strip(),
                    "expected_answer": (r.get("expected_answer") or "").strip(),
                    "relevant_clause_ids": [x.strip() for x in clause_ids_raw.split(";") if x.strip()],
                    "relevant_source_pages": [int(x) for x in pages_raw.split(";") if x.strip().isdigit()],
                }
            )
    return rows


def run(
    pdf_path: Path = PDF_PATH,
    questions_csv: Path = QUESTIONS_CSV,
    ground_truth_path: Path = GOLDEN_TRUTH_PATH,
    results_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Run the corrected EXP-001 retrieval evaluation against verified evidence."""
    results_dir = Path(results_dir) if results_dir is not None else RESULTS_DIR_V3

    pages = load_pages(pdf_path)
    chunks = make_chunks(pages)
    questions = load_questions(questions_csv)
    units = load_ground_truth(ground_truth_path)
    valid_by_question, problems = validate_ground_truth(units, pages)

    verification_md = generate_ground_truth_verification_markdown(questions, valid_by_question, problems)

    if problems:
        raise RuntimeError(
            f"Ground-truth verification failed: {len(problems)} evidence mapping(s) could not be "
            f"verified against the extracted source text. REFUSING to run retrieval evaluation / "
            f"calculate corrected metrics. First problem: {problems[0]}."
        )

    collection = build_vectorstore(chunks, model=SentenceTransformer(EMBEDDING_MODEL))
    model = SentenceTransformer(EMBEDDING_MODEL)

    retrieval_results: list[dict[str, Any]] = []
    per_question_metrics: list[dict[str, Any]] = []
    dataset_questions: list[dict[str, Any]] = []

    for q in questions:
        qid = q["question_id"]
        relevant = get_relevant_evidence(qid, valid_by_question)
        if not relevant:
            raise RuntimeError(
                f"Ground-truth validation error: no valid evidence unit for question {qid}. "
                f"Problems: {problems}. REFUSING to compute retrieval metrics."
            )

        retrieved = dense_retrieve(collection, q["question"], TOP_K, chunks=chunks, model=model)
        retrieved_chunks: list[dict[str, Any]] = []
        retrieved_objs: list[Chunk] = []
        for r in retrieved:
            chunk = r.get("chunk")
            overlaps = [u for u in relevant if _unit_overlaps(u, chunk)] if chunk is not None else []
            retrieved_chunks.append(
                {
                    "rank": r["rank"],
                    "chunk_id": r["chunk_id"],
                    "text": r["text"],
                    "metadata": r["metadata"],
                    "relevant": len(overlaps) > 0,
                    "overlap_chars": sum(_unit_overlap_chars(u, chunk) for u in overlaps) if chunk is not None else 0,
                    "evidence_ids": [u.get("evidence_id") for u in overlaps],
                }
            )
            if chunk is not None:
                retrieved_objs.append(chunk)

        metrics = compute_metrics(retrieved_objs, relevant, TOP_K)
        retrieval_results.append(
            {
                "question_id": qid,
                "question": q["question"],
                "expected_answer": q["expected_answer"],
                "relevant_clause_ids": q["relevant_clause_ids"],
                "relevant_source_pages": q["relevant_source_pages"],
                "retrieved_chunks": retrieved_chunks,
                "relevant_flags": metrics["relevant_flags"],
                "recall@5": metrics["recall@5"],
                "precision@5": metrics["precision@5"],
                "mrr@5": metrics["mrr@5"],
                "covered_evidence_ids": metrics["covered_clause_ids"],
                "relevant_evidence_ids": metrics["relevant_clause_ids"],
                "overlap_chars": metrics["overlap_chars"],
                "evidence_overlaps": metrics["evidence_overlaps"],
            }
        )
        per_question_metrics.append(
            {
                "question_id": qid,
                "recall@5": metrics["recall@5"],
                "precision@5": metrics["precision@5"],
                "mrr@5": metrics["mrr@5"],
            }
        )
        dataset_questions.append(
            {
                "question_id": qid,
                "question": q["question"],
                "expected_answer": q["expected_answer"],
                "relevant_clause_ids": q["relevant_clause_ids"],
                "relevant_source_pages": q["relevant_source_pages"],
            }
        )

    aggregate = {
        "recall@5": round(sum(p["recall@5"] for p in per_question_metrics) / len(per_question_metrics) if per_question_metrics else 0.0, 4),
        "precision@5": round(sum(p["precision@5"] for p in per_question_metrics) / len(per_question_metrics) if per_question_metrics else 0.0, 4),
        "mrr@5": round(sum(p["mrr@5"] for p in per_question_metrics) / len(per_question_metrics) if per_question_metrics else 0.0, 4),
    }

    config = {
        "experiment_id": EXPERIMENT_ID,
        "chunking": {
            "strategy": "naive_fixed_character_sliding_window",
            "chunk_size": CHUNK_SIZE,
            "overlap": OVERLAP,
            "step": STEP,
        },
        "embedding_model": EMBEDDING_MODEL,
        "vector_db": "chroma",
        "collection_name": COLLECTION_NAME,
        "persist_directory": str(PERSIST_DIR),
        "retrieval": {"method": "dense_vector_similarity", "top_k": TOP_K},
        "metrics": ["Recall@5", "Precision@5", "MRR@5"],
        "pdf_path": str(pdf_path),
        "questions_csv": str(questions_csv),
        "ground_truth": {
            "path": str(ground_truth_path),
            "manifest_version": "3",
            "status": "verified" if not problems else "invalid",
        },
    }

    results = {
        "experiment_id": EXPERIMENT_ID,
        "config": config,
        "embedding_model": {"name": EMBEDDING_MODEL},
        "evaluation_dataset": {
            "path": str(questions_csv),
            "question_count": len(questions),
            "question_ids": [q["question_id"] for q in questions],
            "status": "approved_15_question_set_v3",
        },
        "ground_truth": {
            "path": str(ground_truth_path),
            "status": "verified" if not problems else "invalid",
            "entries": [u.to_dict() for u in units],
            "valid_by_question": {k: sorted(v, key=lambda u: int(u.get("start_char") or 0)) for k, v in valid_by_question.items()},
        },
        "retrieval_results": retrieval_results,
        "per_question_metrics": per_question_metrics,
        "aggregate_metrics": aggregate,
    }

    verification_md = generate_ground_truth_verification_markdown(questions, valid_by_question, problems, version="3")

    # ---- write corrected outputs under experiments/results/<run>/ ----
    results_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        results_dir / "dataset.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "config": config,
            "evaluation_dataset": results["evaluation_dataset"],
            "questions": dataset_questions,
        },
    )
    write_json(
        results_dir / "ground_truth.json",
        generate_ground_truth_manifest([u.to_dict() for u in units]),
    )
    with open(results_dir / "ground_truth_verification.md", "w", encoding="utf-8") as f:
        f.write(verification_md)
    write_json(
        results_dir / "chunks.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "config": config,
            "embedding_model": {"name": EMBEDDING_MODEL},
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "source": c.source,
                    "page": c.page,
                    "start": c.start,
                    "end": c.end,
                    "text": c.text,
                    "chunk_index": c.chunk_index,
                    "article": c.article,
                    "clause": c.clause,
                }
                for c in chunks
            ],
        },
    )
    write_json(results_dir / "retrieval_results.json", results)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP-001 corrected ground-truth experiment runner")
    parser.add_argument("--generate-ground-truth", action="store_true", help="Generate experiments/exp001_ground_truth.json from the verified anchors.")
    parser.add_argument("--pdf", type=Path, default=PDF_PATH, help="Source PDF path.")
    parser.add_argument("--questions-csv", type=Path, default=QUESTIONS_CSV, help="Approved evaluation dataset CSV path.")
    parser.add_argument("--ground-truth", type=Path, default=GOLDEN_TRUTH_PATH, help="Verified ground-truth manifest path.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR_V3, help="Directory for corrected experiment outputs.")
    args = parser.parse_args()

    if args.generate_ground_truth:
        entries = generate_ground_truth_entries(args.pdf)
        write_json(GOLDEN_TRUTH_PATH, generate_ground_truth_manifest(entries))
        print(f"Ground-truth manifest written to {GOLDEN_TRUTH_PATH} with {len(entries)} entries.")
        return

    res = run(
        pdf_path=args.pdf,
        questions_csv=args.questions_csv,
        ground_truth_path=args.ground_truth,
        results_dir=args.results_dir,
    )
    print(json.dumps(res["aggregate_metrics"], indent=2))


if __name__ == "__main__":
    main()
