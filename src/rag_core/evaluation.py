"""Verified evidence loading and retrieval metrics shared by experiments.

The approved EXP-001 ground-truth manifest stores character offsets relative to
the *stripped* text of each PDF page (the representation used by the frozen
EXP-001 runner).  The canonical document core stores unstripped page text and
global document offsets.  This module is the single place that translates
between the two and refuses to proceed when an excerpt does not exactly match
the source.

Metric definitions are intentionally identical to EXP-001:

- Recall@K: distinct required evidence units retrieved in the top K / total.
- Precision@K: relevant retrieved chunks / K.
- MRR@K: reciprocal rank of the first retrieved chunk overlapping any evidence.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .document import SourceDocument
from .index import RetrievedChunk


@dataclass(frozen=True)
class Question:
    question_id: str
    question: str
    expected_answer: str
    relevant_clause_ids: Tuple[str, ...]
    relevant_source_pages: Tuple[int, ...]


@dataclass(frozen=True)
class EvidenceUnit:
    """A verified supporting span in global document coordinates."""

    question_id: str
    evidence_id: str
    source: str
    page: int
    clause_id: str
    start_offset: int
    end_offset: int
    page_relative_start: int
    page_relative_end: int
    excerpt: str
    status: str = "valid"

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "evidence_id": self.evidence_id,
            "source": self.source,
            "page": self.page,
            "clause_id": self.clause_id,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "page_relative_start": self.page_relative_start,
            "page_relative_end": self.page_relative_end,
            "exact_supporting_excerpt": self.excerpt,
            "status": self.status,
        }


def load_questions(csv_path: Path) -> List[Question]:
    questions: List[Question] = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            clause_ids_raw = (row.get("relevant_clause_ids") or "").strip()
            pages_raw = (row.get("relevant_source_pages") or "").strip()
            questions.append(
                Question(
                    question_id=(row.get("question_id") or "").strip(),
                    question=(row.get("question") or "").strip(),
                    expected_answer=(row.get("expected_answer") or "").strip(),
                    relevant_clause_ids=tuple(x.strip() for x in clause_ids_raw.split(";") if x.strip()),
                    relevant_source_pages=tuple(int(x) for x in pages_raw.split(";") if x.strip().isdigit()),
                )
            )
    return questions


def load_exp001_evidence(
    manifest_path: Path,
    document: SourceDocument,
) -> Tuple[List[EvidenceUnit], List[dict[str, Any]]]:
    """Load the approved EXP-001 evidence manifest into global coordinates.

    Returns ``(units, problems)``.  ``problems`` is non-empty when an excerpt
    cannot be reproduced exactly from the canonical document; callers must
    refuse to run metrics in that case.
    """
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    entries = payload.get("entries", payload) if isinstance(payload, dict) else payload
    page_by_number = {page.page_number: page for page in document.pages}

    units: List[EvidenceUnit] = []
    problems: List[dict[str, Any]] = []
    for entry in entries:
        evidence_id = str(entry.get("evidence_id") or "")
        page_number = int(entry.get("page") or 0)
        page = page_by_number.get(page_number)
        if page is None:
            problems.append({"evidence_id": evidence_id, "problem": f"page {page_number} does not exist"})
            continue

        raw = page.text
        stripped = raw.strip()
        leading = len(raw) - len(raw.lstrip())
        page_start = int(entry.get("start_char") or 0)
        page_end = int(entry.get("end_char") or 0)
        excerpt = str(entry.get("exact_supporting_excerpt") or "")
        if not (0 <= page_start <= page_end <= len(stripped)):
            problems.append(
                {
                    "evidence_id": evidence_id,
                    "problem": "page-relative offsets are outside the stripped page text",
                    "page": page_number,
                    "page_start": page_start,
                    "page_end": page_end,
                    "stripped_page_length": len(stripped),
                }
            )
            continue

        global_start = page.start_char + leading + page_start
        global_end = global_start + (page_end - page_start)
        actual = document.text[global_start:global_end]
        if actual != excerpt:
            problems.append(
                {
                    "evidence_id": evidence_id,
                    "problem": "source excerpt mismatch after global offset mapping",
                    "page": page_number,
                    "global_start": global_start,
                    "global_end": global_end,
                    "expected_prefix": excerpt[:120],
                    "actual_prefix": actual[:120],
                }
            )
            continue

        units.append(
            EvidenceUnit(
                question_id=str(entry.get("question_id") or ""),
                evidence_id=evidence_id,
                source=str(entry.get("source") or document.source),
                page=page_number,
                clause_id=str(entry.get("clause_id") or ""),
                start_offset=global_start,
                end_offset=global_end,
                page_relative_start=page_start,
                page_relative_end=page_end,
                excerpt=excerpt,
                status=str(entry.get("status") or "valid"),
            )
        )
    return units, problems


def evidence_for_question(units: Iterable[EvidenceUnit], question_id: str) -> List[EvidenceUnit]:
    return [unit for unit in units if unit.question_id == question_id]


def overlap_chars(unit: EvidenceUnit, retrieved: RetrievedChunk) -> int:
    if unit.source != retrieved.source:
        return 0
    return max(0, min(unit.end_offset, retrieved.end_offset) - max(unit.start_offset, retrieved.start_offset))


def compute_retrieval_metrics(
    retrieved: Sequence[RetrievedChunk],
    evidence: Sequence[EvidenceUnit],
    k: int = 5,
) -> Dict[str, Any]:
    """Compute evidence-span retrieval metrics for one question."""
    if k <= 0:
        raise ValueError("k must be positive")
    if not evidence:
        raise ValueError("at least one evidence unit is required")

    top = list(retrieved[:k])
    flags: List[bool] = []
    overlap_by_rank: List[int] = []
    covered: set[str] = set()
    evidence_overlaps: List[dict[str, Any]] = []
    coverage: Dict[str, dict[str, Any]] = {
        unit.evidence_id: {"covered": False, "best_rank": None, "overlap_chars": 0, "clause_id": unit.clause_id}
        for unit in evidence
    }

    for index, hit in enumerate(top):
        matched = [unit for unit in evidence if overlap_chars(unit, hit) > 0]
        chars = sum(overlap_chars(unit, hit) for unit in matched)
        flags.append(bool(matched))
        overlap_by_rank.append(chars)
        if matched:
            evidence_overlaps.append(
                {
                    "rank": index + 1,
                    "chunk_id": hit.chunk_id,
                    "evidence_ids": [unit.evidence_id for unit in matched],
                    "overlap_chars": chars,
                }
            )
        for unit in matched:
            covered.add(unit.evidence_id)
            record = coverage[unit.evidence_id]
            record["covered"] = True
            record["overlap_chars"] += overlap_chars(unit, hit)
            if record["best_rank"] is None or index + 1 < record["best_rank"]:
                record["best_rank"] = index + 1

    recall = len(covered) / len(evidence)
    precision = sum(1 for flag in flags if flag) / k
    mrr = 0.0
    for index, flag in enumerate(flags):
        if flag:
            mrr = 1.0 / (index + 1)
            break

    required_ids = [unit.evidence_id for unit in evidence]
    return {
        f"recall@{k}": round(recall, 4),
        f"precision@{k}": round(precision, 4),
        f"mrr@{k}": round(mrr, 4),
        "retrieved_count": len(top),
        "relevant_flags": flags,
        "overlap_chars_by_rank": overlap_by_rank,
        "evidence_overlaps": evidence_overlaps,
        "required_evidence_ids": required_ids,
        "covered_evidence_ids": sorted(covered),
        "missing_evidence_ids": sorted(set(required_ids) - covered),
        "evidence_coverage": coverage,
    }


def evaluate_answer_support(answer: Any, evidence: Sequence[EvidenceUnit]) -> Dict[str, Any]:
    """Measure statement/citation spans against verified evidence units.

    Shared by the production slice and answer-layer experiments so both arms are
    scored by exactly the same rule.  ``answer`` is a ``GroundedAnswer``-like
    object with ``statements``, ``citations``, ``status``, ``reason``, ``text``,
    ``token_support`` and ``best_distance`` attributes.
    """

    def overlapping(start: int, end: int) -> List[str]:
        return [unit.evidence_id for unit in evidence if unit.start_offset < end and start < unit.end_offset]

    statement_rows = []
    supported_statements = 0
    for statement in answer.statements:
        evidence_ids = overlapping(statement.start_offset, statement.end_offset)
        if evidence_ids:
            supported_statements += 1
        statement_rows.append(
            {
                **statement.to_dict(),
                "evidence_ids": evidence_ids,
                "supported": bool(evidence_ids),
            }
        )

    citation_rows = []
    supported_citations = 0
    for citation in answer.citations:
        evidence_ids = overlapping(citation.start_offset, citation.end_offset)
        if evidence_ids:
            supported_citations += 1
        citation_rows.append(
            {
                **citation.to_dict(),
                "evidence_ids": evidence_ids,
                "supported": bool(evidence_ids),
            }
        )

    return {
        "status": answer.status,
        "reason": answer.reason,
        "text": answer.text,
        "token_support": round(answer.token_support, 4),
        "best_distance": answer.best_distance,
        "statements": statement_rows,
        "citations": citation_rows,
        "statement_count": len(statement_rows),
        "supported_statement_count": supported_statements,
        "citation_count": len(citation_rows),
        "supported_citation_count": supported_citations,
        "answer_supported": supported_statements > 0,
    }


def aggregate_metrics(per_question: Sequence[Dict[str, Any]], k: int = 5) -> Dict[str, float]:
    if not per_question:
        return {f"recall@{k}": 0.0, f"precision@{k}": 0.0, f"mrr@{k}": 0.0}
    count = len(per_question)
    return {
        f"recall@{k}": round(sum(row[f"recall@{k}"] for row in per_question) / count, 4),
        f"precision@{k}": round(sum(row[f"precision@{k}"] for row in per_question) / count, 4),
        f"mrr@{k}": round(sum(row[f"mrr@{k}"] for row in per_question) / count, 4),
    }


def evidence_coverage_summary(
    per_question: Sequence[Dict[str, Any]],
    questions: Sequence[Question],
) -> List[dict[str, Any]]:
    """Per-question evidence-unit coverage summary for multi-clause inspection."""
    rows: List[dict[str, Any]] = []
    question_by_id = {question.question_id: question for question in questions}
    for metrics in per_question:
        question_id = metrics["question_id"]
        question = question_by_id.get(question_id)
        rows.append(
            {
                "question_id": question_id,
                "required_units": len(metrics["required_evidence_ids"]),
                "covered_units": len(metrics["covered_evidence_ids"]),
                "missing_units": metrics["missing_evidence_ids"],
                "required_clause_ids": list(question.relevant_clause_ids) if question else [],
                "coverage": metrics["evidence_coverage"],
            }
        )
    return rows
