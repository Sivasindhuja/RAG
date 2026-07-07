"""
Retrieval-quality metrics without generation (fast ablations).

Uses golden Q/A: checks whether retrieved context contains key terms from the reference answer.
"""

from __future__ import annotations

import re
from typing import Iterable

from langchain_core.documents import Document

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with", "as", "by",
    "is", "are", "was", "were", "be", "been", "being", "that", "this", "these", "those",
    "it", "its", "at", "from", "may", "will", "shall", "not", "only", "such", "under",
}


def _keywords(text: str, min_len: int = 4) -> set[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", text.lower())
    return {t for t in tokens if t not in STOPWORDS and len(t) >= min_len}


def keyword_recall_in_context(docs: Iterable[Document], ground_truth: str) -> float:
    """
    Fraction of ground-truth keywords found in retrieved text (proxy for recall).
    """
    keys = _keywords(ground_truth)
    if not keys:
        return 0.0
    blob = " ".join(d.page_content.lower() for d in docs)
    hits = sum(1 for k in keys if k in blob)
    return hits / len(keys)


def hit_at_k(docs: list[Document], ground_truth: str, threshold: float = 0.35) -> bool:
    """True if keyword recall meets threshold for this query."""
    return keyword_recall_in_context(docs, ground_truth) >= threshold


def evaluate_retrieval(
    questions: list[str],
    ground_truths: list[str],
    retrieve_fn,
) -> dict:
    recalls = []
    hits = []
    for q, gt in zip(questions, ground_truths):
        docs = retrieve_fn(q)
        r = keyword_recall_in_context(docs, gt)
        recalls.append(r)
        hits.append(hit_at_k(docs, gt))
    n = len(recalls) or 1
    return {
        "mean_keyword_recall": round(sum(recalls) / n, 4),
        "hit_rate": round(sum(hits) / n, 4),
        "n": n,
    }
