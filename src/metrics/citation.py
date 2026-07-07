"""Citation coverage (complements Ragas faithfulness)."""

from __future__ import annotations

import re


def citation_coverage(answer: str) -> float:
    sentences = [
        s.strip()
        for s in re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", answer)
        if len(s.strip()) > 10
    ]
    if not sentences:
        return 0.0
    cited = sum(1 for s in sentences if re.search(r"\[Source \d+\]", s))
    return round(cited / len(sentences), 4)
