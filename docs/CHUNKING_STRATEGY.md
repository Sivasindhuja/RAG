# Chunking strategy analysis (SatCom-NGP)

## Document structure

The policy PDF (~15 pages) is **not** flat prose:

- **Articles** (`ARTICLE-1`, `ARTICLE-2`, …) define major policy areas.
- **Numbered clauses** (`3.4.2`, `3.5`, …) are legally meaningful units.
- **ALL-CAPS headings** mark responsibilities and definitions.
- **Acronyms** (GMPCS, INSAT, CAISS, WPC) must stay with their defining clause.

## Options considered

| Strategy | Pros | Cons | Fit |
|----------|------|------|-----|
| Fixed-size recursive split | Simple, fast | Splits mid-clause; hurts recall on cross-sentence facts | Baseline only |
| Page-level chunks | Stable page citations | Pages are long; mixed topics per page | Weak |
| Semantic chunking (embedding breakpoints) | Adapts to “topic” shifts | Expensive index build; unstable on legal lists | Overkill for 15 pages |
| **Structure-aware + parent–child** | Respects articles/clauses; small children for search, large parents for LLM | Needs domain-specific regex | **Chosen** |

## Decision

**Structure-aware parent–child retrieval** (small-to-big):

1. **Parents** = splits on `ARTICLE-N`, numbered sections, and major headers (with merge/split size guards).
2. **Children** = recursive splits *within* each parent for dense retrieval.
3. At query time: retrieve children → dedupe to parent context for generation.

This matches how regulators read the document and keeps acronyms attached to definitions.

## Validation

Run ablations:

```bash
python -m experiments.benchmark --rebuild-index --sample 20
```

Compare `mean_keyword_recall` and `hit_rate` for `recursive` vs `structure` on `tests/golden_dataset.csv`.

**Note:** A keyword-overlap proxy often favors recursive splits (smaller, denser parents). Structure-aware chunking is still preferred for **generation**: parents align with legal clauses, which improves citation grounding and reduces cross-clause hallucinations. Validate with Ragas faithfulness on the full golden set, not keyword recall alone.
