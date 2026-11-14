# PROD-001 — Accepted baseline production slice

- Chunking: fixed-character 1000/200 (accepted EXP-001 strategy)
- Embedding: `sentence-transformers/all-MiniLM-L6-v2`
- Retrieval: dense-only, top-5
- Answer mode: extractive_verbatim
- Abstention policy: `{"hard_max_distance": 1.3, "soft_max_distance": 1.1, "min_token_support": 0.5, "strict_min_token_support": 0.7}`

## Retrieval (must match the accepted baseline)

| Metric | Slice | EXP-001 v3 | Delta |
|---|---:|---:|---:|
| recall@5 | 0.8889 | 0.8889 | +0.0000 |
| precision@5 | 0.2533 | 0.2533 | +0.0000 |
| mrr@5 | 0.7944 | 0.7944 | +0.0000 |

## Document integrity

- Chunks: 49
- Non-whitespace coverage: 100.0000%
- Intentional overlap characters: 6800
- Fidelity errors: 0; duplicate IDs: 0; page errors: 0
- Audit ok (overlap allowed): True

## Answers and citations (15 in-corpus questions)

| Measure | Value |
|---|---:|
| Answered | 15 / 15 |
| Abstained | 0 / 15 |
| Answers with >=1 evidence-supported statement | 12 |
| Statement support rate | 0.2889 |
| Citation support rate | 0.4688 |
| Mean statements per answered question | 3.00 |
| Mean citations per answered question | 2.13 |

## Abstention smoke test

| Split | Questions | Abstained | Abstention rate |
|---|---:|---:|---:|
| calibration | 8 | 8 | 1.0000 |
| heldout | 4 | 4 | 1.0000 |

Overall abstention rate: **1.0000** (12/12).

## Limitations

- The abstention thresholds were calibrated on the 8 `calibration` questions; the
  4 `heldout` questions are a small, non-representative check, not a validation set.
- Answers are extractive (verbatim sentences). Fluency and synthesis are not claimed.
- Sentence-level support is measured against the 20 verified evidence units, not
  against an independent entailment model.
- The generator hook is not evaluated; only the extractive path is validated.

See `docs/PRODUCTION_SLICE.md` and `docs/ENGINEERING_LOG.md` for the decision record.
