# EXP-003 — Sibling-clause answer selection

- Experiment id: `EXP-003`
- Changed variable: `sibling_score_ratio` (baseline `None` → candidate `0.5`)
- Retrieval: fixed-character 1000/200, MiniLM, dense top-5 (unchanged)
- Decision: **ACCEPTED**

## Primary metrics

| Measure | Baseline | Sibling-clause | Delta |
|---|---:|---:|---:|
| Supported answers | 12 | 12 | +0 |
| Supported statements | 13/45 | 15/45 | +2 |
| Statement support rate | 0.2889 | 0.3333 | +0.0444 |
| Supported citations | 15/32 | 16/31 | +1 |
| Citation support rate | 0.4688 | 0.5161 | +0.0474 |
| Mean statements | 3.00 | 3.00 | +0.00 |
| Mean citations | 2.13 | 2.07 | -0.07 |

## Decision checks

| Check | Result |
|---|---|
| `statement_support_delta_ge_0.02` | PASS |
| `supported_answers_not_lower` | PASS |
| `citation_support_delta_ge_minus_0.02` | PASS |
| `abstention_unchanged` | PASS |

## Per-question statement support

| ID | Recall@5 | Baseline | Sibling | Delta | Missing evidence (sibling) |
|---|---:|---:|---:|---:|---|
| EXP001-Q01 | 1.0000 | 1/3 | 1/3 | +0 | — |
| EXP001-Q02 | 1.0000 | 1/3 | 1/3 | +0 | — |
| EXP001-Q03 | 1.0000 | 1/3 | 1/3 | +0 | — |
| EXP001-Q04 | 1.0000 | 1/3 | 1/3 | +0 | — |
| EXP001-Q05 | 0.0000 | 0/3 | 0/3 | +0 | EXP001-Q05-E01 |
| EXP001-Q06 | 1.0000 | 1/3 | 1/3 | +0 | — |
| EXP001-Q07 | 1.0000 | 1/3 | 1/3 | +0 | — |
| EXP001-Q08 | 0.3333 | 0/3 | 0/3 | +0 | EXP001-Q08-E02, EXP001-Q08-E03 |
| EXP001-Q09 | 1.0000 | 1/3 | 1/3 | +0 | — |
| EXP001-Q10 | 1.0000 | 0/3 | 0/3 | +0 | — |
| EXP001-Q11 | 1.0000 | 1/3 | 1/3 | +0 | — |
| EXP001-Q12 | 1.0000 | 1/3 | 1/3 | +0 | — |
| EXP001-Q13 | 1.0000 | 1/3 | 3/3 | +2 | — |
| EXP001-Q14 | 1.0000 | 2/3 | 2/3 | +0 | — |
| EXP001-Q15 | 1.0000 | 1/3 | 1/3 | +0 | — |

## Abstention smoke test

| Split | Questions | Baseline abstained | Sibling abstained |
|---|---:|---:|---:|
| calibration | 8 | 8 | 8 |
| heldout | 4 | 4 | 4 |

Overall: baseline 12/12; sibling 12/12.

## Interpretation

- The rule only activates when the top-ranked sentence starts with a numbered
  clause.  It promotes same-parent clauses scoring at least 50% of the top
  sentence, then fills any remaining slots from the global ranking.
- Q13 improves because `4.3.2` and `4.3.3` are retrieved and now selected
  alongside `4.3.4`.
- Q05 and Q08 are retrieval failures; answer-layer selection cannot recover
  evidence that is not in the top-5.
- Q10 remains unsolved: its required evidence sits inside a `3.6.8` body
  sentence that does not begin with the clause ID, so sibling detection does
  not find it.
- The 15-question set is small.  The 0.5 ratio is a natural threshold but was
  chosen during exploratory sweeps; this experiment is a controlled check, not
  independent confirmation.

See `docs/EXP-003.md` and `docs/ENGINEERING_LOG.md` for the decision record.
