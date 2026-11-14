# EXP-005 — Clause-body-aware answer-unit selection

- Experiment id: `EXP-005`
- Baseline: EXP-003 accepted selection (`sibling_score_ratio=0.5`, sentence units)
- Candidate: `sibling_score_ratio=0.5`, depth>=3 clause-body answer units
- Retrieval: fixed-character 1000/200, MiniLM, dense top-5, identical precomputed hits
- Decision: **REJECTED**

## Root cause (Q10)

Q10 evidence `EXP001-Q10-E01` is inside clause 3.6.8 and is retrieved at rank 3.
Sentence segmentation split 3.6.8 into four units; body procedure sentences had
no clause identity. Lexical selection favored the 3.6.7(a)/(b) list and 3.6.10,
so the accepted selector returned 0/3 supported statements for Q10.

## Primary metrics

| Measure | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| Supported answers | 12 | 12 | +0 |
| Supported statements | 15/45 | 14/45 | -1 |
| Statement support rate | 0.3333 | 0.3111 | -0.0222 |
| Supported citations | 16/31 | 16/32 | +0 |
| Citation support rate | 0.5161 | 0.5000 | -0.0161 |

## Q10 checks

| Check | Result |
|---|---|
| `q10_evidence_covered` | FAIL |
| `q10_supported_statement_present` | FAIL |
| `q10_cites_clause_chunk` | FAIL |
| `q10_all_key_phrases_present` | FAIL |

Key phrases present: first come first served=no, government owned system=no, priority=no, discussions with the applicants=no, common advanced publication of information=no, ap s4 filing=no

## Other checks

| Check | Result |
|---|---|
| `no_non_q10_regressions` | FAIL |
| `supported_answers_not_lower` | PASS |
| `statement_support_not_lower` | FAIL |
| `citation_support_delta_ge_minus_0.02` | PASS |
| `abstention_unchanged` | PASS |

Regressions: EXP001-Q14:supported_statement_count

## Per-question statement support

| ID | Recall@5 | Baseline | Candidate | Delta |
|---|---:|---:|---:|---:|
| EXP001-Q01 | 1.0000 | 1/3 | 1/3 | +0 |
| EXP001-Q02 | 1.0000 | 1/3 | 1/3 | +0 |
| EXP001-Q03 | 1.0000 | 1/3 | 1/3 | +0 |
| EXP001-Q04 | 1.0000 | 1/3 | 1/3 | +0 |
| EXP001-Q05 | 0.0000 | 0/3 | 0/3 | +0 |
| EXP001-Q06 | 1.0000 | 1/3 | 1/3 | +0 |
| EXP001-Q07 | 1.0000 | 1/3 | 1/3 | +0 |
| EXP001-Q08 | 0.3333 | 0/3 | 0/3 | +0 |
| EXP001-Q09 | 1.0000 | 1/3 | 1/3 | +0 |
| EXP001-Q10 | 1.0000 | 0/3 | 0/3 | +0 |
| EXP001-Q11 | 1.0000 | 1/3 | 1/3 | +0 |
| EXP001-Q12 | 1.0000 | 1/3 | 1/3 | +0 |
| EXP001-Q13 | 1.0000 | 3/3 | 3/3 | +0 |
| EXP001-Q14 | 1.0000 | 2/3 | 1/3 | -1 |
| EXP001-Q15 | 1.0000 | 1/3 | 1/3 | +0 |

## Abstention smoke test

Overall: baseline 12/12; candidate 12/12.

## Interpretation

- Clause-body units are exact verbatim spans: a depth>=3 clause start is grouped
  with following body sentences until the next numbered clause start in the same chunk.
- Depth-2 headings are left as sentence units to avoid swallowing child clauses.
- The experiment is controlled: retrieval hits are identical and only answer-unit
  segmentation/selection input changes.

See `docs/EXP-005.md` and `docs/ENGINEERING_LOG.md` for the decision record.
