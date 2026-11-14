# EXP-004 decision

- Mechanical gate: **FAIL**
- Production default: **unchanged** (extractive pipeline)
- Generator: **rejected for production** by this experiment

## Manual audit counts (15 positive questions)

| Correctness | Extractive | Generated |
|---|---:|---:|
| correct | 0 | 5 |
| partial | 11 | 6 |
| incorrect | 4 | 4 |

| Faithfulness | Extractive | Generated |
|---|---:|---:|
| grounded | 15 | 14 |
| partially_grounded | 0 | 1 |
| ungrounded | 0 | 0 |

## Reasons

- pre-registered mechanical gate failed: claim_grounding_rate_ge_0_9
- 4/15 generated answers were judged incorrect

## Interpretation

- The generator was constrained to the extractive selector's own evidence,
  so it could not fix retrieval or selection failures.
- The audit found no fabricated material claims, but four answers were still
  incorrect because the selected evidence was wrong or incomplete.
- A fluent rewrite of an incomplete heading (Q04, Q12) is not an improvement
  in answer quality.
- The pre-registered grounding proxy also failed (14/16 claims at >=0.80
  token coverage, rate 0.875 vs required 0.90).
