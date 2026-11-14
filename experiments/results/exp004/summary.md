# EXP-004 — Constrained generator evaluation

- Experiment id: `EXP-004`
- Generator: `qwen2.5:1.5b` (local Ollama, selected evidence only)
- Retrieval/selection: accepted PROD-001/EXP-003 path, unchanged
- Candidate input: only the extractive selector's verbatim evidence sentences
- Mechanical gate: **FAIL**
- Manual correctness/faithfulness audit: see `manual_audit.json` / `manual_audit_summary.json`

## Extractive baseline vs generated synthesis

| Measure | Extractive | Generated |
|---|---:|---:|
| Answered | 15/15 | 15/15 |
| Model abstained | — | 0/15 |
| Generation errors | — | 0/15 |
| Statements / claims | 45 | 16 |
| Citations | 31 | 17 |
| Mean words per answer | — | 29.00 |
| Mean cited-token coverage | — | 0.9149 |
| Claims grounded at >=0.80 | — | 14/16 |
| Uncited claims | — | 0 |
| Invalid source IDs | — | 0 |
| Duplicate claims | — | 0 |
| Truncated generations | — | 0 |

## Mechanical gate

| Check | Result |
|---|---|
| `negative_abstention_12_of_12` | PASS |
| `no_generation_errors` | PASS |
| `all_positives_parse` | PASS |
| `zero_invalid_sources` | PASS |
| `zero_uncited_claims` | PASS |
| `claim_grounding_rate_ge_0_9` | FAIL |
| `at_least_12_positives_answered` | PASS |

## Per-question mechanical view

| ID | Recall@5 | Extractive supported | Gen status | Claims/cited | Grounded >=0.80 |
|---|---:|---:|---|---:|---:|
| EXP001-Q01 | 1.0000 | 1/3 | answered | 1/1 | 1/1 |
| EXP001-Q02 | 1.0000 | 1/3 | answered | 1/1 | 1/1 |
| EXP001-Q03 | 1.0000 | 1/3 | answered | 1/1 | 1/1 |
| EXP001-Q04 | 1.0000 | 1/3 | answered | 1/1 | 1/1 |
| EXP001-Q05 | 0.0000 | 0/3 | answered | 1/1 | 1/1 |
| EXP001-Q06 | 1.0000 | 1/3 | answered | 1/1 | 1/1 |
| EXP001-Q07 | 1.0000 | 1/3 | answered | 1/1 | 1/1 |
| EXP001-Q08 | 0.3333 | 0/3 | answered | 1/1 | 1/1 |
| EXP001-Q09 | 1.0000 | 1/3 | answered | 1/1 | 1/1 |
| EXP001-Q10 | 1.0000 | 0/3 | answered | 2/2 | 2/2 |
| EXP001-Q11 | 1.0000 | 1/3 | answered | 1/1 | 0/1 |
| EXP001-Q12 | 1.0000 | 1/3 | answered | 1/1 | 1/1 |
| EXP001-Q13 | 1.0000 | 3/3 | answered | 1/1 | 0/1 |
| EXP001-Q14 | 1.0000 | 2/3 | answered | 1/1 | 1/1 |
| EXP001-Q15 | 1.0000 | 1/3 | answered | 1/1 | 1/1 |

## Abstention smoke test

Extractive: 12/12 abstained.

The generator is never called when the shared abstention policy already declines
the question, so the negative set measures the same gate for both arms.

## Interpretation limits

- The generator sees only selected sentences. It cannot fix Q05/Q08 retrieval
  misses or Q10's wrong selection; those remain upstream failures.
- Cited-token coverage is a lexical proxy, not entailment or human faithfulness.
- Expected-answer token recall is a weak surface metric and is not used as a
  decision gate.
- The manual audit in `manual_audit.json` is a single-reviewer judgment, not
  an independent gold standard.
- A local 1.5B model is evaluated here; results do not generalise to larger
  hosted models.

Artifacts: `answers_extractive.json`, `answers_generated.json`,
`comparison.json`, `per_question_metrics.csv`, `manual_audit_template.csv`.
