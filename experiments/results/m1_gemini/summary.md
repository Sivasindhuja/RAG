# M1 — Validated Gemini Answer Path

- Experiment: `M1-gemini`
- Gemini available: `True`
- Selected model: `gemini-flash-lite-latest`
- Prompt version: `m1-v1`
- Evidence sufficiency gate: `selected_support >= 0.5`

## Extractive baseline (EXP-003 selection, unchanged)

- Answered: 15/15, supported answers: 12
- Supported statements: 15/45 (0.3333)
- Supported citations: 16/31 (0.5161)

## Gemini generation

- Answered: 11, model abstained: 3, insufficient-evidence abstained: 1, blocked: 0, errors: 0
- Claims: 18, grounded claims (>=0.80 overlap): 14 (0.7778)
- Uncited claims: 0, invalid source IDs: 0

## Abstention (out-of-corpus)

- extractive: 12/12 abstained
- gemini: 12/12 abstained

## Notes

- Retrieval/chunking/embeddings/ground truth unchanged from accepted baseline.
- Use current 15-question set as initial evaluation only; not general production readiness.
- Manual correctness/completeness/faithfulness/citation audit required for Gemini outputs.
