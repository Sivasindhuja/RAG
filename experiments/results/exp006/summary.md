# EXP-006 — Cross-reference applicability selection

- Decision: **REJECTED**
- Changed variable: selection only (`cross_reference_selection`)
- Frozen: PDF, questions, ground truth, chunking, embeddings, retrieval hits, top-k, citations, abstention, metrics

## Aggregate
- Supported answers: 12 -> 13 (+1)
- Statement support: 15/45 (0.3333) -> 16/45 (0.3556) (+0.0222)
- Citation support: 16/31 (0.5161) -> 17/32 (0.5312) (+0.0151)

## Q10 checks
- evidence_covered: True
- supported_statement_present: True
- cites_q10_chunk: True
- all_key_phrases_present: False
- key phrases: first come first served=True, government owned system=True, priority=False, discussions with the applicants=True, common advanced publication of information=True, ap s4 filing=True

## Checks
- q10_evidence_covered: True
- q10_supported_statement_present: True
- q10_cites_clause_chunk: True
- q10_all_key_phrases_present: False
- no_non_q10_regressions: True
- supported_answers_not_lower: True
- statement_support_not_lower: True
- citation_support_delta_ge_minus_0_02: True
- abstention_unchanged: True


## Post-run review note (2026-11-14)
- Automated decision remained **REJECTED** only because `q10_all_key_phrases_present` failed on `priority` (extracted text contains `prio rity`, a PDF whitespace artifact). The 3.6.8 procedure text includes government-owned-system priority and all other key phrases are present.
- Substantive result: Q10 covers `EXP001-Q10-E01`, has 1/3 supported statements, cites `satcom-ngp::page_8::chunk_0::off17796`, and no non-Q10 regressions; abstention unchanged; aggregate support improved.
- Candidate kept **opt-in** (`cross_reference_selection=False` default). See `docs/EXP-006.md` and engineering log (EXP-006/DR-016).
