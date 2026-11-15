# EXP-007 — Heading-introduced associated answer units (Q04/Q12)

- Decision (promotion consideration): **REJECTED**
- Changed variable: answer-unit construction/selection only (`associated_heading_units`)
- EXP-006 kept separate (opt-in comparison, not combined). EXP-003 remains default.

## Aggregate (baseline -> EXP-007)
- Supported answers: 12 -> 12 (+0)
- Statement support: 15/45 (0.3333) -> 15/45 (0.3333) (+0.0000)
- Citation support: 16/31 (0.5161) -> 16/32 (0.5000) (-0.0161)

## Target checks
- Q04 all four classifications present: True (4/4) {'telecommunications': True, 'broadcasting': True, 'education and developmental communications': True, 'security communications': True}
- Q04 evidence covered: True, answer supported: True, supported stmts: 1/3
- Q12 all required definition elements present: False (0/7) {'pstn': False, 'gmpcs': False, 'value added services': False, 'programme exchange': False, 'foreign country': False, 'maritime mobile': False, 'aeronautical mobile': False}
- Q12 evidence covered: True, answer supported: True, supported stmts: 1/3

## Q10 (separate comparison)
- EXP-006 only: supported=True, supported_statements=1, covered=['EXP001-Q10-E01'], cites_q10_chunk=True
- EXP-007 only: supported=False, supported_statements=0, covered=[]

## Checks
- q04_all_four_classifications_present: True
- q04_evidence_covered: True
- q12_all_required_definition_elements_present: False
- q12_evidence_covered: True
- no_non_target_regressions_vs_baseline: True
- no_regressions_vs_baseline_any: True
- supported_answers_not_lower: True
- statement_support_not_lower: True
- citation_support_delta_ge_minus_0_02: True
- abstention_unchanged: True
- all_offsets_valid_exp007: True
