# EXP-008 — Q12 definitional evidence selection

- Decision (promotion consideration): **ACCEPTED**
- Changed variable: selection/ranking only (`definition_intent_boost`)
- EXP-006 and EXP-007 kept separate (not combined). EXP-003 default unchanged.

## Aggregate (baseline -> EXP-008)
- Supported answers: 12 -> 12 (+0)
- Statement support: 15/45 (0.3333) -> 17/45 (0.3778) (+0.0444)
- Citation support: 16/31 (0.5161) -> 17/32 (0.5312) (+0.0151)

## Q12
- All required elements present: True (7/7)
- Term hits: {'pstn': True, 'gmpcs': True, 'value added services': True, 'programme exchange': True, 'foreign country': True, 'maritime mobile': True, 'aeronautical mobile': True}
- Evidence covered: True, answer supported: True, supported stmts: 3/3, valid offsets: True

## Separate checks (not combined)
- Q04 EXP-008: supported=True, stmts=1, all terms=False {'telecommunications': False, 'broadcasting': False, 'education and developmental communications': False, 'security communications': False}
- Q04 EXP-007 only: supported=True, stmts=1, all terms=True
- Q10 EXP-008: supported=False, stmts=0, covered=[]
- Q10 EXP-006 only: supported=True, stmts=1, covered=['EXP001-Q10-E01'], cites_q10_chunk=True

## Checks
- q12_all_required_elements_present: True
- q12_evidence_covered: True
- q12_answer_supported: True
- q12_valid_offsets_traceability: True
- no_regressions_vs_baseline: True
- supported_answers_not_lower: True
- statement_support_not_lower: True
- citation_support_delta_ge_minus_0_02: True
- abstention_unchanged: True
- all_offsets_valid_exp008: True
