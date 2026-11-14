# Known Issues and Deferred Improvements Backlog

**Status:** living backlog. Last updated 2026-11-14.  
**Purpose:** track important problems and improvements that are known but not in the current milestone, so they are not silently discarded and do not derail active work.

Rules (DR-013, `docs/PROJECT_BRIEF.md`):
1. Document newly discovered important problems.
2. Assess impact and urgency.
3. Add to backlog with status and target milestone.
4. Decide when to address; do not derail the current milestone for every issue.
5. Do not silently discard problems. Rejected experiments remain documented.

---

## Open issues

| ID | Title | Type | Impact | Urgency | Status | Target | Evidence / notes |
|---|---|---|---|---|---|---|---|
| BL-001 (Q05) | `2.5.6` evidence not retrieved | Retrieval miss | High — question answered without supported evidence (Recall@5 0.0) | High | Open | M6 (controlled retrieval experiment) | Q05 consistently misses; abstention does not catch in-corpus retrieval failure. |
| BL-002 (Q08) | Incomplete child-clause retrieval | Retrieval miss | High — multi-clause question incomplete (Recall@5 0.3333, 1/3) | High | Open | M6 | Requires evidence-driven retrieval/context experiment. |
| BL-003 (Q10) | Evidence retrieved but wrong unit selected | Selection/ranking | High — Recall@5 1.0 but 0/3 supported; 3.6.8 procedure not selected | High | Open | M6 | EXP-003 did not apply; EXP-005 formed clause-body unit but lexical/context ranking favored 3.6.7(a)/(b) list and 3.6.10. Candidate: controlled reference-aware/ranking selection. |
| BL-004 (Q04) | Heading-only answer unit selected | Answer unit/segmentation | Medium — answer lacks following list | Medium | Open | M6 | Selected evidence is heading; needs list/definition following it. Not a generation problem. |
| BL-005 (Q12) | Heading-only answer unit selected | Answer unit/segmentation | Medium — classification heading without definition | Medium | Open | M6 | Same class as BL-004. |
| BL-006 | LLM answer faithfulness and citation enforcement | Generation | High — blocks Phase 1/2 grounded Gemini answers | High | Open | M1 | EXP-004 rejected constrained local synthesis (0.875 grounding < 0.90 gate; 4/15 incorrect). Gemini path unvalidated; citation validity/enforcement required. |
| BL-007 | Abstention misses in-corpus retrieval failure | Abstention/safety | High — Q05 answered despite missing evidence | High | Open | M1/M6 | Current policy is distance/token-support heuristic (12/12 out-of-corpus smoke test) and not a validated answerability classifier. |
| KNOWN-001 | Legacy backend defect and drift | Integration/bug | High — blocks safe app exposure | High | Open | M2 | `backend/main.py:154` uses `status_status=429` (should be `status_code`), uses legacy collection `satcom_ngp_clauses`, `gemini-1.5-pro`, unvalidated paths; file currently modified in working tree (preserve). Isolate/retire or fix in planned task. |
| BL-008 | No Gemini model/config selected and documented for production path | Integration | High — required by fixed constraints | High | Open | M1 | Settings reference `gemini-2.0-flash`; legacy backend uses `gemini-1.5-pro`. Must select/document model based on quality, availability, cost, latency. |
| BL-009 | Vector store not suitable for Vercel Hobby serverless | Deployment | High — blocks public deployment | High | Open | M3 | Validated path uses local persistent Chroma (`chroma_db_*`). Serverless has no persistent filesystem; need Hobby-compatible/free-tier strategy (prebuilt artifact, hosted free tier, or documented alternative) with explicit trade-offs. |
| BL-010 | Golden evaluation dataset missing (need ~50–200) | Evaluation | Critical for Phase 3/DoD | Medium (sequence) | Open | M4 | Current: 15 approved Q/A + 20 evidence units + 12 out-of-corpus. Expansion requires verified evidence and recorded rationale; do not modify approved set to improve metrics. |
| BL-011 | Offline LLM evaluation harness missing | Evaluation | High — cannot measure correctness/faithfulness/completeness/citation/abstention for Gemini answers | High | Open | M4 (design can start earlier) | Retrieval metrics exist for small set only. Thresholds must be derived from baseline evidence, not invented. |
| BL-012 | CI quality gates absent | Quality/CI | High — regressions can slip in | Medium | Open | M5 | No active quality-evaluation workflow in repo; unrelated deleted workflow in working tree. |
| BL-013 | Observability/monitoring absent | Operations | High — failures/cost/latency/quality hard to diagnose | Medium | Open | M5 (basic tracing earlier in M1/M2 useful) | Need traces, retrieved chunks/ranking, prompt/response handling, token/cost, P50/P95, citation coverage, failure rates, degradation investigation. |
| BL-014 | Frontend not verified against validated pipeline | UI/integration | Medium — Phase 1 requires functional UI | Medium | Open | M2 | React/Vite app exists in `frontend/`; API contract and evidence inspection UX unverified. |
| BL-015 | Vercel Hobby feasibility unknown (bundle, cold start, limits, CORS/env) | Deployment | High — fixed deployment target | High | Open | M3 | Must account for serverless execution limits, cold starts, function bundle size, Python runtime constraints, and secret handling. Document measurements/constraints. |
| BL-016 | Q14 evidence completeness/granularity | Answer quality | Low-medium — evidence covered but statement granularity/count can vary | Low | Open | M6/deferred | EXP-005 showed granularity/count changes while preserving evidence coverage (R-014). Inspect evidence coverage/content, not counts alone. |
| BL-017 | Small evaluation set and threshold validity | Evaluation risk | Medium — EXP-003/004/005 conclusions on reused 15-question set | Medium | Open | M4 | Expand and independently label before broader claims. |

## Deferred / watch items

| ID | Title | Status | Notes |
|---|---|---|---|
| DEF-001 | Hybrid retrieval (BM25+dense) | Deferred, evidence-gated | Only if exact-terminology retrieval failures persist after controlled diagnosis (BL-001/002). |
| DEF-002 | Cross-encoder reranking | Deferred, evidence-gated | Only if irrelevant chunks rank above relevant evidence and experiment justifies it. |
| DEF-003 | Query expansion | Deferred, evidence-gated | Not a default fix; must target observed retrieval problem. |
| DEF-004 | Parent-child/semantic chunking changes | Deferred, evidence-gated | EXP-002 and EXP-005 show chunking/unit changes can regress; require controlled evidence. |
| DEF-005 | Multi-PDF / arbitrary uploads | Out of scope (initial) | Fixed corpus constraint. |

## Resolved / historical (kept for traceability)

- BUG-001..BUG-009: fixed and documented in `docs/ENGINEERING_LOG.md`.
- EXP-002, EXP-004, EXP-005: rejected with recorded evidence; do not re-adopt without new controlled justification.
- Legacy experimental generator (`src/rag_core/generation.py`, Ollama): not production; retained as experimental artifact only.