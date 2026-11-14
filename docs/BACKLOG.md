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
| BL-001 (Q05) | `2.5.6` evidence not retrieved | Retrieval miss | High — question answered without supported evidence (Recall@5 0.0) | High | Open (mitigated for M1 generation) | M6 (controlled retrieval experiment) | Live M1: recall 0.0, selected support 0.3333, sufficiency gate abstained before generation (trustworthy given evidence); retrieval still unfixed. |
| BL-002 (Q08) | Incomplete child-clause retrieval | Retrieval miss | High — multi-clause question incomplete (Recall@5 0.3333, 1/3) | High | Open | M6 | Live M1: Gemini answered but incorrect/incomplete (omitted required DoS/WPC/operating-license authorisations); claims grounded to selected but insufficient evidence. Requires evidence-driven retrieval/context experiment. |
| BL-003 (Q10) | Evidence retrieved but wrong unit selected | Selection/ranking | High — Recall@5 1.0 but 0/3 supported; 3.6.8 procedure not selected | High | Open | M6 | Live M1: lexical gate passed (0.7778) but selected evidence wrong; model abstained (faithful to supplied evidence) so no hallucination, but question unanswered. Confirms semantic insufficiency not caught by lexical support. Candidate: controlled reference-aware/ranking selection. |
| BL-004 (Q04) | Heading-only answer unit selected | Answer unit/segmentation | Medium — answer lacks following list | Medium | Open | M6 | Live M1: Gemini answered faithfully to heading text but incorrect/incomplete (omitted sector classifications). Needs heading+list/definition answer unit. |
| BL-005 (Q12) | Heading-only answer unit selected | Answer unit/segmentation | Medium — classification heading without definition | Medium | Open | M6 | Live M1: model abstained; selected evidence lacked international services definition. Needs heading+definition/list answer unit or better selection. |
| BL-006 | LLM answer faithfulness and citation enforcement | Generation | High — blocks Phase 1/2 grounded Gemini answers | High | **Partially addressed (M1 live run, quality not met)** | M1/M6 | Gemini path validated live with `gemini-flash-lite-latest` (explicit, probe ok). Mechanical: 18 claims, 0 uncited, 0 invalid source IDs, 14/18 token-overlap≥0.8. Manual audit (15q): faithfulness 9 grounded/2 partial/0 unfaithful/4 abstained, citation quality 8 correct/3 partial/0 incorrect/4 n/a; correctness 5/4/2/4, completeness 5/2/4/4. App-resolved citation IDs correct, but valid IDs do not guarantee claim support and lexical overlap does not prove semantic sufficiency. Upstream evidence/selection failures cause incomplete/incorrect/abstained answers. Not accepted for production quality. |
| BL-007 | Abstention misses in-corpus retrieval failure | Abstention/safety | High — Q05 answered despite missing evidence | High | **Partially addressed (heuristic)** | M1/M6 | M1 sufficiency gate abstains before generation when pipeline abstained, no selected statements, or selected lexical support < 0.5. Live: Q05 correctly abstained (0.3333, recall 0.0); out-of-corpus 12/12 abstained. Q10 (wrong selection, support 0.7778) passed gate but model abstained on evidence; Q12/Q15 model-abstained. Heuristic, not validated answerability classifier; semantic insufficiency not detected. |
| KNOWN-001 | Legacy backend defect and drift | Integration/bug | High — blocks safe app exposure | High | Open | M2 | `backend/main.py:154` uses `status_status=429` (should be `status_code`), uses legacy collection `satcom_ngp_clauses`, `gemini-1.5-pro`, unvalidated paths; file currently modified in working tree (preserve). Isolate/retire or fix in planned task. |
| BL-008 | No Gemini model/config selected and documented for production path | Integration | High — required by fixed constraints | High | **Partially resolved (model selected for M1 run)** | M1 | Credentials loaded from local `.env` (not exposed). 49 generateContent models listed. Probes: `gemini-2.5-flash` unavailable (404), `gemini-3.8-flash` 504 timeout, `gemini-flash-latest` connection abort/timeout. Selected **`gemini-flash-lite-latest`** (explicit, probe ok 3182 ms, tokens 10/9/19). Live run recorded latency/tokens (14 calls, mean 2504.93 ms, total tokens 6015); cost not computed. Model is a working lite choice, not proven optimal for quality/cost/latency; rerun with better available model when feasible. |
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
- M1 Gemini path (`m1-v1`, `experiments/m1_gemini_eval.py`, `experiments/results/m1_gemini/`): architecture/contract implemented and unit-tested (97 tests total pass). **Live validation executed** (DR-015) with `gemini-flash-lite-latest` (explicit, probe ok; other probes failed/unavailable). Results: 11 answered, 3 model-abstained, 1 sufficiency-abstained, 0 errors; mechanical grounding 0.7778, 0 uncited/invalid IDs; manual audit 5 correct/4 partial/2 incorrect/4 abstained, 9 grounded/2 partial/0 unfaithful. **M1 exit criteria not met** due to correctness/completeness and upstream Q04/Q05/Q08/Q10/Q12 blockers; not accepted for production quality.