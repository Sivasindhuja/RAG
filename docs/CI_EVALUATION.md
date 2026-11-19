# CI Evaluation (Fixed 15-Question Suite)

## Dataset
- **File:** `evaluation/ci_golden_dataset.json` (`ci_golden_v1`, version 1.0)
- **Verification status:** `draft_not_independently_verified` (current dataset remains a draft until required independent verification is resolved; do not mark independently verified).
- **Size:** exactly 15 questions (routine CI/PR requirement).

### Composition revision (documented)
To satisfy the final delivery requirement that the 15-question CI suite exercises unanswerable/out-of-scope questions while remaining limited to 15 total, the CI suite was composed from preserved historical sources:
- **12 answerable (from EXP-001 verified set, artifacts preserved):** EXP001-Q01, Q03, Q04, Q05, Q07, Q08, Q09, Q10, Q12, Q13, Q14, Q15.
  - Covers all four articles (Q01 Art1; Q03/Q04/Q05 Art2; Q07/Q08/Q09/Q10 Art3; Q12/Q13/Q14/Q15 Art4).
  - Includes exact clause/terminology and known lexical retrieval weakness (Q05, 2.5.6), compound multi-evidence questions (Q08 3 clauses, Q13 4 clauses), cross-reference/procedural reasoning (Q10), definitions/classifications (Q04, Q07, Q12), exceptions/conditions (Q09, Q14, Q15), and historical regression/quality cases (Q04,Q05,Q08,Q10,Q12,Q13).
  - Q02, Q06, Q11 omitted from CI subset to make room for unanswerable coverage; their historical artifacts and full EXP-001 set remain preserved (`experiments/exp001_questions.csv`, `experiments/exp001_ground_truth.json`, prior results).
- **3 unanswerable/out-of-scope (from preserved unanswerable set):** UNANS-01 (clearly out of domain), UNANS-03 (near domain), UNANS-11 (near domain/out of scope).

Historical experiment datasets/artifacts were not rewritten.

## Evaluation command
```bash
PYTHONPATH=. python -m evaluation.ci_eval \
  --dataset evaluation/ci_golden_dataset.json \
  --out evaluation/results/ci \
  --retrieval-modes dense,hybrid \
  --generation-mode extractive \
  --k 5 \
  --baseline evaluation/baseline_metrics.json \
  --check-regression
```

## Measured baselines (selected 800/160 chunking, CI set)
Source: `evaluation/baseline_metrics.json` (dense, extractive, k=5), measured 2026-11-19 on `satcom_ngp_production_fixed_char_800_160` (61 chunks).

| Metric | Value |
|---|---:|
| recall@5 | 0.9236 |
| precision@5 | 0.3 |
| mrr@5 | 0.8611 |
| evidence_coverage | 0.9236 |
| supported_answer_rate | 0.5833 (7/12) |
| statement_support_rate | 0.25 |
| citation_support_rate | 0.4828 |
| unanswerable abstention_accuracy | 1.0 (3/3) |
| invalid_citation_count_total | 0 |
| failure_rate | 0.0 |

Previous 1000/200 baseline (preserved): `evaluation/baseline_metrics_1000_200.json` (recall@5 0.8611, mrr@5 0.743, supported_answer_rate 0.75, abstention 1.0).

## Retrieval comparison (measured, 800/160)
Source: `evaluation/results/ci_compare_800_160/report.json`

| Mode | recall@5 | precision@5 | mrr@5 | evidence_coverage | supported_answer_rate | citation_support_rate | abstention_accuracy | invalid citations | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 0.9236 | 0.3 | 0.8611 | 0.9236 | 0.5833 | 0.4828 | 1.0 | 0 | ~24 | ~3011 |
| hybrid (dense+BM25 RRF) | 0.9236 | 0.3333 | 0.8472 | 0.9236 | 0.8333 | 0.5484 | 1.0 | 0 | ~40 | ~44 |

Notable per-question effects (hybrid vs dense): Q05 retrieved in both with new chunking (recall 1.0) and better supported (2/3 vs 1/3 statements); Q07, Q12, Q14 became supported; Q08 remains partial retrieval (1/3 evidence units), Q10 evidence retrieved but not selected/supported by extractive selector (known selection issue), Q13 covers 3/4 units. All unanswerables correctly abstained.

### Cohere reranking (live validation)
- **Integration:** implemented (`src/rag_core/retrieval.py`, mode `hybrid_rerank`, Cohere `rerank-english-v3.0` default), with automated mocked tests (`tests/test_cohere_reranker.py`).
- **Live validation (completed with configured key):** `CO_API_KEY` present in `.env` and reranking applied successfully (observed `rerank.applied=true`, scores/latency returned; example request latency ~1.57s for rerank stage). Measured comparison on fixed 15-question CI set (`evaluation/results/ci_compare_800_160_rerank_live_final/report.json`, extractive, k=5):

| Mode | recall@5 | precision@5 | mrr@5 | evidence_coverage | supported_answer_rate | citation_support_rate | abstention_accuracy | invalid citations | p50 ms | p95 ms | failure rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 0.9236 | 0.3 | 0.8611 | 0.9236 | 0.5833 | 0.4828 | 1.0 | 0 | 20.225 | 3770.367 | 0.0 |
| hybrid | 0.9236 | 0.3333 | 0.8472 | 0.9236 | 0.8333 | 0.5484 | 1.0 | 0 | 26.321 | 33.210 | 0.0 |
| hybrid_rerank (live Cohere) | 0.9236 | 0.3167 | 0.9028 | 0.9236 | 0.6667 | 0.4839 | 1.0 | 0 | 1144.883 | 3858.898 | 0.0 |

**Interpretation (no unsupported improvement claim):** live reranking improved ranking quality (MRR@5 0.9028 vs hybrid 0.8472) at substantially higher latency/cost, but **supported answer/citation rates decreased** versus hybrid on this small draft CI set (0.6667/0.4839 vs 0.8333/0.5484). Evidence is mixed; do not claim overall quality improvement without further evidence/tuning. Hybrid remains the measured default for answer-support tradeoff on this suite.
- Earlier fallback run (no live key): `evaluation/results/ci_compare_800_160_rerank_live/report.json` (not quality evidence).

## Regression gates (CI)
- Baseline file: `evaluation/baseline_metrics.json` (measured, dense). CI checks **do not invent thresholds**; they enforce non-regression vs measured baseline (recall/mrr/supported/citation/abstention minimums, invalid citations/failures not above baseline) with floating tolerance.
- CI workflow: `.github/workflows/ci.yml` (runs on PR/push). Routine CI uses deterministic extractive mode and no required live external APIs; optional comparison job is non-blocking.

## RAGAS (executed)
- Integrated in evaluation script (`evaluation/ci_eval.py`, `--enable-ragas`).
- **Compatibility issue resolved:** upgraded `ragas` (0.2.5→0.2.15 in-session) to remove Pydantic v2 `__modify_schema__` failure; updated RAGAS integration to map `GEMINI_API_KEY`→`GOOGLE_API_KEY` for LangChain Google wrappers, pass API keys explicitly, support configurable metrics/models/embedding defaults, and run with serialized/rate-limit-tolerant settings (`RunConfig`, `max_workers=1`, `batch_size=1`, `raise_exceptions=False`).
- **Execution attempts (live Gemini key available):**
  - `evaluation/results/ci_ragas_dense_800_160/report.json`: Pydantic v2 compatibility error (pre-fix).
  - `evaluation/results/ci_ragas_dense_800_160_v2/report.json`: `google_api_key` mapping issue (pre-fix).
  - `evaluation/results/ci_ragas_dense_800_160_v3/report.json`: executed but blocked by quota/embedding-model incompatibility during run.
  - **Final executed run:** `evaluation/results/ci_ragas_dense_800_160_final/report.json` (dense, extractive, 12 answerable rows). Pipeline executed to completion and produced metrics: `context_precision=0.3333333333041667`, `context_recall=0.4166666666666667`, `faithfulness=0.8605442176870749`, `answer_relevancy=null`, `answer_correctness=null` (LLM model `gemini-flash-lite-latest`; embedding model recorded as `models/text-embedding-004` in artifact while environment verification showed `models/gemini-embedding-001` working — null metrics indicate per-metric computation issues/NaNs under `raise_exceptions=False`, likely influenced by quota/embedding/runtime behavior and should be interpreted cautiously).
- **Conclusion:** RAGAS integration is no longer unexecuted; compatibility resolved and pipeline successfully executed with recorded outputs/artifacts. Some metrics are unavailable/null in this run and are reported honestly (not as zero). For stable/complete metric coverage, rerun in a compatible environment with sufficient quota and validated evaluator embedding/model configuration.

## Live Gemini evaluation (separate, measured)
- **Artifacts:** `evaluation/results/gemini_live_eval/summary.json`, `responses.json`, `analysis.json`, `analysis_summary.json` (hybrid retrieval, `mode=gemini`, configured provider/model `google_generativeai` / `gemini-flash-lite-latest`, 15-question CI set).
- **Aggregate:** 15 requests, 12 answered, 3 abstained, 0 errors. Latency (request total, ms): mean 3766.529, p50 3162.029, p95 5231.978. Generation latency observed for answered/generation-attempted cases: 926.8–10151.6 ms. Tokens (12 requests with usage): prompt 4613, completion 673, total 5286. **Estimated cost USD:** not computed in-code (pricing/SKU not auto-fetched); token counts recorded where available.
- **Honest quality assessment (automated response review, not independent human adjudication):**
  - **Correct/abstained as expected:** all 3 unanswerable questions correctly abstained (no unsupported out-of-scope answers). No invalid source IDs or uncited-claim markers detected in generated claims by automated checks.
  - **Incorrect/incomplete answerable responses (4/12):** `EXP001-Q04` incomplete (failed to list the user-sector categories present in evidence), `EXP001-Q08` incomplete (did not clearly include RTI/Remote Terminal Infrastructure component from supported evidence), `EXP001-Q10` incomplete procedural details, `EXP001-Q15` incomplete/underspecified relative to expected condition details. These are quality gaps despite grounding/citation plumbing; not claimed production-accepted.
  - **Unsupported answers:** no explicit invalid-citation/uncited-claim evidence from automated checks, but incompleteness means some answers do not fully satisfy expected information needs.

## CI dataset verification pass (separate, automated)
- **Artifact:** `evaluation/results/ci_dataset_verification.json` (scripted pass). **Result:** pass (15 questions, 12 answerable + 3 unanswerable, unique IDs, all evidence spans matched source PDF, abstention/answer fields consistent). **Independent verification not claimed.**

## CI workflow validation
- Workflow: `.github/workflows/ci.yml` (PR/push) runs deterministic tests, index build/check, fixed 15-question evaluation, and regression gate against measured baseline.
- **Local validation (this environment):** commands equivalent to CI executed successfully, including regression gate: `evaluation/results/ci_local_validation/` → `regression.passed=true` vs `evaluation/baseline_metrics.json`.
- **Actual remote PR/push run:** not executed in-session (no authenticated GitHub CLI / push-PR credentials available). Workflow syntax/commands validated locally; remote gate execution remains to be confirmed in GitHub Actions.

## Deterministic checks retained
- Evidence-span retrieval metrics (recall/precision/MRR/evidence coverage).
- Required evidence coverage and per-question evidence IDs.
- Citation validity (offsets/pages/ids/source) and citation support overlap.
- Unsupported/unstated answer proxies via statement-citation support.
- Abstention correctness for labeled unanswerables and in-corpus abstention behavior.

## Limitations
- 15-question CI set is small and draft (not independently verified); results are regression/diagnostic signals, not general production-quality proof.
- Extractive answer support metrics depend on selector behavior and answerability guard; they measure grounded support, not full semantic correctness.
