# M4 — Evaluation Expansion Progress (2026-11-15)

**Status:** foundation implemented, baseline measured on preserved datasets. **No retrieval/chunking/EXP-003/threshold/production-behavior changes made.**  
**Preserved:** `experiments/exp001_questions.csv`, `experiments/exp001_ground_truth.json`, `experiments/unanswerable_questions.csv`, and all accepted experiment history/artifacts.

## Implemented (foundation)
- `docs/M4_EVALUATION_PLAN.md`: expansion plan (50–100 verified questions initially, path to 200), required per-question fields/labels, coverage targets (all four policy articles; answerable/unanswerable; compound/definitions/exceptions/cross-references), v2 dataset schema direction, reproducible harness design, metric definitions, and threshold policy (measured baselines only, no arbitrary targets).
- `experiments/m4_eval_harness.py`: reproducible offline harness running against local `RAGService` (extractive default, EXP-003, corrected prod index `satcom_ngp_production_fixed_char`) without changing production behavior.
  - Inputs: preserved `exp001_questions.csv` + `exp001_ground_truth.json` + `unanswerable_questions.csv` (adapters for legacy format).
  - Outputs: `answerable_results.json`, `unanswerable_results.json`, `summary.json`, `summary.md`.
  - Measures: retrieval (recall@k/precision@k/MRR@k, evidence overlap), answer support/faithfulness proxies via span overlap with verified evidence (`evaluate_answer_support`), completeness proxies (supported statement/citation counts), citation structural correctness/coverage proxies, abstention behavior/accuracy on labeled set, and per-request total/stage latency (p50/p95/mean/max).

## Baseline run (executed, measured)
- **Command:** `PYTHONPATH=. python -m experiments.m4_eval_harness --out experiments/results/m4_baseline --mode extractive --k 5`
- **Artifacts:** `experiments/results/m4_baseline/{answerable_results.json,unanswerable_results.json,summary.json,summary.md}`
- **Config/defaults:** extractive, k=5, `sentence-transformers/all-MiniLM-L6-v2`, `satcom_ngp_production_fixed_char`, EXP-003 `sibling_score_ratio=0.5`, answerability threshold 0.5, opt-ins off.

### Measured summary (27 questions total: 15 answerable + 12 unanswerable)
- **Answerable (15):** answered 14, abstained 1 (Q05, guard), supported answers 12 (**supported_answer_rate 0.80**), statement support rate **0.3333**, citation support rate **0.5161**.
- **Retrieval (answerable):** mean recall@5 **0.8889**, mean precision@5 **0.2533**, mean MRR@5 **0.7944** (consistent with EXP-001 baseline on preserved set).
- **Unanswerable (12, preserved set):** correct abstentions 12 (**abstention_accuracy 1.00**) under current extractive+guard behavior.
- **Latency (ms, includes cold first-call/model/index load):** total count 27, mean 658.412, p50 19.704, p95 22.648, max 17265.834; retrieval mean 1168.481 / p50 18.911 / p95 5194.283 / max 17263.756; selection+extractive mean 1.655 / p50 1.606 / p95 2.003. Cold-start outlier should be separated in future steady-state measurement.

### Interpretation notes / limitations
- Baseline is on the preserved 15-question set only; **not a production-readiness claim**.
- Span-overlap/support metrics are grounding proxies, not full semantic correctness/completeness/faithfulness judgments; human/manual review remains required for expanded dataset sign-off.
- Statement/citation support rates are sensitive to extractive statement granularity vs verified evidence spans; investigate as measurement refinement during curation (without changing production behavior).
- No thresholds established yet beyond baseline references; thresholds/gates must be derived from measured results on expanded, verified datasets.

## Next steps (M4, pending manual curation)
1. Author and manually verify 35–85 additional questions (to reach 50–100) covering ARTICLE-1–ARTICLE-4 with required tags/types/evidence/citation expectations/abstention labels and verification metadata; do not fabricate.
2. Add v2 JSONL dataset(s) under `experiments/eval/datasets/` and extend harness adapters to consume them alongside legacy sets.
3. Re-run harness to produce expanded baseline, analyze per-category/article gaps, and propose measured (non-arbitrary) thresholds/CI quality gates.
4. Scale curated dataset toward 200 with same verification standard and coverage tracking.
5. Keep Gemini-mode evaluation separate/optional (M1 not accepted); record model/config/cost when available. No production behavior changes as part of dataset expansion.