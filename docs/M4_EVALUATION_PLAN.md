# M4 — Evaluation Expansion Plan & Harness (initial)

**Date:** 2026-11-15  
**Status:** planning and harness foundation implemented; no retrieval/chunking/EXP-003/threshold/production-behavior tuning performed.  
**Constraints:** preserve existing 15-question dataset and ground truth; do not alter source/offsets/ground truth to improve metrics; thresholds derived from measured results only.

## 1. Current baseline (preserved)
- **Answerable (in-corpus):** `experiments/exp001_questions.csv` (15 questions, Q01–Q15) + verified evidence manifest `experiments/exp001_ground_truth.json` (20 evidence units, global offsets mapped/validated against `data/satcom-ngp.pdf`).
- **Unanswerable/out-of-corpus:** `experiments/unanswerable_questions.csv` (12 questions: 8 calibration, 4 held-out; near-domain and out-of-domain).
- **Existing metrics/utilities:** `src/rag_core/evaluation.py` (retrieval recall@k/precision@k/MRR@k, evidence coverage, `evaluate_answer_support`), experiment runners (`experiments/production_slice.py`, `experiments/m1_gemini_eval.py`), RAG service API (`/query`).

## 2. Expansion target and coverage
- **Phase 1 (initial):** expand to **50–100 manually verified questions** total (including preserved 15) with a clear path to **200**.
- **Coverage requirements (all four policy articles):** ARTICLE-1..ARTICLE-4 represented; include answerable and unanswerable questions; compound/multi-clause questions; definitions; exceptions/conditions; cross-references/procedural dependencies.
- **Question types/labels:** `answerable`, `unanswerable_in_corpus` (insufficient evidence in document), `unanswerable_out_of_scope` (outside SATCOM NGP), plus difficulty/category tags (definition, procedure, exception, cross-reference, compound, numeric/detail, abstention/near-miss).
- **Per-question required fields (verified):**
  - `id`, `question`, `type`, `article_coverage`, `category/tags`
  - for answerable: `expected_answer` (reference), `expected_evidence_units[]` with verified `source/page`, global `start_offset/end_offset` (or derivable page-relative), `excerpt`, `clause/section` labels
  - `citation_expectations`: required clause/section/page/chunk constraints and/or evidence-unit coverage expectations
  - `abstention_expected`: bool/label with rationale
  - review metadata: `verified_by`, `verification_date`, `notes`

## 3. Dataset format (v2, additive)
- Canonical evaluation dataset: JSONL (`experiments/eval/datasets/*.jsonl`) or JSON with schema version, preserving legacy CSV/manifest compatibility via adapters.
- Proposed schema (summary): `{schema_version, id, question, type, expected_answer, expected_evidence:[{evidence_id, source,page,clause_id,start_offset,end_offset,excerpt}], citation_expectations:{required_sections/pages/evidence_ids,min_citations?}, abstention:{expected,label,reason}, tags:{articles,categories,compound,cross_reference,...}, provenance:{verified_by,date,notes}}`
- **Do not fabricate verification.** New entries must be added only after manual verification against `data/satcom-ngp.pdf`.

## 4. Reproducible offline evaluation harness
- **Harness:** `experiments/m4_eval_harness.py` (foundation) runs queries against the local `RAGService` (default extractive, EXP-003, corrected prod index) without changing production behavior.
- **Inputs:** dataset file(s), config (mode/top_k/model/index), optional baseline results for comparison.
- **Outputs (artifacts):** per-question results JSONL, summary JSON/Markdown, metrics CSV/JSON, latency/timing aggregates, failure/abstention details.
- **Metrics measured:**
  - **Retrieval:** recall@k, precision@k, MRR@k, evidence coverage/missing evidence (for answerable with verified evidence).
  - **Answer support/correctness proxies:** evidence-supported statement/citation counts and rates via span overlap with verified evidence (`evaluate_answer_support`), selected-evidence coverage.
  - **Completeness:** required evidence-unit coverage ratio and missing required evidence/clauses.
  - **Faithfulness:** supported vs unsupported statement/citation rates against verified evidence (span-grounded); app citation validity checks.
  - **Citation correctness/coverage:** citation presence for answered claims, offset/page/section consistency with source metadata and verified evidence overlap, coverage of required evidence/citations.
  - **Abstention:** abstention rate/accuracy on labeled abstention cases and answerable cases (false abstentions), abstention reasons.
  - **Latency:** per-request total and stage timings (p50/p95/mean/max) from service trace/timings.
- **Generation-mode note:** Gemini evaluation remains optional and **M1 not accepted**; harness records mode/config and reports separately. No production default change.

## 5. Thresholds (measured, not arbitrary)
- Run baseline on preserved datasets with fixed config and record measured metrics as **baseline references**.
- Define candidate quality gates/thresholds only after sufficient dataset size and stable measurements (e.g. non-regression vs baseline, minimum evidence/citation validity, abstention behavior targets) — to be set in follow-up once measurements exist.
- Any threshold proposal must cite measured baseline and variance; no invented targets.

## 6. Execution plan (incremental)
1. **Foundation (done in this step):** schema/plan doc, harness scaffold, baseline run on existing 15q+12 unanswerable set.
2. **Curation (manual, pending):** author/verify 35–85 additional questions to reach 50–100, balanced across articles/types; peer/author verification recorded.
3. **Scale to 200:** continue curation with same verification standard and coverage tracking.
4. **CI integration (M5/CI):** add reproducible evaluation command and measured-threshold gates once baselines/thresholds are established.

## 7. Risks/limitations
- 15-question set too small for general claims (known). Manual verification is labor-intensive and must not be shortcut.
- Span-overlap/lexical support metrics are proxies, not full semantic correctness judgments; human review remains necessary for correctness/completeness/faithfulness sign-off.
- Harness evaluates current system behavior; it must not be used to justify untested RAG technique changes.
