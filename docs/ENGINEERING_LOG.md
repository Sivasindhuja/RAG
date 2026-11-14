# Engineering Log — SATCOM NGP RAG

Status: living document. Last updated 2026-11-14.
Scope: decisions, defects, experiments, rejected approaches, verification evidence,
risks and next actions for the SATCOM NGP RAG project.

Companion documents: `docs/PROJECT_BRIEF.md`, `docs/EXP-001.md`,
`docs/EXP-002.md`, `docs/CHUNKING_STRATEGY.md`.

Convention: entries are append-only in spirit. Do not rewrite history to make a
later decision look inevitable; record what was known at the time.

---

## A. Purpose and how to read this log

This log is the audit trail for *why* the system looks the way it does. It exists
so that:

- a defect fix can be traced to a root cause and a regression test;
- an experiment can be reproduced and judged on its recorded numbers;
- a rejected approach is not silently reintroduced later;
- a reader can tell the difference between **validated**, **legacy** and
  **not yet built** components.

Sections:

- **B** — architecture decisions.
- **C** — defect log.
- **D** — experiment log.
- **E** — rejected approaches.
- **F** — verification and test evidence.
- **G** — risks and open items.
- **H** — change history and next actions.

---

## B. Architecture decisions

### DR-001 — Page-aware heading scanning

**Context.** The PDF extractor can place a standalone page number immediately
before the first clause of the next page. Scanning the concatenated document text
line-by-line fused the footer into the clause label (page footer `2` + next-page
clause `2.4` → `22.4`; page footer `12` + `4.1` → `124.1`). The fused label also
attached the clause to the wrong page.

**Decision.** Detect headings by scanning lines **per `SourceDocument` page**,
using page-local line offsets, and treat a standalone line equal to the current
page number as page furniture rather than heading content.

**Consequences.**
- Clause IDs and page metadata reflect real page boundaries.
- The scanner can no longer see across pages, so multi-page clauses are handled by
  the document model (global offsets), not by fusing lines.
- Regression coverage:
  `test_page_footer_number_is_not_fused_into_next_page_clause_id`,
  `test_real_pdf_detects_all_four_articles_and_normalised_clause_ids`.

### DR-002 — Conservative, article-aware clause-ID normalisation

**Context.** Even with page-aware scanning, some extractor artefacts remain. A
naive "strip the page-number prefix" fix would corrupt legitimate IDs: a real
clause `12.4.1` inside `ARTICLE-12` could be mistaken for page `12` + clause
`4.1`.

**Decision.** Only normalise a raw ID when **all** of the following hold:

1. the raw ID does **not** already start with the current article number;
2. the raw ID starts with the current page number;
3. the remainder is a valid dotted numeric path;
4. the remainder's **first component equals the current article number**.

Otherwise the raw ID is kept unchanged.

**Consequences.**
- `22.4` → `2.4` and `124.1` → `4.1` on article boundaries.
- `12.4.1` inside `ARTICLE-12` is preserved.
- Without article context, no stripping is attempted.
- Regression coverage:
  `test_normalise_section_id_removes_page_number_prefix_artefacts`,
  `test_normalise_section_id_preserves_legitimate_clause_ids`.

### DR-003 — Heading metadata must not swallow body prose

**Context.** An early heading-block implementation absorbed any following line
into the heading, which pulled ordinary body prose (and page-number furniture)
into `heading` metadata.

**Decision.** Only absorb a continuation line when it is heading-like (≥75% of its
letters are upper-case). Ordinary body prose and page furniture close the heading
block. Heading text is metadata only; the source text in the chunk is unchanged.

**Consequences.**
- `3.4.1` heading is `3.4.1 Authorisation from Department of Space …`, not the
  following UN Outer Space Treaty paragraph.
- `4.3.3` heading does not contain the page footer `13`.
- Multi-line ALL-CAPS headings are still joined correctly.
- Regression coverage: `test_clause_heading_metadata_is_not_the_whole_body`,
  `test_page_footer_number_is_not_part_of_heading_metadata`,
  `test_heading_wrapped_across_lines_is_absorbed`.

### DR-004 — Chunk IDs include the source offset

**Context.** Duplicate or malformed section labels can occur (e.g. two `2.1`
blocks, or a split clause). An ID built only from article/section/part can
collide, which breaks Chroma upserts and evidence mapping.

**Decision.** Chunk IDs are
`<source>::<article|PREAMBLE>::<section|BODY>[::part_N]::off<global_start>`.
The `::off` suffix makes IDs unique even for duplicate labels and stable across
runs.

**Consequences.**
- Unique IDs verified on the real PDF and on a synthetic duplicate-label case.
- IDs remain traceable to the exact global character offset.
- Regression coverage: `test_chunk_ids_are_unique_even_for_duplicate_section_labels`,
  `test_real_pdf_chunk_ids_are_unique_and_stable`.

### DR-005 — Strict source-coverage audit is a release gate

**Context.** The original audit only checked that chunks were slices of the
source. It could not detect dropped content, overlaps, duplicate IDs, wrong page
metadata or wrong ordering.

**Decision.** `audit_source_coverage` verifies, for every chunk set:

- exact source slices (fidelity);
- offset bounds;
- no overlaps;
- **complete non-whitespace coverage**;
- unique IDs;
- contiguous `chunk_sequence` and source order;
- page metadata derivable from offsets;
- no empty or oversized chunks.

Experiments fail before retrieval if the audit is not `ok`.

**Consequences.**
- Real PDF: 107 chunks, 90 distinct sections, 4 articles, 100% non-whitespace
  coverage, 0 overlaps, 0 fidelity/page/ID errors.
- Regression coverage:
  `test_real_pdf_integrity_audit_passes_every_check`,
  `test_audit_detects_uncovered_non_whitespace_content`,
  `test_audit_detects_chunk_text_mutation`.

### DR-006 — One shared core for experiments and production

**Context.** The repository contained multiple chunking/retrieval paths and a
legacy backend with advanced unvalidated components. Divergent implementations
make results incomparable and let production silently differ from what was
measured.

**Decision.** Canonical document, chunking, dense index and evaluation code lives
in `src/rag_core/`. New experiments import it. `experiments/exp001.py` is the
single frozen exception because it is the accepted historical baseline and must
remain byte-for-byte reproducible.

**Consequences.**
- EXP-002 uses the same metrics and dense retrieval as the baseline via
  `src.rag_core.evaluation` and `src.rag_core.index`.
- The legacy `backend/` and `src/{chunking,retrieval,generation,...}` paths are
  **not** the validated production path and must not be presented as such.
- Regression coverage: `test_dense_index_build_search_and_rebuild`,
  `test_chunk_metadata_is_scalar_and_preserves_traceability`.

### DR-007 — Productionise the accepted strategy, do not invent a new one

**Context.** EXP-002 was rejected. Production therefore needs the accepted
fixed-character 1000/200 strategy without importing the frozen EXP-001 runner.
Its intentional overlap also conflicted with the strict "no overlaps" audit rule.

**Decision.** Add `FixedCharacterChunker` to the shared `src/rag_core/chunker.py`
and add `allow_overlap` to `audit_source_coverage`. `allow_overlap` suppresses
only the overlap *issue*; overlap counts are still reported and complete
non-whitespace coverage, fidelity, ordering, ID uniqueness and page mapping
remain mandatory.

**Consequences.**
- PROD-001 reproduces all 49 accepted EXP-001 v3 chunk texts and page assignments
  exactly.
- Retrieval parity is exact: 0.8889 / 0.2533 / 0.7944.
- Regression coverage: `tests/test_fixed_chunker.py` (6 tests), including
  chunk-by-chunk equivalence with the frozen accepted chunks.

### DR-008 — Extractive answers first; citations only for contributing chunks

**Context.** A generation path was not justified yet, and the project requires
traceable claims. A naive implementation cited every retrieved chunk, which
implied support from chunks that contributed nothing to the answer.

**Decision.** The default answer mode is deterministic extractive assembly:

1. consider at most 5 retrieved chunks;
2. split them into sentences and score candidate sentences with IDF-weighted
   question-token overlap, a clause/list bonus, and a penalty for ALL-CAPS
   headings;
3. select at most 3 distinct sentences;
4. assign markers only to chunks that actually contributed a selected sentence;
5. record each statement's global source offsets.

**Consequences.**
- Every extractive statement is a verbatim source slice; faithfulness is
  guaranteed by construction.
- Retrieved-but-unused chunks are not cited (BUG-009).
- Relevance is not guaranteed. The lexical selector misses relevant sentences in
  Q10/Q13 and cannot recover evidence absent from the top-5 (Q05/Q08).
- Regression coverage: `tests/test_pipeline.py`, including
  `test_citations_omit_retrieved_chunks_that_do_not_contribute` and
  `test_all_caps_heading_is_deprioritised_below_body_text`.

### DR-009 — Abstention is a small, model-specific heuristic

**Context.** The application must not answer from parametric memory. A general
answerability classifier was not available or justified.

**Decision.** Use a configurable retrieval-confidence policy with decision order:
empty question → no hits → hard distance limit → minimum lexical support →
soft distance plus strict lexical support. Defaults:
`hard=1.30`, `soft=1.10`, `min_support=0.50`, `strict_support=0.70`.

**Consequences.**
- 12/12 out-of-corpus smoke questions abstained (8 calibration, 4 held-out).
- All 15 in-corpus questions were answered, so no false abstention was observed
  on that set.
- The policy does not detect in-corpus retrieval failure (Q05): retrieved text
  shares vocabulary and passes the thresholds.
- Thresholds must be recalibrated for any embedding or index change.
- Regression coverage: abstention branches in `tests/test_pipeline.py`.

---

## C. Defect log

All defects below were found during the Phase-2 independent audit and fixed in
this work. "Test" refers to an executed regression test in `tests/`.

| ID | Severity | Symptom | Root cause | Fix | Regression test | Status |
|---|---|---|---|---|---|---|
| BUG-001 | High | False clause IDs `22.4`, `32.5.7`, `124.1`; clause attached to wrong page | Heading scan over concatenated text fused page footer with next-page clause | Per-page line scan + page-furniture detection (DR-001) | `test_page_footer_number_is_not_fused_into_next_page_clause_id` | Fixed |
| BUG-002 | High | Naive prefix stripping could corrupt legitimate IDs such as `12.4.1` | Unconditional page-prefix removal | Conservative article-aware normalisation (DR-002) | `test_normalise_section_id_preserves_legitimate_clause_ids` | Fixed |
| BUG-003 | High | Duplicate chunk IDs broke uniqueness/upsert/evidence mapping | ID built only from label + part | Append `::off<global_start>` (DR-004) | `test_chunk_ids_are_unique_even_for_duplicate_section_labels` | Fixed |
| BUG-004 | Medium | `heading` metadata contained body paragraphs and page numbers | Heading block absorbed any continuation line | Absorb only heading-like uppercase lines; close on page furniture (DR-003) | `test_clause_heading_metadata_is_not_the_whole_body`, `test_page_footer_number_is_not_part_of_heading_metadata` | Fixed |
| BUG-005 | Medium | `page_for_offset` misattributed boundary offsets and end-of-document offsets | Ambiguous interval/end handling | Half-open intervals; document end attributed to last page; reject negative/out-of-range | `test_page_for_offset_boundaries_and_invalid_offsets` | Fixed |
| BUG-006 | High | Audit passed despite dropped content, overlaps, duplicate IDs or wrong page metadata | Audit only checked source slices | Full strict audit + experiment gate (DR-005) | `test_real_pdf_integrity_audit_passes_every_check`, `test_audit_detects_uncovered_non_whitespace_content`, `test_audit_detects_chunk_text_mutation` | Fixed |
| BUG-007 | High | Malformed/unrecognised headings could lose body text | Body text after an unrecognised heading had no owner | Emit unsectioned body chunks; only structurally valid clause regex creates headings; content preserved | `test_headingless_document_yields_single_body_chunk_without_loss`, `test_lowercase_after_clause_number_does_not_lose_content`, `test_unexpected_numeric_heading_format_is_not_treated_as_clause`, `test_ocr_corrupted_clause_heading_is_still_detected` | Fixed |
| BUG-008 | Medium | Experiments and production could drift into separate chunking/retrieval implementations | Duplicated logic and legacy paths | Shared `src/rag_core` index/evaluation; EXP-001 frozen as the documented exception (DR-006) | `test_dense_index_build_search_and_rebuild`, `test_chunk_metadata_is_scalar_and_preserves_traceability` | Fixed |
| BUG-009 | Medium | Citation list included retrieved-but-unused chunks, overstating support | Initial assembly cited all top-K hits before sentence selection | Select statements first; cite only contributing chunks with renumbered markers (DR-008) | `test_citations_omit_retrieved_chunks_that_do_not_contribute` | Fixed |

### Open known issue (not part of BUG-001…008)

- **KNOWN-001 — legacy backend keyword error.** `backend/main.py` contains
  `raise HTTPException(status_status=429, ...)`; the correct keyword is
  `status_code`. The backend also references a legacy `satcom_ngp_clauses`
  collection and an unvalidated advanced pipeline. It is not on the validated
  experiment path. Must be fixed/isolated before Phase 6 (application).

---

## D. Experiment log

### EXP-001 — Fixed-character baseline (ACCEPTED)

**Hypothesis / role.** Establish a naive fixed-character retrieval baseline
against which later chunking changes are measured.

**Configuration (frozen).**

| Parameter | Value |
|---|---|
| Chunking | Fixed-character sliding window |
| Chunk size / overlap / step | 1000 / 200 / 800 |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | Chroma, isolated collection |
| Retrieval | Dense-only, top-5 |
| Metrics | Recall@5, Precision@5, MRR@5 |
| Dataset | 15 approved questions / 20 verified evidence units |

**Results (EXP-001 v3).**

| Metric | Value |
|---|---:|
| Recall@5 | 0.8889 |
| Precision@5 | 0.2533 |
| MRR@5 | 0.7944 |

Per-question notes: Q05 Recall@5 = 0.0, Q08 = 0.3333, Q13 = 1.0.

**Decision.** **Accepted** as the reference baseline. Artifacts in
`experiments/results/exp001_v3/` are frozen; `experiments/exp001.py` and the
approved questions/ground truth must not be modified. See `docs/EXP-001.md`.

**Caveats.** The original `experiments/results/exp001/` run is invalid
(wrong question set, heading-only ground truth) and must not be used.

### EXP-002 — Structure-aware chunking (REJECTED as default)

**Hypothesis.** Clause-anchored, structure-aware chunks (max 1000 chars,
paragraph-aware splitting, no overlap) improve retrieval quality over the
fixed-character baseline by keeping legally meaningful units intact.

**Controlled change.** Only chunking changed. Held constant: PDF, 15 questions,
20 evidence units, embedding model, dense-only retrieval, top-5, metric
definitions.

**Configuration.**

| Parameter | Value |
|---|---|
| Strategy | `structure_aware` |
| Max chunk size | 1000 chars |
| Overlap | 0 |
| Splitting | Paragraph-aware; character fallback for a single oversized paragraph |
| Heading scan | Page-aware |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | Chroma, `satcom_ngp_exp002_structure_aware` |
| Retrieval | Dense-only, top-5 |

**Document integrity (pre-retrieval gate passed).**

- 107 chunks, 90 distinct sections, 4 articles.
- 100% non-whitespace coverage, 0 overlaps, 0 fidelity errors, 0 page-mapping
  errors, 0 duplicate IDs, 0 empty/oversized chunks.
- Character lengths: min 22, median 256, p90 761, max 1000.
- 16 chunks < 100 chars; 7 multi-page chunks; 5 clauses split into parts
  (`2.2`, `3.5`, `3.6.1`, `3.7.2.1`, `4.2`).

**Results.**

| Metric | EXP-002 | EXP-001 v3 | Delta |
|---|---:|---:|---:|
| Recall@5 | 0.8833 | 0.8889 | −0.0056 |
| Precision@5 | 0.1867 | 0.2533 | −0.0666 |
| MRR@5 | 0.6244 | 0.7944 | −0.1700 |

Per-question movement: 1 improved, 8 regressed, 6 unchanged.

Key per-question observations:

- **Q05** improved 0.0 → 1.0 Recall@5 (the clause chunk exposed the answer that
  fixed-character chunks missed).
- **Q08** fell 0.3333 → 0.0. The retriever found parent clause `3.4` but not the
  required child clauses `3.4.1`, `3.4.2`, `3.4.3`.
- **Q13** fell 1.0 → 0.25. The retriever found `4.1`, `4.3` and `4.3.4` but not
  `4.3.1`, `4.3.2`, `4.3.3`.
- MRR dropped broadly because small clause chunks often ranked below less-specific
  neighbours.

**Cause analysis.** Clause-level splitting improves pinpoint precision for
single-clause questions but fragments compound answers. Fixed-character chunks
crossed clause boundaries and accidentally covered several required child clauses
in one hit. EXP-002 chunks are often too small to carry enough embedding signal
(16 chunks < 100 chars) and separate sibling clauses that the question needs
together.

**Decision.** **Rejected as the production default for this configuration.**
The aggregate Recall change is small, but Precision and MRR are materially worse
and the compound-question regressions are real. The structure-aware chunker is
retained as a metadata-rich baseline and a foundation for a justified
parent-context/sibling-expansion experiment. See `docs/EXP-002.md` and
`experiments/results/exp002/`.

### PROD-001 — Production slice over the accepted baseline

**Role.** Productionise the accepted EXP-001 strategy in the shared core and
evaluate the grounded answer layer. This is not a retrieval experiment.

**Configuration.**

| Parameter | Value |
|---|---|
| Chunking | `fixed_character`, 1000 / 200 / step 800, per page |
| Chunks | 49 |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` |
| Retrieval | dense-only, top-5 |
| Answer mode | extractive verbatim, max 3 sentences from 5 chunks |
| Abstention | hard 1.30 / soft 1.10 / min support 0.50 / strict 0.70 |

**Document integrity.**

- 49 chunks; 100% non-whitespace coverage (29,250 / 29,250 chars).
- 6,800 intentional overlap characters across 34 adjacent windows.
- 0 fidelity/bounds/duplicate-ID/page-mapping errors; 0 empty/oversized chunks.
- Audit passes with `allow_overlap=True`.

**Retrieval.** Exact parity with EXP-001 v3: Recall@5 0.8889, Precision@5 0.2533,
MRR@5 0.7944 (delta 0.0000 on all three).

**Answer layer (15 in-corpus questions).**

| Measure | Value |
|---|---:|
| Answered | 15 / 15 |
| Answers with ≥1 evidence-supported statement | 12 / 15 |
| Statements supported | 13 / 45 (0.2889) |
| Citations supported | 15 / 32 (0.4688) |
| Mean statements / citations per answer | 3.00 / 2.13 |

**Abstention smoke test.** 8/8 calibration and 4/4 held-out out-of-corpus
questions abstained (12/12). All 15 in-corpus questions were answered.

**Observed failures.**

- Q05: Recall@5 0.0. The required `2.5.6` evidence is absent, so the extractive
  answer is unsupported. The abstention policy does not catch it.
- Q08: Recall@5 0.3333. Only one of the three required `3.4.1`–`3.4.3` clauses is
  retrieved.
- Q10: Recall@5 1.0 but 0/3 supported statements; the selector chooses a long
  list block and later clauses instead of the conflict procedure.
- Q13: Recall@5 1.0 but 1/3 supported statements; sibling categories
  `4.3.2`/`4.3.3` are retrieved but not selected.

**Decision.** Accept the slice as the **validated retrieval-and-provenance
foundation**. Do not present it as a complete answer engine. Answer relevance and
multi-clause completeness are open defects, not hidden by the exact retrieval
parity. See `docs/PRODUCTION_SLICE.md` and
`experiments/results/production_slice/`.

**Reproduction.** `python -m experiments.production_slice` writes config, chunks,
integrity audit, retrieval results, answers, abstention results,
per-question metrics and summary. Observed latency in the recorded run: index
build 1,927.89 ms; pipeline mean 32.44 ms, median 32.50 ms, max 34.67 ms.

---

## E. Rejected approaches

| ID | Approach | Why rejected / status |
|---|---|---|
| RA-001 | Naive page-prefix stripping to fix fused IDs | Corrupts legitimate IDs (`12.4.1` in `ARTICLE-12`). Replaced by DR-002. |
| RA-002 | Scan headings over concatenated page text | Causes footer/clause fusion and wrong page attribution. Replaced by DR-001. |
| RA-003 | Accept EXP-002 structure-aware as the production default because recall is nearly flat | Precision −0.0666, MRR −0.1700, Q08/Q13 regressions. Rejected. |
| RA-004 | Add hybrid retrieval / reranking / query expansion / parent-child now | No controlled evidence that these fix the observed failure; violates the evidence-first rule. Deferred until a specific experiment justifies each. |
| RA-005 | Use heading-only spans as ground truth | Invalid: headings do not support the answer. Used only by the invalid original EXP-001 run. |
| RA-006 | Adjust questions/ground truth to raise metrics | Prohibited. Ground truth may only be corrected with explicit, verified evidence and a recorded rationale. |
| RA-007 | Treat the legacy `backend/` pipeline as validated production | Unvalidated advanced components and an open keyword bug (KNOWN-001). Not accepted. |
| RA-008 | Tune the lexical answer selector further, or add a reranker/cross-encoder, on the 15-question set | Exploratory sweeps (sentence count, heading penalty, length normalisation, relative score floors) did not remove the Q10/Q13 selection failures and risk overfitting a small set. Deferred to a controlled answer-layer experiment with explicit acceptance criteria. |

---

## F. Verification and test evidence

### F.1 Executed test suite

Command: `python -m pytest tests -q`

Result (executed 2026-11-14, after PROD-001): **78 passed, 25 warnings** in
~28.7 s. The earlier Phase-2/3 suite result was 57 passed; PROD-001 added 21
tests.

Warnings are non-fatal Pydantic deprecation notices raised by Chroma
(`Accessing the 'model_fields' attribute on the instance is deprecated`).

Coverage added across this work:

- `tests/test_rag_core.py` — 25 tests: document offsets, page-boundary mapping,
  real-PDF fidelity/order/bounds, all four articles, footer fusion regressions,
  multi-page clauses, heading metadata, oversized-clause splitting, headingless
  documents, conservative normalisation, strict coverage audit, duplicate IDs,
  stable IDs, page-relative offsets, malformed/OCR headings, lowercase
  continuation, wrapped headings, numbered body lists.
- `tests/test_rag_evaluation.py` — 10 tests: approved 20-unit ground truth maps
  exactly to global offsets, 15-question set preserved, metric definitions
  (single/multi evidence, missing unit, no hits, empty evidence), scalar
  traceable metadata, dense index build/search/rebuild, empty-query rejection.
- `tests/test_fixed_chunker.py` — 6 tests: chunk-by-chunk equivalence with the
  frozen 49 accepted EXP-001 v3 chunks, overlap/page boundaries, audit behaviour,
  unique/stable IDs, empty pages, invalid configuration.
- `tests/test_pipeline.py` — 15 tests: extractive verbatim answers, sentence
  provenance, citation-only-contributors, heading deprioritisation, abstention
  branches, generator hook, token support, limit/dedupe, fallback.

### F.2 Real-document integrity (EXP-002 chunk set)

- Source: `data/satcom-ngp.pdf`, 15 pages, 35,407 extracted characters.
- 107 chunks; 90 distinct clause sections; articles 1–4.
- 100% non-whitespace coverage; 0 uncovered non-whitespace characters.
- 0 overlaps; 0 fidelity errors; 0 bounds errors; 0 duplicate IDs;
  0 page-mapping errors; 0 empty/oversized chunks.
- 7 multi-page chunks; 5 clauses split into multiple parts.

### F.2b Real-document integrity (PROD-001 fixed-character chunk set)

- 49 chunks across 15 pages; 35,407 extracted characters.
- 100% non-whitespace coverage (29,250 / 29,250); 6,800 intentional overlap
  characters across 34 adjacent windows.
- 0 overlaps treated as errors (`allow_overlap=True`), 0 fidelity errors,
  0 bounds errors, 0 duplicate IDs, 0 page-mapping errors, 0 empty/oversized
  chunks.
- Chunk texts and page assignments match the accepted EXP-001 v3 set exactly.

### F.3 Ground-truth verification

- 20/20 approved evidence units reproduce their exact excerpt from the canonical
  document after page-relative → global offset mapping.
- No unit is missing; no excerpt mismatch.

### F.4 Reproducibility

- `experiments/exp002.py` writes `config.json`, `chunks.json`,
  `integrity_audit.json`, `chunk_stats.json`, `evidence.json`,
  `retrieval_results.json`, `evidence_coverage.json`, `comparison.json`,
  `failure_analysis.json`, `per_question_metrics.csv`, `summary.md` under
  `experiments/results/exp002/`.
- CSV/metric comparison keys follow the configured `--top-k`; baseline comparison
  is only reported when the baseline contains the same metric key (prevents
  misleading deltas at a different K).
- Retrieval latency (observed): mean 35.63 ms, median 34.94 ms, p90 40.61 ms.
  Index build: 2,714.7 ms for 107 chunks.
- PROD-001 writes `config.json`, `chunks.json`, `integrity_audit.json`,
  `retrieval_results.json`, `answers.json`, `abstention_results.json`,
  `per_question_metrics.csv` and `summary.md` under
  `experiments/results/production_slice/`.
- PROD-001 observed latency: pipeline mean 32.44 ms, median 32.50 ms, max
  34.67 ms; index build 1,927.89 ms for 49 chunks.

---

## G. Risks and open items

| ID | Risk / open item | Impact | Mitigation / next step |
|---|---|---|---|
| R-001 | Evaluation set is small (15 Q / 20 units) | Metric deltas are noisy; overfitting to the set is easy | Require manual failure inspection; prefer clear directional results; expand dataset with verified units before large claims |
| R-002 | Compound questions lose child-clause coverage under clause-level chunking | Wrong/incomplete answers for Q08/Q13-type questions | Next controlled experiment: sibling/parent-context expansion, measured on the same fixed set |
| R-003 | Very small chunks (<100 chars) weaken embeddings | Poor ranking / MRR | Consider minimum-size merge rules inside the structure-aware strategy, as a controlled change |
| R-004 | Legacy backend has `status_status` bug and unvalidated pipeline (KNOWN-001) | App path may fail or bypass validation | Fix keyword; isolate legacy advanced components; build validated ingestion on `src/rag_core` |
| R-005 | Generation and LLM faithfulness are not validated | Cannot claim generative answer quality | Extractive citations/abstention now have a baseline; evaluate any generator separately for faithfulness and citation validity |
| R-006 | README performance claims (faithfulness 1.00, context recall 0.60) are not from this pipeline | Misleading project status | Treat as legacy; supersede with validated experiment reports |
| R-007 | Chroma telemetry warnings | Noise in logs | Ignore; does not affect correctness |
| R-008 | Environment drift (Python/Chroma/sentence-transformers versions) | Non-reproducible numbers | Record versions; pin before final release |
| R-009 | Lexical sentence selection misses evidence that retrieval already found (Q10, Q13) | Incomplete/irrelevant answers despite perfect recall | Controlled answer-layer experiment; do not add a reranker without measurement |
| R-010 | Multi-clause completeness not met; Q05 retrieval recall is 0.0 | Q08/Q13 answers lack required child clauses; Q05 has no evidence in top-5 | Treat as open acceptance failure; decide retrieval vs answer-layer fix by experiment |
| R-011 | Abstention policy does not detect in-corpus retrieval failure | Q05 is answered without supporting evidence | Add answerability/relevance checks only with a labelled evaluation set; do not tune on the 15 questions |

---

## H. Change history and next actions

### H.1 Change history (this work)

1. Audited repository, PDF, EXP-001 artifacts, legacy backend/frontend and the
   new `src/rag_core`.
2. Reproduced the existing test suite; then found and fixed the integrity defects
   BUG-001…BUG-008 with regression tests.
3. Built shared `src/rag_core` modules: `index.py` (dense index) and
   `evaluation.py` (verified evidence + generic metrics).
4. Added `tests/test_rag_core.py` and `tests/test_rag_evaluation.py`.
5. Ran EXP-002 (structure-aware chunking) with full audit and comparison.
6. Rejected EXP-002 as the production default based on metrics **and** failure
   inspection.
7. Wrote `docs/PROJECT_BRIEF.md`, `docs/ENGINEERING_LOG.md`, `docs/EXP-002.md`.
8. Cleaned EXP-002 runner: dynamic `top_k` metric keys/CSV headers; guarded
   baseline comparison when K differs; added `chroma_db_exp002/` to `.gitignore`.
9. Added `FixedCharacterChunker` and overlap-aware strict audit; verified exact
   reproduction of the 49 accepted EXP-001 v3 chunks and retrieval parity.
10. Built `GroundedPipeline`: extractive verbatim answers, statement-level global
    offsets, contributing-chunk citations, heading-aware sentence scoring and a
    configurable abstention policy (DR-007…DR-009).
11. Ran PROD-001 over 15 in-corpus and 12 out-of-corpus questions; recorded
    integrity, retrieval, answer/citation support and abstention results; fixed
    BUG-009; documented the answer-layer failures rather than hiding them.
12. Wrote `docs/PRODUCTION_SLICE.md`; updated brief and this log; full suite now
    78 passed.

### H.2 Next actions (ordered)

1. **Run a controlled answer-layer experiment, retrieval fixed.** Observed
   justification: Q10 and Q13 have the required evidence in the top-5 but the
   lexical selector misses it. Test one change at a time (for example sibling
   clause expansion or sentence reranking) on the same questions/evidence.
2. **Decide retrieval work from evidence.** Q05 (Recall@5 0.0) and Q08 (0.3333)
   are genuine retrieval misses. Any expansion/reranking must be a separate
   controlled experiment and must not silently replace the accepted baseline.
3. Fix KNOWN-001 and isolate or retire unvalidated legacy pipeline components
   before exposing the application.
4. Expand the verified evaluation set before making broader quality claims.
5. Add generation-phase evaluation (faithfulness, answer correctness, citation
   validity, abstention) only when a generation path is deliberately introduced.

### H.3 Phase status

| Phase | Description | Status |
|---|---|---|
| 1 | Independent audit | Done |
| 2 | Document-processing foundation + strict integrity | Done |
| 3 | EXP-002 structure-aware chunking experiment | Done (rejected as default) |
| 4 | Justified chunking refinement (if pursued) | Not started |
| 5 | Validated production ingestion | Done for accepted baseline (PROD-001) |
| 6 | Application: generation, citations, abstention | Partial: extractive citations + abstention validated; generation not started |
