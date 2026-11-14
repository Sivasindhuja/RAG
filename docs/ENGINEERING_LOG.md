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

---

## F. Verification and test evidence

### F.1 Executed test suite

Command: `python -m pytest tests -q`

Result (executed 2026-11-14): **57 passed, 25 warnings** in ~28 s.

Warnings are non-fatal Pydantic deprecation notices raised by Chroma
(`Accessing the 'model_fields' attribute on the instance is deprecated`).

Coverage added in this work:

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

### F.2 Real-document integrity (EXP-002 chunk set)

- Source: `data/satcom-ngp.pdf`, 15 pages, 35,407 extracted characters.
- 107 chunks; 90 distinct clause sections; articles 1–4.
- 100% non-whitespace coverage; 0 uncovered non-whitespace characters.
- 0 overlaps; 0 fidelity errors; 0 bounds errors; 0 duplicate IDs;
  0 page-mapping errors; 0 empty/oversized chunks.
- 7 multi-page chunks; 5 clauses split into multiple parts.

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

---

## G. Risks and open items

| ID | Risk / open item | Impact | Mitigation / next step |
|---|---|---|---|
| R-001 | Evaluation set is small (15 Q / 20 units) | Metric deltas are noisy; overfitting to the set is easy | Require manual failure inspection; prefer clear directional results; expand dataset with verified units before large claims |
| R-002 | Compound questions lose child-clause coverage under clause-level chunking | Wrong/incomplete answers for Q08/Q13-type questions | Next controlled experiment: sibling/parent-context expansion, measured on the same fixed set |
| R-003 | Very small chunks (<100 chars) weaken embeddings | Poor ranking / MRR | Consider minimum-size merge rules inside the structure-aware strategy, as a controlled change |
| R-004 | Legacy backend has `status_status` bug and unvalidated pipeline (KNOWN-001) | App path may fail or bypass validation | Fix keyword; isolate legacy advanced components; build validated ingestion on `src/rag_core` |
| R-005 | Generation, citation rendering and abstention are not validated | Cannot yet claim grounded answers end-to-end | Build on validated retrieval; add citation-validity + faithfulness + abstention evaluation |
| R-006 | README performance claims (faithfulness 1.00, context recall 0.60) are not from this pipeline | Misleading project status | Treat as legacy; supersede with validated experiment reports |
| R-007 | Chroma telemetry warnings | Noise in logs | Ignore; does not affect correctness |
| R-008 | Environment drift (Python/Chroma/sentence-transformers versions) | Non-reproducible numbers | Record versions; pin before final release |

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

### H.2 Next actions (ordered)

1. **Decide the next experiment from evidence, not preference.** Leading
   candidate: structure-aware refinement that preserves clause metadata but
   expands retrieval context to siblings/parent for compound questions. Must hold
   PDF, questions, ground truth, embedding, dense retrieval, top-K and metrics
   constant; record config and failure analysis.
2. Alternatively, if retrieval is good enough for the first end-to-end slice:
   build validated production ingestion on `src/rag_core`, then generation with
   chunk-ID citations and explicit abstention.
3. Fix KNOWN-001 and isolate or retire unvalidated legacy pipeline components
   before exposing the application.
4. Expand the verified evaluation set before making large quality claims.
5. Add generation-phase evaluation (faithfulness, answer correctness, citation
   validity, abstention) once the generation path exists.

### H.3 Phase status

| Phase | Description | Status |
|---|---|---|
| 1 | Independent audit | Done |
| 2 | Document-processing foundation + strict integrity | Done |
| 3 | EXP-002 structure-aware chunking experiment | Done (rejected as default) |
| 4 | Justified chunking refinement (if pursued) | Not started |
| 5 | Validated production ingestion | Not started |
| 6 | Application: generation, citations, abstention | Not started |
