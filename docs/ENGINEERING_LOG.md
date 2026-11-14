# Engineering Log — SATCOM NGP RAG

Status: living document. Last updated 2026-11-14.
Scope: decisions, defects, experiments, rejected approaches, verification evidence,
risks and next actions for the SATCOM NGP RAG project.

Companion documents: `docs/PROJECT_BRIEF.md`, `docs/EXP-001.md`,
`docs/EXP-002.md`, `docs/EXP-003.md`, `docs/EXP-004.md`, `docs/EXP-005.md`,
`docs/PRODUCTION_SLICE.md`, `docs/CHUNKING_STRATEGY.md`.

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
  Q10 and cannot recover evidence absent from the top-5 (Q05/Q08). Q13 was later
  resolved by sibling-clause selection (EXP-003, DR-010).
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

### DR-010 — Sibling-clause answer selection (EXP-003)

**Context.** PROD-001 retrieved all required Q13 clauses (`Recall@5 = 1.0`) but
selected only one of the four category clauses. The lexical selector chose
`4.3.4` plus two non-clause sentences, covering `1/3` evidence units. This was a
concrete answer-layer failure, not a retrieval failure.

**Decision.** When the highest-scoring sentence starts with a numbered clause,
promote same-parent clauses scoring at least 50% of that top score, then fill any
remaining answer slots from the global ranking. The rule is opt-out via
`sibling_score_ratio=None` for controlled comparison. It is lexical and depends
on source clause IDs; it adds no model and no retrieval change.

**Experiment.** EXP-003 held PDF, questions, 20 evidence units, chunks, embedding,
top-5 retrieval, answer caps, citations, abstention and metrics constant. Both
arms received identical precomputed hits via `answer_from_hits`.

**Results.**

| Measure | Baseline | Sibling-clause |
|---|---:|---:|
| Supported answers | 12/15 | 12/15 |
| Supported statements | 13/45 (0.2889) | 15/45 (0.3333) |
| Supported citations | 15/32 (0.4688) | 16/31 (0.5161) |
| Q13 evidence coverage | 1/3 | 3/3 |
| Abstention | 12/12 | 12/12 |

All four pre-registered decision checks passed. The change is accepted as the
production default.

**Consequences.**
- Q13 now selects `4.3.2`, `4.3.3` and `4.3.4`.
- Q10 remains unsolved because its required `3.6.8` evidence does not begin with
  the clause ID.
- Q05/Q08 remain retrieval failures; answer selection cannot recover missing
  evidence.
- The 15-question set is small and the 0.5 ratio was chosen during exploratory
  analysis; the improvement is not independently confirmed.
- Regression coverage: `tests/test_pipeline.py` sibling tests and
  `test_answer_from_hits_reuses_supplied_retrieval_without_searching`.

See `docs/EXP-003.md` and `experiments/results/exp003/`.

### DR-011 — Constrained generator evaluation rejected (EXP-004)

**Context.** The user selected generator evaluation as the next controlled piece
of work. No hosted-model API key was available, so the experiment used local
Ollama. Full-context pilot prompts were rejected during preparation because the
available small models selected wrong clauses, omitted citation markers or echoed
source metadata. The full experiment was therefore limited to synthesis over the
accepted extractive selector's own evidence.

**Decision.** Do **not** replace the extractive production answer path with the
evaluated generator. Keep `GroundedPipeline` extractive as the default. The new
`src/rag_core/generation.py` module is experimental, not wired into production.

**Experiment.** EXP-004 held PDF, questions, 20 evidence units, 49 chunks,
MiniLM embedding, dense top-5 retrieval, EXP-003 sibling selection, answer caps,
citations and abstention constant. The candidate used `qwen2.5:1.5b` with a JSON
claim contract over the selected evidence. Every claim named a selected evidence
sentence; the harness attached the original citation markers.

**Results.**

| Measure | Extractive | Generated |
|---|---:|---:|
| Answered | 15/15 | 15/15 |
| Model abstained / errors | — | 0 / 0 |
| Claims | 45 statements | 16 claims |
| Claims grounded at >=0.80 | — | 14/16 (0.8750) |
| Uncited / invalid-source claims | — | 0 / 0 |
| Citations | 31 | 17 |
| Citations overlapping verified evidence | — | 14/17 (0.8235) |
| Manual correctness (correct / partial / incorrect) | 0 / 11 / 4 | 5 / 6 / 4 |
| Manual faithfulness (grounded / partial / ungrounded) | 15 / 0 / 0 | 14 / 1 / 0 |
| Abstention (12 out-of-corpus) | 12/12 | 12/12 |
| Mean generation latency | — | 3,757 ms |

The pre-registered mechanical gate failed on claim grounding (`0.875` vs
required `0.90`). The manual audit found no fabricated material claim, but the
same four questions remained incorrect (Q04, Q05, Q10, Q12), and Q11 completed a
truncated source with the unsupported word “requirements.” The generator could
not fix retrieval or selection failures because it never saw unselected chunks.

**Consequences.**
- Production remains extractive; no generator dependency or latency is added.
- `tests/test_generation.py` covers the experimental contract but no test calls
  the local model.
- Full-context generation remains unevaluated end to end; the pilot observations
  are not a substitute for a controlled run.
- Revisit only after upstream Q05/Q08/Q10/Q04/Q12 coverage improves, with an
  independent audit and a grounding metric that is not satisfied by wrong
  evidence.

See `docs/EXP-004.md` and `experiments/results/exp004/`.

### DR-012 — Clause-body-aware answer units investigated and rejected (EXP-005)

**Context.** Q10 retrieved the required 3.6.8 evidence (`Recall@5 = 1.0`) but the
accepted extractive selector returned 0/3 supported statements. The evidence is
the complete 3.6.8 procedure for 3.6.7(a)/(b): first come, first served,
government-owned priority in conflict, discussions with applicants, common API,
and decision at AP S4 filing or later.

**Root cause (recorded before implementation).**
1. Sentence segmentation fragmented 3.6.8 into four answer units; only the first
   carried clause ID `3.6.8`, so the three body procedure sentences had no clause
   identity.
2. Lexical scoring favored the 3.6.7(a)-(e) list (`(a) new orbit-spectrum`,
   `(b) unplanned bands`) and 3.6.10 because the question names "new or unplanned
   bands", while 3.6.8 refers to them as "3.6.7(a) and 3.6.7(b)".
3. The 3-sentence cap could not assemble the complete four-step procedure even if
   body sentences were partially ranked.
4. Sibling-clause selection did not apply to the selected top units.

**Options considered.**
- Keep sentence units (EXP-003 baseline): known to miss Q10.
- Group every numbered clause start with its body: rejected during inspection
  because depth-2 headings (e.g. `4.3`, `4.4`) can swallow child clauses or broad
  introductions and changed Q13 selection badly in a quick check.
- Group only depth>=3 subclauses with contiguous body sentences until the next
  numbered clause start in the same chunk: smallest clause-body-aware change that
  preserves 3.6.8 as one verbatim unit while leaving depth-2 headings as sentence
  units.

**Decision.** Run EXP-005 with depth>=3 clause-body grouping as the candidate
against the accepted EXP-003 selector. **Reject** the candidate as the
production default.

**Experiment.** Retrieval, chunks, embeddings, questions, evidence, citations,
abstention and answer caps were held fixed; both arms received identical
precomputed hits from the production index (`sibling_score_ratio=0.5`).

**Results.**

| Measure | Baseline (EXP-003) | Candidate (clause-body units) | Delta |
|---|---:|---:|---:|
| Supported answers | 12/15 | 12/15 | 0 |
| Supported statements | 15/45 (0.3333) | 14/45 (0.3111) | -1 (-0.0222) |
| Supported citations | 16/31 (0.5161) | 16/32 (0.5000) | 0 (-0.0161) |
| Q10 evidence covered / supported statements | no / 0/3 | no / 0/3 | 0 |
| Abstention | 12/12 | 12/12 | 0 |

Q10 remained unanswered by evidence (0/3 supported, 3.6.8 chunk not cited, no key
procedure phrase present). The grouped 3.6.8 unit scored 4.914 but ranked behind
the 3.6.7(a)-(e) list (11.323), the grouped 3.6.10 unit (9.176) and a 3.6.9-tail
sentence (5.801). Q14 regressed in supported-statement count (2/3 -> 1/3)
because 4.4.4 became a selected clause-body unit while the 4.4.3+body merge still
covered `EXP001-Q14-E01` but no longer produced a second supported statement;
evidence coverage for Q14 remained present.

**Consequences.**
- Production selection remains EXP-003 sentence units + sibling promotion.
- Clause-body segmentation alone is **not sufficient** for Q10: the needed unit is
  created, but lexical/context ranking still does not select it.
- Future Q10 work must address answer-unit ranking/context (e.g. reference-aware
  selection, clause-reference resolution, or a different controlled selection
  rule), not only segmentation granularity.
- Regression coverage: `tests/test_pipeline.py` adds depth-3 grouping and depth-2
  non-grouping tests.

See `docs/EXP-005.md` and `experiments/results/exp005/`.

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

### Open known issue (not part of BUG-001…009)

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
| Answer mode | extractive verbatim, sibling-clause selection, max 3 sentences from 5 chunks |
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
| Statements supported | 15 / 45 (0.3333) |
| Citations supported | 16 / 31 (0.5161) |
| Mean statements / citations per answer | 3.00 / 2.07 |

**Abstention smoke test.** 8/8 calibration and 4/4 held-out out-of-corpus
questions abstained (12/12). All 15 in-corpus questions were answered.

**Observed failures.**

- Q05: Recall@5 0.0. The required `2.5.6` evidence is absent, so the extractive
  answer is unsupported. The abstention policy does not catch it.
- Q08: Recall@5 0.3333. Only one of the three required `3.4.1`–`3.4.3` clauses is
  retrieved.
- Q10: Recall@5 1.0 but 0/3 supported statements; the selector chooses a long
  list block and later clauses instead of the conflict procedure. Its evidence
  does not begin at a clause ID, so sibling selection does not help.
- Q13: resolved by EXP-003. The baseline selected only `4.3.4` (`1/3`); sibling
  clause selection now covers `4.3.2`–`4.3.4` (`3/3`).

**Decision.** Accept the slice as the **validated retrieval-and-provenance
foundation**, with EXP-003 accepted as the answer-selection default. Do not
present it as a complete answer engine. Q08 multi-clause completeness and Q10
relevance remain open defects, not hidden by the exact retrieval parity. See
`docs/PRODUCTION_SLICE.md`, `docs/EXP-003.md` and
`experiments/results/production_slice/`.

**Reproduction.** `python -m experiments.production_slice` writes config, chunks,
integrity audit, retrieval results, answers, abstention results,
per-question metrics and summary. Observed latency in the recorded run: index
build 2,422.49 ms; pipeline mean 38.60 ms, median 38.52 ms, max 44.97 ms.

### EXP-003 — Answer-layer sibling-clause selection (accepted)

**Role.** Controlled answer-layer experiment on the observed Q13 failure.
Retrieval, chunks, embedding, questions, evidence and answer caps were held
constant.

**Arms.** Baseline pure score order vs sibling-clause promotion: when the
top-ranked sentence is a numbered clause, promote same-parent clauses scoring at
least 50% of the top score, then fill from the global ranking. Both arms received
the same precomputed hits via `answer_from_hits`.

**Pre-registered acceptance.** Statement support +0.02 or better; supported
answers not lower; citation support no worse than −0.02; abstention unchanged.

**Results.**

| Measure | Baseline | Sibling |
|---|---:|---:|
| Supported answers | 12/15 | 12/15 |
| Supported statements | 13/45 (0.2889) | 15/45 (0.3333) |
| Supported citations | 15/32 (0.4688) | 16/31 (0.5161) |
| Q13 evidence coverage | 1/3 | 3/3 |
| Abstention | 12/12 | 12/12 |

All checks passed. Only Q13 changed; every other question had the same supported
statement count. The rule is accepted as the production default (DR-010).

**Not fixed.** Q05 (retrieval miss), Q08 (partial retrieval), Q10 (body-level
clause evidence), Q14 (two statements overlapping one evidence unit). See
`docs/EXP-003.md` and `experiments/results/exp003/`.

### EXP-004 — Constrained generator over selected evidence (rejected)

**Role.** Controlled answer-synthesis experiment on the accepted extractive
path. Retrieval and selection were held fixed; the candidate rewrote only the
selector's selected verbatim sentences into JSON claims, each naming the evidence
sentence(s) it used.

**Model.** Local Ollama `qwen2.5:1.5b`, `temperature=0`, JSON format. Full-context
free-text and JSON prompts were rejected during pilots because the available
models selected wrong clauses, omitted citations or produced malformed/repeated
JSON. Those pilots were not advanced to the full 15-question run.

**Pre-registered mechanical gate.** 12/12 negative abstention; no generation
errors and all 15 positives parse; zero invalid source IDs; zero uncited claims;
>=90% of claims at >=0.80 cited-token coverage; >=12/15 positives answered.

**Results.**

| Measure | Extractive | Generated |
|---|---:|---:|
| Answered | 15/15 | 15/15 |
| Claims | 45 statements | 16 claims |
| Claims at >=0.80 cited coverage | — | 14/16 (0.8750) |
| Uncited / invalid-source claims | — | 0 / 0 |
| Citations | 31 | 17 |
| Manual correctness (correct / partial / incorrect) | 0 / 11 / 4 | 5 / 6 / 4 |
| Manual faithfulness (grounded / partial / ungrounded) | 15 / 0 / 0 | 14 / 1 / 0 |
| Abstention | 12/12 | 12/12 |

The mechanical gate failed only on grounding rate (`0.875` < `0.90`). The manual
audit found no fabricated material claim, but the same four questions remained
incorrect (Q04, Q05, Q10, Q12), and Q11 was partially grounded because it
completed a truncated source with “requirements.” Q10's wrong claims had `1.0`
token coverage, showing the proxy does not measure relevance.

**Decision.** Reject as the production default (DR-011). The generator is not
wired into production. Q04/Q12 headings, Q05/Q08 retrieval misses and Q10
selection remain upstream problems. See `docs/EXP-004.md` and
`experiments/results/exp004/`.

### EXP-005 — Clause-body-aware answer-unit selection (rejected)

**Role.** Controlled answer-layer experiment targeting Q10, whose required 3.6.8
evidence is retrieved but not selected by the accepted EXP-003 extractive path.

**Root cause (before implementation).** 3.6.8 was fragmented into four sentence
units; body sentences lacked clause identity. Lexical selection favored the
3.6.7(a)-(e) definitional list and 3.6.10 over the 3.6.8 procedure body because
the question states "new or unplanned bands" while 3.6.8 references "3.6.7(a)
and 3.6.7(b)". The 3-sentence cap and sibling rule could not assemble the full
procedure.

**Hypothesis.** Grouping a depth>=3 clause start with contiguous body sentences
(until the next numbered clause start in the same chunk) preserves clause
identity and complete procedure as one verbatim answer unit.

**Controlled variable.** Answer-unit segmentation only: EXP-003 sentence units
vs depth>=3 clause-body units. Retrieval, chunks, embeddings, questions,
evidence, citations, abstention, answer caps and sibling ratio (0.5) held fixed;
identical precomputed hits.

**Pre-registered acceptance.** Q10 covers `EXP001-Q10-E01`, cites
`satcom-ngp::page_8::chunk_0::off17796`, and contains all key procedure phrases
(first come first served, government-owned priority, discussions with applicants,
common API/Advanced Publication of Information, AP S4 filing/later decision);
no non-Q10 regression in supported statements/citations/answer support; supported
answers not lower; statement support not lower; citation support delta >= -0.02;
abstention 12/12 unchanged.

**Results.**

| Measure | Baseline (EXP-003) | Candidate | Delta |
|---|---:|---:|---:|
| Supported answers | 12/15 | 12/15 | 0 |
| Supported statements | 15/45 (0.3333) | 14/45 (0.3111) | -1 |
| Supported citations | 16/31 (0.5161) | 16/32 (0.5000) | 0 |
| Q10 supported statements / evidence | 0/3 / no | 0/3 / no | 0 |
| Abstention | 12/12 | 12/12 | 0 |

Q10 failed all Q10 checks. The grouped 3.6.8 unit scored 4.914 but ranked below
the 3.6.7(a)-(e) list (11.323), grouped 3.6.10 (9.176) and a 3.6.9-tail sentence
(5.801), so it was not selected. Q14 regressed in supported-statement count
(2 -> 1) after 4.4.4 became a selected clause-body unit; `EXP001-Q14-E01`
remained covered by a merged 4.4.3+body statement.

**Decision.** Reject (DR-012). Clause-body segmentation alone is insufficient:
it creates the required unit but lexical/context ranking still does not select it.
Production remains EXP-003. See `docs/EXP-005.md` and `experiments/results/exp005/`.

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
| RA-008 | Tune the lexical answer selector without a controlled protocol on the 15-question set | Exploratory sweeps (sentence count, heading penalty, length normalisation, relative score floors) risk overfitting a small set. Deferred to EXP-003, which tested one pre-registered selection rule and accepted it only after all decision checks passed. |
| RA-009 | Adopt cross-encoder sentence reranking after an informal check | `cross-encoder/ms-marco-MiniLM-L-6-v2` raised statement support to 17/45 at three sentences but regressed Q13 to 0/3 and cost ~456 ms per query on CPU. Not adopted without a full controlled experiment and because it did not fix the main observed failure. |
| RA-010 | Adopt a local LLM generator over selected evidence | EXP-004's `qwen2.5:1.5b` synthesis passed the citation/uncited-claim checks but failed the pre-registered grounding rate (0.875 < 0.90). The manual audit found 4/15 answers still incorrect (same as extractive), 1/15 partially grounded, and no evidence it fixes upstream retrieval/selection failures. Rejected for the production default; revisit only after upstream coverage improves. |
| RA-011 | Replace sentence units with depth>=3 clause-body units as the default selector input | EXP-005 showed this creates the needed 3.6.8 procedure unit but does not fix Q10 (0/3 supported; 3.6.8 not selected due lexical/context ranking) and regresses Q14 supported-statement count (2 -> 1) and aggregate statement/citation support. Rejected; Q10 needs a controlled ranking/reference-aware selection change, not segmentation alone. |

---

## F. Verification and test evidence

### F.1 Executed test suite

Command: `python -m pytest tests -q`

Result (executed 2026-11-14, after EXP-005): **91 passed, 25 warnings** in
~10.7 s. The earlier Phase-2/3 suite result was 57 passed; PROD-001 added 21
tests, EXP-003 added 4, EXP-004 added 7 generation-contract tests, and EXP-005
added 2 pipeline segmentation tests.

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
- `tests/test_pipeline.py` — 21 tests: extractive verbatim answers, sentence
  provenance, citation-only-contributors, heading deprioritisation, sibling-clause
  selection and threshold behaviour, `answer_from_hits` reuse, abstention branches,
  generator hook, token support, limit/dedupe, fallback, and depth-aware
  clause-body answer-unit grouping (EXP-005).
- `tests/test_generation.py` — 7 tests: selected-evidence prompt construction,
  source-ID → original-marker mapping, invalid-source recording, empty-model
  abstention, malformed JSON, duplicate-key detection and missing-field
  validation. No local model is called.

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
- PROD-001 observed latency: pipeline mean 38.60 ms, median 38.52 ms, max
  44.97 ms; index build 2,422.49 ms for 49 chunks.
- EXP-003 writes `config.json`, `integrity_audit.json`, `retrieval_results.json`,
  `answers_baseline.json`, `answers_sibling.json`, `abstention_results.json`,
  `comparison.json`, `per_question_metrics.csv` and `summary.md` under
  `experiments/results/exp003/`. Answer-assembly latency (hits held fixed):
  baseline 2.93 ms mean, sibling 3.06 ms mean (negligible; varies by run).
- EXP-004 writes `config.json`, `integrity_audit.json`, `retrieval_results.json`,
  `answers_extractive.json`, `answers_generated.json`, `abstention_results.json`,
  `comparison.json`, `per_question_metrics.csv`, `latency.json`, `summary.md`,
  `manual_audit_template.csv`, `manual_audit.json`, `manual_audit_summary.json`
  and `decision.md` under `experiments/results/exp004/`. The generator run used
  an isolated `chroma_db_exp004/` collection. Observed generator latency:
  mean 3,757.21 ms, median 3,256.99 ms, max 7,818.54 ms for 15 answered questions.
- EXP-005 writes `config.json`, `integrity_audit.json`, `retrieval_results.json`,
  `answers_baseline.json`, `answers_candidate.json`, `abstention_results.json`,
  `comparison.json`, `per_question_metrics.csv`, `latency.json` and `summary.md`
  under `experiments/results/exp005/`. It reused the production index with
  identical precomputed retrieval hits. Observed selection latency:
  baseline 2.00 ms mean, candidate 1.86 ms mean (negligible; varies by run).

---

## G. Risks and open items

| ID | Risk / open item | Impact | Mitigation / next step |
|---|---|---|---|
| R-001 | Evaluation set is small (15 Q / 20 units) | Metric deltas are noisy; overfitting to the set is easy | Require manual failure inspection; prefer clear directional results; expand dataset with verified units before large claims |
| R-002 | Compound questions can lose child-clause coverage | Wrong/incomplete answers for Q08/Q10-type questions | Q13 is fixed by EXP-003 sibling-clause selection; Q08 remains a retrieval miss. EXP-005 showed clause-body segmentation alone does not fix Q10 (ranking/context still selects 3.6.7(a)/(b)/3.6.10). |
| R-003 | Very small chunks (<100 chars) weaken embeddings | Poor ranking / MRR | Consider minimum-size merge rules inside the structure-aware strategy, as a controlled change |
| R-004 | Legacy backend has `status_status` bug and unvalidated pipeline (KNOWN-001) | App path may fail or bypass validation | Fix keyword; isolate legacy advanced components; build validated ingestion on `src/rag_core` |
| R-005 | Generation quality is not established | Cannot claim generative answer quality | EXP-004 evaluated constrained synthesis and rejected it; full-context generation remains unevaluated. Keep extractive default until a generator beats it under a controlled protocol. |
| R-006 | README performance claims (faithfulness 1.00, context recall 0.60) are not from this pipeline | Misleading project status | Treat as legacy; supersede with validated experiment reports |
| R-007 | Chroma telemetry warnings | Noise in logs | Ignore; does not affect correctness |
| R-008 | Environment drift (Python/Chroma/sentence-transformers versions) | Non-reproducible numbers | Record versions; pin before final release |
| R-009 | Lexical/context selection still misses evidence that retrieval already found (Q10) | Incomplete/irrelevant answers despite perfect recall | EXP-005 tested depth>=3 clause-body answer units and was rejected: the 3.6.8 unit was formed but ranked below the 3.6.7(a)/(b) list and 3.6.10. Next controlled candidate is reference-aware/ranking selection, not segmentation alone. |
| R-010 | Multi-clause completeness partial; Q05 retrieval recall is 0.0 | Q08 answer lacks required child clauses; Q05 has no evidence in top-5 | Treat as open acceptance failure; decide retrieval vs answer-layer fix by experiment |
| R-011 | Abstention policy does not detect in-corpus retrieval failure | Q05 is answered without supporting evidence | Add answerability/relevance checks only with a labelled evaluation set; do not tune on the 15 questions |
| R-012 | EXP-003's 0.5 sibling threshold and +0.044 improvement come from a small, reused 15-question set | Improvement may not generalise | Keep the rule but validate on an expanded/independent set before treating the delta as a general quality claim |
| R-013 | EXP-004's generator conclusion rests on a selected-evidence design, one small local model, one reviewer and a lexical grounding proxy | The result may not generalise to full-context or larger models | State the scope explicitly; do not claim "generation is worse" in general. Re-evaluate with independent labels and an upstream-fixed selector before revisiting. |
| R-014 | EXP-005 shows clause-body grouping can change statement granularity/counts (Q14 2->1 supported statements) while preserving evidence coverage | Statement-count metrics can move without evidence loss; a naive "no regression" gate may reject coherent merges | Inspect per-question evidence coverage and answer content, not only counts; define acceptance to distinguish evidence loss from granularity changes. |

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
12. Wrote `docs/PRODUCTION_SLICE.md`; updated brief and this log; full suite then
    78 passed.
13. Ran EXP-003: sibling-clause answer selection with retrieval held fixed. Q13
    evidence coverage improved 1/3 → 3/3; statement support 0.2889 → 0.3333;
    citation support 0.4688 → 0.5161; all pre-registered checks passed. Accepted
    as the production default (DR-010).
14. Added shared `evaluate_answer_support`, `answer_from_hits`, sibling-selection
    tests and `docs/EXP-003.md`; reran PROD-001 and the full suite. Full suite then
    82 passed.
15. Ran EXP-004: constrained `qwen2.5:1.5b` generation over the selected evidence.
    The mechanical gate failed on grounding rate (0.875 < 0.90); manual audit
    found 5 correct / 6 partial / 4 incorrect, with no fabricated claim but one
    partially grounded completion. Rejected for production (DR-011, RA-010).
    Added `src/rag_core/generation.py`, `tests/test_generation.py`,
    `docs/EXP-004.md` and the EXP-004 audit artifacts. Full suite now 89 passed.
16. Ran EXP-005: depth>=3 clause-body-aware answer-unit segmentation for Q10 with
    retrieval/selection inputs fixed. Root cause: 3.6.8 was fragmented and its
    body lacked clause identity while question terms matched the 3.6.7(a)/(b)
    list. The candidate formed the 3.6.8 unit but did not select it (scores:
    list 11.323, 3.6.10 9.176, 3.6.9-tail 5.801, 3.6.8 4.914), left Q10 at 0/3,
    and regressed Q14 supported-statement count (2 -> 1). Rejected (DR-012,
    RA-011). Added `clause_body_units` support/tests in the pipeline,
    `experiments/exp005.py`, `docs/EXP-005.md` and `experiments/results/exp005/`.
    Full suite now 91 passed.

### H.2 Next actions (ordered, evidence-based)

1. **Q10 selection/ranking (retrieval fixed, controlled).** EXP-005 shows
   segmentation is necessary but insufficient: the 3.6.8 clause-body unit exists
   but is outranked by the 3.6.7(a)/(b) list and 3.6.10, which lexically match
   "new/unplanned bands" while 3.6.8 references "3.6.7(a) and 3.6.7(b)". A
   follow-up experiment should test a controlled ranking/selection rule that uses
   clause-reference context (e.g. include referenced clause text when scoring or
   prefer clauses that explicitly answer "For 3.6.7(a) and 3.6.7(b)") without
   changing retrieval, ground truth or caps.
2. **Fix Q04/Q12 answer units.** The selected evidence is a heading; the answer
   needs the following list or definition. This is an answer-unit/sentence
   segmentation problem, not generation.
3. **Decide retrieval work from evidence.** Q05 (Recall@5 0.0) and Q08 (0.3333)
   are genuine retrieval misses. Any expansion/reranking must be a separate
   controlled experiment and must not silently replace the accepted baseline.
4. Fix KNOWN-001 and isolate or retire unvalidated legacy pipeline components
   before exposing the application.
5. Expand the verified evaluation set before making broader quality claims; the
   EXP-003/004/005 results are on a small, reused 15-question set.
6. Revisit generation only after upstream coverage improves, with an independent
   audit and a grounding metric that is not satisfied by wrong evidence.

### H.3 Phase status

| Phase | Description | Status |
|---|---|---|
| 1 | Independent audit | Done |
| 2 | Document-processing foundation + strict integrity | Done |
| 3 | EXP-002 structure-aware chunking experiment | Done (rejected as default) |
| 4 | Justified chunking refinement (if pursued) | Not started |
| 5 | Validated production ingestion | Done for accepted baseline (PROD-001) |
| 6 | Application: generation, citations, abstention | Partial: extractive citations + abstention + sibling-clause selection validated; constrained generation evaluated and rejected (EXP-004); Q10 clause-body segmentation evaluated and rejected (EXP-005) |
