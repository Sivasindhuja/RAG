# Engineering Log — SATCOM NGP RAG

Status: living document. Last updated 2026-11-14.
Scope: decisions, defects, experiments, rejected approaches, verification evidence,
risks and next actions for the SATCOM NGP RAG project.

Companion documents: `docs/PROJECT_BRIEF.md`, `docs/BACKLOG.md`,
`docs/EXP-001.md`, `docs/EXP-002.md`, `docs/EXP-003.md`, `docs/EXP-004.md`,
`docs/EXP-005.md`, `docs/PRODUCTION_SLICE.md`, `docs/CHUNKING_STRATEGY.md`.

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
in `src/rag_core/`. New experiments import it. `removed-experiment-artifacts/exp001.py` is the
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

See `docs/EXP-003.md` and `removed-experiment-artifacts/results/exp003/`.

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

See `docs/EXP-004.md` and `removed-experiment-artifacts/results/exp004/`.

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

See `docs/EXP-005.md` and `removed-experiment-artifacts/results/exp005/`.

### DR-013 — Authoritative project objective, fixed constraints and milestone roadmap

**Date.** 2026-11-14.

**Context.** After EXP-001–EXP-005, the repository had a validated extractive
retrieval/selection slice but no bounded path to a complete, deployable,
evaluated product. Continued technique-by-technique experimentation risked drift
without delivering the required application.

**Problem.** Define a fixed destination, fixed constraints, and a measured
roadmap so experiments and engineering decisions serve completion rather than
indefinite exploration.

**Options considered.**
- Continue opportunistic experiments from the observed backlog (Q10/Q04/Q12/Q05/Q08)
  before defining end-state requirements: rejected as likely to optimize local
  metrics without reaching a deployable, grounded Gemini application.
- Treat the current extractive baseline as sufficient and skip generation/eval/deployment:
  rejected; contradicts the ultimate objective.
- Adopt the full RAG technique stack (BM25, reranking, query expansion, etc.)
  upfront: rejected; violates evidence-first engineering philosophy.
- **Selected:** define an authoritative objective with three phases (functional
  app, production-quality RAG, evaluation/QA) plus observability, fixed
  constraints, milestone roadmap, and a maintained backlog.

**Decision.**
- **Objective:** build a production-grade, domain-specific RAG application for
  `data/satcom-ngp.pdf` (ISRO SATCOM NGP) with grounded Gemini answers,
  verifiable PDF-linked citations, abstention, functional UI, evaluation, CI
  quality gates, observability, and Vercel Hobby deployment.
- **Fixed constraints:** single corpus (`data/satcom-ngp.pdf`), LLM Google Gemini
  API (model selected/documented by quality/availability/cost/latency), Vercel
  Hobby deployment target, free/free-tier preference with tracked costs and
  documented trade-offs, clean frontend/backend separation (React + Python/FastAPI
  RAG core where appropriate).
- **Technique policy:** hybrid retrieval, reranking, query expansion, etc. are
  potential solutions only; introduce only when a specific observed problem is
  diagnosed, hypothesized, experimentally tested, and justified by measured results.
- **Roadmap:** M0 direction/backlog (this step), M1 validated Gemini answer path,
  M2 end-to-end integration, M3 Vercel Hobby deployment readiness, M4 golden
  dataset (~50–200) + offline evaluation, M5 quality gates + observability, M6
  targeted evidence-driven improvements.
- **Issue management:** maintain `docs/BACKLOG.md`; document/assess/backlog
  important out-of-milestone problems; do not derail milestones, do not silently
  discard issues; rejected experiments remain documented.

**Evidence/reconciliation (2026-11-14, repo/Git inspected):** EXP-001–EXP-005
history preserved; extractive EXP-003 path accepted for selection baseline;
Gemini not integrated/validated; no golden dataset/eval harness/CI gates/
observability/deployment; legacy backend remains defective/unvalidated
(KNOWN-001, legacy collection/model); unrelated working-tree changes preserved.

**Trade-offs.** Defers immediate Q10/Q05/Q08/Q04/Q12 technique work until it is
scheduled against milestones, but prevents unbounded local optimization. Vercel
Hobby and free-tier constraints may force storage/architecture trade-offs that
must be measured and documented (BL-009/BL-015).

**Impact.** `docs/PROJECT_BRIEF.md` is authoritative for objective/constraints/
roadmap/status; `docs/BACKLOG.md` tracks known issues and deferred improvements;
future removed-experiment-artifacts/changes must cite the milestone/problem they serve.

**Status.** Accepted.

### DR-014 — M1 validated Gemini answer path (implemented, live validation blocked)

**Date.** 2026-11-14.

**Context.** M1 requires a Gemini-based, evidence-grounded answer path over the
accepted retrieval/selection baseline (EXP-001/EXP-003) with verifiable
citations, abstention, and evaluation on the 15-question set. No chunking or
retrieval changes are permitted in M1.

**Problem.** Provide a trustworthy Gemini generation contract and evaluation
harness: versioned prompt, supplied-evidence-only claims, app-validated citation
IDs resolved to actual source metadata, abstention for out-of-corpus and
in-corpus insufficient-evidence cases, and mechanical/manual evaluation.

**Options considered.**
- Reuse legacy `backend/main.py` Gemini path (`gemini-1.5-pro`, unvalidated
  collection/pipeline): rejected (KNOWN-001, not on validated core, no claim/citation contract).
- Let model emit section/page citations: rejected; model-generated metadata is
  untrusted. **Selected:** opaque evidence IDs in prompt, app validates IDs and
  resolves markers/citations from selected evidence objects (actual chunk_id,
  section, page, offsets).
- Rely only on pipeline distance/token-support abstention: rejected as
  insufficient for trustworthiness because Q05 (retrieval miss, Recall@5 0.0)
  was answered with unsupported evidence. **Selected:** add M1 selected-evidence
  sufficiency gate before generation (abstain if pipeline abstained, no selected
  statements, or lexical support of question against selected statements < 0.5);
  record as heuristic, not a general answerability classifier.
- Block M1 entirely until API credentials available: rejected; implement and
  test the path/contract/harness so live validation can run immediately when
  credentials are provided.

**Decision/implementation.**
- Added `GeminiGenerator`, `GEMINI_PROMPT_VERSION='m1-v1'`,
  `build_gemini_evidence_prompt`, and sufficiency-gate handling in
  `src/rag_core/generation.py`; exports in `src/rag_core/__init__.py`.
- Prompt/contract: JSON `{"sentences":[{"text":"...","sources":[1]}]}` over
  selected evidence only; no outside knowledge; no page/section/metadata in model
  text; empty list on insufficient evidence. Citation IDs validated and mapped to
  app-owned markers/citations by `assemble_generated_answer`.
- Added `removed-experiment-artifacts/m1_gemini_eval.py` and artifacts in
  `removed-experiment-artifacts/results/m1_gemini/` (config, integrity audit, retrieval, extractive
  answers, Gemini answers, abstention, manual audit template, per-question
  comparison, summary). Fixed/changed: retrieval/selection none; generation
  backend Gemini (blocked in run).
- Model selection procedure implemented (actual availability required): list
  `generateContent` models, preference
  `gemini-2.0-flash` → `gemini-2.0-flash-lite` → `gemini-1.5-flash*` → flash
  fallback, then probe for availability/latency/token metadata. Final model not
  selected.

**Results (2026-11-14, no API key).**
- Tests: `python -m pytest tests -q` → **97 passed, 27 warnings** (added 6 Gemini
  generation-contract tests; warnings include existing Chroma/Pydantic notices
  and google/grpc deprecation warnings).
- Extractive baseline unchanged: 15/15 answered, 12/15 supported answers,
  15/45 statements (0.3333), 16/31 citations (0.5161).
- Gemini live run **blocked** (`missing_api_key`, no `GOOGLE_API_KEY`/
  `GEMINI_API_KEY`): 0 answered, 0 model-abstained, 1 insufficient-evidence
  abstained (Q05, selected support 0.3333 < 0.5), 15 blocked, 0 generation
  errors. Out-of-corpus abstention 12/12 in both arms.
- Per-question M1 sufficiency: Q01–Q04,Q06–Q15 sufficient; **Q05 insufficient**
  (in-corpus retrieval failure, correctly abstained by M1 gate). Q04/Q08/Q10/Q12
  remain quality failures in extractive/selection evidence (see backlog) but were
  sufficient for generation attempt under lexical gate; live Gemini
  correctness/faithfulness/completeness/citation quality not yet evaluated.

**Trade-offs/limits.** Sufficiency gate is lexical/heuristic (catches Q05-like
irrelevant selected evidence, not semantically wrong-but-lexically-related cases
like Q10). No live Gemini availability/cost/latency/quality data yet; cannot
claim M1 validation complete or production readiness. Small 15-question set.

**Status.** Superseded by DR-015 (live validation executed). Architecture and
contract implemented and tested; live run completed with explicit model selection
after probe failures; M1 exit criteria not fully met.

### DR-015 — M1 live Gemini validation results (executed, exit criteria not met)

**Date.** 2026-11-14.

**Context.** Continued M1 live validation using credentials present in local
`.env` (`GEMINI_API_KEY` present; value not exposed). Frozen retrieval, chunking,
embeddings, ground truth and EXP-003 selection unchanged.

**Model availability/selection (actual).**
- `google.generativeai.list_models()` returned 49 `generateContent` models for
  the configured key (recorded in `removed-experiment-artifacts/results/m1_gemini/config.json`).
- Probes attempted (actual availability/latency/cost signals): `gemini-2.5-flash`
  selected by fallback but **unavailable** to account (404 "no longer available to
  new users"); `gemini-3.8-flash` (explicit, API-suggested) **probe timeout/504**
  (~599s); `gemini-flash-latest` (explicit) **probe failed** (connection abort/
  retry timeout ~740s). `gemini-flash-lite-latest` (explicit) **probe succeeded**
  (3182 ms, prompt/candidate/total tokens 10/9/19, JSON `{"sentences":[]}`).
- **Selected model:** `gemini-flash-lite-latest` (explicit, based on successful
  probe after availability failures). Pricing not configured in harness; cost not
  computed.

**Execution/results (15-question frozen set).**
- `python -m experiments.m1_gemini_eval --model gemini-flash-lite-latest` (with
  `.env` loaded): extractive baseline unchanged (15/15 answered, 12/15 supported,
  15/45 statements 0.3333, 16/31 citations 0.5161). Gemini: **11 answered, 3
  model-abstained, 1 insufficient-evidence abstained (Q05), 0 errors, 0 blocked**.
- Mechanical (generated claims): 18 claims, 14 grounded by token overlap ≥0.8
  (**0.7778**), 0 uncited claims, 0 invalid source IDs. Latency (14 API calls):
  mean 2504.93 ms, median 2435.07 ms, max 5175.49 ms. Tokens: prompt 5311,
  candidates 704, total 6015 (counts from API usage metadata where available).
- Out-of-corpus abstention: 12/12 (extractive and Gemini path).
- Manual audit (`removed-experiment-artifacts/results/m1_gemini/manual_audit.{json,csv,summary}`; not
  independent external audit): correctness **5 correct / 4 partial / 2 incorrect /
  4 abstained**; completeness **5 complete / 2 partial / 4 incomplete / 4 n/a**;
  faithfulness **9 grounded / 2 partially grounded / 0 unfaithful / 4 abstained**;
  citation quality **8 correct / 3 partial / 0 incorrect / 4 n/a**. Lexical
  support/overlap and valid citation IDs were **not** treated as proof of semantic
  sufficiency or claim-citation support.

**Per-question findings (Q04,Q05,Q08,Q10,Q12 and key cases).**
- **Q05 (recall 0.0):** M1 sufficiency gate abstained (selected support 0.3333) —
  trustworthy given missing/wrong evidence; corpus answer exists but not
  retrievable/selected (BL-001/BL-007).
- **Q10 (recall 1.0, wrong selection):** model abstained despite lexical gate pass
  (0.7778); selected evidence lacked 3.6.8 procedure. Faithful to supplied evidence
  but question unanswered (BL-003).
- **Q12:** model abstained; selected evidence lacked international services
  definition (BL-005/insufficient selected evidence).
- **Q15:** model abstained; selected evidence included frequency/orbit coordination
  but missed security concerns (conservative/incomplete).
- **Q04:** incorrect/incomplete (heading-only answer unit; omitted sector list;
  faithful to selected text but not a complete answer) (BL-004).
- **Q08:** incorrect/incomplete (retrieval 0.3333; omitted required specific
  authorisations DoS/WPC/operating licenses) (BL-002).
- **Q13:** partial/incomplete (omitted Broadcasting 4.3.1; paraphrased headings,
  mechanical overlap <0.8 though semantically close).
- **Q11:** partial/incomplete and partially grounded (omitted construction/launch
  cost, business plan, funding/revenues; truncated evidence/generalized claim).
- **Q09:** partial (SCC in India and 2-year shift stated; CAISS-authorised other
  location exception not included).
- **Q14, Q02,Q03,Q06,Q07:** correct/complete/grounded. **Q01:** partial (core
  correct, added relevant foreign-ops preference detail).

**Comparison (measured vs manual).** EXP-003 extractive baseline unchanged
(12/15 supported answers, 15/45, 16/31). EXP-004 (local qwen2.5:1.5b, rejected):
15 answered, mechanical claim grounding 0.875, manual 5 correct/6 partial/4
incorrect, 14 grounded/1 partial/0 unfaithful. M1 Gemini (`gemini-flash-lite-latest`,
single run): 11 answered + 4 abstentions, mechanical grounding 0.7778, manual 5
correct/4 partial/2 incorrect/4 abstained, 9 grounded/2 partially grounded/0
unfaithful. Differences reflect model behavior, abstentions, and the same
upstream retrieval/selection failures; results are on the small 15-question set
and not general production evidence.

**Bug fixed during live run.** `removed-experiment-artifacts/m1_gemini_eval.py`:
`_evaluate_generated/_claim_grounded` assumed statement objects; Gemini
`GeneratedAnswer.to_dict()` stores evidence statements as dicts, causing
`AttributeError` (`'dict' object has no attribute 'text'`). Fixed to accept
dict/object (`_stmt_text`).

**Decision.** **M1 exit criteria not met.** Gemini path and app-validated
citations work with a real model, and no unfaithful claims were observed in
manual audit; however correctness/completeness remain insufficient (5/15 correct,
5/15 complete; 2 incorrect, 4 incomplete), key questions fail due to upstream
retrieval/selection/evidence-unit issues (Q04,Q05,Q08,Q10,Q12) or conservative
abstention (Q15), and results are limited to one explicitly selected model/run on
15 questions. Lexical sufficiency did not ensure semantic sufficiency (Q10).

**Smallest justified next actions (evidence-driven).** (1) Treat M1 as
implemented-but-not-validated-for-quality: document selected model limitations
and rerun on a more capable/available Gemini model if/when probe succeeds, with
same frozen pipeline (model/config choice remains M1). (2) Address M1-blocking
quality only where justified: improve evidence selection/answer-unit coverage for
BL-003/BL-004/BL-005 and retrieval for BL-001/BL-002 as controlled M6 experiments
before claiming grounded Gemini quality. (3) Do not start M2 or add retrieval/
chunking techniques outside controlled experiments.

**Status.** Live validation executed; **M1 not accepted as meeting exit criteria**.

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
`removed-experiment-artifacts/results/exp001_v3/` are frozen; `removed-experiment-artifacts/exp001.py` and the
approved questions/ground truth must not be modified. See `docs/EXP-001.md`.

**Caveats.** The original `removed-experiment-artifacts/results/exp001/` run is invalid
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
`removed-experiment-artifacts/results/exp002/`.

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
`removed-experiment-artifacts/results/production_slice/`.

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
`docs/EXP-003.md` and `removed-experiment-artifacts/results/exp003/`.

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
`removed-experiment-artifacts/results/exp004/`.

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
Production remains EXP-003. See `docs/EXP-005.md` and `removed-experiment-artifacts/results/exp005/`.


### DR-016 — Cross-reference applicability selection investigated; accepted as opt-in candidate (EXP-006)

**Date.** 2026-11-14 (investigation/run), 2026-11-15 (review/evaluator normalization).

**Review decision (2026-11-15).** Accepted as successful opt-in candidate; **EXP-003 remains production default** and `cross_reference_selection` not promoted globally. Before EXP-007: fixed Q10 phrase-check normalization in `removed-experiment-artifacts/exp006.py` evaluator to tolerate PDF whitespace artifacts (e.g. `prio rity`) while preserving original source text and ground truth; added `removed-experiment-artifacts/results/exp006/normalized_review.json` (`all_key_phrases_present=true` under normalization). Original EXP-006 artifacts/decision history preserved (`comparison.json`/`summary.md` unchanged). Complete test suite on working tree: **97 passed, 27 warnings** (`python -m pytest tests -q`, ~83s, 2026-11-15).

**Date.** 2026-11-14.

**Context.** Live M1 (DR-015) left BL-003/Q10 open: retrieval fixed (`Recall@5 = 1.0`, 3.6.8 at rank 3) but selected evidence wrong; EXP-003 selector returned 0/3 supported statements and M1 Gemini abstained. User requested one controlled investigation, selection-only.

**Root cause (established before implementation).**
1. Q10 asks for procedure when multiple applicants request same orbit-spectrum in new or unplanned bands. Required procedure is 3.6.8 (`EXP001-Q10-E01`, `satcom-ngp::page_8::chunk_0::off17796`, offsets 17802–18522).
2. Highest lexical evidence is 3.6.7(a)-(e) list (rank 1, score 11.323) containing “(a) for new orbit-spectrum…” and “(b) in the unplanned bands…”. Clause 3.6.8 states “For 3.6.7(a) and 3.6.7(b)…” and gives the procedure but does not repeat those terms.
3. Sentence units fragmented 3.6.8; its body procedure sentences scored low (best ~3.414) and were outranked by 3.6.7 list, 3.6.10 (6.790), 3.6.11 (3.375) etc. 3.6.8 chunk not cited; 0 evidence covered.

**Hypothesis.** Document-grounded applicability cross-references (`^(\d+(?:\.\d+)+)\s+For\s+\d+(?:\.\d+)+` with subclause refs such as `(a),(b)`) link to query-relevant conditions present in retrieved segments (e.g. 3.6.7(a)/(b) list items matching new/unplanned bands). Promote matching applicability clause together with contiguous body as one verbatim answer unit (general rule, no Q10-specific logic).

**Controlled variable.** Selection only: EXP-003 baseline (`cross_reference_selection=False`, default) vs candidate (`cross_reference_selection=True`, opt-in). Frozen: PDF, questions, 20 evidence units, fixed-character 1000/200 chunks, MiniLM embeddings, dense top-5 retrieval/precomputed hits, sibling ratio 0.5, answer caps, citations, abstention, metrics. No Gemini evaluation.

**Implementation.** `src/rag_core/pipeline.py`: added `cross_reference_selection` parameter (default `False`) and `_apply_cross_reference_selection`. Candidate identifies query-relevant submarkers from retrieved segments (meaningful overlap: >=2 query tokens or weighted segment score >=4.0), finds numbered applicability clauses whose referenced submarkers intersect them, merges clause start + contiguous body (until next clause start) into a single exact source span, suppresses components, promotes unit near top score (0.95 * top direct score). `removed-experiment-artifacts/exp006.py` and `removed-experiment-artifacts/results/exp006/*` record config, audit, retrieval, answers, abstention, comparison, CSV, summary.

**Results.**
| Measure | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| Supported answers | 12/15 | 13/15 | +1 |
| Supported statements | 15/45 (0.3333) | 16/45 (0.3556) | +1 |
| Supported citations | 16/31 (0.5161) | 17/32 (0.5312) | +1 |
| Abstention (12/12) | 12/12 | 12/12 | 0 |

Q10: baseline 0/3 supported, no evidence covered, 3.6.8 not cited. Candidate 1/3 supported, covers `EXP001-Q10-E01`, cites `satcom-ngp::page_8::chunk_0::off17796` (3.6.8, offsets 17802–18522), and selects full merged 3.6.8 procedure including first come first served, government-owned-system priority, discussions with applicants, common Advanced Publication of Information, and AP S4 filing/later decision. Automated key-phrase check flagged `priority` as missing because extracted text contains `prio rity` (PDF whitespace artifact); procedure element is present. Only Q10 changed; no non-Q10 regressions in supported statements/citations/answer support/covered evidence.

**Decision.** **Accepted as opt-in candidate; not promoted to production default.** Substantive acceptance criteria met (required evidence covered, supported statement/citation correct, full procedure traceable, no regressions, abstention unchanged, aggregate improved). Automated script marked REJECTED on formatting-artifact phrase check; recorded as evaluation-note, not a content failure. **EXP-003 remains production default** pending explicit promotion review/sign-off on the 15-question set (BL-017). See `docs/EXP-006.md`.

**Trade-offs/limits.** Lexical/heuristic submarker relevance and promotion thresholds; merges applicability body into one unit (granularity/count effects possible elsewhere); small evaluation set; does not fix retrieval misses (BL-001/002) or heading-only units (BL-004/005).

### EXP-006 — Cross-reference applicability selection for Q10 (accepted as opt-in candidate)

**Role.** Controlled selection-only experiment addressing Q10 evidence-selection failure (retrieval correct, EXP-003 selection wrong).

**Acceptance.** Substantive checks passed: Q10 evidence covered/supported/correct citation/full procedure present; no non-Q10 regressions; supported answers and statement support not lower; citation delta +0.0151 (>= -0.02); abstention 12/12 unchanged. Automated `all_key_phrases_present` failed only on `priority` vs extracted `prio rity` artifact. Kept opt-in; production default unchanged.

**Not fixed by EXP-006.** BL-001 (Q05 retrieval), BL-002 (Q08 retrieval), BL-004 (Q04 heading+list), BL-005 (Q12 heading+definition), BL-017 (small set). Gemini not rerun.




### DR-017 — Heading-introduced associated answer units investigated; rejected for promotion (EXP-007)

**Date.** 2026-11-15.

**Context.** M1/EXP-006 left BL-004 (Q04 heading-only, omitted four classifications) and BL-005 (Q12 missing international-services definition elements) open. Controlled investigation requested: selection/answer-unit only, frozen retrieval/embeddings/chunking/ground truth/precomputed hits/top-k/EXP-003 settings/evaluation; full 15q regression + 12q abstention; EXP-006 kept separate (not combined); no Gemini.

**Diagnosis (recorded before implementation).**
- **Q04 (Recall@5 1.0, rank 1 `satcom-ngp::page_3::chunk_0::off4699`, E01 4740–4938):** heading `2.4.1 ... broadly classified as:` (global 4734–4786) selected with high score, but immediately following list span `(a)-(d)` (4790–4937) not selected due low lexical overlap; answer omitted four classifications.
- **Q12 (Recall@5 1.0, ranks 1–2 overlap E01 30425–31507):** key definition elements split across chunks/spans (4.2 body/definition start in chunk 0; `International Communications Services will include PSTN, GMPCS, Value Added Services, Programme Exchange... foreign country...` and MMSS/AMSS treatment in chunk 1). Baseline selected introductory/non-definitional units (4.1 classification, 4.2 practice sentence supported, 4.3.4) and missed required definition elements.

**Hypothesis.** Introductory spans ending with `:` signal an immediately associated enumeration/definition block in the same chunk; grouping intro span with contiguous following spans until next numbered clause start preserves heading-associated lists/definitions as one verbatim unit (general, document-grounded).

**Implementation.** `src/rag_core/pipeline.py`: opt-in `associated_heading_units` (default `False`) groups colon-ending intro spans with contiguous associated content until next numbered clause start. `removed-experiment-artifacts/exp007.py`, `removed-experiment-artifacts/results/exp007/*`.

**Results (frozen 15q + 12q abstention).** Baseline vs EXP-007: supported answers 12→12, statements 15/45→15/45 (0.3333), citations 16/31→16/32 (0.5161→0.5000, delta -0.0161), abstention 12/12 unchanged, no non-target regressions, offsets/citation traceability valid 15/15. **Q04:** all four classifications present (4/4), evidence covered/cited, supported 1/3; selected merged unit `2.4.1 ... (a)-(d)` offsets 4734–4937 (verbatim, traceable). **Q12:** required definition elements present 0/7 in answer text despite evidence covered by supported non-definitional statement; key retrieved definition sentences not selected/ranked. **Q10 separate comparison:** EXP-006-only still supported/covers E01/cites 3.6.8; EXP-007-only unchanged from baseline (no combination).

**Decision.** **Rejected for promotion; EXP-003 remains default; EXP-006 remains separate opt-in.** Q04 content/traceability improved but Q12 primary target unmet and no aggregate gain.

**Limitations.** Colon-intro rule does not capture Q12 definition sentences (not led by selected colon-intro in constructed/selected units); supported-statement count unchanged for Q04 despite content completeness (metric granularity). Next: controlled Q12 selection/ranking experiment targeting retrieved definition sentences.

### EXP-007 — Heading-introduced associated answer units (Q04/Q12, rejected for promotion)

Controlled answer-unit construction/selection only (`associated_heading_units`, opt-in). Q04 includes all four classifications in one verbatim supported/cited unit; Q12 required definition elements not selected. No regressions, abstention unchanged, traceability valid. Not promoted/combined. See `docs/EXP-007.md`, `removed-experiment-artifacts/results/exp007/`.

### DR-018 — Q12 definitional evidence selection investigated; accepted as opt-in candidate (EXP-008)

**Date.** 2026-11-15 (experiment/review), 2026-11-15 (post-review citation metadata correction).

**Context.** Q12 retrieval correct (Recall@5 1.0, E01 ranks 1–2) but baseline EXP-003 selection missed required definition elements (PSTN, GMPCS, Value Added Services, Programme Exchange, foreign-country origination/termination, MMSS/AMSS treatment). EXP-007 did not fix Q12.

**Diagnosis (established before implementation).** Key definitional sentences were present but scored 2.962 and outranked (top 4.049/3.751/3.674): chunk 1 spans 31006–31254 (PSTN/GMPCS/VAS/Programme Exchange + foreign origination/termination) and 31255–31506 (MMSS/AMSS treated as international), chunk 0 span 31006–31178 (partial definition start). Query intent token `defined` not in definition text; definition expressed via patterns (`will include`, `treated as`, `where...`) with partial query-content overlap.

**Hypothesis.** For definitional-intent questions, boost candidates containing document-grounded definitional/explanatory markers with query-content overlap (general, not Q12-specific).

**Controlled variable.** Selection/ranking only: `definition_intent_boost` off vs on (opt-in). Frozen: PDF, questions/evidence/unanswerable, chunking, embeddings, retrieval/precomputed hits, top-k, EXP-003 sibling ratio, caps, citations, abstention, metrics. EXP-006/EXP-007 kept separate.

**Implementation.** `src/rag_core/pipeline.py` opt-in `definition_intent_boost` (+1.5 when definitional intent, marker, and query-content overlap present). `removed-experiment-artifacts/exp008.py`, `removed-experiment-artifacts/results/exp008/*`.

**Results.** Checks all passed. Baseline→EXP-008: supported answers 12→12, statements 15/45 (0.3333)→17/45 (0.3778, +0.0444), citations 16/31 (0.5161)→17/32 (0.5312), abstention 12/12 unchanged, no regressions. **Q12:** all 7 required elements present, evidence covered (`EXP001-Q12-E01`), answer supported, 3/3 supported statements, valid offsets/pages/traceability. EXP-008 result artifacts preserved as executed history (`removed-experiment-artifacts/results/exp008/*`). **Separate checks:** Q04 EXP-008 baseline-like (not combined with EXP-007); Q10 EXP-008 baseline-like while EXP-006-only still fixes Q10.

**Post-review citation metadata correction (separate, 2026-11-15).** Audit against original PDF verified Q12 statement offsets 31006–31254, 31255–31506, 31006–31178 are page 13, verbatim 4.2 content (4.2 starts 30381, 4.3 starts 31512); ground truth unchanged. Recorded EXP-008 citations labeled 4.2 statements as 4.3/4.1 due to **chunk metadata assignment issue with fixed-character windowing**: chunk `...off30978` starts inside 4.2 but contained later 4.3 heading and was labeled 4.3; chunk `...off30178` starts at 4.1 but cited statement is in 4.2 region. Correction (smallest justified): `FixedCharacterChunker` now assigns start-governing clause metadata; pipeline resolves citation section/heading from governing clause of first selected statement offset (preferring in-chunk clause at/before offset; retaining start-governing metadata when statement precedes first in-chunk clause). Production index `chroma_db_production/satcom_ngp_production_fixed_char` rebuilt (49 chunks). Corrected Q12 citations resolve to **4.2 / “4.2 INTERNATIONAL COMMUNICATION SERVICES”** with unchanged chunk ranges/pages/offsets. See `docs/EXP-008-citation-metadata-correction.md`, `tests/test_citation_metadata.py`.

**Decision.** **Accepted for promotion consideration as opt-in candidate; not promoted. EXP-003 remains production default.** EXP-006/EXP-007/EXP-008 remain separate.

**Limitations.** Heuristic marker/intent boost **validated only on the current 15-question evaluation set** (BL-017). Possible redundant overlapping selected spans. Fixed-char chunks may span multiple clauses; citation now reflects governing clause of first selected statement per chunk while offsets/pages/text remain traceability anchors. No retrieval-miss fix.

### EXP-008 — Q12 definitional evidence selection (accepted as opt-in candidate)

Controlled selection/ranking only (`definition_intent_boost`, opt-in). Q12 all required elements selected/cited with traceability; no regressions; abstention unchanged. Not promoted/combined. See `docs/EXP-008.md`, `removed-experiment-artifacts/results/exp008/`.

**Verification note (2026-11-15, post-EXP-008 correction):** complete test suite `python -m pytest tests -q` → **101 passed, 29 warnings** (~15.09s) after adding citation-metadata regression tests and rebuilding production index. `tests/test_pipeline.py` restored from HEAD (accidentally emptied during audit session) before final run.



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

Latest full run (executed 2026-11-14, after M1 live validation / EXP-006 code changes): **97 passed, 27 warnings** (`python -m pytest tests -q`). Prior milestones: 91 passed after EXP-005 (+2 segmentation tests); 97 after M1 generation-contract tests (+6). EXP-006 added opt-in selection logic without changing default behavior; existing pipeline tests remained compatible. Warnings include existing Chroma/Pydantic notices and google-api-core/grpc deprecation/future warnings (non-fatal).

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

- `removed-experiment-artifacts/exp002.py` writes `config.json`, `chunks.json`,
  `integrity_audit.json`, `chunk_stats.json`, `evidence.json`,
  `retrieval_results.json`, `evidence_coverage.json`, `comparison.json`,
  `failure_analysis.json`, `per_question_metrics.csv`, `summary.md` under
  `removed-experiment-artifacts/results/exp002/`.
- CSV/metric comparison keys follow the configured `--top-k`; baseline comparison
  is only reported when the baseline contains the same metric key (prevents
  misleading deltas at a different K).
- Retrieval latency (observed): mean 35.63 ms, median 34.94 ms, p90 40.61 ms.
  Index build: 2,714.7 ms for 107 chunks.
- PROD-001 writes `config.json`, `chunks.json`, `integrity_audit.json`,
  `retrieval_results.json`, `answers.json`, `abstention_results.json`,
  `per_question_metrics.csv` and `summary.md` under
  `removed-experiment-artifacts/results/production_slice/`.
- PROD-001 observed latency: pipeline mean 38.60 ms, median 38.52 ms, max
  44.97 ms; index build 2,422.49 ms for 49 chunks.
- EXP-003 writes `config.json`, `integrity_audit.json`, `retrieval_results.json`,
  `answers_baseline.json`, `answers_sibling.json`, `abstention_results.json`,
  `comparison.json`, `per_question_metrics.csv` and `summary.md` under
  `removed-experiment-artifacts/results/exp003/`. Answer-assembly latency (hits held fixed):
  baseline 2.93 ms mean, sibling 3.06 ms mean (negligible; varies by run).
- EXP-004 writes `config.json`, `integrity_audit.json`, `retrieval_results.json`,
  `answers_extractive.json`, `answers_generated.json`, `abstention_results.json`,
  `comparison.json`, `per_question_metrics.csv`, `latency.json`, `summary.md`,
  `manual_audit_template.csv`, `manual_audit.json`, `manual_audit_summary.json`
  and `decision.md` under `removed-experiment-artifacts/results/exp004/`. The generator run used
  an isolated `chroma_db_exp004/` collection. Observed generator latency:
  mean 3,757.21 ms, median 3,256.99 ms, max 7,818.54 ms for 15 answered questions.
- EXP-005 writes `config.json`, `integrity_audit.json`, `retrieval_results.json`,
  `answers_baseline.json`, `answers_candidate.json`, `abstention_results.json`,
  `comparison.json`, `per_question_metrics.csv`, `latency.json` and `summary.md`
  under `removed-experiment-artifacts/results/exp005/`. It reused the production index with
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
    `removed-experiment-artifacts/exp005.py`, `docs/EXP-005.md` and `removed-experiment-artifacts/results/exp005/`.
    Full suite now 91 passed.

17. Established authoritative project direction (DR-013): updated
    `docs/PROJECT_BRIEF.md`, created `docs/BACKLOG.md`, reconciled implementation
    against repo/Git history, and replaced ad-hoc next actions with milestone
    roadmap M0–M6. No new RAG experiment or technique implemented.
18. Started M1 Gemini answer path (DR-014): added `GeminiGenerator` and
    `m1-v1` evidence-only JSON claim contract with app-validated citation-ID
    resolution, selected-evidence sufficiency gate, `removed-experiment-artifacts/m1_gemini_eval.py`,
    and 6 generation-contract tests. `python -m pytest tests -q` → 97 passed.
    Ran M1 eval on 15-question set with retrieval/selection fixed; extractive
    baseline unchanged (12/15 supported, 15/45, 16/31). **Live Gemini run
    blocked** by missing `GOOGLE_API_KEY`/`GEMINI_API_KEY`: model availability/
    quality/cost/latency selection and generation not executed. Q05 correctly
    abstained by sufficiency gate (0.3333 < 0.5); 14 sufficient questions blocked
    pending credentials; out-of-corpus 12/12 abstained. Artifacts in
    `removed-experiment-artifacts/results/m1_gemini/`.
19. Completed M1 live validation run (DR-015) with local `.env` credentials
    (key not exposed). Actual availability: 49 generateContent models listed.
    Probes: `gemini-2.5-flash` unavailable (404), `gemini-3.8-flash` 504 timeout,
    `gemini-flash-latest` connection abort/timeout; selected
    `gemini-flash-lite-latest` (explicit, probe ok 3182 ms, tokens 10/9/19).
    Frozen pipeline unchanged. Gemini: 11 answered, 3 model-abstained, 1
    insufficient-evidence abstained (Q05), 0 errors; mechanical claims 18,
    grounded>=0.8 14 (0.7778), 0 uncited, 0 invalid IDs; latency mean/median/max
    2504.93/2435.07/5175.49 ms (14 calls), tokens 5311/704/6015. Manual audit:
    correctness 5/4/2/4, completeness 5/2/4/4, faithfulness 9/2/0/4, citation
    quality 8/3/0/4. Q04 incorrect/incomplete (heading unit), Q05 abstained
    (retrieval 0.0, gate), Q08 incorrect/incomplete (retrieval 0.3333), Q10
    model-abstained despite lexical sufficiency (wrong selection), Q12/Q15
    model-abstained (insufficient selected evidence). Fixed eval script
    dict/object evidence-text bug. **M1 exit criteria not met.**

### H.2 Milestone roadmap (authoritative, evidence-bounded)

Roadmap defined by DR-013. Each milestone requires pre-defined exit evidence; do
not declare complete without it. Detailed issue tracking: `docs/BACKLOG.md`.

| Milestone | Objective | Key outputs | Exit evidence | Status |
|---|---|---|---|---|
| **M0** | Direction & backlog | Authoritative brief/log, backlog, reconciled status | Documented; no new RAG technique added | **Done** |
| **M1** | Validated Gemini answer path (local/integration) | Gemini model/config documented, versioned prompt contract, retrieval→context→Gemini flow, PDF-linked citations, citation/abstention enforcement, faithfulness checks | Controlled evaluation on current set shows grounded inspectable citations; failures recorded; no silent regression vs extractive traceability | **Executed, not accepted** (live run complete with `gemini-flash-lite-latest`; correctness/completeness insufficient and upstream Q04/Q05/Q08/Q10/Q12 failures block quality; no M2/new retrieval/chunking) |
| **M2** | End-to-end application integration | Clean API contract, React UI evidence/citation display, local reproducible run, KNOWN-001/legacy path handled per plan | User can ask and inspect exact evidence/citations locally; API/integration tests pass | Not started (blocked until M1 quality/exit evidence justified) |
| **M3** | Deployment-ready (Vercel Hobby) | Hobby-compatible storage/index strategy, serverless-safe packaging, env/secret handling, cold-start/latency/cost measurements, runbook | Publicly accessible app or documented blocking constraint with approved alternative; measured limits recorded | Not started |
| **M4** | Golden dataset & offline evaluation | ~50–200 verified Q/A with evidence/citations/abstention labels, retrieval/answer/faithfulness/completeness/citation/abstention metrics, baseline report | Reproducible evaluation with thresholds derived from baseline evidence (not invented) | Not started |
| **M5** | Quality gates & observability | CI quality checks, tracing, prompt/response/token/cost/latency/citation/failure instrumentation, monitoring/investigation path | CI blocks important regressions per defined thresholds; traces explain request outcomes | Not started |
| **M6** | Targeted quality improvements | Only justified fixes/experiments for observed gaps (e.g. BL-001..BL-007) | Each change accepted/rejected by controlled evidence; backlog updated | Backlog-driven, not started |

**Recommended immediate next step (smallest justified):** keep frozen pipeline and
run controlled, evidence-driven **M6** fixes for the observed blockers before
claiming M1 complete/M2: (a) Q10 reference-aware/selection fix (retrieval fixed),
(b) Q04/Q12 answer-unit fix (heading + following list/definition), (c) Q05/Q08
retrieval experiments (separate, controlled). Optionally rerun M1 eval with a
more capable available Gemini model when probe succeeds (M1 config choice only,
no retrieval/chunking changes). Do not start M2 or add untested RAG techniques.

### H.3 Historical phase status (pre-DR-013)

| Phase | Description | Status |
|---|---|---|
| 1 | Independent audit | Done |
| 2 | Document-processing foundation + strict integrity | Done |
| 3 | EXP-002 structure-aware chunking experiment | Done (rejected as default) |
| 4 | Justified chunking refinement (if pursued) | Superseded by milestone roadmap/backlog (evidence-gated) |
| 5 | Validated production ingestion | Done for accepted extractive baseline (PROD-001, EXP-003) |
| 6 | Application: generation, citations, abstention | Partial: extractive citations + abstention validated; local generator rejected (EXP-004); Gemini path, enforcement, UI, eval, deployment pending (M1–M5) |

---

## I. Strategy pivot — prioritize end-to-end application completion (2026-11-15)

**Decision (user-directed, 2026-11-15):** Pause component-level retrieval/selection/chunking/reranking experiments and prioritize completing the full production-quality RAG application for `data/satcom-ngp.pdf`.

| Item | Decision |
|---|---|
| EXP-009 (Q05 retrieval investigation) | **Planning authorized earlier but implementation postponed.** No implementation started. |
| BL-001 (Q05), BL-002 (Q08) | Kept in backlog as known retrieval limitations; do not run new component experiments unless required to resolve a demonstrated integration or production-quality blocker. |
| EXP-003 | **Remains production default.** |
| EXP-006 (Q10), EXP-008 (Q12) | **Remain separate opt-in candidates** (validated on current 15-question set only); not promoted, not combined. |
| EXP-007 | **Rejected for promotion** (Q04 content finding only). |
| Historical artifacts/decisions | **Preserved intact** (EXP-001–008, M1, correction records, results). |
| M1 status | **Still not accepted** (grounded generation quality/completeness and broader evaluation insufficient); progress on selection/citation correctness/abstention does not substitute for project quality requirements. |

**Rationale:** Sufficient component diagnosis exists (EXP-006/007/008, citation metadata correction). Continuing isolated optimization on the 15-question set delays delivery of a demonstrable, observable, deployable end-to-end system. Future optimization must be driven by measured end-to-end/integration evidence.

**Immediate next workstream (authorized):** repository audit and architecture/integration report, then incremental implementation toward complete backend, React/FastAPI integration, end-to-end local validation, observability/metrics, evaluation expansion + CI quality gates, and Vercel Hobby-compatible deployment (reusing validated modules; no duplicate implementations).

---

## J. WP0/WP1 — Backend integration on validated core (2026-11-15)

**Objective (authorized):** run validated RAG core through a clean FastAPI endpoint; extractive end-to-end first (Gemini optional/controlled), no EXP-009, no new retrieval/chunking/reranking, no promotion/combination of EXP-006/008, preserve unrelated work/history.

### J.1 What already existed / reused
- **Reused (validated):** `src/rag_core/document.py`, `chunker.py` (corrected fixed-char 1000/200, start-governing metadata), `index.py` (DenseIndex/Chroma, MiniLM), `pipeline.py` (GroundedPipeline, **EXP-003 default** `sibling_score_ratio=0.5`, opt-ins off, statement-governing citation resolution), `generation.py` (`GeminiGenerator`, m1-v1 contract, app-validated citation IDs/resolution), `evaluation.py`, corrected production index `chroma_db_production/satcom_ngp_production_fixed_char`.
- **Replaced for API path:** legacy `backend/main.py` (KNOWN-001 `status_status`, legacy `satcom_ngp_clauses`, `gemini-1.5-pro`, unvalidated Cohere/chat/metrics flow) rewritten as thin FastAPI app over new service; legacy parallel stacks (`src/generation/models.py`, `src/retrieval/*`, `src/chunking/structure.py`, `src/config/prompts.yaml` RAG flow) **not integrated**.
- **Preserved:** unrelated working-tree changes, legacy experiment artifacts/results/decisions, git history.

### J.2 Implemented (files changed/added)
- **Added `src/rag_app/`** (single application service layer):
  - `config.py`: `AppConfig` (env-driven; defaults: corrected prod index/collection, top_k=5, EXP-003 defaults, opt-ins false, extractive default mode, Gemini config/sufficiency threshold 0.5, allow_gemini).
  - `schemas.py`: Pydantic API contract (`QueryRequest`, `QueryResponse`, citations/statements/retrieved, `CitationValidationModel`, `GenerationModel`, `HealthResponse`).
  - `observability.py`: lightweight per-request trace (`request_id`, stage timings, events/errors) — stages include `retrieval`, `selection_and_extractive`, `gemini_generation` when attempted.
  - `service.py`: `RAGService` composing DenseIndex + GroundedPipeline(EXP-003) + optional GeminiGenerator. Flow: retrieve → extractive select/cite/abstain → (if `mode=gemini`) sufficiency gate (pipeline abstain/no statements/selected lexical support < threshold) then Gemini with `evidence_sufficient=True`; **app-side citation validation/resolution preserved**, extractive evidence retained, safe fallback to extractive grounded result on Gemini abstain/error/insufficiency. Exposes retrieved/selected IDs, citation validation, timings, reasons.
- **Modified `backend/main.py`**: clean FastAPI app (`/`, `/health`, `POST /query`, compat `/ask`), env/config via `AppConfig.from_env()`, optional dotenv load, CORS, lazy cached service, response schema from `RAGService`.
- **Added tests:** `tests/test_api.py` (4 fast contract tests + 1 representative real-index smoke test).
- **Added docs:** `docs/RUN_LOCAL.md` (run/config/endpoints/smoke/tests/limitations). `docs/ENGINEERING_LOG.md` appended (this entry).

### J.3 Architecture (WP1)
`Client -> FastAPI (/query) -> RAGService -> DenseIndex.search (corrected prod Chroma, MiniLM) -> GroundedPipeline.answer_from_hits (EXP-003, verbatim statements, statement-governing app-resolved citations, abstention) -> [optional sufficiency gate + GeminiGenerator over selected evidence + app citation ID validation/resolution + safe fallback] -> structured response + trace (request_id, timings, retrieved/selected IDs, citation validation, abstention/errors)`.

### J.4 Commands/tests executed and actual results (2026-11-15)
- `python -m py_compile src/rag_app/*.py backend/main.py` → **OK**.
- `python -m pytest tests/test_api.py -q -k "not real_index"` → **4 passed, 1 deselected** (1.52s). (Initial run had 1 failure due test cache-clear after monkeypatch; fixed; rerun passed.)
- `python -m pytest tests/test_api.py -q -k "real_index"` → **1 passed, 4 deselected, 4 warnings** (11.27s) against `chroma_db_production` (chroma.sqlite3, collection count 49).
  - **Q05 extractive (representative):** `status=answered`, `abstained=False`, `token_support=0.5`, `selected_lexical_support=0.3333`, `best_distance=0.8424`; retrieved top-5 = `['satcom-ngp::page_12::chunk_2::off29601','satcom-ngp::page_14::chunk_1::off33296','satcom-ngp::page_10::chunk_0::off22860','satcom-ngp::page_12::chunk_1::off28801','satcom-ngp::page_5::chunk_2::off11188']`; **required `satcom-ngp::page_3::chunk_2::off6299` not retrieved** (BL-001 persists). Response was traceable/cited to retrieved wrong evidence (BL-007 semantic gap: pipeline distance/token thresholds satisfied). Gemini mode would apply additional sufficiency gate (<0.5 selected support) before generation.
  - **Q12 extractive (representative, EXP-003 default):** `status=answered`, 3 statements / 2 app-resolved citations (`4.1` @ chunk `...off30178` p13, `4.3.4` @ chunk `...off32496` p14), offsets valid and consistent (`[30208,30275],[30346,30377],[32783,32850]`), `citation_validation.app_resolved=True`, timings recorded. (Under EXP-003 default, Q12 does not select full 4.2 definition set; EXP-008 candidate not promoted — known BL-005 state preserved.)
- `python -m pytest tests -q` → **106 passed, 33 warnings** (19.21s). New: 5 API tests. Warnings non-fatal (Chroma/Pydantic deprecation, google-api-core/grpc/Python 3.11 future warnings as previously observed).

### J.5 Known limitations / blockers after WP1
- **M1 not accepted:** Gemini path integrated but quality/completeness/model suitability unproven; extractive is grounded baseline. Do not claim M1 complete.
- **BL-001/Q05:** retrieval miss remains; extractive pipeline can answer with wrong evidence (threshold-based abstention only). Needs evidenced fix in later work if prioritized as blocker.
- **BL-002/Q08:** incomplete retrieval. **BL-003/004/005:** opt-in candidates separate, not default/promoted/combined.
- **Index/deployment:** local Chroma + local embedding model load not Vercel Hobby-ready (M3/BL-009/BL-015).
- **Frontend:** not integrated to new contract (WP2). **Eval/CI/observability depth:** WP4/WP5 pending (lightweight tracing added in-service).
- **KNOWN-001:** legacy defective behavior removed from active API implementation; legacy history preserved in git.

### J.6 Next work package
**WP2 — Complete frontend/backend integration:** connect React UI to `/health`+`/query`, render answer/citations (page/section/offset/chunk/quote), abstention reasons and errors/loading states; remove dependence on legacy `/chat`/`/metrics` contract; validate full local UI→API flow (extractive default).

---

## K. WP1.1 — Conservative answerability/abstention guard (2026-11-15)

**Trigger:** WP1 real-index smoke showed Q05 (`2.5.6` evidence not retrieved, BL-001) returned `answered` with selected lexical support 0.3333 and traceable-but-wrong citations (BL-007 semantic/answerability gap). Traceability alone is insufficient for a policy-document chatbot.

### K.1 Investigation (no retrieval/chunking/selection changes)
- Extractive path: `GroundedPipeline.answer_from_hits` answered because retrieval distance/token-support thresholds were satisfied (`token_support=0.5`, `best_distance=0.8424`) even though required evidence chunk `satcom-ngp::page_3::chunk_2::off6299` was absent and selected statements weakly supported the question (`selected_lexical_support=0.3333`).
- Existing sufficiency signal already computed in service and used for Gemini precheck: selected lexical support vs `RAG_GEMINI_SUFFICIENCY_THRESHOLD` (default 0.5) and selected-statement presence.

### K.2 Implementation (general-purpose, conservative)
- **`src/rag_app/service.py`:** added application-level answerability guard after extractive selection (before authoritative response/generation):
  - If pipeline answered but `no selected statements` or `selected_lexical_support < threshold` → abstain with explicit reason (`insufficient_selected_evidence:*`), return `ABSTENTION_MESSAGE`, preserve statements/citations/retrieved/selected IDs for inspection, record `answerability_guard_applied/reason` and pipeline status in debug.
  - Gemini path reconciled to same authoritative abstention/sufficiency state (no call when insufficient; reports `abstained_insufficient_evidence` fallback metadata).
- **No changes** to retrieval, chunking, index, EXP-003 defaults, or experimental opt-ins. No Q05/question-specific logic.

### K.3 Tests added/updated (`tests/test_api.py`, 7 tests total)
- Added `test_service_extractive_abstains_on_insufficient_selected_evidence` and `test_service_extractive_keeps_supported_answer`.
- Updated Gemini insufficient-evidence test to expect authoritative abstention + reason.
- Updated real-index smoke: Q05 now must abstain with `insufficient_selected_evidence` reason, selected support < 0.5, guard applied, BL-001 chunk still absent, evidence preserved for inspection; Q12 still answered with valid app-resolved citations/offsets/timings.

### K.4 Results (executed 2026-11-15)
- `python -m pytest tests/test_api.py -q` → **7 passed, 4 warnings** (34.40s).
- `python -m pytest tests -q` → **108 passed, 33 warnings** (47.65s). Warnings non-fatal (known Chroma/Pydantic, google-api-core/grpc/Python 3.11).
- **15-question extractive regression (RAGService, EXP-003 defaults, corrected prod index):**
  - **Q05:** `abstained=true`, `selected_support=0.3333`, reason `insufficient_selected_evidence:selected_lexical_support=0.3333<threshold=0.5`, guard applied. Retrieval miss unchanged.
  - **Q01–Q04, Q06–Q15:** `answered`, guard not applied. Selected supports: Q01 0.75, Q02 1.0, Q03 0.6, Q04 0.6, Q06 0.875, Q07 1.0, Q08 1.0, Q09 1.0, Q10 0.7778, Q11 0.75, Q12 0.8, Q13 0.6, Q14 0.5455, Q15 0.5 (boundary, passes `<0.5`).
  - **No additional abstentions/regressions** on current 15-question set. Q12 citation metadata/behavior unchanged under defaults.

### K.5 Trade-offs / remaining limitations
- Heuristic lexical-overlap guard, **not semantic proof** of answerability/correctness/completeness; can still allow weak-but-above-threshold or semantically wrong answers (e.g. Q08 incomplete retrieval, Q10 selection issues remain under defaults).
- Threshold boundary behavior (0.5 passes) is intentional/conservative relative to current signal; may need evidence-driven calibration on larger labeled set (M4), not tuned here.
- BL-001 retrieval remains open (guard mitigates unsafe answering, not retrieval). BL-007 improved but not solved.

### K.6 Next
Proceed to **WP2 — frontend integration** with new `/query` API (extractive default), inspectable citations/abstention/errors/loading states, validated local UI→API flow.

---

## L. WP2 — Frontend integration with validated `/query` API (2026-11-15)

**Objective:** connect React frontend to WP1 backend, display grounded answers with inspectable citations, abstention/error/loading states; extractive default.

### L.1 Implemented
- **Replaced `frontend/src/App.jsx`** with clean UI integrated to `POST ${VITE_API_BASE_URL||http://localhost:8000}/query` (`mode:"extractive"` default):
  - Query input, loading state, request error handling, sample questions (Q05/Q12), response history.
  - Result rendering: status (answered/abstained/error), answer text, abstention reason, **app-resolved inspectable citations** (marker, section/heading, page range, chunk ID, source, offsets, distance, quote), selected evidence statements, retrieved chunks, request ID/timings, guard/debug diagnostics.
  - Health check button (`GET /health`), guide tab with run instructions.
  - Removed dependency on legacy `/chat` and `/metrics` contracts and prior IndexedDB token/session logic from active UI.
- **Reused:** existing Vite/React/Tailwind/UI component structure (`frontend/src/components/ui/*`, styles, config).

### L.2 Verification (executed / not executed)
- **Executed (backend):** `python -m pytest tests/test_api.py -q` → **7 passed** (incl. real-index smoke Q05 abstain/Q12 traceable citations); `python -m pytest tests -q` → **108 passed, 33 warnings** after WP1.1. Backend contract supports frontend fields.
- **Not executed:** `cd frontend && npm run build` failed in this environment (`vite` not found / dependencies not installed). Frontend source updated but build/runtime not verified here; requires `npm install` + local dev/build with backend running.

### L.3 Limitations / next
- Gemini UI toggle not exposed (extractive default; M1 not accepted). No automated browser E2E tests added.
- Next: **WP3** end-to-end local validation runbook/smoke (API+UI), integration hardening; **WP4** observability depth; **WP5** eval expansion/CI; **WP6** Vercel Hobby deployment feasibility/implementation.

---

## M. WP2 verification and WP3 end-to-end local validation (2026-11-15)

**Objective:** verify actual frontend integration/build and complete local E2E validation without modifying retrieval/chunking/EXP-003/opt-ins, without EXP-009, and without threshold tuning.

### M.1 Frontend package/install/build (executed)
- **Package manager/lockfile:** `frontend/package.json` + `frontend/package-lock.json` (npm). Reproducible install: `npm ci`.
- `cd frontend && npm ci` → **success** (170 packages added, audited 171; vulnerabilities reported by npm audit, not remediated in this pass).
- `cd frontend && npm run build` → **success** (Vite 8.1.3, 36 modules, `dist/index.html`, `dist/assets/index-BVkQV0Vb.css` 24.64 kB gzip 5.60 kB, `dist/assets/index-J4qS0Kjf.js` 256.36 kB gzip 79.34 kB; built 223ms initial / 275ms final).

### M.2 Running application checks (executed)
- Started **backend:** `PYTHONPATH=. python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000` (background). Health ready: `{"status":"ok","default_mode":"extractive","collection_name":"satcom_ngp_production_fixed_char","top_k":5,"index_available":true,"gemini_configured":true}`.
- Started **frontend:** `npm run dev -- --host 127.0.0.1 --port 5173` (Vite dev served app at 127.0.0.1:5173).
- **Browser-equivalent HTTP/CORS flow tested** (requests matching frontend `fetch` with `Origin: http://127.0.0.1:5173`, JSON POST to `/query`, GET `/health`):
  - **CORS preflight:** `OPTIONS /query` → 200 with `access-control-allow-origin: http://127.0.0.1:5173`, allow POST/content-type.
  - **Health-check:** `GET /health` → ok, index available, extractive default.
  - **Q12 (answered, inspectable citations):** `POST /query {"question":"What satellite communication services are classified as domestic services?","mode":"extractive"}` → `status=answered`, `abstained=false`, 3 statements, 2 app-resolved citations with markers/sections/pages/offsets/chunk IDs/quotes (`[1] sec 4.1 heading 4.1 SERVICE CLASSIFICATION p13 30178–31178 chunk satcom-ngp::page_13::chunk_0::off30178`; `[2] sec 4.3.4 p14 32496–33496 chunk satcom-ngp::page_14::chunk_0::off32496`), timings recorded. (EXP-003 default selection preserved; not EXP-008-promoted answer.)
  - **Q05 (abstention, clear explanation, inspectable retrieved evidence):** → `status=abstained`, reason `insufficient_selected_evidence:selected_lexical_support=0.3333<threshold=0.5`, guard applied, preserved citations/statements/retrieved evidence and debug (BL-001 retrieval miss remains, required chunk not present).
  - **Normal supported question:** `"Who can allot capacity in INSAT/GSAT satellites to non-government users?"` → `status=answered`, citations to `2.6.2` and `2.6.4` (p4) with timings.
  - **Repeated requests:** same supported question twice → consistent answers, different `request_id`s, both answered.
  - **Backend unavailable:** backend stopped; `POST /query` → connection failure (`curl exit 7`, could not connect), consistent with frontend `fetch` error/catch UI path. Frontend dev server remained reachable.
- **Not executed (limitation):** no headless browser automation; in-browser DOM rendering, visual loading/disabled states, and manual click-through UX were not directly observed. Frontend code implements these states; API-level browser-flow behavior verified.

### M.3 Automated validation and runbook
- **Python:** `python -m pytest tests -q` → **108 passed, 33 warnings** (19.99s). Existing integration/contract coverage: `tests/test_api.py` (mocked + real-index smoke, Q05/Q12 representative cases).
- **Frontend:** `npm run build` → **success** (275ms final run).
- **Runbook:** `docs/RUN_LOCAL.md` updated (prereqs, reproducible npm ci, backend/frontend start, build, smoke, WP2/WP3 verification record, limitations).

### M.4 Boundaries respected
No retrieval/chunking/EXP-003/opt-in changes, no EXP-009, no sufficiency-threshold tuning, extractive default preserved, experiment history/unrelated working-tree changes preserved. No internal debug/sensitive config added for public exposure.

### M.5 Next
Move to **deployment feasibility (M3/WP6 planning)**, **observability expansion (WP4)**, and **evaluation/CI planning (M4/M5)** per roadmap; targeted optimization only with measured evidence.

---

## N. M3 — Deployment feasibility & architecture assessment (2026-11-15, evidence-based, zero-cost strict)

**Objective:** assess genuinely zero-cost deployment options for the current RAG application (preserve fixed-char/EXP-003/citations/abstention behavior; extractive default; optional Gemini M1 not accepted) without consequential infrastructure changes.

### N.1 Evidence gathered (repo + authoritative/public sources)
- **Repo baseline:** FastAPI+`rag_app`+`rag_core`; React/Vite frontend (dist ~296 KB, build pass); `data/satcom-ngp.pdf` 68 KB; corrected Chroma prod index `chroma_db_production/` ~3.89 MB (49 chunks); query embeddings via `sentence-transformers/all-MiniLM-L6-v2`; local persistent Chroma; heavy deps in `requirements.txt` (`torch 2.12.0`, `sentence-transformers 3.0.1`, `chromadb 0.5.23`, `onnxruntime 1.24.4` + eval/legacy libs). WP2/WP3 validated locally (**108 passed**, frontend build pass, running API/CORS scenarios exercised). Bundle/RSS/cold-start not measured on targets.
- **Vercel (docs/search, incl. fetched limits):** Hobby $0 allotments (e.g. 4 active CPU-hrs, 360 GB-hrs, 1M invocations), 300s max, 2 GB/1 vCPU (Hobby), Python 500 MB uncompressed bundle (standard; large functions beta exists with conditions), ephemeral serverless FS; compatibility of current ML stack **unverified/likely constrained**.
- **Hugging Face Spaces (official docs/search evidence):** “Gradio and Docker Spaces run on compute and require a paid plan to create: PRO for personal accounts, Team or Enterprise for organizations. Free personal accounts in good standing can still host up to 2 Gradio Spaces running on ZeroGPU.” CPU Basic listed $0/hr but Docker/Gradio compute creation gated on free accounts; 50 GB non-persistent disk, sleep on free hardware. **Docker Spaces not zero-cost eligible** for personal free accounts; ZeroGPU Gradio-only unsuitable for preserving current stack.
- **Gemini (ai.google.dev pricing/rate-limits):** Free tier available for eligible models with project/model RPM/TPM/RPD quotas (AI Studio), RPD midnight PT reset, no spend-based limit for Free tier, no billing required to remain on Free tier; free-tier content may be used to improve products and some features unavailable. Must avoid paid billing/paid features and handle quotas/429.
- **Render (official-doc excerpts via search):** free web services (512 MB/0.1 CPU, 750 instance hrs/mo, 15 min spin-down, ephemeral FS, bandwidth/build limits/suspension risks) — likely insufficient for ML stack; terms require dashboard confirmation.

### N.2 Comparison and decision
- **Vercel Hobby:** monetarily plausible within limits but **not claimed compatible** (bundle/memory/cold-start/model+Chroma untested for heavy deps). **HF Docker Spaces:** **rejected** (requires PRO $9/mo, violates strict zero-cost). **Render free:** likely resource-incompatible. **Oracle Always Free VM:** technically closest to local with minimal change but tenant/eligibility/ops caveats prevent confident guaranteed-$0 claim without verification.
- **Recommendation:** no consequential deployment now. Preserve local validated app; provide reproducible $0 demo/runbook; if public hosting mandatory, only after tenant-verified Oracle Always Free (or other verified $0) confirmation. No DB/embedding/RAG rewrites.

### N.3 Deliverables and next
- **Report:** `docs/M3_DEPLOYMENT_FEASIBILITY.md` (detailed comparison, risks, conditional implementation steps, alongside-work: M4 eval planning, M5 observability/CI/hardening, minimal runtime packaging/measurement).
- **Status:** awaiting user review/approval before any infrastructure changes or deployment attempts.

### J. 2026-11-15 — M4 evaluation expansion foundation and M5 observability (parallel, no prod-behavior changes)

**M4 (foundation + baseline measured, no retrieval/chunking/EXP-003/threshold tuning):** Added `docs/M4_EVALUATION_PLAN.md`, `removed-experiment-artifacts/m4_eval_harness.py`, and `docs/M4_PROGRESS.md`. Preserved existing datasets/ground truth (`exp001_questions.csv`, `exp001_ground_truth.json`, `unanswerable_questions.csv`). Executed baseline on local validated `RAGService` (extractive, EXP-003, corrected prod index, k=5): **answerable 15** — answered 14, abstained 1 (Q05 guard), supported_answer_rate 0.80, statement support 0.3333, citation support 0.5161, mean recall@5 0.8889 / precision@5 0.2533 / MRR@5 0.7944; **unanswerable 12** — abstention accuracy 1.00. Latency (includes cold first-call outlier): total p50 19.704 ms / p95 22.648 ms / max 17265.834 ms. Artifacts: `removed-experiment-artifacts/results/m4_baseline/`. Thresholds not set (must be derived from measured expanded baselines). Pending: manual curation/verification of 35–85 additional questions (50–100 initial, path to 200) covering all four articles and required types/tags/evidence/citation/abstention labels.

**M5 (lightweight, $0, exportable):** Extended `src/rag_app/observability.py` (trace details/total), added `src/rag_app/metrics.py` (in-memory aggregated collector: totals/rates, p50/p95 latency for total/retrieval/selection/generation, recent records), instrumented `src/rag_app/service.py` (retrieval/selection/abstention/citation-validation/generation details; metrics recording), added `GET /metrics` in `backend/main.py`, and documented in `docs/M5_OBSERVABILITY.md`. Token/cost fields recorded when available (Gemini cost not computed). **Tests:** `tests/test_api.py` 8 passed (incl. metrics endpoint), `tests -q -k "not real_index"` 108 passed / 1 deselected.

**Deployment investigation (separate, no provisioning):** Added `docs/ORACLE_ALWAYS_FREE_INVESTIGATION.md`. Direct authoritative OCI page retrieval blocked in-session (websearch unavailable; oracle docs fetch transport/403) — findings marked best-effort and **UNVERIFIED pending current official verification**. Conclusion: Oracle Always Free plausibly closest VM path for current stack but **not confidently guaranteed $0/obtainable** without tenant-specific eligibility/quota/capacity/payment/idle-reclamation verification and resource measurement; **no account created, no resources/billing/deployment**. Local app remains validated source of truth.

### K. 2026-11-15 — Post-commit verification (commit `03c79d4`, no additional commit created)

**Commit/remote sync (verified):** `03c79d4cd920206ff092bdcc1d772ca956f05d42` on `main`, synchronized with `origin/main` (`https://github.com/Sivasindhuja/RAG`); `git fetch` clean, `git status -sb` shows only pre-existing unrelated unstaged deletions.

**1) Dedicated secret scan (entire history / repository):**
- **Gitleaks:** not installed/available in environment.
- **History pattern scan (all refs, full history):** no matches for `AIza*`, GitHub/cloud token patterns, private key blocks, or non-placeholder `GEMINI_API_KEY=<value>` assignments. (Prior audit noted old commit `2b814b` contains env-var name/placeholder usage only, not real credentials.)
- **detect-secrets v1.5.0 (installed for this pass):** scanned repository files (`package.json`, `frontend/package.json`, `backend`, `docs`, `src`, `experiments`, `tests`; large generated artifacts/caches excluded). **1 finding:** `tests/test_generation.py:230` flagged as `Secret Keyword` (unverified, hashed). Manual inspection: line contains a **test fixture/dummy value assignment** (`"sk-or-v1-placeholder"`) used for unit-test construction, **not an actual credential**. Classified as **false positive/test placeholder**; no rotation/history rewrite required. **No actual credentials discovered.**
- **Limitation:** no dedicated full-history secret scanner (e.g. gitleaks) executed over all historical blobs; pattern scans + detect-secrets provide additional assurance but not proof of absence.

**2) Frontend build (clean dependency environment, no package-version changes):**
- Performed `cd frontend && rm -rf node_modules && npm ci && npm run build` (lockfile-resolved install).
- **Result:** `npm ci` completed (170 packages), **`vite build` succeeded** (Vite 8.1.3, 36 modules, `dist/index.html` 0.47 kB gzip 0.30 kB, `dist/assets/index-BVkQV0Vb.css` 24.64 kB gzip 5.60 kB, `dist/assets/index-J4qS0Kjf.js` 256.36 kB gzip 79.34 kB, built in 188 ms).

**3) `npm audit` (inspected; no `npm audit fix --force` run):**
- `npm audit` (registry reachable in this pass) reported **11 vulnerabilities: 4 moderate, 7 high** (fixes available via `npm audit fix`, not applied).
- **Classification (by package/dependency role):**
  - **Dev/build toolchain (not shipped in production bundle):** `eslint`/`@eslint/config-array`/`minimatch` (moderate/chain), `brace-expansion` (high, via eslint minimatch chain), `zod` (high advisory; present via eslint plugin/tooling chain), `vite` (high advisories incl. dev-server/path/middleware issues), `browserslist` (high), `baseline-browser-mapping` (moderate, via browserslist), `@vitejs/plugin-react` chain — used for lint/build/dev only.
  - **Build-time CSS/asset processing deps (not runtime app logic shipped to users, but affect build pipeline):** `postcss` (high; devDependency) and transitive `nanoid` (high advisories), `source-map-js` (high). `@tailwindcss/vite` is declared in `dependencies` but functions as a Vite/build plugin; production artifact is static `dist/`.
  - **Direct production runtime dependencies (`react`, `react-dom`, Radix UI, `clsx`, `tailwind-merge`, `class-variance-authority`, `uuid`):** **no vulnerabilities reported** in this audit output for these direct runtime packages.
- **Risk note:** dev-server/build-chain vulnerabilities can matter in local/CI/dev contexts (e.g. malicious project content, untrusted inputs in dev tooling) and supply-chain/build integrity; they do not by themselves indicate exploitation in the deployed static frontend bundle. Remediation should be evaluated via normal dependency updates (respecting compatibility/lockfile), not `--force`.

**4) Necessary corrections:** none identified requiring a new commit. Current commit and accepted history preserved (no force-push, no rewrite). No credentials discovered; no rotation/history rewriting performed.

**Status:** post-commit verification completed with noted limitations (no gitleaks over full history; detect-secrets finding reviewed as test placeholder). Returned to M4 dataset expansion work.

### DR-017 — Final delivery implementation: chunking selection, hybrid retrieval, prompts, CI eval, tracing, and deployment artifacts

**Date.** 2026-11-19

**Context.** Final project delivery required completing ingestion-to-answer pipeline, chunking experiment, hybrid retrieval + Cohere reranking integration, externalized/versioned prompts, RAGAS-integrated evaluation on fixed 15-question CI suite, PR CI with regression gates, Langfuse tracing, metrics dashboard, and public deployment compatibility.

**Decisions and evidence.**
- **Chunking (measured):** controlled experiment (`removed-experiment-artifacts/chunking_experiment.py`, `removed-experiment-artifacts/results/chunking_experiment/`) compared fixed-character 500/100, 800/160, 1000/200, 1200/240 on CI answerable subset (same PDF/embedding/dense k=5). Selected **800/160** (61 chunks) for highest recall@5 (0.9236) and MRR@5 (0.8611), audit ok. Q05 (2.5.6) became retrievable at 800/160 (and other tested sizes except 1000/200). Default config and ingestion/index rebuild updated to 800/160; collection `satcom_ngp_production_fixed_char_800_160` built. Historical 1000/200 artifacts/baselines preserved.
- **Hybrid retrieval (measured):** implemented BM25+RRF fusion and configurable retriever (`src/rag_core/retrieval.py`, `HybridRetriever`, modes dense/hybrid/hybrid_rerank). On 800/160 CI set, hybrid improved supported_answer_rate 0.5833→0.8333 and citation_support_rate 0.4828→0.5484 vs dense with same recall/evidence coverage (0.9236) and precision 0.3333 (`evaluation/results/ci_compare_800_160/report.json`). Hybrid adopted as default retrieval mode. Q08 remains partial retrieval; Q10 remains selection-limited despite evidence retrieval.
- **Cohere reranking:** integrated (`cohere_rerank`, mode `hybrid_rerank`, default `rerank-english-v3.0`) with mocked tests (`tests/test_cohere_reranker.py`). Live credentialed comparison not completed because `CO_API_KEY`/`COHERE_API_KEY` unavailable (empty in `.env`); fallback runs not evidence of reranker quality. Pending valid key for live latency/usage/cost/quality measurement.
- **Prompts:** externalized to `prompts/` with manifest and versioned template (`prompts/generation_grounded_v1.txt`, `grounded_json_v1`), loader (`src/rag_app/prompts.py`) and generation integration; prompt version recorded in responses/trace metadata.
- **CI fixed 15-question suite:** `evaluation/ci_golden_dataset.json` (ci_golden_v1, draft_not_independently_verified) composed as 12 preserved answerable EXP-001 questions + 3 preserved unanswerable/out-of-scope questions to satisfy 15-question limit and required coverage; historical artifacts unchanged. Evaluation harness `evaluation/ci_eval.py` computes retrieval/answer/citation/abstention/latency metrics, retrieval comparisons, and regression gates against measured baseline (`evaluation/baseline_metrics.json`, 800/160 dense). Workflow `.github/workflows/ci.yml` runs on PR/push with deterministic tests/eval and regression check.
- **RAGAS:** integrated in `evaluation/ci_eval.py` (`--enable-ragas`); live run attempted (`evaluation/results/ci_ragas_dense_800_160/`) but failed in this environment due to ragas/langchain/pydantic dependency incompatibility. Metrics not measured; integration retained.
- **Langfuse tracing:** implemented best-effort/no-op tracer (`src/rag_app/langfuse_tracing.py`) instrumented across request, retrieval, selection, evidence sufficiency, generation, citation validation stages with metadata capture (request/question/mode/retrieval, prompt/model info, chunk IDs/ranks/scores, errors). Live tracing unavailable because Langfuse credentials empty in `.env`; mock/no-op behavior tested (`tests/test_observability_langfuse.py`).
- **Dashboard/API:** `/metrics`, `/dashboard` (live + offline artifacts), `/ingest`, `/query` with retrieval_mode/generation mode, trace IDs/timings/debug. Frontend remains integrated to backend API.
- **Deployment:** Vercel-oriented artifacts added (`vercel.json`, `api/index.py`) with documented Hobby/serverless constraints (bundle/model/cold start, no persistent local Chroma). No paid infrastructure used; public deployment not completed due to external hosting/provisioning/account constraints. Precise remaining steps documented in `docs/DEPLOYMENT.md`.

**Test evidence (2026-11-19).**
- `python -m pytest tests -q -k "not real_index"`: **118 passed, 1 deselected, 29 warnings**
- `python -m pytest tests/test_api.py::test_real_index_extractive_smoke_q05_and_q12_representative -q`: **1 passed, 4 warnings** (hybrid, 800/160 index)
- New tests: prompts, hybrid RRF/tokenization, Cohere reranker mocks, Langfuse no-op tracer.

**Known limitations (honest).** Cohere live rerank and Langfuse live traces pending credentials; RAGAS metrics pending compatible runtime; public deployment pending verified free-tier hosting/provisioning; Gemini quality limitations remain (extractive grounded path default); Q08 partial retrieval and Q10 selection issues persist; CI dataset draft/not independently verified.

---

### DR-018 — Final gap closure: Langfuse skill/v4 tracing, RAGAS execution, live evals, CI verification, metrics/dashboard, and deployment status (2026-11-19)

**Scope.** Closed remaining delivery gaps without starting new RAG experiments or rebuilding completed components. Preserved pre-existing unrelated working-tree deletions and historical artifacts.

**Langfuse (skill + implementation).**
- Installed Langfuse AI skill from `github.com/langfuse/skills` into `.agents/skills/langfuse` (SKILL.md + references) and followed skill guidance (documentation-first, v4 SDK/manual instrumentation, meaningful trace/span names, input/output/metadata capture, flush, no-op safety).
- Rewrote tracing integration for Langfuse Python SDK v4 (OTEL-based, 4.2.0) in `src/rag_app/langfuse_tracing.py` (root request observation + child observations with stage-appropriate types: retriever/chain/evaluator/generation), added generation usage details capture in `src/rag_app/service.py`, maintained no-op behavior when disabled/unconfigured. Tests: `tests/test_observability_langfuse.py` and API tests passed.
- **Live Langfuse tracing:** blocked in-session (`LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` empty in `.env`); integration is credential-ready but actual cloud traces/pipeline-stage visibility could not be verified without valid credentials.

**RAGAS / evaluation pipeline.**
- Resolved RAGAS/Pydantic compatibility by upgrading `ragas` to 0.2.15 and updating `evaluation/ci_eval.py` (Gemini key mapping for LangChain wrappers, explicit API keys, configurable metrics/models/embedding, serialized `RunConfig`, `raise_exceptions=False`).
- Successfully executed RAGAS-enabled evaluation pipeline (not left unexecuted): final artifact `evaluation/results/ci_ragas_dense_800_160_final/report.json` with metrics `context_precision=0.3333333333041667`, `context_recall=0.4166666666666667`, `faithfulness=0.8605442176870749`, `answer_relevancy=null`, `answer_correctness=null` (12 rows, `gemini-flash-lite-latest`; embedding recorded as `models/text-embedding-004`, null metrics indicate partial computation issues/quota/runtime effects). Earlier failures recorded in `ci_ragas_dense_800_160*` artifacts.

**Frontend and production configuration (Gemini path).**
- Verified backend/default production config retains Gemini provider/model path (`google_generativeai` / `gemini-flash-lite-latest`, configurable via env; `RAG_ALLOW_GEMINI`, `GEMINI_API_KEY`/`GOOGLE_API_KEY`). No alternative LLM provider/model searched or introduced.
- Updated frontend (`frontend/src/App.jsx`) to support selectable generation mode (`extractive` default, `gemini` available) against existing `/query` API and clarified UI messaging; frontend build passed (`npm run build`).

**Live Gemini evaluation (measured, honest).**
- Artifacts: `evaluation/results/gemini_live_eval/{responses.json,summary.json,analysis.json,analysis_summary.json}` (15-question CI set, hybrid retrieval, mode gemini).
- Aggregate: 15 requests, 12 answered, 3 abstained, 0 errors. Request latency mean 3766.529 ms, p50 3162.029 ms, p95 5231.978 ms. Generation latency 926.8–10151.6 ms (answered cases). Tokens (12 requests with usage): prompt 4613, completion 673, total 5286. Estimated cost USD not computed (pricing/SKU not auto-fetched).
- Quality issues (automated review): all unanswerables correctly abstained; no invalid source IDs/uncited claims detected; **incomplete/quality-deficient answerable responses** observed for `EXP001-Q04`, `EXP001-Q08`, `EXP001-Q10`, `EXP001-Q15` (missing expected details/categories/components/procedural specificity). Not production-accepted.

**Live Cohere reranker validation (measured).**
- `CO_API_KEY` available; live rerank applied (`rerank-english-v3.0`, observed scores/latency). Comparison: `evaluation/results/ci_compare_800_160_rerank_live_final/report.json` (dense/hybrid/hybrid_rerank, extractive, k=5, 15-question CI).
- Results: dense recall@5 0.9236/mrr@5 0.8611/supported 0.5833/citation 0.4828; hybrid recall@5 0.9236/mrr@5 0.8472/supported 0.8333/citation 0.5484; hybrid_rerank recall@5 0.9236/mrr@5 0.9028/supported 0.6667/citation 0.4839 (latency much higher). **Mixed evidence; no overall improvement claimed** (ranking improved vs hybrid but answer/citation support decreased).

**CI dataset and workflow validation.**
- Fixed 15-question suite preserved (`evaluation/ci_golden_dataset.json`, draft_not_independently_verified). Performed separate **automated** verification pass: `evaluation/results/ci_dataset_verification.json` (pass: count/split/IDs/evidence-span-to-PDF mapping/field consistency). Independent human verification **not claimed**.
- GitHub Actions workflow `.github/workflows/ci.yml` reviewed and locally validated with CI-equivalent commands including regression gate: `evaluation/results/ci_local_validation/report.json` → `regression.passed=true` against measured baseline. Actual remote PR/push execution not performed (no authenticated `gh`/push credentials in environment).

**Dashboard/metrics and live verification.**
- Enhanced `src/rag_app/metrics.py` and `backend/main.py` dashboard to aggregate/report P50/P95/mean latency, token consumption, cost-per-request, citation coverage, failure/abstention rates; **null used for unavailable metrics (not zero)** with explicit notes. Generated actual request/evaluation data and snapshot: `evaluation/results/live_metrics_snapshot.json` (4 requests: 3 extractive + 1 Gemini, failure_rate 0.0, citation_coverage 1.0, latency p50 1603.017 ms/p95 12112.304 ms, token mean 516 where available, cost mean null).

**Deployment.**
- Deployment artifacts/config remain in place (`vercel.json`, `api/index.py`, env-driven config, CORS). Architecture documented as Vercel frontend + separate free-tier backend host with persistent storage/model cache (or equivalent) due to Hobby/serverless constraints for local Chroma + embedding model. **Public deployment not completed** (no verified free-tier backend provisioning/account access/Vercel deployment credentials in-session); no public URL. Remaining steps and constraints recorded in `docs/DEPLOYMENT.md`.

**Test results (executed).** `python -m pytest tests -q` → **120 passed, 42 warnings**. `cd frontend && npm run build` → **passed**. Local CI regression eval passed.

**Blockers (external).** Missing Langfuse API credentials (live trace verification in Langfuse UI/cloud). No authenticated GitHub access to create/run actual remote PR/push workflow run. No verified free-tier backend hosting/account provisioning/Vercel deployment access for public URL. RAGAS metrics partially null in executed run (quota/runtime/evaluator constraints). Gemini cost not auto-priced.

---

### DR-019 — CI pypdf consistency, Q12 integration-test setup, and definition-intent boost guard (2026-11-21)

**Problem.** PR #1 GitHub Actions failures reported for commit `2c3003b` (branch `deploy/render-vercel`): (1) RAG Evaluation CI evidence-mapping mismatch on `EXP001-Q03-E01`, and CI pytest failures including source-excerpt mismatches after global offset mapping (e.g. `EXP001-Q13-E02` expected `4.3.2`/actual `.3.2`, `EXP001-Q13-E03` `4.3.3`/`.3.3`, `EXP001-Q15-E01` identical-looking prefix but comparison failure). (2) After later fixes, `tests/test_citation_metadata.py::test_q12_definition_statements_resolve_to_clause_4_2` failed in CI because it hardcoded collection `satcom_ngp_production_fixed_char` and `chroma_db_production` while `AppConfig` defaults to `satcom_ngp_production_fixed_char_800_160`; `.gitignore` excludes `chroma_db_production/`; and CI ran pytest before production-index build/verify. With current production config (800/160, 61 chunks, `pypdf==6.10.2`), Q12 retrieved expected 4.2 statements (offsets 31006, 31255) but also selected an unrelated 3.2 fragment (offset 10868).

**Investigation / root cause (measured).**
- **Extraction determinism (local diagnosis):** `data/satcom-ngp.pdf` (tracked, 68,396 bytes) extracted via `src/rag_core/document.py:extract_source_document` (`pypdf.PdfReader.extract_text()` per page, as-is, cumulative offsets). Local env `pypdf 6.10.2` (win32, Python 3.11.0): 15 pages, 35,407 chars, per-page `[2517,2182,2166,2723,2360,2982,2864,2609,2457,2840,2301,2177,2316,2692,221]`; all 17 `evaluation/ci_golden_dataset.json` units and all 20 `tests/fixtures/exp001/exp001_ground_truth.json` units matched exactly; `tests/test_exp001.py` + `tests/test_rag_evaluation.py` = 32 passed locally. CI used unpinned/loose `pypdf` (`requirements.txt` unpinned, `requirements.runtime.txt>=4.0`), consistent with environment-dependent `extract_text()` drift causing reported mapping failures. Exact CI extracted text/offsets not available locally; cause attributed with high confidence to extractor/version drift under non-deterministic dependency resolution.
- **Q12 trace (isolated temp index, not `chroma_db_production/`):** temp `C:\Users\sindh\AppData\Local\Temp\opencode\q12_ci_idx_xkh2rq2j`, collection `satcom_ngp_production_fixed_char_800_160`, 800/160, 61 chunks. Hits for “How are international satellite communication services defined?” included 4.2 chunks (off30818, off31458) and 3.2 chunk `satcom-ngp::page_5::chunk_2::off10868` (rank 4). `GroundedPipeline(index, sibling_score_ratio=0.5, definition_intent_boost=True).answer_from_hits` selected 31006/31255 (4.2, correct per expected answer/evidence) and 10868–11204 (3.2, **incorrect/unrelated**: “defined as  the satellites…” defining Indian satellites, chunk-boundary mid-sentence fragment). Base score 5.0123 (tokens `defined,international,satellite`), boosted to 6.5123 by definition-intent marker `defined as` (`src/rag_core/pipeline.py` `_extractive_answer` boost at scoring loop; top candidate had no numbered-clause parent so global score selection applied). Citations/offset mapping consistent; retrieval included correct evidence but selection admitted unrelated fragment.

**Decisions / implementation (uncommitted at time of log, local working tree).**
- **Dependency determinism:** pinned `pypdf==6.10.2` in `requirements.txt` and `requirements.runtime.txt`; added diagnostic step in `.github/workflows/ci.yml` and `.github/workflows/rag_eval.yml` (Python/pypdf version, PDF page count, total/per-page extracted char counts via `extract_source_document`).
- **CI/test setup:** moved “Build/verify index and ingestion integrity…” before pytest in `.github/workflows/ci.yml`; updated `tests/test_citation_metadata.py::test_q12_definition_statements_resolve_to_clause_4_2` to use `AppConfig.from_env()` for `persist_dir/collection_name/embedding_model` and assert `index.count() > 0` before query. **All original Q12 assertions preserved** (not weakened/rewritten).
- **Answer-selection correction:** in `src/rag_core/pipeline.py` added `_is_mid_sentence_chunk_boundary_fragment(chunk_text, local_start, sentence)` and suppressed **definition-intent boost only** when candidate is at chunk start (allowing leading whitespace) with lowercase initial character (mid-sentence chunk-boundary artifact). Base score/selection eligibility unchanged. Added focused regression tests in `tests/test_pipeline.py` covering: fragment-at-chunk-start boost suppression, complete definition sentence later in chunk still boosted, non-boundary chunk-start candidate retains boost behavior, and boost-disabled behavior unchanged for fragment.

**Verification (local; CI/public not re-verified in this step).**
- Diagnostics: `python_version=3.11.0`, `pypdf_version=6.10.2`, `pdf_page_count=15`, `total_extracted_char_count=35407`, per-page `[2517,2182,2166,2723,2360,2982,2864,2609,2457,2840,2301,2177,2316,2692,221]`.
- Evidence mapping: `evaluation/ci_golden_dataset.json` 17 units/0 mismatches; `tests/fixtures/exp001/exp001_ground_truth.json` 20 units/0 problems (`load_exp001_evidence`).
- `python -m pytest tests/test_exp001.py tests/test_rag_evaluation.py -q` → **32 passed, 25 warnings** (exit 0).
- `python -m pytest tests/test_pipeline.py -q` → **25 passed** (exit 0) including new definition-intent boost regression tests.
- `RAG_PERSIST_DIR=<temp isolated index> python -m pytest tests/test_citation_metadata.py -q` → **1 failed, 3 passed** (exit 1): Q12 still fails original assertion (`by_chunk[s.chunk_id].section == "4.2"` got `3.2`) because 3.2 fragment base score (5.0123, boost now suppressed) still outranks 4.2 boosted statements (4.3267/4.1260); suppressed boost alone is **necessary but not sufficient** under current production config. Q12 now reaches retrieval/answer with expected 31006/31255 present but also includes 10868/3.2.
- `RAG_PERSIST_DIR=<temp isolated index> RAG_GENERATION_MODE=extractive RAG_RETRIEVAL_MODE=dense LANGFUSE_ENABLED=false TOKENIZERS_PARALLELISM=false python -m pytest tests -q -k "not real_index" --maxfail=5` → **1 failed, 122 passed, 1 deselected, 29 warnings** (exit 1), same Q12 failure.
- `PYTHONPATH=. RAG_GENERATION_MODE=extractive RAG_RETRIEVAL_MODE=dense LANGFUSE_ENABLED=false TOKENIZERS_PARALLELISM=false python -m evaluation.ci_eval --dataset evaluation/ci_golden_dataset.json --out evaluation/results/ci_local_q12fix --retrieval-modes dense --generation-mode extractive --k 5 --baseline evaluation/baseline_metrics.json --check-regression` → exit 0, regression **passed** (dense recall@5 0.9236, mrr@5 0.8611, evidence_coverage 0.9236, supported_answer_rate 0.5833, citation_support_rate 0.4828, abstention_accuracy 1.0, invalid_citations 0, failure_rate 0.0; `evaluation/results/ci_local_q12fix` untracked local artifact).

**Notes / side effects.** During earlier temp-index setup for verification, a CI-equivalent build was mistakenly executed without `RAG_PERSIST_DIR` pointed at temp, rebuilding local `chroma_db_production/` (gitignored, `chroma.sqlite3` updated) with equivalent current config; subsequent verification used isolated temp index as required. Untracked `nul` entry observed and preserved untouched. CI and public deployment verification **not executed** in this session. Q12 remains unresolved pending approved follow-up (authorized boost-suppression implemented but insufficient given base-score dominance of unrelated chunk-boundary fragment; any further selection/extraction change requires explicit authorization and must preserve Q12 intent/assertions constraints).

---

### Investigation (2026-11-21, investigation-only; no implementation) — Q12 residual selection of unrelated 3.2 fragment (base-score dominance)

**Status/constraints.** Investigation-only (no code/config/data/dataset/baseline/retrieval/chunking/embedding/RRF/provider/model changes; no rebuild of `chroma_db_production/`; no staging/commit/push/merge/deploy). Preserves DR-019 and all existing working-tree changes (including uncommitted `.github/workflows/ci.yml`, `tests/test_citation_metadata.py`, `src/rag_core/pipeline.py`, `tests/test_pipeline.py`, docs updates), seven pre-existing unstaged deletions, and unrelated untracked files (including `nul`). Proposed correction below is **not implemented**.

**Current measured state (local, isolated temp index `C:\Users\sindh\AppData\Local\Temp\opencode\q12_ci_idx_xkh2rq2j`, `satcom_ngp_production_fixed_char_800_160`, 800/160, 61 chunks).**
- Q12 (“How are international satellite communication services defined?”, `EXP001-Q12`, expected/ground-truth: clause 4.2, page 13) retrieves 4.2 evidence and selects correct statements at 31006–31254 and 31255–31506, but also selects 10868–11204 from `satcom-ngp::page_5::chunk_2::off10868` (section/heading `3.2` / “3.2 DEFINITION OF INDIAN SATELLITES”): fragment “defined as  the satellites…” (defines Indian satellites; **unrelated** to requested subject).
- With DR-019 guard, 3.2 fragment: base 5.0123, **boost suppressed** (`frag=True`), still outranks 4.2 boosted candidates (4.3267, 4.1260). `pytest tests/test_citation_metadata.py -q` (temp index) → 1 failed/3 passed; `pytest tests -q -k "not real_index" --maxfail=5` → 1 failed/122 passed/1 deselected; `python -m evaluation.ci_eval ... --check-regression` → regression passed (dense metrics unchanged vs baseline). CI/public not verified.

**1) Section-heading metadata availability to scoring/selection (code references).**
- `src/rag_core/index.py:RetrievedChunk` carries `article`, `section`, `heading`, `text`, offsets/pages/distance/metadata (chunk-level metadata from index).
- `src/rag_core/pipeline.py:_extractive_answer` (~569–706): candidates are `(score, position, sentence_index, sentence, hit, local_start, local_end)`; `hit` is available per candidate, but **scoring uses only sentence text vs query tokens/IDF** (`_sentence_score` ~381–397: token overlap, clause-start/list bonuses, ALL-CAPS-heading penalty on sentence text via `_is_heading` ~315–319) plus definition-intent boost (~599–605) and duplicate filtering. **`hit.heading`/`hit.section`/`hit.article` are not used in scoring or `_select_candidates` (~796–838)**.
- `_governing_metadata_for_offset` (~513–559) resolves governing `(article, section, heading)` from in-chunk headings or chunk start-governing metadata; used for citation/statement attribution (~656–695), not for ranking. `_answer_unit_spans` (~225–305) can group by structural markers (`associated_heading_units`, `clause_body_units`) but does not score by heading semantics.
- For Q12 3.2 candidate: chunk metadata and governing heading at offset 10868 (chunk start, before first in-chunk clause) are “3.2 DEFINITION OF INDIAN SATELLITES” — **available** on `hit.heading` and derivable, but currently unused by scoring/selection.

**2) Relevance to the actual concept asked about (vs generic token overlap).**
- Current mechanism is lexical: `content_tokens`/`_idf_weights`/`_sentence_score` reward overlapping content tokens (`defined`, `international`, `satellite` in 3.2 fragment) irrespective of whether the sentence defines the *requested* concept. Definition-intent boost rewards definitional markers (`_DEFINITIONAL_MARKER_RE`, e.g. `defined as`) with any query-token overlap, not subject alignment.
- Q12 requested concept: *international satellite communication services* (question tokens `communication, defined, international, satellite, services`; expected answer/GT clause 4.2 “INTERNATIONAL COMMUNICATION SERVICES”). 3.2 fragment defines *Indian Satellites* — shares “satellite/defined/international” lexically but denotes a different definiendum; base score exceeds 4.2 because it matches high-weight tokens (including “defined”) while 4.2 statements match fewer query tokens in the same window.
- No existing generic “definiendum/subject” comparison between question intent and candidate/heading subject in pipeline scoring.

**3) Heading “DEFINITION OF INDIAN SATELLITES” as negative relevance signal (generic assessment).**
- **Potentially informative:** 3.2 chunk/gov heading explicitly encodes subject “Indian Satellites”; Q12’s requested subject is “international satellite communication services” (4.2 heading aligns: `['communication','international','services']` overlap in heading trace). Comparing question concept tokens against governing/section heading tokens could downweight subject-misaligned definition candidates.
- **Not reliably negative without care:** (a) lexical overlap is partial (“satellite” appears in both headings/subjects; morphology/tokenization differences like defined/definition); (b) must not penalize **Q07** (“How is an Indian Satellite defined in the policy?”, clause 3.2) where 3.2/“DEFINITION OF INDIAN SATELLITES” is the correct match; (c) headings vary in form (not all are `DEFINITION OF X`; some definitions appear in non-heading-named clauses like 4.2 body); (d) a hard negative based only on “heading topic differs” risks suppressing relevant definitional text in mixed/compound clauses. Conclusion: heading can provide a **supporting/general relevance signal** (especially positive alignment or comparative alignment across candidates) but not a standalone, assumption-free negative rule without semantic subject extraction or careful calibration.

**4) Existing generic mechanisms for “definition of different subject”.**
- **None found.** Sibling selection (`_clause_parent`/`_select_candidates`) groups by clause-number hierarchy; cross-reference selection targets applicability refs; definition-intent boost is marker+overlap based; `_governing_metadata_for_offset` resolves metadata for citation only. No subject/definiendum matching between question and heading/sentence.

**5) Effect on other definition-intent questions/tests (risk assessment).**
- Definition-intent in EXP-001: Q04 (classifications, 2.4.1), Q07 (Indian Satellite defined, 3.2 — correct target), Q10 (procedure), Q12 (international satellite communication services, 4.2), Q13 (categories). `definition_intent_boost` is opt-in (`AppConfig.definition_intent_boost` default `False`, `src/rag_app/config.py`) and currently used in `tests/test_citation_metadata.py::test_q12_definition_statements_resolve_to_clause_4_2` and `tests/test_pipeline.py` definition-intent tests (synthetic). Any scoring/selection change in boosted path can affect Q12/Q07-like definition queries and synthetic tests; must be validated against `tests/test_pipeline.py`, `tests/test_citation_metadata.py` (all original assertions), full deterministic suite `pytest tests -q -k "not real_index"`, and CI eval regression (`evaluation/ci_eval.py`, baseline `evaluation/baseline_metrics.json`) on fixed 15-question set. Particular care: Q07 must continue to prefer 3.2.

**6) Smallest generalizable correction (proposed, not implemented) and focused regression tests**
- **Recommendation:** incorporate **governing-heading/section relevance** into extractive candidate scoring for definition-intent (or general extractive) ranking in `GroundedPipeline._extractive_answer` (~569–706) by computing a lightweight alignment signal between question content tokens and the candidate’s governing heading (from `_governing_metadata_for_offset(hit, hit.start_offset + local_start)` or `hit.heading`) and/or clause context, then applying a **bounded adjustment** (e.g. additive bonus/penalty or multiplicative factor) so that candidates whose governing definition heading’s subject better matches the asked concept outrank subject-misaligned definitional fragments, without discarding candidates. This is generalizable (uses existing metadata, no Q12/clause hardcoding) and can demote 3.2 “DEFINITION OF INDIAN SATELLITES” for Q12 while supporting Q07.
- **Key design/risk constraints:** avoid false negatives for valid definitions (Q07, headings with different phrasing), keep adjustment small/bounded and evidence-based, prefer comparative/positive alignment over absolute negative filtering, and ensure interaction with existing definition boost and sibling selection is controlled.
- **Candidate code touchpoints:** `_extractive_answer` candidate scoring (~590–605, ~640–706), optionally helper near `_governing_metadata_for_offset`/`_sentence_score`; inputs already available (`hit`, `local_start`, `query_tokens`/`weights`).
- **Focused regression tests (proposed):** (1) definition-intent query for concept A selects definition sentence under heading aligned to A over a higher-overlap/boosted fragment under heading defining B (generic synthetic headings, not Q12-specific IDs); (2) query for concept matching heading B still selects B (guards Q07-like case); (3) candidates without definition-style heading or with comparable alignment fall back to existing behavior; (4) non-definition-intent path unchanged. Validate against existing Q12 assertions and full deterministic/CI-eval regressions.

**Risks of any such change:** over/under-weighting headings, sensitivity to heading phrasing/tokenization, interaction with boosted/non-boosted scores and sibling selection, and potential regressions on definition-intent questions beyond Q12 (including Q07) — requires measured validation before acceptance. No implementation performed.

---

### Fix (2026-11-21, implemented; narrowly scoped) — Governing-heading relevance for definition-intent extractive scoring (Q12 residual)

**Status/constraints.** Implemented only the authorized narrow correction in `src/rag_core/pipeline.py` (+ focused regression tests/docs). **No changes** to chunking (800/160), embedding model, retrieval strategy, RRF, provider/model, datasets/ground truth/evidence annotations, baseline metrics, or Q12 original assertions. `chroma_db_production/` not rebuilt/overwritten. No staging/commit/push/merge/deploy; no `git add .`. Preserved all pre-existing uncommitted changes (including `.github/workflows/ci.yml`, `tests/test_citation_metadata.py` prior edits), seven pre-existing unstaged deletions, and unrelated untracked files (including `nul`, `.agents/`, `evaluation/results/`, experiment artifacts, `vercel.frontend.json`, `vercel.json`). CI/public deployment **not executed** in this session.

**Problem (from investigation + DR-019).** DR-019 chunk-boundary definition-boost guard correctly suppressed the Q12 3.2 lowercase fragment boost, but the fragment still won on base lexical score (5.0123) over correct 4.2 boosted candidates (4.3267/4.1260) because extractive scoring used token overlap/markers only, not the requested definition subject. RetrievedChunk carried heading/section metadata and `_governing_metadata_for_offset` could resolve governing headings, but `_extractive_answer` did not use heading relevance. Q12 correct evidence is clause 4.2; Q07 correct evidence is clause 3.2.

**Implementation rationale.** Add a small, generic, bounded comparative heading-alignment signal for **definition-intent boosted candidates** so governing headings aligned with the requested subject are preferred over subject-misaligned definitional fragments, while preserving existing behavior when heading metadata is missing/ambiguous and avoiding hardcoded Q12/clause/subject rules. This follows the investigation recommendation with the minimum scope needed to resolve the demonstrated Q12 ranking issue.

**Scoring approach (implemented).** In `src/rag_core/pipeline.py`:
- Added `_DEFINITION_SUBJECT_STOPWORDS = {"define","defined","definition","meaning","means"}` and helpers: `_normalize_subject_token` (lightweight singular/plural normalization: `ies->y`, common `ses/xes/zes/ches/shes->es`, else trailing `s` stripped when safe), `_definition_subject_tokens(question)`, `_governing_heading_for_candidate(hit, local_start)` using existing `_governing_metadata_for_offset(hit, hit.start_offset + local_start)`, and `_heading_subject_alignment(subject_tokens, heading)`.
- In `_extractive_answer`, when `definition_intent_boost` is active and a candidate receives the definitional marker boost (`is_definition_candidate`), resolve governing heading and compute alignment as overlap count between normalized meaningful question subject tokens and normalized heading tokens (excluding definitional stopwords). If alignment is computable (`alignment is not None`), apply bounded comparative bonus: `score += min(3.0, 1.0 * float(alignment))`. Missing/empty heading, no meaningful subject tokens, or no heading tokens return `None` and **retain previous behavior** (no bonus). No candidate hard-filtering; base score and eligibility unchanged.

**Limitations / risks.** Heuristic lexical heading-subject alignment (not semantic NLU); sensitive to heading phrasing/tokenization and cases where definitions lack explicit aligned headings. Bonus applies only to boosted definition-intent candidates, reducing scope but still able to shift close rankings; bounded to cap over-bias. Comparative bonus only (no negative penalty for alignment 0, no hard filter) to reduce false negatives. Requires ongoing regression coverage for Q07/Q12-like definition queries and sibling selection.

**Files changed (uncommitted; not staged).**
- `src/rag_core/pipeline.py` — heading-alignment helpers + bounded bonus in definition-intent scoring path.
- `tests/test_pipeline.py` — `_hit` helper extended with optional `heading`/`article` args (default-preserving); added 4 focused regression tests: Q12-style aligned heading over higher-base unrelated fragment, Q07-style Indian Satellite preference, missing/ambiguous heading retains behavior, sibling-selection no-regression under definition_intent_boost.
- `docs/ENGINEERING_LOG.md` — investigation preserved; this implementation entry appended.
- Pre-existing uncommitted changes preserved: `.github/workflows/ci.yml`, `tests/test_citation_metadata.py` (Q12 original assertions unchanged).

**Tests / exact results (local).**
- Focused: `PYTHONPATH=$PWD python -m pytest tests/test_pipeline.py -q` → **29 passed**.
- Temp-index citation metadata: `RAG_PERSIST_DIR=C:\Users\sindh\AppData\Local\Temp\opencode\q12_ci_idx_xkh2rq2j PYTHONPATH=$PWD python -m pytest tests/test_citation_metadata.py -q` → **4 passed, 2 warnings** (includes Q12).
- Full deterministic suite: `RAG_PERSIST_DIR=...q12_ci_idx_xkh2rq2j PYTHONPATH=$PWD python -m pytest tests -q -k "not real_index" --maxfail=5` → **127 passed, 1 deselected, 29 warnings**.
- Deterministic CI eval regression: `RAG_PERSIST_DIR=...q12_ci_idx_xkh2rq2j PYTHONPATH=$PWD python -m evaluation.ci_eval --dataset evaluation/ci_golden_dataset.json --out evaluation/results/ci_local_q12_heading_fix --retrieval-modes dense --generation-mode extractive --k 5 --baseline evaluation/baseline_metrics.json --check-regression` → **regression passed** (`passed=true`, no failures), metrics unchanged vs baseline: recall@5 0.9236, precision@5 0.3, mrr@5 0.8611, evidence_coverage 0.9236, supported_answer_rate 0.5833, citation_support_rate 0.4828, abstention_accuracy 1.0, invalid_citations 0, failure_rate 0.0.

**Q07 / Q12 verification.** Q12: original `tests/test_citation_metadata.py::test_q12_definition_statements_resolve_to_clause_4_2` assertions **pass unchanged** against isolated temp index; selected statements resolve to 4.2 and no longer include 3.2 fragment. Q07: synthetic Q07-style regression test passes (prefers Indian Satellites heading definition); temp-index sanity check with `definition_intent_boost=True` selected the 3.2-governed Indian Satellite definition sentence as top candidate. No dataset/ground-truth/baseline modifications.

**CI/public deployment status.** Not executed/verified in this session (no commit/push/merge/deploy performed; GitHub Actions for pushed commits not verified; no Render/Vercel public deployment/smoke tests). Local deterministic verification only. Untracked local eval artifact `evaluation/results/ci_local_q12_heading_fix/` may exist from regression command; no baseline/dataset files changed.

---

### Deployment hardening (2026-11-21, post-merge Task 1, local implementation) — Protect public `POST /ingest`

**Status/constraints.** Post-merge hardening on latest merged `main` (`4187f4d`, origin/main fast-forwarded locally) before this task. **No commit/push/merge/deploy.** No `git reset/clean/stash/discard`. Preserved pre-existing unrelated unstaged deletions (`removed-experiment-artifacts/rag.py`, `removed-experiment-artifacts/results/eval_report.json`, `src/memory/working.py`, `src/metrics/citation.py`, `src/restart.py`, `tests/eval_pipeline.py`, `todo.txt`) and unrelated untracked files (`.agents/`, `evaluation/results/`, `removed-experiment-artifacts/chunking_experiment.py`, `removed-experiment-artifacts/eval/`, `removed-experiment-artifacts/results/chunking_experiment.log`, `removed-experiment-artifacts/results/chunking_experiment.run.log`, `removed-experiment-artifacts/results/chunking_experiment/`, `nul`, `vercel.frontend.json`, `vercel.json`). No Gemini provider/model change; no RAG architecture/chunking/retrieval default redesign.

**Security issue (before).** `POST /ingest` in `backend/main.py` was publicly reachable without authentication and triggered index rebuild from configured PDF (`extract_source_document` -> chunk -> audit -> `index.build(..., reset=True)` -> reset retriever). On a publicly deployed backend this allowed unauthorized callers to trigger expensive CPU/memory/disk work and index mutation/reset behavior.

**Root cause.** The endpoint was exposed for reproducible ingestion convenience but had no access control, no disable-by-default guard, and no deployment secret/token requirement.

**Decision.** Protect `POST /ingest` with a dedicated ingestion token (`RAG_INGEST_TOKEN`) and **fail closed** when token is missing/unconfigured. This preserves controlled HTTP ingestion for authorized operators while disabling public ingestion by default. Internal ingestion/recovery behavior remains internal: `src/rag_app/service.py::_ensure_index_ready` still rebuilds missing/empty index from the configured source PDF; `scripts/ingest.py` CLI remains available for local/controlled ingestion. No paid services/dependencies added.

**Implementation.**
- `src/rag_app/config.py`: added `ingest_token: Optional[str] = None` and `RAG_INGEST_TOKEN` env loading.
- `backend/main.py`: `POST /ingest` now requires `X-Ingest-Token` header. If `RAG_INGEST_TOKEN` is unset/empty after strip -> **403** `"Ingestion endpoint is disabled. Configure RAG_INGEST_TOKEN to enable authorized ingestion."` (fail closed). If header missing/wrong -> **403** `"Unauthorized ingestion request."` Token compared with `secrets.compare_digest`; token not logged, not returned in response, not exposed in frontend. Authorized requests execute existing ingestion flow unchanged.
- `tests/test_api.py`: added 4 tests covering disabled-by-default fail-closed, missing/wrong token rejection, authorized rebuild, and no impact on `/health`/`/query`.
- Docs/config examples: `.env.example`, `render.yaml`, `docker-compose.yml`, `docs/DEPLOYMENT.md`, `docs/PROJECT_BRIEF.md`, `docs/CHUNKING_EXPERIMENT.md` updated to document protected ingestion.

**Exact security behavior before/after.**
- **Before:** `POST /ingest` unauthenticated; any caller could trigger ingestion/index rebuild. No token required.
- **After (local implementation, verified):** `POST /ingest` fails closed by default. No `RAG_INGEST_TOKEN` configured -> 403 disabled, no rebuild. Wrong/missing `X-Ingest-Token` when configured -> 403 unauthorized, no rebuild. Correct `X-Ingest-Token` matching configured `RAG_INGEST_TOKEN` -> authorized ingestion executes (200 on successful build path), rebuild/reset behavior preserved. `/health`, `/healthz`, `/query`, `/ask`, `/metrics`, `/dashboard` behavior unaffected.

**Tests executed and exact results (local).**
- `PYTHONPATH=$PWD python -m pytest tests/test_api.py -q` → **13 passed, 4 warnings in 18.61s**.
- `PYTHONPATH=$PWD python -m pytest tests -q -k "not real_index" --maxfail=5` → **131 passed, 1 deselected, 29 warnings in 19.37s**.

**Limitations / remaining work.**
- `/ingest` remains an operationally sensitive endpoint even when token-protected; deployers should set a strong `RAG_INGEST_TOKEN`, restrict CORS/network exposure as appropriate, and avoid exposing the endpoint unnecessarily. No rate limiting/WAF/authn layer added (no paid dependency introduced).
- Public deployment verification **not executed** (no Render/Vercel deploy/smoke in this session). GitHub Actions/CI for this post-merge hardening commit not executed because **no commit/push performed**.
- Internal auto-recovery in `service._ensure_index_ready` still performs rebuild on missing/empty index (by design for Render Free ephemeral FS); this is internal, not an unauthenticated HTTP control surface, but still has cold-start/resource implications on free tier.

**CI/public deployment status.** Not executed/verified. Local deterministic verification only; no staging/commit/push/merge/deploy.

---

### Investigation (2026-11-21, post-merge Task 2, investigation-only) — `definition_intent_boost` / Q12 heading-alignment production effectiveness

**Status/constraints.** Investigation-only; **no implementation/config change**. No commit/push/merge/deploy; no staging. Preserved Task 1 local changes and all pre-existing unrelated unstaged deletions/untracked files. No Gemini provider/model change; no chunking/retrieval default/RAG architecture changes.

**1) Configuration trace / effective production default (exact code refs).**
- `src/rag_app/config.py:33` — `AppConfig.definition_intent_boost: bool = False` (dataclass default; comment: “Defaults: EXP-003 production behavior, no opt-in experimental candidates”).
- `src/rag_app/config.py:64-93` — `AppConfig.from_env()` does **not** read any env var for `definition_intent_boost` (no `RAG_DEFINITION_INTENT_BOOST`/similar). Effective runtime default from env/config is therefore **False**.
- `src/rag_app/service.py:181-193` — `RAGService.pipeline` constructs `GroundedPipeline(..., definition_intent_boost=self.config.definition_intent_boost, ...)`.
- `src/rag_app/service.py:454-460` — debug `selection_defaults` reports `definition_intent_boost`.
- `src/rag_core/pipeline.py:530,551,656+` — `GroundedPipeline.__init__` stores `definition_intent_boost` (default False). In `_extractive_answer`, `definitional_intent = self.definition_intent_boost and bool(_DEFINITION_INTENT_RE.search(question))`; only then may marker boost apply, and only boosted definition candidates (`is_definition_candidate`) receive bounded governing-heading alignment bonus (`min(3.0, 1.0*alignment)`) when alignment is computable. If boost off, heading-alignment path is never entered.
- Deployment/env exposure checked: `render.yaml`, `docker-compose.yml`, `.env.example`, `backend/main.py` contain no `RAG_DEFINITION_INTENT_BOOST`/`definition_intent_boost` control; no env var currently enables it.

**2) Code paths depending on it.** Only extractive selection scoring in `GroundedPipeline._extractive_answer` (`src/rag_core/pipeline.py`): definition-intent marker boost gate, mid-sentence chunk-boundary fragment guard interaction, and heading-subject alignment bonus (Task 2-fix code). No retrieval (dense/hybrid/RRF/rerank), chunking, embedding, citation validation schema, or Gemini provider/model path depends on it. Service only passes config through and reports debug.

**3) Q12 heading-alignment improvement dormancy.** **Yes, dormant under current effective production/default config.** The heading-alignment bonus is applied only when `definition_intent_boost` is active and a candidate is boosted (`is_definition_candidate=True`). With default `False`, the Q12 fix added in the prior PR does not affect production selection. Local read-only check on existing `chroma_db_production/` (800/160, `satcom_ngp_production_fixed_char_800_160`, no rebuild) with `AppConfig.from_env()` default pipeline settings (`sibling_score_ratio=0.5`) and `answer_from_hits` on same retrieved hits for EXP001-Q12:
- `definition_intent_boost=False` (effective default): selected 3 statements — `satcom-ngp::page_13::chunk_0::off30178@30208` sec `4.1`, `satcom-ngp::page_5::chunk_2::off10868@10868` sec `3.2`, `satcom-ngp::page_14::chunk_0::off32496@32783` sec `4.3.4`.
- `definition_intent_boost=True`: selected 3 statements — `satcom-ngp::page_13::chunk_1::off30818@31006` sec `4.2`, `satcom-ngp::page_13::chunk_1::off30818@31255` sec `4.2`, `satcom-ngp::page_13::chunk_0::off30178@30569` sec `4.2`.

**4) Q12 regression test / evaluation dataset review.**
- `tests/test_citation_metadata.py::test_q12_definition_statements_resolve_to_clause_4_2` explicitly constructs `GroundedPipeline(index, sibling_score_ratio=0.5, definition_intent_boost=True)` and asserts Q12 statements resolve to clause 4.2/page 13 and include offsets 31006/31255. It validates the **opt-in boosted path**, not default production config. It passes locally (Task 1: 13 passed in test_api.py; full suite 131 passed) but does not prove Q12 under default config.
- CI evaluation (`evaluation/ci_eval.py`) uses `AppConfig.from_env()` -> `RAGService`; with no env override and no config field for boost, CI eval runs with **default boost=False**. The committed baseline `evaluation/baseline_metrics.json` / CI comparison artifacts were measured under default config; they do not provide evidence for boost-on behavior. The 15-question CI set includes Q12 but does not currently fail the regression gate because supported/citation metrics are aggregate and Q12’s default behavior has not been isolated as a gate failure in current baseline.

**5) Risk of enabling by default.** **Unmeasured risk exists.** Benefit evidence is narrow (Q12/EXP-008 target + synthetic heading-alignment regressions). Risks: definition-intent queries may receive boosted/heading-aligned selection that changes answer composition for other definition-like questions (Q04,Q07,Q10,Q12,Q13 in EXP-001 category), possible over-weighting/misalignment where headings are ambiguous/missing/mismatched, and interaction with sibling selection/answerability guard. Current automated evidence does **not** show broad improvement; no boost-on CI eval/baseline comparison was executed in this investigation. Enabling by default now would change production behavior without measured CI-set evidence.

**6) Env/deployment exposure.** **Not exposed.** No `RAG_DEFINITION_INTENT_BOOST` in `AppConfig.from_env()`, `.env.example`, `render.yaml`, `docker-compose.yml`, or docs deployment env lists. Enabling today requires code/config construction change.

**Recommendation (investigation-only; no implementation).** **Expose an environment variable and keep the default unchanged (`False`)**. Rationale: preserves current measured/default behavior and avoids unmeasured production-wide change; provides controlled rollout/observability for deployment experiments; aligns with existing opt-in pattern and limited evidence. Before enabling by default in any environment, run a measured boost-on vs boost-off comparison on the fixed CI set (and review Q07/Q04/Q10/Q12/Q13 and unanswerables), capture regression impact on supported_answer_rate/citation_support_rate/abstention/invalid citations, and document a decision. “Enable by default” is **not recommended now** due to insufficient evidence. “Keep disabled with no env exposure” is safe but limits controlled validation.

**Evidence (factual).** Code refs above; local read-only Q12 default-vs-boost behavior observed on existing production index; Task 1 local deterministic suite **131 passed, 1 deselected, 29 warnings** (default config paths); CI eval uses `AppConfig.from_env()`; no boost-on CI baseline artifact produced here. No new metrics invented.

**Unresolved decision.** Whether/when to expose `RAG_DEFINITION_INTENT_BOOST` (or equivalent) and whether to enable it after measured CI comparison. No config change made.

**CI/public deployment status.** Not executed/verified. Investigation-only documentation change; no staging/commit/push/merge/deploy.

---

### Deployment hardening (2026-11-21, post-merge Task 3, local implementation) — Expose `definition_intent_boost` env var

**Status/constraints.** Implements Task 2 recommendation (expose env var, keep default unchanged). No commit/push/merge/deploy; no staging. Preserved Task 1/Task 2 local changes and all pre-existing unrelated unstaged deletions/untracked files. No Gemini provider/model change; no chunking defaults, hybrid retrieval defaults, or RAG architecture changes. No retrieval algorithm/heading-alignment bonus changes.

**Implementation.**
- `src/rag_app/config.py`: added `_env_bool_strict(name, default)` (accepts true/false,1/0,yes/no,on/off; raises `ValueError` on invalid values). `AppConfig.from_env()` now reads `RAG_DEFINITION_INTENT_BOOST` via strict parsing; **default remains `False`** when unset. Direct `AppConfig(...)` construction behavior/default unchanged.
- Propagation verified by code/tests: `AppConfig.definition_intent_boost` -> `RAGService.pipeline` (`GroundedPipeline(..., definition_intent_boost=self.config.definition_intent_boost)`) -> `GroundedPipeline._extractive_answer` gating/bonus path; debug `selection_defaults` already reports it.
- Config/docs exposure: `.env.example`, `render.yaml` (value `"false"`), `docker-compose.yml`, `docs/DEPLOYMENT.md` updated with `RAG_DEFINITION_INTENT_BOOST` (default false, opt-in only; not promoted baseline).
- Tests: `tests/test_config.py` added (15 tests) covering unset default False, explicit true/false, alternative boolean representations, invalid value raises clear `ValueError`, and propagation to service/pipeline.

**Tests executed and exact results (local).**
- `PYTHONPATH=$PWD python -m pytest tests/test_config.py -q` → **15 passed in 1.31s**.
- `PYTHONPATH=$PWD python -m pytest tests -q -k "not real_index" --maxfail=5` → **146 passed, 1 deselected, 29 warnings in 19.06s**.

**Fixed 15-question deterministic evaluation comparison (local, not promoted as baseline).** Same dataset/baseline/generation/retrieval settings (`evaluation/ci_golden_dataset.json`, dense, extractive, k=5, baseline `evaluation/baseline_metrics.json`, `--check-regression`); only `RAG_DEFINITION_INTENT_BOOST` differed. Baseline file unchanged.

| Run | recall@5 | precision@5 | mrr@5 | evidence_coverage | supported_answer_rate | citation_support_rate | abstention_accuracy | invalid_citations | failure_rate | regression |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Default (`RAG_DEFINITION_INTENT_BOOST=false`, `evaluation/results/ci_defboost_off`) | 0.9236 | 0.3 | 0.8611 | 0.9236 | 0.5833 | 0.4828 | 1.0 | 0 | 0.0 | passed |
| Opt-in (`RAG_DEFINITION_INTENT_BOOST=true`, `evaluation/results/ci_defboost_on`) | 0.9236 | 0.3 | 0.8611 | 0.9236 | 0.75 | 0.5185 | 1.0 | 0 | 0.0 | passed |

Per-question changes (answer support/citation/statement rates): **EXP001-Q07** improved (supported False->True, cite 0.5->0.5, stmt 0.0->0.3333); **EXP001-Q12** improved (supported False->True, cite 0.3333->1.0, stmt 0.0->1.0); **EXP001-Q13** mixed/regressed on support rates while remaining supported (supported True->True, cite 0.6667->0.5, stmt 0.6667->0.3333). EXP001-Q04/Q10 unchanged on these fields. Interpretation: **small-sample, mixed question-level effect**; do not claim broad improvement or promote boost-on as default/baseline.

**CI/public deployment status.** Not executed/verified. Local deterministic verification only; no staging/commit/push/merge/deploy.

---

### Deployment hardening (2026-11-21, post-merge Task 4, local implementation) — CORS configuration for Render+Vercel

**Status/constraints.** Investigation + local implementation. No commit/push/merge/deploy; no staging. Preserved Task 1-3 local changes and all pre-existing unrelated unstaged deletions/untracked files. No Gemini provider/model change; no chunking/retrieval defaults/RAG architecture changes; `RAG_DEFINITION_INTENT_BOOST` default remains `false`.

**Investigation (before).**
- `backend/main.py` configured CORS at import with `allow_origins=os.getenv("RAG_CORS_ORIGINS", "*").split(",")`, `allow_credentials=True`, `allow_methods=["*"]`, `allow_headers=["*"]`.
- `render.yaml` set `RAG_CORS_ORIGINS="*"`; `docker-compose.yml` set `RAG_CORS_ORIGINS=*`; `.env.example` had no `RAG_CORS_ORIGINS`.
- Wildcard origins were permitted by default (unsafe as intended production config). With wildcard + credentials, browser behavior is restrictive; frontend does not use credentialed requests.
- Actual Vercel frontend origin is unknown in-repo and must not be guessed.

**Implementation.**
- `backend/main.py`: added `DEFAULT_DEV_CORS_ORIGINS` (localhost/127.0.0.1 ports 5173/3000) and `_parse_cors_origins(raw)`. Behavior: unset/empty -> local-dev defaults (no wildcard production default); explicit `*` allowed only when intentionally configured; comma-separated explicit origins supported with whitespace trim/empty ignore. `allow_credentials` now `False` for normal/wildcard-safe behavior (frontend does not use browser credentials).
- `render.yaml`: replaced wildcard with placeholder `https://REPLACE_WITH_VERCEL_FRONTEND_ORIGIN` (must be replaced before public frontend integration).
- `docker-compose.yml`: local-dev explicit origins.
- `.env.example`: added local-dev `RAG_CORS_ORIGINS` default.
- `docs/DEPLOYMENT.md`: documented CORS behavior, requirement to set real Vercel origin before public deployment, and non-wildcard production expectation.
- `tests/test_cors.py`: added 9 tests covering parser defaults/empty/multiple/explicit wildcard, allowed origin header, unlisted origin omission, multiple origins, explicit wildcard behavior, and local-dev Vite origin usability.

**Exact configuration behavior before/after.**
- **Before:** default/unset effectively wildcard (`*`), credentials allowed; render/docker defaulted to wildcard.
- **After (local implementation, verified):** unset/empty -> local-dev defaults only; explicit comma-separated origins accepted; explicit `*` only when configured; unlisted browser origins do not receive `Access-Control-Allow-Origin`. `allow_credentials=False`. Render placeholder is fail-closed for real Vercel origin until replaced with actual deployed frontend origin.

**Tests executed and exact results (local).**
- `PYTHONPATH=$PWD python -m pytest tests/test_cors.py -q` → **9 passed in 1.57s**.
- `PYTHONPATH=$PWD python -m pytest tests/test_api.py -q` → **13 passed, 4 warnings in 27.58s**.
- `PYTHONPATH=$PWD python -m pytest tests -q -k "not real_index" --maxfail=5` → **155 passed, 1 deselected, 29 warnings in 33.98s**.

**Unresolved deployment requirements.** Actual Vercel frontend origin must be set in `RAG_CORS_ORIGINS` (Render/Vercel env) before public frontend integration; placeholder `https://REPLACE_WITH_VERCEL_FRONTEND_ORIGIN` must not be used as-is for production. Public CORS/browser integration verification not executed.

**CI/public deployment status.** Not executed/verified. Local deterministic verification only; no staging/commit/push/merge/deploy.

---

### Render Free deployment failure investigation (2026-11-21, read-only investigation)

**Status/constraints.** Read-only code investigation; no application code changed; no deploy/staging/commit of app code performed during investigation. Preserved existing work and unrelated working-tree changes. Did not change embedding model, chunking (800/160), hybrid dense+BM25 retrieval, or Gemini provider/model. No paid infrastructure introduced. This entry documents investigation only; **no proposed fix is implemented**.

**User-reported Render evidence (external, not independently verified in this environment):**
- Commit: `e478a1b`. Render service `satcom-ngp-rag-backend`, plan Free, 512 MB memory, 0.15 CPU.
- Deployment became Live, then instance `qmmb9` failed ~7 minutes later.
- Logs showed SentenceTransformer model weights downloaded and loaded immediately before instance restart.
- `/health` returned `index_available: false`.
- App has automatic Chroma index recovery from `data/satcom-ngp.pdf`.

**Files inspected (code/config):**
- `backend/main.py` — `/health` (`get_service().health()`), `/healthz`, startup/lifespan behavior (no startup preload), `get_service()` lru_cache singleton.
- `src/rag_app/service.py` — `model` property lazy-loads `SentenceTransformer(self.config.embedding_model)`; `index` property creates `DenseIndex(..., model=self.model)` then calls `_ensure_index_ready()`; `_ensure_index_ready()` checks `index.count()`, if missing/empty extracts `data/satcom-ngp.pdf`, chunks (800/160 fixed char default), audits, calls `index.build(chunks, reset=True)`, clears retriever cache; `health()` catches exceptions and returns `index_available` based on `self.index.count() > 0`.
- `src/rag_core/index.py` — `DenseIndex` uses `chromadb.PersistentClient`, loads ST model if not injected, `build()` encodes all chunk texts at once via `model.encode`, adds to Chroma; `search()` encodes query.
- `src/rag_app/config.py` — defaults: `sentence-transformers/all-MiniLM-L6-v2`, 800/160 fixed_char, hybrid, collection `satcom_ngp_production_fixed_char_800_160`, persist dir `chroma_db_production`.
- `requirements.runtime.txt`, `Dockerfile`, `render.yaml` — CPU torch installed; runtime includes torch-indirect deps through sentence-transformers, chromadb, onnxruntime/chromadb deps as resolved; Render free 512MB not configurable in yaml.
- `data/satcom-ngp.pdf` present (68,396 bytes, tracked).

**Execution-path findings:**
1. **`/health` can trigger heavy initialization.** `/health` -> `get_service().health()` -> `self.index.count()`; first access constructs service/index and loads model + runs index readiness check. `/healthz` is lightweight and does not touch model/index.
2. **No duplicate model load in inspected path.** `RAGService.model` loads once; passed into `DenseIndex(model=self.model)`. `_ensure_index_ready` uses same index/model for build.
3. **Index recovery can be expensive on cold/ephemeral start.** If Chroma persist dir missing/empty (expected on Render Free ephemeral FS), recovery extracts PDF, chunks 61 chunks (small), encodes all chunks with ST model, writes to Chroma. PDF/chunking overhead small; embedding/build adds CPU/memory/transient allocations.
4. **Retriever/BM25 not triggered by `/health`.** `health()` does not access `retriever`; query does. BM25 corpus build in `HybridRetriever` would add memory on first query.

**Memory requirement estimate (rough, not measured on Render):**
- **Python + FastAPI/uvicorn baseline:** ~40-80 MB (estimate).
- **PyTorch CPU + sentence-transformers/transformers runtime:** often ~150-250+ MB resident depending on wheels/imports/allocator fragmentation (estimate).
- **all-MiniLM-L6-v2 model weights in memory:** ~90 MB (fp32 parameters) plus tokenizer/runtime overhead; can be higher with framework overhead (estimate).
- **ChromaDB/client/SQLite/collections:** ~20-60+ MB depending on state (estimate).
- **PDF extraction + 61 chunks + embedding batch vectors:** small (PDF 68KB, 61 chunks, 384-dim embeddings ~0.1 MB raw) but transient overhead during encode/build (estimate low).
- **HybridRetriever BM25 corpus (on query):** additional memory proportional to corpus/tokenization; likely modest for 61 chunks but non-zero.
- **Rough total:** likely ~300-500+ MB at rest after model load, with spikes during first load/recovery/query possibly exceeding 512 MB. **This is an estimate, not a Render measurement.**

**Most likely failure explanation:**
- **Likely OOM / memory pressure on Render Free (512 MB) during or after SentenceTransformer model load/index recovery path.** Confidence: **medium**. Evidence aligns (model load logged immediately before restart; `/health` reported `index_available:false`; health path triggers lazy model+index+recovery; 512MB is tight for torch+ST+Chroma). **Not confirmed as OOM** without Render memory metrics/events/logs explicitly stating OOM/killed; restart could also be caused by health-check timeout, process crash, or platform instability, but model-load timing makes OOM/memory exhaustion the leading hypothesis.

**Render Free viability assessment:**
- **Not viable as currently configured with high confidence of reliability.** It may occasionally boot, but 512 MB/0.15 CPU is unlikely to reliably support torch CPU + sentence-transformers + Chroma + index recovery/query under cold starts. Canonical RAG quality settings (embedding/chunking/hybrid/Gemini) are constrained; infra fit is the main risk.

**Potential optimizations (preserve RAG architecture/quality; NOT implemented):**
1. **Low risk / operational:** Prebuild and package/bake `chroma_db_production` into deploy artifact/image (if size acceptable) to avoid recovery embedding/build on cold start; keep same collection/chunking/embedding. Still requires model load for query unless avoided.
2. **Low-medium risk:** Lazy-load model only on `/query` (or background warmup) and keep `/health` lightweight; `/healthz` already lightweight. Improves liveness but does not reduce query memory need; can delay first-query failure.
3. **Medium risk:** Reduce runtime memory footprint without changing retrieval quality: use smaller CPU torch build options already used; tune sentence-transformers/torch threads (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `TOKENIZERS_PARALLELISM`), disable unnecessary model features, ensure `low_cpu_mem_usage`/fp32 defaults understood, clear caches after recovery; measure locally with RSS profiling. Uncertain gain.
4. **Medium-high risk:** Move embedding inference to external free API/provider while keeping same vector space? **Not allowed** by "Do not change the embedding model" / quality constraints; rejected.
5. **Medium-high risk:** Split services (separate lightweight API + embedding/index worker) — likely conflicts with "no paid infrastructure"/Render Free limits and complexity; not viable on free.
6. **High-confidence alternative (if Render Free remains insufficient):** identify another verified free-tier host with >=1GB RAM or different container limits; requires user account/access and evidence before migration. No code/architecture change required beyond deploy config.

**Next steps (investigation only):** capture Render dashboard memory graph/events, deploy logs around restart, and OOM/killed signal; if available, reproduce locally with 512MB container limit (`docker run --memory=512m`) and profile RSS during `/health` -> model load -> recovery -> `/query`. Do not implement fixes without explicit approval.

---

### Render Free resource-limit reproduction attempt (2026-11-21, investigation only)

**Status/constraints.** Investigation/documentation only. No application code modified. No embedding model/Gemini provider-model/chunking/retrieval/RAG architecture changes. No paid infrastructure introduced. No commit/push/deploy/reset/clean/stash/discard. Preserved all existing unrelated modifications, deletions, and untracked files. Did not use `git add .`. No measurements manufactured.

**Environment checked (local):**
- OS: Windows (win32), workspace `C:\Users\sindh\RAG`.
- Docker CLI: `Docker version 28.0.4, build b8034c0` (`docker --version`).
- Docker daemon: **unavailable/not running**. `docker info` failed with: `error during connect: Get "http://%2F%2F.%2Fpipe%2FdockerDesktopLinuxEngine/v1.48/info": open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified.`

**Planned test configuration (not executed):**
- Use existing `Dockerfile` / `requirements.runtime.txt` / runtime config.
- Container limits: `--memory=512m --cpus=0.15` (Render Free approximation).
- Stages to measure: process startup, `/healthz`, `/health` + index recovery, one representative query.
- Metrics to capture: process/container exit status, logs, peak RSS/container memory (`docker stats`/cgroup), elapsed time.

**Exact commands used:**
- `docker --version 2>&1 | tail -n 5; echo "==="; docker info 2>&1 | tail -n 20`
- `git status --short` (before/after documentation)

**Measured results:** **None.** Resource-limited container test could not be run because Docker daemon/engine was unavailable.

**Failures/evidence:** Docker daemon connection failure (above). No container created/run; no process exit status, peak memory, or stage timings obtained.

**Limitations of local reproduction:**
- Cannot reproduce Render Free cgroup limits without running Docker engine.
- Local Windows process RSS would not enforce 512MB/0.15 CPU and would not be comparable to Render container behavior.
- Even with Docker, local reproduction would approximate but not identical to Render Free (kernel/cgroup behavior, filesystem, networking, CPU scheduling, platform health checks).

**Effect on Render OOM hypothesis:** **Neutral (no strengthening/weakening evidence).** The hypothesis remains **medium confidence, unconfirmed** based on prior code-path analysis and user-reported Render symptoms. This attempt provides no new runtime measurement.

**Render Free viability:** Unchanged from prior investigation: **not viable with high confidence of reliability in current configuration**; still unconfirmed by local 512MB-limited runtime reproduction.

---

### Render Free confirmed OOM and remediation planning (2026-11-21, investigation only; no implementation)

**Status/constraints.** Investigation/planning only. No application code modified. No commit/push/deploy/stage. Preserved all pre-existing unrelated deletions and untracked files. Preserved `google_generativeai/gemini-flash-lite-latest`, `sentence-transformers/all-MiniLM-L6-v2`, fixed-character chunking 800/160, hybrid dense+BM25 RRF retrieval, existing Chroma collection/grounded-answer behavior. No paid services/infrastructure; no model/provider substitution; no retrieval/chunking experiments.

**Confirmed Render event (user-reported, Render platform message):**
- **Exact error:** `Instance failed: qmmb9 — Ran out of memory (used over 512MB) while running your code.`
- **Service:** `satcom-ngp-rag-backend` (Render Free).
- **Resource limit:** 512 MB RAM / 0.15 CPU (Render Free, as reported).
- **Instance:** `qmmb9`.
- **Timestamp:** not provided in user message; event reported by user on 2026-11-21.
- **Root cause status:** **Confirmed OOM** (platform-reported memory limit exceeded). Prior local reproduction was not possible because Docker daemon was unavailable; this platform event confirms memory exhaustion.
- **Remediation status:** **Not implemented.** No code/config changes applied in this task.

**Memory-heavy code paths (file/line refs, code-observed):**
- `backend/main.py:79-82,101-108` — `get_service()` lru_cache singleton; `/health` calls `get_service().health()` and can trigger lazy heavy init. `/healthz` (`:95-98`) is lightweight.
- `src/rag_app/service.py:99-104` — `model` property lazy-loads `SentenceTransformer(self.config.embedding_model)`.
- `src/rag_app/service.py:107-118` — `index` property creates `DenseIndex(..., model=self.model)` and calls `_ensure_index_ready()` on first access.
- `src/rag_app/service.py:120-165` — `_ensure_index_ready()` on missing/empty index: PDF extraction, chunking, coverage audit, `index.build(chunks, reset=True)`, retriever reset. Runs on ephemeral FS when prebuilt index absent.
- `src/rag_app/service.py:167-179` — `retriever` property lazily creates `HybridRetriever`, which builds/uses `CorpusIndex` (BM25).
- `src/rag_core/index.py:89-103,111-131` — `DenseIndex` holds Chroma PersistentClient; model load if not injected; `build()` encodes all chunk texts at once (`_encode`) then `collection.add(...)`.
- `src/rag_core/retrieval.py:44-76,235` — `CorpusIndex.load()` pulls all docs/metadatas from Chroma into memory, tokenizes all docs, builds `BM25Okapi` over tokenized corpus; `HybridRetriever` creates `CorpusIndex(dense_index)`.
- `requirements.runtime.txt`/`Dockerfile` — CPU torch + sentence-transformers + chromadb + onnxruntime (chromadb dep) contribute base RSS.

**Evidence vs estimates:**
- **Evidence:** Render platform error confirms >512MB usage and OOM kill. Code paths above show lazy model load on first service/index/health/query use; index recovery embeds/builds if index missing; BM25 corpus loads full corpus into memory on first hybrid retrieval. `/healthz` is already lightweight and configured as Render healthCheckPath (`render.yaml:8`).
- **Estimates (not measured):** base Python/runtime 40-80MB; torch CPU+ST/transformers 150-250+MB; all-MiniLM-L6-v2 weights ~90MB+overhead; Chroma 20-60+MB; BM25 tokenized corpus small for 61 chunks but non-zero; transient encode/build spikes possible. Total can exceed 512MB.

**Proposed changes (NOT implemented; ranked by expected memory benefit / risk / quality impact):**

| Rank | Proposal | Expected memory benefit | Impl. risk | Quality/behavior impact | Notes |
|---|---|---:|---:|---:|---|
| 1 | **Bake prebuilt `chroma_db_production` into deploy image/artifact** (include in Docker build context; remove/adjust ignore as needed for deploy image only) so recovery does not encode/build on cold start. | Medium (removes recovery encode/build spike; avoids duplicate model+build work) | Low-medium | **None to retrieval quality** if same collection/chunks/embedding used | Does not remove model load. Requires deploy artifact to include index; verify image size/build time. Render Free ephemeral FS still fine if index inside image. |
| 2 | **Make `/health` lightweight/readiness-safe by default** (do not trigger model/index load; report `index_available` only if already initialized or via cheap check) while keeping `/healthz` as liveness. | Medium on health probes (prevents accidental heavy load) | Low-medium | **No RAG answer quality change**; health semantics change (readiness less deep) | Render already uses `/healthz`. Need clear API/docs; avoid breaking existing clients expecting deep health. |
| 3 | **Runtime memory tuning (env/process)**: set `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `TOKENIZERS_PARALLELISM=false`, `PYTORCH_ENABLE_MPS_FALLBACK` irrelevant; consider `ORT_*`/Chroma settings; keep CPU torch. | Low-medium (reduces thread/allocator overhead; uncertain) | Low | **None expected** | Easy to test; gains uncertain on 0.15 CPU/512MB. |
| 4 | **Lazy/conditional heavy initialization boundaries**: ensure no import-time model load; keep model load only on first query or explicit warmup endpoint (protected) — not on health. | Medium for idle/liveness | Medium | **No quality change**; first query latency increases | Must not create unauthenticated heavy trigger; `/ingest` already protected. |
| 5 | **Reduce BM25/corpus duplication**: avoid loading full corpus when dense-only; for hybrid, consider lazy BM25 build on first hybrid query and careful data structures. | Low-medium (61 chunks small) | Medium | **Must preserve hybrid RRF results** | Current corpus small; benefit limited. Any change needs equivalence test. |
| 6 | **Model-load memory controls** (same model): evaluate `torch.set_num_threads(1)`, low-memory load patterns supported by sentence-transformers/torch for same model/weights; clear unused refs after init. | Low-medium, uncertain | Medium | **Must preserve embeddings/quality** | Must not change model or quantization unless proven identical outputs; risky to assume. |

**Quality impact:** All ranked proposals above are intended to preserve answer quality/retrieval behavior. Any implementation touching retrieval/index/model loading would require equivalence/regression verification against existing deterministic eval and citation behavior.

**Render Free salvageability:** **Uncertain-to-low.** OOM is confirmed. Optimizations 1-4 may reduce spikes and avoid unnecessary health-triggered loads, but base torch+ST+Chroma footprint may still be near/over 512MB on first real query. Prebaked index helps recovery spike but not model-resident memory. A 512MB-limited local Docker reproduction is still needed to confirm post-change viability; Docker daemon unavailable here. If optimizations fail under 512MB, Render Free is not viable for this stack without violating constraints.

**Decision required before implementation:** Presenting proposed changes for approval. **No implementation performed.**

---

### Safe memory optimizations for Render Free (2026-11-21, Task 7, investigation/planning only)

**Status/constraints.** Investigation/planning only. No application code modified. No commit/push/deploy. Preserved `all-MiniLM-L6-v2`, CPU-only PyTorch, Gemini `gemini-flash-lite-latest`, 800/160 fixed-character chunking, hybrid dense+BM25 RRF retrieval, ChromaDB/existing collection, answer-generation/grounding behavior. No paid infrastructure. All pre-existing unrelated git changes preserved.

**1) PyTorch / Sentence Transformers**
- **Imports/init (evidence):** `service.py:99-104` and `index.py:97-102` lazy-import/load `SentenceTransformer` only when model property accessed. `service.py` passes model into `DenseIndex(model=self.model)` (`:107-116`), avoiding duplicate model load in inspected app path. No module-import-time ST model load found in app runtime files.
- **Thread limits:** No `OMP_NUM_THREADS`/`MKL_NUM_THREADS`/`TOKENIZERS_PARALLELISM`/`torch.set_num_threads` set in `Dockerfile`, `render.yaml`, or app code. Setting thread caps can reduce thread-stack/allocator overhead (qualitative); unlikely to fix 512MB alone but low risk.
- **Dependencies:** `requirements.runtime.txt` contains required runtime deps (fastapi, uvicorn, chromadb, sentence-transformers, rank_bm25, google-generativeai, cohere, langfuse, numpy, httpx, pypdf, dotenv). No obvious unused required dep removal; removing any could break behavior. CPU torch already installed separately.

**2) ChromaDB / index initialization**
- **Prebuilt index:** `_ensure_index_ready()` (`service.py:120-165`) skips rebuild when `index.count() > 0`. If `chroma_db_production` containing the same collection is baked into image/build artifact, recovery encode/build is avoided. For 61 chunks, rebuild is small but still an avoidable spike (encode + Chroma add). Prebuild does **not** eliminate model-load memory.
- **Duplication:** `index.build()` encodes all chunk texts into vectors in memory then adds to Chroma (`index.py:111-131`). Single encode batch for 61 chunks is small; no obvious duplicate copy beyond normal Python/NumPy/torch objects. Index recovery creates doc/chunks/audit objects transiently.

**3) Health endpoints**
- **Current behavior (evidence):** `backend/main.py:101-108` `/health` calls `get_service().health()`; `service.py:205-210` accesses `self.index.count()`, triggering lazy model load, DenseIndex init, and `_ensure_index_ready()`. `backend/main.py:95-98` `/healthz` is lightweight. `render.yaml:8` uses `/healthz` for platform health check.
- **Proposal:** Make `/health` non-loading by default (return process/service liveness/readiness metadata without forcing model/index init; report deep index status only if already initialized). Preserve `/healthz`. No retrieval/generation behavior change. Requires careful API contract/docs update.

**4) BM25**
- **Current behavior (evidence):** `HybridRetriever.__init__` creates `CorpusIndex(dense_index)` (`retrieval.py:235`) but `CorpusIndex.load()` is lazy (`:58-76`), invoked by `bm25`/`bm25_search`/`get`/`to_retrieved_chunk`. Dense mode in `HybridRetriever.retrieve()` returns before BM25 (`:259-267` earlier section). Service dense fast path (`service.py:224-230`) bypasses HybridRetriever entirely.
- **Deferral:** BM25 corpus build is already deferred until hybrid retrieval needs it. Further deferral limited; creating `CorpusIndex` object is cheap. No change needed for correctness; any refactor must preserve identical hybrid RRF candidates/scores.

**Proposals (not implemented) — see final report for ranking/test plan.** No measurements invented.

---

### Lightweight health checks and conservative CPU/thread limits (2026-11-21, Task 8, local implementation)

**Status/constraints.** Local implementation only. No commit/push/deploy/stage. Preserved all pre-existing unrelated deletions and untracked files. No changes to embedding model/provider, Gemini config, chunking (800/160), retrieval/BM25/RRF, Chroma collection, or answer-generation/grounding behavior. No prebuilt index packaged. No dependencies removed.

**What changed and why:**
- **`/health` lightweight semantics (`backend/main.py`, `src/rag_app/service.py`, `src/rag_app/schemas.py`):** `GET /health` no longer triggers SentenceTransformer loading, Chroma initialization, index recovery, or embedding generation. It reports deep index availability only when the index is already initialized in-process. This reduces avoidable memory pressure from health probes on Render Free (confirmed OOM 512MB). `GET /healthz` remains lightweight liveness endpoint (Render `healthCheckPath`).
- **Health response fields:** preserved existing fields where practical; added `index_status` (`unknown|available|unavailable`, default `unknown`) and `index_initialized` (bool). `index_available` remains bool but is **never falsely true**; it is true only when index already initialized and `count()>0`, false otherwise (including unknown/not initialized).
- **Conservative thread limits:** set `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `TOKENIZERS_PARALLELISM=false` in `render.yaml`, `Dockerfile`, and `docker-compose.yml` for consistent low-overhead runtime. Did not add `torch.set_num_threads()` (not necessary based on inspection; safe init point not required).
- **Docs/tests:** updated `docs/DEPLOYMENT.md`; added API tests proving `/health` does not initialize model/index and `/healthz` remains lightweight.

**Tests and exact results (local):**
- `PYTHONPATH="$PWD" python -m pytest tests/test_api.py tests/test_config.py tests/test_cors.py -q` → **41 passed, 4 warnings in 28.31s**.
- `PYTHONPATH="$PWD" python -m pytest tests -q -k "not real_index" --maxfail=5` → **159 passed, 1 deselected, 29 warnings in 27.95s**.
- CI eval comparison (no baseline overwrite): `PYTHONPATH="$PWD" LANGFUSE_ENABLED=false RAG_ALLOW_GEMINI=false python -m evaluation.ci_eval --out evaluation/results/ci_task8_healthcheck --baseline evaluation/baseline_metrics.json --check-regression` → **regression passed** (`passed: true`, `failures: []`, `baseline: evaluation\\baseline_metrics.json`). Dense metrics matched baseline: recall@5 0.9236, precision@5 0.3, mrr@5 0.8611, evidence_coverage 0.9236, supported_answer_rate 0.5833, citation_support_rate 0.4828, abstention_accuracy 1.0, invalid_citations 0, failure_rate 0.0. (Hybrid also reported in comparison output; no regression failures.)

**RAG quality evidence:** Deterministic CI eval regression passed against baseline; no RAG behavior/quality change claimed beyond this measured comparison. Not a full RAGAS/production-traffic validation.

**Limitations / remaining OOM risk:**
- `/health` no longer triggers heavy init, and Render already uses `/healthz`; this removes one avoidable trigger but **does not reduce model-resident memory** required for first query/index recovery.
- Thread env vars are conservative/low-risk but **expected benefit uncertain**; not measured in 512MB container here (Docker daemon unavailable).
- Render Free OOM remains **confirmed risk**; 512MB may still be insufficient for torch+sentence-transformers+Chroma on query/recovery. Prebuilt index packaging was not done in this task.
- Local verification only; no CI/public deploy verification performed.

---

### Prebuilt Chroma index packaging feasibility (2026-11-21, Task 9, investigation only; no implementation)

**Status/constraints.** Investigation only. No implementation. No modification/deletion/regeneration of existing index. No stage/commit/push/deploy. Preserved all pre-existing unrelated deletions and untracked files. Preserved RAG behavior constraints (all-MiniLM-L6-v2, CPU PyTorch, Gemini, 800/160 fixed-character chunking, hybrid dense+BM25 RRF, ChromaDB/existing collection, grounded-answer behavior).

**Index location/size/structure (evidence):**
- **Path:** `chroma_db_production/` (repo root; default `RAG_PERSIST_DIR` resolves here).
- **Disk size:** ~6.3M (`du -sh`); 9 files total.
- **Structure:** `chroma.sqlite3` (~3.1M), two HNSW segment dirs: `dc220631-1c37-42f8-b4fc-45cce513acf7/` (data_level0.bin ~1.6M, header.bin, length.bin, link_lists.bin) and `f43144c4-4050-4e58-9fa7-63dcd953216e/` (similar).
- **Inspection method:** copied `chroma_db_production` to temp dir `C:\Users\sindh\AppData\Local\Temp\opencode\chroma_db_production_inspect` and queried SQLite read-only (`mode=ro`); no writes to original index.

**Collection metadata (evidence, sqlite read-only):**
- Collections in DB: `satcom_ngp_production_fixed_char` (`cd6d9962-...`, dim 384) and `satcom_ngp_production_fixed_char_800_160` (`bc04931c-f898-424a-b8f0-023d1ab20173`, dim 384, HNSW l2, ef_construction 100, M 16).
- Target collection `satcom_ngp_production_fixed_char_800_160`: metadata segment `fabb5d6a-5159-4123-9729-0205db55dd89` has **61 embeddings**; vector segment `dc220631-1c37-42f8-b4fc-45cce513acf7` is HNSW-persisted.
- Embedding metadata keys: `article, chroma:document, chunk_id, chunk_sequence, chunking_strategy, end_offset, end_page, end_page_char, heading, part, section, section_hierarchy, source, start_offset, start_page, start_page_char`.
- Sample metadata: `chunking_strategy='fixed_character'`, `source='satcom-ngp.pdf'`, `start_offset=2,end_offset=802` (800 chars), next `642-1442`, `1282-2082` => **800 char size, 160 overlap/640 step**. Confirms 800/160 fixed-character.

**Provenance verification (evidence/limits):**
- **800/160 fixed-character:** confirmed by metadata (`chunking_strategy=fixed_character`, offset/size pattern) and chunk count 61; consistent with `docs/CHUNKING_EXPERIMENT.md` and config defaults.
- **all-MiniLM-L6-v2:** **partially evidenced**. Collection dimension is 384, consistent with MiniLM-L6-v2; app config/default and docs specify `sentence-transformers/all-MiniLM-L6-v2`. Collection config/metadata does not explicitly store model name. **Not proven by directory name alone**; provenance is supported by code/config/docs + dimension, but not a cryptographic/model-id stamp in Chroma metadata.
- **Collection match:** exact name `satcom_ngp_production_fixed_char_800_160` present.

**Packaging/deploy feasibility (evidence):**
- `.gitignore:10` ignores `chroma_db_production/`; git-based Render deploy would not include it unless committed or fetched at build.
- `.dockerignore:17` ignores `chroma_db` but **not** `chroma_db_production`; Docker build context would include it if present.
- `Dockerfile` copies app (`COPY . .`) and creates `/app/chroma_db_production`; image would include index if in context. `render.yaml` is git-based Python runtime (not Docker image deploy) with `pip install ... && pip install -r requirements.runtime.txt`; no index fetch step.
- **Render git deploy implication:** packaging requires either (a) committing index to repo (size/tradeoff), (b) downloading from external storage during build (must be free/public/no paid infra; adds availability dependency), or (c) Docker-image-based deploy path. All need careful validation.

**Chroma portability/version concerns:**
- `requirements.runtime.txt` pins `chromadb>=0.5.23` (open upper bound). Chroma persistent format/sqlite+HNSW may be sensitive across versions; unpinned minor/major upgrades can break compatibility. **Risk:** deploy-time dependency resolution could install incompatible Chroma version.
- Index files are small (6.3M) and should be portable as files, but runtime compatibility must be validated against exact deployed Chroma version.

**Expected effect (estimates, not measured):**
- **Deployment artifact/image size:** +~6.3M raw; minor.
- **Build time:** neutral for git deploy if committed; Docker context slightly larger. Avoids recovery embedding/build CPU time on cold start.
- **Cold-start memory:** **reduces avoidable recovery spike** (no PDF extract/chunk/embed/build). **Does not reduce SentenceTransformer model load memory** required for dense query embeddings.
- **Index recovery behavior:** if index present and valid, `_ensure_index_ready()` returns early; recovery path unused. Need fallback behavior decision if packaged index missing/corrupt.

**Safe validation approach (no modification to existing local index):**
- Copy `chroma_db_production` to temp/staging dir or mount read-only copy in Docker; set `RAG_PERSIST_DIR` to copy; run app/tests against copy. Alternatively build Docker image with index included in image layer and run `--memory=512m --cpus=0.15` locally when Docker daemon available. Do not point validation writes at original `chroma_db_production`.

**Recommendation (investigation only, no implementation):**
- **Feasible in principle, with caveats.** Packaging the prebuilt index can remove cold-start recovery embedding/build work, but **will not solve confirmed model-memory OOM** by itself. Recommended only as part of a measured Render Free memory-reduction plan after: (1) pinning/confirming Chroma version compatibility; (2) deciding git-commit vs build-fetch packaging (no paid infra); (3) validating in 512MB-limited container; (4) keeping recovery fallback. Do not treat as sufficient fix for 512MB OOM.

**Cleanup:** temp inspection copy `C:\Users\sindh\AppData\Local\Temp\opencode\chroma_db_production_inspect` removed after inspection. No changes to original `chroma_db_production/`.

---

### Safe free prebuilt-index delivery mechanism (2026-11-21, Task 10, planning/documentation only; no implementation)

**Status/constraints.** Planning/documentation only. No implementation. No change to index contents/RAG behavior. No change to all-MiniLM-L6-v2, CPU PyTorch, Gemini, 800/160 chunking, hybrid dense+BM25 RRF, or grounded-answer behavior. Did not modify/regenerate original `chroma_db_production/`. No commit/push/deploy. Preserved all pre-existing unrelated deletions and untracked files.

**Additional evidence gathered (read-only):**
- **Local installed ChromaDB:** `chromadb==0.5.23` (`pip show chromadb` / `python -c "import chromadb; print(chromadb.__version__)"`). `requirements.runtime.txt` specifies `chromadb>=0.5.23` (open upper bound).
- **Index sqlite migrations (temp read-only copy):** sysdb migrations 00001-00009, metadb 00001-00004, embeddings_queue 00001-00002 present; consistent with Chroma 0.5.x schema. **Not a definitive creator-version stamp**; supports compatibility with 0.5.23 but does not prove exact build version.
- **Original index checksums (read-only, computed on original files; no modification):** SHA256 manifest
  - `34fd9d20fdf50c31ce329e67264660634078c597d8f2ab76a3266b7dd4ca9af0  chroma_db_production/chroma.sqlite3`
  - `d3c9fd302f000d7790aa403c2d0d8fec363fe46f30b07d53020b6e33b22435a9  .../dc220631-1c37-42f8-b4fc-45cce513acf7/data_level0.bin`
  - `e87a1dc8bcae6f2c4bea6d5dd5005454d4dace8637dae29bff3c037ea771411e  .../dc220631-1c37-42f8-b4fc-45cce513acf7/header.bin`
  - `a29ec4facde1e7b598b7c8d104314d6f93b4e9551d20a84d9ceca7950ee868d7  .../dc220631-1c37-42f8-b4fc-45cce513acf7/length.bin`
  - `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  .../dc220631-1c37-42f8-b4fc-45cce513acf7/link_lists.bin`
  - `d3c9fd302f000d7790aa403c2d0d8fec363fe46f30b07d53020b6e33b22435a9  .../f43144c4-4050-4e58-9fa7-63dcd953216e/data_level0.bin`
  - `e87a1dc8bcae6f2c4bea6d5dd5005454d4dace8637dae29bff3c037ea771411e  .../f43144c4-4050-4e58-9fa7-63dcd953216e/header.bin`
  - `a23d5308dba44ab7cab6f53d8fa640a017846e9de5e5fc261e0da3040a6c1ef6  .../f43144c4-4050-4e58-9fa7-63dcd953216e/length.bin`
  - `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  .../f43144c4-4050-4e58-9fa7-63dcd953216e/link_lists.bin`
  Temp inspection copy `chroma_db_production_inspect2` removed after inspection.

## Options comparison

| Option | Free availability / limits (evidence) | Reliability / accessibility | Build/image implications | Chroma persistence compatibility | Security / integrity | Render Blueprint fit |
|---|---|---|---|---|---|---|
| 1. Versioned index artifact fetched during Render build (e.g. GitHub Release asset / raw repo object / other free host) | **Needs verification.** GitHub provides free public-repo features, but release-asset/raw bandwidth/rate limits/terms must be confirmed at implementation time; do not assume unlimited/reliable. Other "free" hosts not verified. | External dependency; build fails if download unavailable/rate-limited. Pin URL + SHA256. | Adds download/extract step to buildCommand; +~6.3M download; small extraction disk. No repo bloat if not committed. | Same files; validate against deployed chromadb version. | Strong if SHA256 verified in build; use HTTPS; pin immutable version/tag/commit. | **Works with current native Python Blueprint** (`render.yaml` buildCommand/startCommand). |
| 2. Include index in Git repository | **Free** for normal git repo usage; repo size grows by ~6.3M (small). GitHub repo hosting limits/quotas should be confirmed if repo grows. | High (artifact travels with source). No external runtime fetch. | Clone/build context includes index; small increase. `.gitignore` currently ignores `chroma_db_production/` (`.gitignore:10`) — would need explicit exception/commit. `.dockerignore` does not ignore it. | Same files; version compatibility risk remains. | Integrity verifiable by git history + optional SHA256 manifest checked in CI/build. | **Works with current native Python Blueprint**; simplest for Render git deploy. |
| 3. Switch Render service to Docker deployment and package index in image | Render Docker deploy is supported; Render Free resource limits still apply. No paid infra by itself. | High once image built; build depends on base image/registry. | Requires Docker runtime/image build; image already large (torch/ST/chromadb). +~6.3M small; build time/memory may be higher on free tier. | Same files; Dockerfile already `COPY . .`; .dockerignore currently allows `chroma_db_production`. | SHA256 manifest can be validated during build; image provenance via git commit. | **Requires Blueprint/config change** (`runtime: docker` / Docker settings) — not current native Python Blueprint as-is. |

## Recommendation (planning only)
- **Recommended: Option 2 (commit index in Git) as primary candidate, with checksum manifest and Chroma pin**, *if repo policy accepts a 6.3M binary artifact*. Reasons: smallest operational complexity for current Render native-Python Blueprint; no external download dependency at build; high reliability; easy integrity verification (git + SHA256 manifest).
- **Fallback:** Option 1 only with immutable pinned URL and SHA256 verification, after confirming host free/reliability/terms; adds external dependency.
- **Option 3** not recommended now because it adds deploy-mode change and Docker-build overhead on Render Free while not addressing model-memory OOM.
- **Important limitation:** prebuilt index does **not** eliminate SentenceTransformer model memory; confirmed 512MB OOM risk remains.

## Exact required future file/config changes (if implemented later)
- Add `chroma_db_production/` to repo (or selected delivery artifact) **without modifying contents**; update `.gitignore` to allow it (or specific path exception).
- Add checksum manifest (e.g. `chroma_db_production.sha256` or `deployment/index-manifest.json`) and CI/build verification step.
- Pin `chromadb` to a compatible tested version in `requirements.runtime.txt` / deployment env (avoid open `>=0.5.23` for persisted index portability).
- Optionally add startup integrity check/logging (read-only validation) and keep `_ensure_index_ready()` fallback policy explicit.
- Update `docs/DEPLOYMENT.md`/`ENGINEERING_LOG.md` with implemented evidence. If Option 1: modify `render.yaml` buildCommand to download+verify+extract. If Option 3: modify `render.yaml` to Docker runtime and verify Docker build.

## Safe temp-copy validation (no touching original)
- Copy `chroma_db_production/` to temp/staging dir; verify SHA256 manifest against copy; set `RAG_PERSIST_DIR` to copy; run app/tests/CI eval read-only against copy. For Docker, mount/copy into image layer and test in constrained container. Never write to original `chroma_db_production/`.

## Remaining risks
- Chroma version drift/incompatibility with persisted files.
- Render Free 512MB OOM still likely during model load/query despite index prepackaging.
- Git repo binary bloat/policy concerns (Option 2) or external dependency/rate limits (Option 1).
- Need explicit integrity verification to avoid corrupted/stale index.

---

### Repository cleanup batch 1 — redundant deployment files and generated artifacts (2026-11-21, Task 13, local cleanup)

**Status/constraints.** Narrow cleanup only. No application source code modified. No `api/index.py` deleted. No legacy modules deleted/moved. No CI workflow consolidation. No Chroma index modified. No stage/commit/push/reset/stash/deploy. No `git add .`, `git clean`, or broad destructive commands. Preserved all unrelated modifications, deletions, and untracked files. Task 12 backup exists outside repo: `C:\Users\sindh\RAG-backup-task12-20261121-161336-e478a1b`.

**Git status before:**
```text
 M Dockerfile
 M docker-compose.yml
 M docs/DEPLOYMENT.md
 M docs/ENGINEERING_LOG.md
 D removed-experiment-artifacts/rag.py
 D removed-experiment-artifacts/results/eval_report.json
 M render.yaml
 D src/memory/working.py
 D src/metrics/citation.py
 M src/rag_app/schemas.py
 M src/rag_app/service.py
 D src/restart.py
 D tests/eval_pipeline.py
 M tests/test_api.py
 D todo.txt
?? .agents/
?? evaluation/results/
?? removed-experiment-artifacts/chunking_experiment.py
?? removed-experiment-artifacts/eval/
?? removed-experiment-artifacts/results/chunking_experiment.log
?? removed-experiment-artifacts/results/chunking_experiment.run.log
?? removed-experiment-artifacts/results/chunking_experiment/
?? nul
?? vercel.frontend.json
?? vercel.json
```

**Removed (exact paths):**
- `vercel.json` (untracked, root legacy Vercel config pointing to obsolete `api/index.py` path; backed up in Task 12) — **safe**: not canonical; canonical frontend config is `frontend/vercel.json` (confirmed intact).
- `vercel.frontend.json` (untracked, alternate legacy frontend config; backed up in Task 12) — **safe**: no refs found, non-canonical.
- `.pytest_cache/` — **safe**: regenerable test cache.
- `frontend/dist/` — **safe**: regenerable frontend build output.
- Python `__pycache__/` directories (13 dirs): `backend/__pycache__`, `evaluation/__pycache__`, `removed-experiment-artifacts/__pycache__`, `src/__pycache__`, `src/chunking/__pycache__`, `src/config/__pycache__`, `src/generation/__pycache__`, `src/metrics/__pycache__`, `src/observability/__pycache__`, `src/rag_app/__pycache__`, `src/rag_core/__pycache__`, `src/retrieval/__pycache__`, `tests/__pycache__` — **safe**: regenerable bytecode caches.

**Not removed (intentional):** `frontend/node_modules/`, all Chroma indexes (`chroma_db_production/` untouched), evaluation datasets/reports source files, experiment outputs/source, `api/index.py`, legacy src modules, CI workflows, `nul` (Windows reserved-name handling deferred to separate safe approach).

**Ignore-rule changes (minimal, justified):**
- `.gitignore`: added `.env.*` + `!.env.example` (secret-file variants), `*.pyc`, `.pytest_cache/`, `node_modules/`, `frontend/node_modules/`, `frontend/dist/`, `frontend/.vite/` (generated/caches), `evaluation/results/` (local eval artifacts), `*.log` (temp logs), `nul` (Windows junk), `.DS_Store`, `Thumbs.db`. Rationale: prevent accidental commit of secrets/caches/generated artifacts/local results.
- `.dockerignore`: added `chroma_db_exp001/002/003/004/`, `chroma_db_probe/` (local experiment indexes) and `evaluation/results/` (local eval artifacts). Rationale: reduce Docker build context; exclude local artifacts. Did **not** add `chroma_db_production/` because prebuilt-index packaging decision is pending.

**Minor hygiene:** fixed pre-existing trailing whitespace in prior Task 5/8/10 ENGINEERING_LOG entries to satisfy `git diff --check`.

**Git status after:**
```text
 M .dockerignore
 M .gitignore
 M Dockerfile
 M docker-compose.yml
 M docs/DEPLOYMENT.md
 M docs/ENGINEERING_LOG.md
 D removed-experiment-artifacts/rag.py
 D removed-experiment-artifacts/results/eval_report.json
 M render.yaml
 D src/memory/working.py
 D src/metrics/citation.py
 M src/rag_app/schemas.py
 M src/rag_app/service.py
 D src/restart.py
 D tests/eval_pipeline.py
 M tests/test_api.py
 D todo.txt
?? .agents/
?? removed-experiment-artifacts/chunking_experiment.py
?? removed-experiment-artifacts/eval/
?? removed-experiment-artifacts/results/chunking_experiment/
```

**Verification:**
- `frontend/vercel.json` remains intact. Root `vercel.json` and `vercel.frontend.json` removed. `.pytest_cache`, `frontend/dist`, all `__pycache__` dirs removed (find count 0).
- `git diff --check` passes.
- Existing Task 8 modifications and 7 pre-existing deletions remain intact. `evaluation/results/` and `nul` now ignored by updated `.gitignore` (still present on disk/untracked by prior state; no longer reported by `git status --short`).

**Limitations:** This is only the first cleanup batch. `nul` remains on disk (deferred). `api/index.py`, legacy src modules, duplicate CI workflows, and old experiment artifacts not cleaned. Repository not claimed clean.

---

### Repository cleanup and organization (2026-11-21, Task 16, local cleanup)

**Status/constraints.** Cleanup executed locally. No stage/commit/push/reset/stash/deploy. No secrets modified/exposed; `.env` untouched. Task 12 backup verified present: `C:\Users\sindh\RAG-backup-task12-20261121-161336-e478a1b`. Preserved active application work (Task 8 tracked modifications and pre-existing unrelated deletions) and did not restore files from HEAD merely because modified/deleted. Provider/model, embedding model, chunking config, retrieval defaults, and RAG architecture behavior preserved.

**Pre-cleanup verification:** repo `C:\Users\sindh\RAG`, branch `main`, HEAD `e478a1b`.

**Deleted (removed-experiment-artifacts/obsolete/legacy):**
- Entire `removed-experiment-artifacts/` directory (scripts, datasets, eval helpers, results, logs, artifacts, experiment-specific content). Tracked deletions appear in git status; untracked experiment artifacts removed (backed up in Task 12).
- `.github/workflows/rag_eval.yml` removed; CI consolidated into `.github/workflows/ci.yml` (already superset: PR all branches + push main, Python 3.11, cache, diagnostic, index build/check, deterministic pytest `-k "not real_index" --maxfail=5`, dense CI eval regression gate against `evaluation/baseline_metrics.json`, canonical `ci-evaluation` artifact, optional non-blocking dense-vs-hybrid `ci-evaluation-compare` artifact).
- `api/index.py` removed (legacy Vercel serverless entrypoint; canonical deployment is Render backend + Vercel frontend).
- Legacy src modules removed after dependency inspection (not imported by active app/core/backend/tests/CI/ingest/deploy): `src/chunking/`, `src/config/`, `src/generation/`, `src/observability/`, `src/retrieval/`, `src/memory/`, `src/metrics/retrieval_eval.py`, and empty `src/metrics/`.
- Obsolete test/data: `tests/test_exp001.py`, `tests/golden_dataset.csv` (experiment-legacy dependencies).

**Restored minimal active-test dependencies (from HEAD, worktree only, not staged):** `tests/fixtures/exp001/exp001_ground_truth.json`, `tests/fixtures/exp001/exp001_questions.csv`, `tests/fixtures/exp001/chunks.json` were restored because active tests (`test_citation_metadata.py`, `test_fixed_chunker.py`, `test_rag_evaluation.py`) depend on them; deleting them caused 4 failures. Restoration documented; only these required files kept.

**Retained active application components:** `backend/main.py`, `src/rag_app/`, `src/rag_core/`, `frontend/`, `prompts/`, `data/satcom-ngp.pdf`, `evaluation/` (ci_eval, golden dataset, baselines), `tests/` active suite, `scripts/ingest.py`, `scripts/evaluate.py`, `docs/`, `render.yaml`, `frontend/vercel.json`, `Dockerfile`, `docker-compose.yml`, `requirements.txt`, `requirements.runtime.txt`, `.env.example`, `.gitignore`, `.dockerignore`, `README.md`, `pyproject.toml`.

**Config/doc updates:** narrowed `.gitignore` `*.log` to `evaluation/results/*.log`; updated `README.md` in place; corrected stale docs refs (`docs/DEPLOYMENT.md`, `docs/CHUNKING_STRATEGY.md`, `docs/PROJECT_BRIEF.md` deployment/legacy notes); updated `src/rag_core/__init__.py` docstring.

**Validation (local):**
- `PYTHONPATH="$PWD" python -m py_compile backend/main.py src/rag_app/config.py src/rag_app/service.py src/rag_core/pipeline.py src/rag_core/retrieval.py` → EXIT 0.
- `PYTHONPATH="$PWD" python -m pytest tests -q -k "not real_index" --maxfail=5` → **133 passed, 1 deselected, 27 warnings** (after restoring 3 required experiment data files; 4 failures before restoration were due to missing experiment data).
- `PYTHONPATH="$PWD" LANGFUSE_ENABLED=false RAG_ALLOW_GEMINI=false python -m evaluation.ci_eval --dataset evaluation/ci_golden_dataset.json --out evaluation/results/ci_task16_cleanup --baseline evaluation/baseline_metrics.json --check-regression` → **regression passed** (dense recall@5 0.9236, precision@5 0.3, mrr@5 0.8611, evidence_coverage 0.9236, supported_answer_rate 0.5833, citation_support_rate 0.4828, abstention_accuracy 1.0, invalid_citations 0, failure_rate 0.0).
- `cd frontend && npm run build` → **built in 424ms** (EXIT 0).
- `git diff --check` → EXIT 0.
- Deployment/config presence check: `render.yaml`, `frontend/vercel.json`, `Dockerfile`, `docker-compose.yml`, `.env.example`, `data/satcom-ngp.pdf`, `evaluation/ci_golden_dataset.json`, `evaluation/baseline_metrics.json` present; `.env` exists and untouched.
- Reference grep: no active code refs to deleted legacy modules; remaining `rag_eval.yml`/`api/index.py` mentions are historical in `docs/PROJECT_BRIEF.md` historical section and `docs/ENGINEERING_LOG.md`.

**Known limitations / manual follow-up:** Render Free 512MB OOM remains a confirmed risk; public Render/Vercel deployment verification still requires user account/env/secrets and post-deploy smoke evidence. `evaluation/results/` is ignored/local. `nul` remains on disk on Windows (reserved-name cleanup deferred). Large git deletion set for `removed-experiment-artifacts/` is intentional per Task 16; only 3 active-test-required experiment data files retained.

---

### Final experiments dir removal and fixture relocation (2026-11-21, Task 17, local cleanup)

**Status/constraints.** No application behavior/RAG config/model/prompt/deployment setting changes. Fixture contents preserved exactly (SHA256 verified). `.env` untouched. No stage/commit/push/reset/stash/deploy. All other current changes/deletions preserved.

**Moved fixtures (from deleted `experiments/` to `tests/fixtures/exp001/`):**
- `experiments/exp001_ground_truth.json` -> `tests/fixtures/exp001/exp001_ground_truth.json` (sha256 `4b03f584405c9bd3d398f62586430bbbd1f2ea2e326465fd9dd2b120029bc0be`)
- `experiments/exp001_questions.csv` -> `tests/fixtures/exp001/exp001_questions.csv` (sha256 `beac9bcf4aa25b9bc54d988016180a1575cda59a3012408798b3ff9eec00ffa1`)
- `experiments/results/exp001_v3/chunks.json` -> `tests/fixtures/exp001/chunks.json` (sha256 `75858b45efea69f7be60bfcef7823d4ca3b1099769aeff67a9e6e52e869895a4`)

**Reference updates:** updated active tests `tests/test_rag_evaluation.py`, `tests/test_citation_metadata.py`, `tests/test_fixed_chunker.py`; updated `src/rag_app/config.py` comment; updated docs to remove old `experiments/` path references (historical artifact paths marked as removed-experiment-artifacts; moved fixture paths updated). Cleaned stale `.dockerignore` experiment-log lines. Removed `tests/__pycache__`.

**Verification:** `grep` for `experiments/` in active files (excluding `.git`, `.agents`) returned none. `experiments/` directory removed.

**Validation (local):**
- `PYTHONPATH="$PWD" python -m pytest tests -q -k "not real_index" --maxfail=5` → **137 passed, 1 deselected, 29 warnings in 26.26s** (after fixing `tests/test_fixed_chunker.py` fixture path).
- `PYTHONPATH="$PWD" LANGFUSE_ENABLED=false RAG_ALLOW_GEMINI=false python -m evaluation.ci_eval --dataset evaluation/ci_golden_dataset.json --out evaluation/results/ci_task17_fixtures --baseline evaluation/baseline_metrics.json --check-regression` → **regression passed** (`passed:true`, failures:[]).
- `git diff --check` → **DIFF_CHECK_OK**.

---

### Local experiment/probe ChromaDB cleanup (2026-11-21, Task 18, local data cleanup)

**Status/constraints.** Local data cleanup only. No application code/dependency/env/deployment config changes. No stage/commit/push/reset/stash/deploy. Did not delete/regenerate `chroma_db_production/`. Preserved `.env`, `frontend/node_modules/`, active evaluation fixtures, and all other current work.

**Repo verification:** `C:\Users\sindh\RAG`, branch `main`, HEAD `e478a1b`.

**Targets (verified experiment/probe indexes; not referenced by active app/tests/scripts/Docker/deploy runtime):** `chroma_db_exp001/`, `chroma_db_exp002/`, `chroma_db_exp003/`, `chroma_db_exp004/`, `chroma_db_probe/`. References found only in `.gitignore`/`.dockerignore` ignore rules and historical docs/fixture metadata.

**Production index before/after (unchanged):** `chroma_db_production/` — **9 files, 6.3M** before; **9 files, 6.3M** after.

**Removed:** all 5 experiment/probe directories (each had 5 files; ~4.0M/3.6M/2.8M/2.5M/2.5M before removal).

**Verification:**
- All 5 target dirs absent; `chroma_db_production/` intact.
- `PYTHONPATH="$PWD" python -m pytest tests -q -k "not real_index" --maxfail=5` → **137 passed, 1 deselected, 29 warnings in 27.09s**.
- `git diff --check` → **DIFF_CHECK_OK**.

---

### Documentation repair after cleanup (2026-11-21, Task 20, docs-only)

**Status/constraints.** Documentation-only repair. No application code, tests, CI workflows, dependencies, or deployment config modified. `todo.txt` deletion left untouched (pending confirmation). `.env` not modified/inspected. No stage/commit/push/reset/stash/deploy. Historical experiment findings/decisions preserved; no experiment history rewritten; deleted `experiments/` directory/scripts not reintroduced.

**Defects found (Task 19 audit):**
- `docs/RUN_LOCAL.md:99` garbled text: "No new removed-experiment-artifacts/retrieval/chunking/reranking added."
- `docs/RUN_LOCAL.md:109` stale/broken command `python -m experiments.m4_eval_harness ...` (script removed in Task 16 cleanup).
- `docs/PROJECT_BRIEF.md` contained widespread `removed-experiment-artifacts` wording/path artifacts, a garbled "Current Status" insertion, and an accidental duplicated copy of sections 1–10.
- Multiple docs under `docs/` contained `removed-experiment-artifacts` path artifacts from bulk cleanup replacement; some historical docs also contained stale runnable-looking `python -m experiments.*` commands.

**Files corrected:**
- `docs/RUN_LOCAL.md` — fixed garbled limitation text; replaced retired M4 harness command with current supported CI evaluation command (`python -m evaluation.ci_eval ... --dataset evaluation/ci_golden_dataset.json --baseline evaluation/baseline_metrics.json --check-regression --out evaluation/results/ci_local`) and documented `python scripts/evaluate.py --help` wrapper, with evidence from `.github/workflows/ci.yml`, `evaluation/ci_eval.py --help`, and `scripts/evaluate.py`.
- `docs/PROJECT_BRIEF.md` — removed garbled Current Status blob and accidental duplicated sections 1–10 (kept unique Section 11); fixed garbled path/wording artifacts; updated Commands section to current supported pytest/eval commands and marked historical experiment runners retired (Task 16 cleanup, git history).
- `docs/M4_PROGRESS.md` — marked historical M4 harness command/artifacts retired and pointed to current CI evaluation command.
- `docs/M4_EVALUATION_PLAN.md` — marked historical harness as retired; pointed to current CI eval.
- `docs/BACKLOG.md` — clarified BL-012 historical harness retired and current CI gate path.
- `docs/PRODUCTION_SLICE.md` — updated command section to current deterministic pytest subset; marked historical experiment commands as retired/historical record.
- `docs/EXP-002.md`, `docs/EXP-003.md`, `docs/EXP-004.md`, `docs/EXP-005.md`, `docs/EXP-006.md` — added explicit "Historical (retired)" note in experiment command code blocks (scripts removed in Task 16 cleanup; historical record preserved).
- Other docs (`BACKLOG`, `CHUNKING_EXPERIMENT`, `EXP-001`, `EXP-007`, `EXP-008`, `EXP-008-citation-metadata-correction`, `PRODUCTION_SLICE`, `PROJECT_BRIEF`) — replaced `removed-experiment-artifacts` wording/path artifacts with clear historical `experiments/...` references where appropriate; historical findings/decisions preserved.

**Obsolete evaluation command:** `python -m experiments.m4_eval_harness ...` is retired because `experiments/m4_eval_harness.py` and experiment artifacts were removed in Task 16 cleanup. No identical replacement exists; the current supported deterministic evaluation/regression gate is `evaluation/ci_eval.py` (CI workflow `.github/workflows/ci.yml`, wrapper `scripts/evaluate.py`). Docs now point to the supported command.

**Remaining documentation limitations:**
- `docs/ENGINEERING_LOG.md` retains historical `removed-experiment-artifacts` strings in prior entries; left unchanged intentionally (task forbids rewriting earlier log entries).
- Historical experiment report docs still reference `experiments/...` paths/scripts as historical records; those files do not exist on disk after Task 16 cleanup, but references are preserved/marked historical where appropriate.
- `todo.txt` remains deleted in worktree pending explicit confirmation.

**Validation:**
- `grep -rn "removed-experiment-artifacts" docs` → only `docs/ENGINEERING_LOG.md` historical entries remain.
- RUN_LOCAL eval commands verified: `python -m evaluation.ci_eval --help` (exit 0), `scripts/evaluate.py` exists and wraps `evaluation.ci_eval`, CI workflow uses same command/args. Referenced dataset/baseline paths exist.
- `git diff --check` → **DIFF_CHECK_OK**.
- Reviewed diff: docs-only changes (plus pre-existing unrelated worktree changes already present); no application code/tests/CI/deps/deploy config modified by Task 20.

---

### Align local evaluation docs with CI dense-only baseline (2026-11-25, Task 22, docs-only)

**Status/constraints.** Documentation-only. No evaluation code, tests, CI workflow, dependencies, or deployment files modified. `todo.txt` left untouched. No stage/commit/push/reset/stash/deploy. No full evaluation run; validation used source inspection and `--help`.

**Task 21 finding addressed:** documented local CI evaluation command in `docs/RUN_LOCAL.md` omitted `--retrieval-modes`, `--generation-mode`, and `--k`. `evaluation/ci_eval.py` defaults are `--retrieval-modes=dense,hybrid`, `--generation-mode=extractive`, `--k=5`; `.github/workflows/ci.yml` dense baseline step explicitly passes `--retrieval-modes dense --generation-mode extractive --k 5`. Omitting `--retrieval-modes` changes behavior (runs dense+hybrid, not CI-equivalent dense-only baseline); omitting `--generation-mode`/`--k` currently matches defaults but is ambiguous.

**Exact documentation corrections:**
- `docs/RUN_LOCAL.md` — replaced eval command with explicit CI-equivalent args (`--retrieval-modes dense --generation-mode extractive --k 5 --baseline evaluation/baseline_metrics.json --check-regression --out evaluation/results/ci_local`); added Git Bash and Windows PowerShell command forms; added explanation that explicit args reproduce CI dense-only regression baseline step and that omitting `--retrieval-modes` uses default `dense,hybrid` (not CI-equivalent).
- `docs/M4_PROGRESS.md` — updated current supported evaluation/regression gate to same explicit CI-equivalent args with Git Bash/PowerShell forms; retained historical context and clearly distinguished current command from retired historical harness command/artifacts.
- `docs/ENGINEERING_LOG.md` — appended this Task 22 entry; earlier entries unchanged.

**Validation:**
- Args verified against `.github/workflows/ci.yml` dense baseline step and `evaluation/ci_eval.py` argparse defaults/help.
- PowerShell syntax: `$env:PYTHONPATH='.'; python -m evaluation.ci_eval ...` (single-line). Git Bash syntax: `PYTHONPATH=. python -m evaluation.ci_eval \ ...` with line continuations.
- `git diff --check` → **DIFF_CHECK_OK**.
- Diff inspection: only intended Markdown files modified by Task 22 (`docs/RUN_LOCAL.md`, `docs/M4_PROGRESS.md`, `docs/ENGINEERING_LOG.md`) within pre-existing worktree; no code/tests/CI/deps/deploy files changed.
- `todo.txt` remains ` D todo.txt` (untouched).
---

### Isolated Weaviate Cloud retrieval experiment (2026-11-25, Task 24, experimental/opt-in)

**Status/constraints.** Experimental, opt-in only. Default backend remains `chroma` (existing production dense+BM25+RRF path unchanged). No production migration, no deploy/stage/commit/push, no live Weaviate ingestion/evaluation performed (no credentials provided/authorized), no Render queries. `chroma_db_production/` not modified/deleted. Existing evaluation baselines/historical experiment results not changed. `.env`, `.agents/`, `tests/fixtures/`, generated results, local data preserved. No `git add .`.

**Phase 1 inspection (smallest safe integration point):**
- Embedding/init path: `src/rag_app/service.py` lazily loads `SentenceTransformer` via `service.model` and builds `DenseIndex` via `service.index`; `service._ensure_index_ready()` rebuilds local Chroma from PDF when empty.
- Chunking/metadata: `src/rag_app/config.py` default `chunk_strategy=fixed_character`, `chunk_size=800`, `chunk_overlap=160`; `src/rag_core/chunker.py` provides fixed/structure-aware chunkers; `src/rag_core/index.py` `chunk_to_metadata()` persists citation-relevant metadata (`chunk_id,text,source,start/end_offset,start/end_page,article,section,heading,...`).
- Chroma ingestion/retrieval interfaces: `src/rag_core/index.py` `DenseIndex.build/search/count/reset`; `src/rag_core/retrieval.py` `HybridRetriever.retrieve()` returns `RetrievalResult(hits=[RetrievedChunk...], debug=...)` with RRF fusion and optional Cohere rerank.
- BM25/RRF: `src/rag_core/retrieval.py` uses `rank_bm25.BM25Okapi`, `rrf_fuse`, `DEFAULT_RRF_K=60`, `DEFAULT_DENSE_K/BM25_K=20`.
- Answer-generation interface: `src/rag_app/service.py` calls `self.pipeline.answer_from_hits(question, hits)`; `src/rag_core/pipeline.py` `GroundedPipeline` requires index only for `answer()`; `answer_from_hits()` is index-independent.
- Evaluation harness: `evaluation/ci_golden_dataset.json` fixed 15 questions (10 answerable, 5 unanswerable); `evaluation/ci_eval.py` computes retrieval/context/faithfulness/answer/citation/abstention/latency metrics, supports regression gate; CI workflow runs dense-only baseline by default.
- Config/env conventions: `src/rag_app/config.py` `AppConfig.from_env()` with `RAG_*` vars; `.env.example`; `requirements.txt`/`requirements.runtime.txt`.
- Docs/log: `docs/ENGINEERING_LOG.md` append-only; deployment/run docs in `docs/DEPLOYMENT.md`, `docs/RUN_LOCAL.md`.

**Phase 2 Weaviate compatibility (official docs verified via webfetch, 2026-11-25):**
- Docs: Weaviate Python client v4 (`weaviate-client`), `connect_to_weaviate_cloud` + `Auth.api_key`; collections via `client.collections.create`; vectorizer `Configure.Vectors.text2vec_weaviate(model=..., source_properties=[...])` or `text2vec_weaviate` vectorizer config; hybrid query `collection.query.hybrid(query=..., limit=..., alpha=..., return_metadata=...)`.
- Weaviate Embeddings: managed service available on Weaviate Cloud; docs state "You can try it out **for free on a free cluster**" / free Sandbox/trial limits apply (rate/batch limits documented by Weaviate; verify current cluster quota before live run). Model availability is cluster/account-dependent; this experiment defaults to `Snowflake/snowflake-arctic-embed-l-v2.0` (documented example) but is configurable via `WEAVIATE_EMBEDDING_MODEL`. Do not assume equivalence to local `all-MiniLM-L6-v2`.
- Uncertainties/limitations recorded: exact free-tier hosted-embedding quota/model availability may change; client API minor differences handled defensively; live cluster behavior not verified here.

**Phase 3 implementation (opt-in, isolated):**
- `src/rag_core/weaviate_retriever.py` (new): `WeaviateRetrieverConfig.from_env()` requires `WEAVIATE_URL`, `WEAVIATE_API_KEY`, `WEAVIATE_COLLECTION`; optional `WEAVIATE_EMBEDDING_MODEL`; lazy `weaviate-client` import; no silent fallback; returns `RetrievalResult`/`RetrievedChunk` with citation metadata and debug.
- `src/rag_app/config.py`: added `retriever_backend` default `chroma`, plus optional Weaviate env fields (`RAG_RETRIEVER_BACKEND`/`RETRIEVER_BACKEND`, `WEAVIATE_URL`, `WEAVIATE_API_KEY`, `WEAVIATE_COLLECTION`, `WEAVIATE_EMBEDDING_MODEL`). Defaults unchanged.
- `src/rag_app/service.py`: added `weaviate_retriever` property with explicit config validation; `_retrieve()` routes to Weaviate only when backend explicitly selected and does not initialize local Chroma/BM25/SentenceTransformer; no silent fallback. `pipeline` property passes `index=None` for Weaviate backend because `answer_from_hits()` does not require index. `health()` reports `retriever_backend` and avoids misleading Chroma status for Weaviate.
- `src/rag_app/schemas.py`: added `retriever_backend` to `HealthResponse`; extended `QueryResponse.retrieval_mode` literal to include `weaviate_hybrid` for experimental reporting.
- `src/rag_core/pipeline.py`: `GroundedPipeline.index` made optional; `answer()` raises clear error if index missing; `answer_from_hits()` unchanged. This avoids unnecessary local index/model load for precomputed Weaviate hits.
- `.env.example`: added Weaviate env var names only. `requirements.weaviate.txt`: optional `weaviate-client>=4.6.0,<5.0.0` (isolated install; not added to default runtime requirements to avoid Render default bloat).

**Phase 4 ingestion/evaluation:**
- `scripts/weaviate_ingest.py` (new): explicit experimental ingestion command; reads existing PDF, uses existing AppConfig chunking (default fixed 800/160), audits coverage, creates/reuses dedicated Weaviate collection with `text2vec_weaviate` and explicit citation metadata properties, deterministic UUIDs for chunk IDs, batch ingestion, optional `--recreate`, `--limit` debug only. Does not touch `chroma_db_production` or local Chroma.
- `evaluation/weaviate_eval.py` (new): isolated runner reusing `evaluation.ci_eval` dataset/metric logic. Compares A) baseline `chroma` + `hybrid` (dense+BM25+RRF) and B) `weaviate` + native hybrid, same 15-question dataset. Reports per-mode summaries/comparison JSON; requires explicit `--run-baseline`/`--run-weaviate`; Weaviate mode requires env vars and no silent fallback. Not run with live credentials in this task.

**Phase 5 memory implications (reasoned, not deployment-verified):**
- Weaviate experimental retrieval path in `RAGService._retrieve()` avoids calling `self.index`, `HybridRetriever`/BM25 corpus build, and `SentenceTransformer` model load. `GroundedPipeline` is constructed with `index=None` for Weaviate.
- However, `sentence-transformers`, `chromadb`, `rank_bm25`, and torch-related packages remain installed/importable in default runtime image unless dependency packaging is changed. Importing them is lazy in key paths but process-level memory savings on Render require a separate Weaviate-only deployment image/dependency set and actual measurement. No claim that Render Free 512MB will work without live deployment/measurement.

**Phase 6 tests (executed):**
- `tests/test_weaviate_retriever.py` (new, mocked, no live credentials): 8 passed — default backend remains chroma; explicit backend selection; missing credentials fail clearly; normalized retrieval structure/citation metadata preserved; connection error handled; service Weaviate path does not init chroma; ingestion script has no chroma/PersistentClient/DenseIndex usage; baseline eval defaults unchanged.
- `python -m pytest tests/test_api.py -q` → 17 passed, 4 warnings (includes real_index smoke; no regressions observed).
- Not executed: live Weaviate ingestion/eval, full `pytest -k "not real_index"` after final doc edits in this turn, CI-equivalent dense eval, Render/Vercel deployment smoke.

**Phase 7 docs:**
- Updated `.env.example`, added `requirements.weaviate.txt`, appended this log. `docs/RUN_LOCAL.md` and `docs/DEPLOYMENT.md` updated with isolated experiment setup/run/disable/limitations (names only, no secrets).

**Safety checks:** `chroma_db_production/` present and untouched; no Weaviate credentials used; no live network calls to Weaviate; no stage/commit/push/deploy.
---

### Weaviate-only runtime and import audit (2026-11-26, Task 25, audit/no-deploy)

**Status/constraints.** Audit only. No production Chroma index modified/deleted. Default retriever remains `chroma`. No embedding/chunking/retrieval algorithm/eval baseline changes. No Weaviate cluster created, no live API requests, no deploy/stage/commit/push. `.env` untouched; no credentials exposed. Existing uncommitted work preserved. No `git add .`.

**Phase 1 — Import chain trace (code inspection + subprocess probes, PYTHONPATH=.):**

- Entry/import chain: `backend/main.py` -> `src.rag_app` (`AppConfig`, `RAGService`) -> `src/rag_app/service.py` top-level imports `DenseIndex`, `HybridRetriever`, `RetrievedChunk`, `RetrievalResult` from `src.rag_core` -> `src/rag_core/__init__.py` eagerly imports index/retrieval/pipeline/generation/evaluation modules.
- Eager heavy imports identified: `src/rag_core/index.py` imports `chromadb` at module top; `src/rag_core/retrieval.py` imports `rank_bm25.BM25Okapi` at module top. `service.py` lazy-imports `SentenceTransformer` only inside `model` property; Weaviate path avoids `self.index`/`self.model`/`HybridRetriever`.
- Subprocess probes (no live Weaviate calls; dummy env vars):
  - `import backend.main` with `RAG_RETRIEVER_BACKEND=weaviate`: `torch/sentence_transformers/transformers/scipy` not imported; **`chromadb/rank_bm25/numpy` imported**.
  - `from src.rag_app.service import RAGService` with Weaviate env: same result.
  - `WeaviateRetriever` import only: `torch/sentence_transformers/transformers` not imported; **`chromadb/rank_bm25/numpy` imported** due to `RetrievedChunk`/`RetrievalResult` imports from index/retrieval modules.
  - `RAGService` Weaviate backend mocked query path (`_retrieve` with fake retriever, no network): query returned `weaviate_hybrid`; **`chromadb/rank_bm25/numpy` imported, torch/sentence_transformers/transformers/scipy not imported**.

**Import vs init distinction:** Current Weaviate path avoids **initializing** SentenceTransformer model, loading model weights, allocating local embedding inference tensors, initializing Chroma PersistentClient/index, and building BM25 corpus. It does **not** currently avoid **importing** `chromadb`, `rank_bm25`, and `numpy` because of eager module imports in `rag_core/index.py`, `rag_core/retrieval.py`, and eager exports in `rag_core/__init__.py`/`service.py`.

**Phase 2 — Weaviate dependency/API compatibility:**
- `requirements.weaviate.txt` currently `weaviate-client>=4.6.0,<5.0.0` (isolated, optional). Code uses v4 patterns: `weaviate.connect_to_weaviate_cloud`, `Auth.api_key`, `collections.exists/get/create/delete`, `Configure.Vectors.text2vec_weaviate` with fallback to `Configure.Vectorizer.text2vec_weaviate`, `collection.query.hybrid` with `MetadataQuery` where available, `batch.fixed_size` and `failed_objects`, `aggregate.over_all(total_count=True)`, `client.close()`. Auth/batch/hybrid/metadata/cleanup present. No live API verification performed.
- Compatibility risk: `weaviate-client` v4 has minor API surface differences across releases; `>=4.6.0,<5.0.0` is plausible but untested in this repo. Recommend pinning to a specific tested version after live smoke test (e.g. `==4.x.y`) and recording it in docs; keep upper bound `<5.0.0` until v5 migration reviewed.

**Phase 3 — Runtime separation assessment:**
- **A. Required by Weaviate API/query runtime (if refactored):** `weaviate-client`, app framework/runtime (`fastapi`,`uvicorn`,`pydantic`,`python-dotenv`,`httpx`), generation deps if Gemini used (`google-generativeai`), observability as configured (`langfuse` optional).
- **B. Required only by local Chroma retrieval:** `chromadb`, `sentence-transformers`, `torch`/transformer stack (transitive), `rank_bm25` for hybrid BM25/RRF, local index artifacts/model cache.
- **C. Required only by PDF ingestion/local rebuild:** `pypdf` and local ingestion/chunking/index-build path; Weaviate ingestion script uses existing chunking/PDF extraction but does not need Chroma/SentenceTransformers.
- **D. Required only for evaluation/dev:** `pytest`, `ragas`, eval scripts, notebooks/dev tools.
- Proposed untracked file `requirements.weaviate.runtime.proposed.txt` created as a proposal only; not wired into default deployment. It is not sufficient without import-chain refactor because current app imports still load chromadb/rank_bm25/numpy.

**Phase 4 — Tests (executed):**
- `PYTHONPATH=. python -m pytest tests/test_weaviate_retriever.py -q` → **10 passed**.
- Includes subprocess-based import audits: `weaviate_retriever` module import chain (documents chromadb/rank_bm25/numpy currently imported; torch/sentence_transformers/transformers not imported), and Weaviate-backend service import/query path (same finding). Existing mocked tests cover no Chroma/SentenceTransformer initialization on Weaviate query, missing credentials clear error, no silent fallback, default backend unchanged.

**Phase 5 — Memory assessment (qualitative, code-verified where noted):**
- *Current Chroma+SentenceTransformers runtime:* highest; loads/imports Chroma/BM25/numpy and initializes SentenceTransformer/PyTorch model tensors on local retrieval path. Render Free 512MB OOM previously observed (Task 23).
- *Weaviate backend with existing full dependencies:* medium-high; code-verified no local model/index/BM25 initialization on Weaviate query path, but chromadb/rank_bm25/numpy still imported by current module graph; torch/sentence_transformers/transformers not imported in probed path. Memory reduction likely partial at best; not Render-verified.
- *Weaviate backend with proposed lightweight dependency set + import-chain refactor:* potentially lowest; would exclude torch/sentence-transformers/chromadb/rank_bm25 from image if imports refactored and PDF/local ingest separated. **Not verified by deployment/measurement.** Actual Render Free viability requires measured deploy.

**Remaining risks:** eager import chain prevents true lightweight runtime today; weaviate-client version drift; free-tier quotas/model availability; semantic differences vs MiniLM; need separate deploy image/build target; no live Weaviate or Render measurement.

**Safety checks:** `chroma_db_production/` untouched; default retriever still `chroma`; no deploy/stage/commit/push; no `.env` changes; no credentials exposed.
---

### Weaviate lightweight runtime import refactor (2026-11-26, Task 26, refactor/no-deploy)

**Status/constraints.** Import/runtime separation only; no retrieval redesign. Default backend remains `chroma`. No embedding/chunking/retrieval algorithm/eval baseline changes. `chroma_db_production/` untouched. No Weaviate cluster/live API calls. No deploy/stage/commit/push. `.env` untouched; no credentials exposed. Existing uncommitted work preserved. No `git add .`.

**Previous eager-import findings (Task 25):** `rag_core/__init__` eager exports, `index.py` top-level `chromadb`, `retrieval.py` top-level `rank_bm25`, and `service.py` top-level rag_core imports caused `chromadb/rank_bm25/numpy` to load on Weaviate-selected app path; `torch/sentence_transformers/transformers/scipy` were not loaded.

**Refactoring performed:**
- Added `src/rag_core/retrieval_types.py` (lightweight `RetrievedChunk`, `RetrievalResult`; no heavy deps). Updated `index.py`, `retrieval.py`, `weaviate_retriever.py`, `pipeline.py`, `evaluation.py` to use/share it; removed duplicate dataclasses.
- `index.py`: lazy `import chromadb` in `DenseIndex.__init__` (no top-level chromadb import).
- `retrieval.py`: lazy `rank_bm25` import in `CorpusIndex.load()`; TYPE_CHECKING-only heavy type imports; re-exports `RetrievedChunk`/`RetrievalResult` for compat.
- `rag_core/__init__.py`: replaced eager exports with PEP 562 lazy `__getattr__` map; preserves public names/`__all__`.
- `rag_app/service.py`: moved heavy rag_core imports to TYPE_CHECKING/local lazy imports in properties/methods; lightweight `retrieval_types` import at top; Weaviate path avoids local index/BM25/model initialization. Fixed NameError regression in `_apply_gemini` by local ANSWERED/ABSTAINED import.
- FastAPI/ingest routes unchanged; `/ingest` still local-imports chunking/index build only when called.

**Subprocess import audit (fresh Python subprocesses, PYTHONPATH=., dummy Weaviate env, mocked retrieval, no live calls):**

| Scenario | torch | sentence_transformers | transformers | chromadb | rank_bm25 | numpy | scipy |
|---|---:|---:|---:|---:|---:|---:|---:|
| `weaviate_retriever` module import | False | False | False | False | False | False | False |
| `backend.main` import + Weaviate backend mocked query | False | False | False | False | False | *True\** | False |
| Chroma default backend import (`DenseIndex/HybridRetriever` resolvable) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

\*In this environment, `numpy` was imported during `RAGService` init after `backend.main` due to third-party Langfuse serializer (`langfuse._utils.serializer`) when Langfuse was enabled/configured by env; not imported by Weaviate-only service path without that condition. Documented as environment/dependency-sensitive, not a Weaviate code-path requirement.

**Tests executed:**
- `PYTHONPATH=. python -m pytest tests/test_weaviate_retriever.py -q` → **11 passed**.
- `PYTHONPATH=. python -m pytest tests/test_api.py -q` → **17 passed, 4 warnings**.
- `PYTHONPATH=. python -m pytest tests -q -k "not real_index and not test_api" --maxfail=10` → **132 passed, 17 deselected, 29 warnings**.
- CI-equivalent eval / full real_index not rerun in this turn.

**Lightweight dependency proposal:** updated `requirements.weaviate.runtime.proposed.txt` (proposal only, not wired into deployment). Still requires import-audit + Render measurement; numpy may be pulled by optional observability deps.

**Remaining risks:** Langfuse/numpy third-party import sensitivity; full Render memory/startup not measured; weaviate-client version pin still needs live smoke; separate Weaviate-only deployment image/build target still needed.

**Safety checks:** `chroma_db_production/` untouched; default `chroma` unchanged; no deploy/stage/commit/push.
---

### Weaviate Cloud live-experiment preparation (2026-11-26, Task 27, prep/no-live-calls)

**Status/constraints.** Preparation only. No live Weaviate cluster created, no live API calls, no ingestion, no collection deletion, no embedding quota consumed, no deployment/stage/commit/push. `chroma_db_production/` untouched. Default backend remains `chroma`. No `.env` changes; no credentials exposed. Existing uncommitted work preserved.

**1) Collection deletion safety findings/fixes (`scripts/weaviate_ingest.py`):**
- Before: `--recreate` deleted existing collection without exact-name confirmation; existing collections were reused and ingestion proceeded (potential silent upsert/overwrite).
- After: default execution never deletes. `--recreate` requires `--confirm-delete-collection <exact-name>` matching `WEAVIATE_COLLECTION`; prints destructive warning with object count before delete. `_assert_safe_collection_name` refuses production-like names (`prod`/`production`) and requires clearly experimental name (`exp`/`experiment`/`test`/`sandbox`). Existing collection is reported with object count and script exits unless `--allow-existing` is explicitly provided. No silent overwrite.

**2) Ingestion idempotency findings/fixes:**
- Deterministic UUIDs preserved (`uuid5` from `satcom-ngp-weaviate:{chunk_id}`), so explicit `--allow-existing` re-ingest is idempotent by object ID for same chunk IDs. Batch errors are counted; partial failures reported as `partial_failure`. Final collection count is queried via `aggregate.over_all(total_count=True)` and reported; warning emitted when count < selected chunks (non-debug run, no failed objects). Schema compatibility is best-effort by explicit property creation; existing-schema mismatch is not fully auto-validated (remaining risk). 800/160 chunking remains from existing AppConfig defaults.

**3) Evaluation isolation findings/fixes (`evaluation/weaviate_eval.py`):**
- Same fixed 15-question dataset (`evaluation/ci_golden_dataset.json`) unchanged. Baseline run forces `retriever_backend=chroma`, `retrieval_mode=hybrid`; Weaviate run requires explicit env vars and uses configured Weaviate collection only. Default output now timestamped (`evaluation/results/weaviate_experiment/<UTC timestamp>`); existing non-empty output requires explicit `--overwrite`. Report payload includes dataset/dataset_version/question count, k, generation mode, Weaviate collection, embedding model, alpha, UTC timestamp. No historical baseline files overwritten by default.

**4) Free-tier safeguards (official pricing page https://weaviate.io/pricing, fetched 2026-11-26; verify before live run):**
- Confirmed on pricing page (subject to change): Free/Sandbox tier — $0, 1 cluster per user, 100,000 objects, 1 GB memory, 10 GB disk, 1 collection, up to 3 tenants, Embeddings 2,000 req/day, Query Agent 1,000 req/mo, Basic Support. Flex starts $45/mo pay-as-you-go; Premium prepaid/contact sales.
- Account-dependent/verify: region availability, exact embedding model availability, rate-limit enforcement details, cluster creation workflow, billing/upgrade prompts, and any changes after pricing-page fetch. No code path in this repo creates/upgrades clusters or makes paid API calls; live usage requires explicit user authorization and credentials.

**5) Tests executed (mocked, no live credentials/network):**
- `PYTHONPATH=. python -m pytest tests/test_weaviate_retriever.py -q` → **16 passed** (5.75s). Added safety tests: refuse prod-like/non-experimental collection names; recreate requires exact confirmation; existing collection stops without `--allow-existing`; eval refuses overwrite without `--overwrite`.
- `PYTHONPATH=. python -m pytest tests/test_api.py -q` → **17 passed, 4 warnings** (prior Task 26 run in same session; not rerun after Task 27 doc/script edits, but no API code changed in Task 27).

**6) Exact safe procedure for first authorized live experiment (manual, after explicit approval):**
1. Confirm free-tier cluster exists and embedding model/quota available in Weaviate Cloud console; do not upgrade.
2. Set env vars in shell only (never commit): `WEAVIATE_URL`, `WEAVIATE_API_KEY`, `WEAVIATE_COLLECTION` (clearly experimental, e.g. `SatcomNgpExp800160`), optional `WEAVIATE_EMBEDDING_MODEL`.
3. Install isolated deps: `pip install -r requirements.weaviate.txt`.
4. Dry/safety check: run `python scripts/weaviate_ingest.py --help`; confirm collection name is experimental.
5. First ingestion (no delete): `python scripts/weaviate_ingest.py --allow-existing` only if intentionally reusing an existing experimental collection; for a new empty collection, script will create it. If collection exists and you did not pass `--allow-existing`, script will report count and stop.
6. Destructive reset only with explicit authorization: `python scripts/weaviate_ingest.py --recreate --confirm-delete-collection <exact-name> --allow-existing` (or without `--allow-existing` after recreate creates fresh collection - recreate path creates fresh; then run ingestion as needed). Verify exact name before running.
7. Verify collection count equals expected 61 chunks for default 800/160 config before eval.
8. Run isolated eval: `PYTHONPATH=. python -m evaluation.weaviate_eval --run-baseline --run-weaviate --k 5 --generation-mode extractive` (writes timestamped output dir by default). Inspect `report.json` and per-mode files.
9. Record results in `docs/ENGINEERING_LOG.md`; do not overwrite historical baselines.

**Remaining risks:** live schema/model/quota behavior unverified; free-tier limits may change; Weaviate embedding model differs from MiniLM; import/runtime image still not Render-measured; partial ingestion requires manual verification.
---

### Weaviate experiment diagnostic instrumentation (2026-11-26, Task 28, diagnostics/no-live-eval)

**Status/constraints.** Isolated diagnostics only. No live evaluation, no live Weaviate calls, no ingestion/deletion/collection modification. Default Chroma backend unchanged. No embedding/alpha/chunking/retrieval default/production behavior changes. No `.env`/dependency changes. No stage/commit/push/deploy. Existing evaluation results/historical experiments preserved.

**Changes (additive):**
- `src/rag_app/schemas.py`: `RetrievedChunkModel` gained optional additive diagnostic fields `score`, `retrieval_scores`, `metadata` (default empty/None).
- `src/rag_app/service.py`: `_retrieved_model` now populates diagnostic score/metadata (excludes text/content). No retrieval/abstention behavior changes.
- `evaluation/ci_eval.py`: added `_retrieved_diagnostic`, `_abstention_diagnostics`, and per-question `diagnostics` block in `run_mode` results. Diagnostics include ordered top-k retrieved chunks (IDs/ranks/scores/distance/retrieval_scores/safe metadata), selected citation chunk IDs and expected-evidence satisfaction, abstention decision/reason/raw Weaviate score/distance/best_distance/token_support/selected_lexical_support/retrieval_debug, and latency components (`retrieval_ms`, `selection_and_extractive_ms`, `gemini_generation_ms`, `evaluation_ms`, `total_ms`) with note that remote Weaviate query time is inside `retrieval_ms`.
- `evaluation/weaviate_eval.py`: reuses ci_eval diagnostic helpers; adds `focus_question_diagnostics` for Q07/Q12/Q14/UNANS-11 in report.
- `tests/test_weaviate_retriever.py`: added deterministic tests for retrieved diagnostic serialization (excludes text), abstention diagnostic Weaviate score capture, and read-only diagnostic behavior.

**Verification:** `pytest tests/test_weaviate_retriever.py -q` (19 passed in final run) and `pytest tests/test_api.py -q` (17 passed, 4 warnings) after fixes. No live eval run.

**Proposed next diagnostic evaluation command (do not execute now):**
`PYTHONPATH=. WEAVIATE_URL=... WEAVIATE_API_KEY=... WEAVIATE_COLLECTION=SatcomNgpExp800160 WEAVIATE_EMBEDDING_MODEL=Snowflake/snowflake-arctic-embed-l-v2.0 python -m evaluation.weaviate_eval --run-baseline --run-weaviate --k 5 --generation-mode extractive`
---

### Weaviate connection resilience configuration (2026-11-26, Task 31, experimental/no-live-calls)

**Status/constraints.** Experimental Weaviate retriever only. No live Weaviate calls/evaluations, no ingestion/deletion/collection modification, no production Chroma changes/defaults changed, no `.env` modifications or secrets exposed, no dependency upgrades, no stage/commit/push/deploy.

**Inspected client API (installed, local):** `weaviate-client==4.24.1`. `weaviate.connect_to_weaviate_cloud(cluster_url, auth_credentials, headers=None, additional_config=None, skip_init_checks=False) -> WeaviateClient` supports `additional_config`. `weaviate.classes.init.AdditionalConfig(timeout=Timeout(...))` / `weaviate.config.Timeout` supported; `Timeout` fields in v4.24.1: `query=30`, `insert=90`, `init=2`, `stream=None` (seconds). This supports configurable connection initialization and query timeouts. REST-only fallback is **not** implemented/exposed in this repo; `skip_init_checks` is not exposed (would hide connectivity problems) and gRPC startup checks remain enabled.

**Implementation (justified by 110722Z gRPC DEADLINE_EXCEEDED init failure):**
- `src/rag_core/weaviate_retriever.py`: `WeaviateRetrieverConfig` gained `init_timeout_s` (default `2.0`) and `query_timeout_s` (default `30.0`), preserving weaviate-client v4 defaults. `from_env()` reads `WEAVIATE_INIT_TIMEOUT_S`/`RAG_WEAVIATE_INIT_TIMEOUT_S` and `WEAVIATE_QUERY_TIMEOUT_S`/`RAG_WEAVIATE_QUERY_TIMEOUT_S` with validation. `_connect()` passes `AdditionalConfig(timeout=Timeout(init=..., query=...))` when available; connection errors still raise `WeaviateRetrievalError` with type/message (no silent empty retrieval, no retries). Retrieval debug includes configured timeouts.
- `src/rag_app/config.py`: added `weaviate_init_timeout_s`/`weaviate_query_timeout_s` defaults and env parsing.
- `src/rag_app/service.py`: passes timeout config through to `WeaviateRetrieverConfig`.
- `tests/test_weaviate_retriever.py`: added deterministic mocked tests for timeout config passing, default timeout preservation, connection error propagation as `WeaviateRetrievalError`, env parsing/validation, and service config pass-through.
- `.env.example`, `docs/RUN_LOCAL.md`, `docs/DEPLOYMENT.md`: documented optional experimental timeout env vars.

**Tests run:** `pytest tests/test_weaviate_retriever.py -q` and `pytest tests/test_api.py -q` (see task report for exact results). No live Weaviate calls.

**Remaining connection risks:** gRPC 443 egress/firewall or Weaviate Cloud gRPC instability can still cause DEADLINE_EXCEEDED; increasing init timeout mitigates slow startup but not blocked/unavailable gRPC. No bounded retry added (risk of masking persistent failures / duplicate load). REST-only path not implemented because it requires explicit client-path validation and may change semantics; evaluate separately if gRPC remains unstable after timeout tuning.

**Proposed next evaluation command (do not execute):**
`PYTHONPATH=. LANGFUSE_ENABLED=false WEAVIATE_URL=... WEAVIATE_API_KEY=... WEAVIATE_COLLECTION=SatcomNgpExp800160 WEAVIATE_EMBEDDING_MODEL=Snowflake/snowflake-arctic-embed-l-v2.0 WEAVIATE_INIT_TIMEOUT_S=10 WEAVIATE_QUERY_TIMEOUT_S=30 python -m evaluation.weaviate_eval --run-baseline --run-weaviate --k 5 --generation-mode extractive`
---

### Controlled Weaviate connection/retrieval evaluation (2026-11-26, Task 32, blocked)

**Status/constraints.** No live evaluation run completed. No ingestion/deletion/recreation/rename/modification of collection. No chunking/embedding/alpha/dataset/generation/retrieval architecture changes. No production Chroma changes. No retries/fallback added. No dependency changes. No stage/commit/push/deploy. Existing local files/results/uncommitted work preserved. No credentials exposed.

**Pre-flight (performed):**
- Repo/branch: `main`; worktree dirty with pre-existing uncommitted/untracked work (preserved).
- `weaviate-client==4.24.1` confirmed.
- `.env` contains WEAVIATE_URL/WEAVIATE_API_KEY keys (values not inspected/printed); env presence confirmed after safe load.
- Lightweight Weaviate Cloud connection/collection metadata check attempted with `weaviate.connect_to_weaviate_cloud` default client settings (no Task 31 timeout override applied in this pre-flight script). Result: **failed before collection/count verification**.

**Pre-flight error (exact type):** `WeaviateGRPCUnavailableError` during gRPC health check; gRPC call failed with `DEADLINE_EXCEEDED - Deadline Exceeded` (Weaviate v1.40.5; gRPC 443 health check unavailable). Collection existence and object count could not be confirmed due to connection failure.

**Blocker:** Task 32 requires confirming `SatcomNgpExp800160` exists and contains exactly 61 objects before evaluation; if missing/count differs, stop. Connection failure prevented count verification. Per instructions, stopped and reported; no unrequested workaround applied, no evaluation executed.

**Safety confirmation:** No collection modified/repaired/recreated; no eval run; no secrets exposed.

---

### Controlled Weaviate retrieval strategy experiment (2026-11-26, Task 34, isolated/no-production-change)

**Status/constraints.** Isolated experiment only. No collection creation/ingestion/deletion/recreation/modification. Existing collection `SatcomNgpExp800160` reused (61 objects verified in pre-flight). No production Chroma retriever/default changes. No embedding model/chunking/dataset/generation-model changes. No dependency upgrades. No retries/fallback behavior added. No concurrent runs. No stage/commit/push/deploy. No credentials exposed. Existing uncommitted/untracked work preserved.

**Pre-flight (app config/connection path, init_timeout_s=10, query_timeout_s=30):** `weaviate-client==4.24.1`; connection ready=True; collection `SatcomNgpExp800160` exists; object count=61; client closed cleanly. (Task 32 pre-flight with default connection settings failed gRPC DEADLINE_EXCEEDED; this pre-flight used Task 31 timeout config - insufficient evidence to attribute improvement solely to timeout.)

**API/semantics (weaviate-client 4.24.1):** `collection.query.near_text(query, limit, return_metadata)` used for **dense-only** (server-side text2vec-weaviate vectorizer; avoids local embedding model). `collection.query.bm25(query, limit, return_metadata)` for **BM25-only**. `collection.query.hybrid(query, limit, alpha, return_metadata)` for **hybrid**. alpha=0/1 not used as pure-mode substitutes; explicit methods preferred. Metadata requested: score+distance where supported; BM25 may return score, near_text distance, hybrid score/distance depending on server response.

**Implementation (experimental, backward-compatible):** `WeaviateRetriever.retrieve(..., search_mode, alpha)` supports dense/bm25/hybrid; `WeaviateRetrieverConfig` gained `search_mode`/`alpha` defaults (hybrid, 0.5). `AppConfig` gained `weaviate_alpha`/`weaviate_search_mode` defaults (0.5/hybrid). `RAGService` parses/passes experimental weaviate mode/alpha; defaults unchanged. `QueryResponse.retrieval_mode` broadened to `str` to allow experimental labels (e.g., `weaviate_dense`, `weaviate_hybrid_a0.25`). `evaluation/weaviate_eval.py` supports `--weaviate-modes` and per-run incremental save; stops remaining runs on gRPC/connection failure and preserves partial results.

**Tests:** `pytest tests/test_weaviate_retriever.py -q` => 37 passed. `pytest tests/test_api.py -q` => 17 passed, 4 warnings (existing Pydantic/Chroma deprecation warnings). Added deterministic tests for dense/bm25/hybrid alpha 0.25/0.5/0.75, invalid mode/alpha, default preservation, and Chroma/service behavior; updated two mock retrievers for new retrieve signature.

**Run:** `evaluation/results/weaviate_experiment/20261126T114455Z/` (k=5, extractive, 15-question CI dataset, same collection/model, LANGFUSE_ENABLED=false, WEAVIATE_INIT_TIMEOUT_S=10, WEAVIATE_QUERY_TIMEOUT_S=30). Sequential runs: baseline, dense, bm25, hybrid 0.25, hybrid 0.5 completed (failure_rate 0). `weaviate_hybrid_a0.75` failed with gRPC/connection `DEADLINE_EXCEEDED` for all 15 questions; script stopped early and preserved partial result.

| Run | Recall@5 | Prec@5 | MRR@5 | EvCov | SuppAns | CitSup | AbstAcc | InvCit | p50ms | p95ms | Fail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_hybrid_rrf | 0.9236 | 0.3333 | 0.8472 | 0.9236 | 0.8333 | 0.5484 | 1.0000 | 0 | 24.12 | 3612.31 | 0.0 |
| weaviate_dense | 0.9444 | 0.3167 | 0.7639 | 0.9444 | 0.6667 | 0.5000 | 1.0000 | 0 | 397.58 | 2052.20 | 0.0 |
| weaviate_bm25 | 0.9236 | 0.3000 | 0.7222 | 0.9236 | 0.7500 | 0.5000 | 0.6667 | 0 | 299.19 | 1098.04 | 0.0 |
| weaviate_hybrid_a0.25 | 0.9236 | 0.3167 | 0.7778 | 0.9236 | 0.7500 | 0.5000 | 0.6667 | 0 | 386.52 | 1159.86 | 0.0 |
| weaviate_hybrid (a0.5) | 0.9236 | 0.3167 | 0.7778 | 0.9236 | 0.6667 | 0.4545 | 0.6667 | 0 | 390.67 | 1233.25 | 0.0 |
| weaviate_hybrid_a0.75 | 0.0000* | 0.0000* | 0.0000* | 0.0000* | 0.0000* | null* | 0.0000* | 0 | 40.95 | 130550.34 | 1.0 |

`*` hybrid_a0.75 is infrastructure-failed (gRPC DEADLINE_EXCEEDED on all 15 questions); exclude from retrieval-quality comparison.

**Question-level findings (successful runs):** Q01 retrieved in all modes (dense lower MRR). Q03 good in all. Q07 retrieved (R@5/MRR 1.0) in all Weaviate modes but answer support/citation degraded vs baseline (generation/citation-selection issue, not recall). Q08 weak retrieval/evidence in all modes including baseline. Q12: dense best (ans/cit supported), bm25/hybrid0.25 partial, hybrid0.5 answer unsupported despite evidence. Q14: bm25/hybrid0.25 stable; dense/hybrid0.5 show citation/answer support degradation despite evidence retrieval. UNANS-11: dense abstained correctly (abstention 1.0 overall for dense), bm25 and hybrids incorrectly answered (abstention regression).

**Interpretation:** Dense-only had best recall/evidence coverage and fixed UNANS-11 abstention, but lower MRR and mixed answer/citation support on Q07/Q14. BM25/hybrid 0.25 improved supported-answer rate over dense/hybrid 0.5 but regressed UNANS-11 abstention. Hybrid 0.25 and 0.5 had same recall/MRR but 0.25 had better supported/citation rates in this run. No single mode dominates on all quality axes; retrieval ranking/abstention behavior differs by mode and answer/citation selection remains a weakness when evidence is present. Hybrid 0.75 not assessable due to infrastructure failure.

**Limitations:** single run per mode; hybrid 0.75 inconclusive due to gRPC failures; extractive generation and citation selection may confound supported/citation metrics; abstention guard is score/lexical-heuristic based and sensitive to retrieval score distributions; no live repeat performed.

---

### Weaviate score distribution and abstention diagnosis (2026-11-26, Task 35, read-only)

**Status/constraints.** Read-only diagnostic. No live Weaviate calls, no new evaluation, no retrieval/threshold/code changes, no collection modification, no Chroma/production default changes, no dependency upgrades, no stage/commit/push/deploy. Analyzed `evaluation/results/weaviate_experiment/20261126T114455Z/`.

**Score semantics (observed in artifacts, weaviate-client 4.24.1 / Weaviate v4 Cloud):** `dense` (near_text) populated `distance` (smaller=closer) and `score=0.0` in metadata; `bm25` populated `score` (BM25F score, larger=better), `distance=None`; `hybrid` populated `score` (fusion score, larger=better), `distance=None`. Baseline Chroma hybrid RRF populated `rrf_score`, `bm25_score`, `dense_distance`, and hit `distance`. Scores are **not directly comparable** across modes.

**UNANS-11 abstention trace (confirmed):** Baseline abstained: `best_distance 1.3031 > hard limit 1.3`. Dense abstained: `best_distance 0.7370` (distance present) but `lexical support 0.25 < 0.5`. BM25/hybrid_a0.25/hybrid_a0.5 answered incorrectly: `best_distance None` because no hit distance populated; pipeline distance-based abstention skipped; `token_support=0.5`, `selected_lexical_support=0.5`, answerability guard `selected_lexical_support < 0.5` was false (boundary equality), so response answered. Root cause: Chroma-distance-based hard/soft abstention and selected-lexical guard threshold were calibrated for Chroma distance semantics; Weaviate BM25/hybrid provide score-not-distance, so distance guard does not fire. This is a score-semantics/threshold-portability issue, not a single-question threshold-tuning conclusion.

**Q07/Q12/Q14 (confirmed from artifacts):** Expected evidence was retrieved (Recall@5=1.0, evidence coverage 1.0 in all successful Weaviate modes). Failures were post-retrieval: citation/selection included non-evidence chunks alongside evidence chunks (e.g., Q07 added `page_1::chunk_2`; Q12 hybrid 0.5 added `page_5::chunk_2`, `page_14::chunk_0`; Q14 added `page_13::chunk_0`/`page_12::chunk_2` in some modes), reducing answer/citation support. Retrieval ranking was adequate; generation/citation selection is the primary weakness for these cases.

**Q08 (confirmed):** Partial evidence missing across all modes including baseline (EvCov 0.3333). Only EXP001-Q08-E01 (page 5, clause 3.4.1) consistently covered; E02/E03 (clauses 3.4.2/3.4.3, page 6) not satisfied by top-5 retrieved/cited chunks. Indicates retrieval gap for multi-evidence question, not only Weaviate.

**Recommended next isolated experiment (no production change):** (1) Instrument/normalize Weaviate abstention inputs: for BM25/hybrid, expose/use comparable distance-or-score calibration signal (or disable distance-only hard gate when distance absent and rely on calibrated score+lexical guard) in an experimental branch only; evaluate on full 15-question set plus additional held-out unanswerables before any threshold change. (2) Separately test citation-selection tightening for already-retrieved-evidence cases (Q07/Q12/Q14) without changing retrieval.

**Limitations:** Single run; artifacts lack full chunk text; hybrid 0.75 quality inconclusive; no live re-check.

---

### Weaviate-specific abstention policy separation (2026-11-26, Task 36, experimental/no-calibration)

**Status/constraints.** Experimental, opt-in only. No live Weaviate calls, no new evaluation runs, no collection changes, no ChromaDB changes, no embedding/chunking/citation-selection changes, no dependency upgrades, no stage/commit/push/deploy. Existing uncommitted/untracked work preserved.

**Confirmed root cause (from Task 35 artifacts/code):** `pipeline.AbstentionPolicy` hard/soft gates depend on `best_distance = min(hit.distance where distance is not None)`. Weaviate BM25/hybrid results provide score (higher=better) but no distance in observed artifacts, so `best_distance=None` and distance gates are skipped. Pipeline lexical gate and service answerability guard (`selected_lexical_support < 0.5`) allowed UNANS-11 to pass at boundary (token_support=0.5, selected_lexical_support=0.5). Dense returns distance and abstained via lexical support.

**Implementation (opt-in, default off):**
- `src/rag_core/pipeline.py`: `AbstentionPolicy` extended with `missing_distance_behavior` (`skip|reject|score`, default `skip`) and `min_retrieval_score` (default None). Default `skip` preserves existing behavior exactly. `score` mode requires explicit calibrated threshold and uses best available retrieval score (higher=better) only when distance is missing; `reject` explicitly abstains on missing distance. Added `_best_retrieval_score` helper (checks rerank/rrf/weaviate_score/_weaviate_score/score). Distance-present paths unchanged.
- `src/rag_app/config.py`: added `weaviate_abstention_experimental` (default False), `weaviate_abstention_missing_distance_behavior` (default skip), `weaviate_abstention_min_retrieval_score` (default None).
- `src/rag_app/service.py`: applies experimental policy only when `backend==weaviate` and `weaviate_abstention_experimental` is True; Chroma path unchanged.
- `tests/test_weaviate_retriever.py`: added deterministic tests for default policy unchanged, experimental flag default off, Chroma unaffected, dense distance behavior, BM25/hybrid missing distance default/experimental reject/score/boundary/missing-score, and answerable not rejected merely due to missing distance.

**Calibration:** Not justified with current evidence. CI dataset has only 3 unanswerables (UNANS-01/03/11) and 12 answerables; Task 34 shows hybrid scores overlap answerable/unanswerable (UNANS-11 score 0.867-0.934 within answerable range), while BM25 shows separation in this tiny sample but insufficient for reliable threshold. No threshold tuned against UNANS-11 alone. Any score threshold requires a separately documented held-out unanswerable set with trustworthy labels and per-mode score distributions.

**Tests:** `pytest tests/test_weaviate_retriever.py -q` => 47 passed. `pytest tests/test_api.py -q` => 17 passed, 4 warnings (existing).

**Remaining risks/limitations:** No calibrated Weaviate score threshold; experimental score mode assumes higher=better score semantics and must not be used across incomparable modes without calibration; service guard boundary (`< 0.5`) remains unchanged; no live eval performed.

---

### Held-out unanswerable evaluation dataset design (2026-11-26, Task 37, draft/review-only)

**Status/constraints.** Design/documentation only. No retrieval/abstention logic or threshold changes. No experimental Weaviate abstention enabled. No live Weaviate calls/evaluations. No CI dataset changes. No Chroma/production collection modification. No dependency upgrades. No stage/commit/push/deploy. Existing uncommitted/untracked work preserved.

**Inspected:** `evaluation/ci_golden_dataset.json`, `evaluation/ci_eval.py`, `src/rag_core/pipeline.py`, `src/rag_app/service.py`, `evaluation/weaviate_eval.py`, Task 34/35 artifacts, `docs/ENGINEERING_LOG.md`, local SATCOM NGP text extraction (`data/satcom-ngp.pdf`, 15 pages).

**Deliverables (draft, not validated ground truth):**
- `evaluation/drafts/heldout_unanswerable_candidates_draft.json` — 12 candidate questions (HOLDOUT-U01..U12) across categories: specific value not provided, specific parameter not provided, unsupported detail, outside scope, misleading premise, combined supported+unsupported, unsupported comparison, future/external info. Each candidate includes proposed label, reason, evidence inspected, partial-answerability note, and human-review risk.
- `evaluation/drafts/heldout_unanswerable_review_report.md` — methodology, label semantics, candidate table, leakage prevention, feasibility/human-review requirements, recommendation.

**Leakage prevention:** Proposed split — dev/calibration (U01,U03,U05,U07,U09,U11), held-out validation (U02,U04,U06,U08,U10,U12), existing CI regression (`ci_golden_dataset.json`) unchanged. No question reused across splits; no threshold tuning on held-out validation; UNANS-11 remains in CI only.

**Readiness:** Not ready for calibration. All candidates require human review against original PDF (including tables/annexes), partial-answer scoring policy decisions (U07/U08), and validated labeling with reviewer metadata/evidence before any calibration or held-out reporting.

---

### Production migration to Weaviate for Render Free 512 MB (2026-11-26, Task 38, prepared/no-deploy)

**Status/constraints.** Prepared and validated locally/preflight only. No deployment/push/stage/commit. No collection creation/deletion/reset/re-ingestion. No CI golden dataset changes. No threshold/experimental abstention changes (Task 36 remains default off). No dependency upgrades beyond adding `requirements.weaviate.runtime.txt` for Weaviate runtime image. No secrets exposed. Existing uncommitted/untracked work preserved. Chroma data and historical evaluation results preserved.

**Architecture decision (recorded):** "Chroma produced better results in some offline quality metrics. Weaviate was selected for production because the available Render Free backend has a 512 MB memory limit, and moving vector storage and hosted embedding outside the application is necessary to pursue deployment within that constraint. Retrieval quality is an acknowledged trade-off to be improved through targeted post-deployment experiments."

**Production target:** FastAPI on Render Free (512 MB) + Weaviate Cloud (`SatcomNgpExp800160`, `Snowflake/snowflake-arctic-embed-l-v2.0`, init 10s/query 30s) + Gemini generation (existing config) + Vercel React/Vite frontend. Chroma remains non-default/offline comparison path; local `chroma_db_production/` and historical eval results untouched.

**Changes:**
- `render.yaml`: switched to Weaviate production path — build `requirements.weaviate.runtime.txt`; `RAG_RETRIEVER_BACKEND=weaviate`; Weaviate env vars/secrets; removed torch/Chroma runtime assumptions.
- `Dockerfile`: lightweight Weaviate runtime image (no torch/sentence-transformers/chromadb/rank_bm25); `RAG_RETRIEVER_BACKEND=weaviate` default in container env.
- `requirements.weaviate.runtime.txt`: minimal runtime for Weaviate production path.
- `frontend/src/App.jsx`: localhost fallback only in `import.meta.env.DEV`; production build requires `VITE_API_BASE_URL`; health/query guarded with clear error.
- `frontend/.env.example`: added VITE_API_BASE_URL example.
- `docs/DEPLOYMENT.md`: updated architecture decision, Render checklist, env vars, deployment steps, preflight, constraints.
- `docs/RUN_LOCAL.md`: clarified local default vs Render production selection.

**Preflight (read-only, no secrets, Task 38):** Weaviate ready=true, collection exists, object_count=61, smoke retrieve returned 1 hit with text. No collection modification.

**Tests:** `pytest tests/test_weaviate_retriever.py -q` => 47 passed. `pytest tests/test_api.py -q` => 17 passed, 4 warnings (existing Pydantic/Chroma deprecation). Existing import-isolation tests in weaviate suite pass (Weaviate backend path avoids heavy local stack in subprocess checks; numpy may be imported by third-party deps depending on env as previously documented).

**Remaining memory/runtime risks:** 512 MB fit not proven without actual Render deployment measurement. `requirements.weaviate.runtime.txt` still includes langfuse/google-generativeai/pypdf and transitive deps; import-audit in production image needed. /ingest endpoint remains Chroma-oriented (fail-closed by default) - calling it in Weaviate production would be inappropriate/heavy. Render cold start, Weaviate gRPC/REST latency, and OOM must be validated after deploy.

**Blockers to deployment:** explicit deploy approval required; Render/Vercel account access and secrets/env setup required; actual Vercel origin needed for RAG_CORS_ORIGINS; public smoke tests not run.

---

### Final read-only production deployment readiness audit (2026-11-26, Task 39, docs-only/no-code-changes)

**Audit scope:** Read-only review of Task 38 Weaviate production migration before publish/deploy. No application code/config/runtime file changes. No live deployment. No full evaluation suite. No collection modification. No dependency upgrades. No secrets exposed.

**Files inspected (primary):** `render.yaml`, `Dockerfile`, `requirements.weaviate.runtime.txt`, `frontend/src/App.jsx`, `frontend/.env.example`, `docs/DEPLOYMENT.md`, `docs/RUN_LOCAL.md`, `docs/ENGINEERING_LOG.md`, `src/rag_app/config.py`, `src/rag_app/service.py`, `src/rag_core/weaviate_retriever.py`, `src/rag_core/pipeline.py`, `backend/main.py`, `src/rag_app/schemas.py`, `.env.example`, `docker-compose.yml`, git status/diff.

**Task 38 change verification (git diff/status):** Task 38 touched `render.yaml`, `Dockerfile`, `docs/DEPLOYMENT.md`, `docs/ENGINEERING_LOG.md`, `docs/RUN_LOCAL.md`, `frontend/src/App.jsx` and added untracked `frontend/.env.example`, `requirements.weaviate.runtime.txt`. Worktree also contains many pre-existing dirty/untracked files from prior tasks; they were not attributed to Task 38.

**Configuration verification results (confirmed by code inspection):**
1. `RAG_RETRIEVER_BACKEND=weaviate` selects Weaviate path in `AppConfig.from_env`/`RAGService._retrieve`/`query`; Weaviate query path does not initialize local Chroma/BM25/SentenceTransformer.
2. Missing Weaviate config fails clearly (`ValueError` in `service.weaviate_retriever` for missing URL/API key/collection; `WeaviateConfigError` in config helper paths).
3. No silent fallback to Chroma when Weaviate selected; `_retrieve` returns `self.weaviate_retriever.retrieve(...)`.
4. `WEAVIATE_URL`/`WEAVIATE_API_KEY` read via env in config/retriever (also supports RAG_* aliases).
5. `render.yaml` sets `WEAVIATE_COLLECTION=SatcomNgpExp800160`; config default matches.
6. Hosted embedding model set to `Snowflake/snowflake-arctic-embed-l-v2.0` in render/config.
7. `WEAVIATE_INIT_TIMEOUT_S`/`WEAVIATE_QUERY_TIMEOUT_S` wired through AppConfig -> service -> WeaviateRetrieverConfig -> `AdditionalConfig(timeout=Timeout(init=..., query=...))`.
8. Gemini config: `RAG_GEMINI_MODEL`, `RAG_GEMINI_TEMPERATURE`, `RAG_GEMINI_MAX_OUTPUT_TOKENS`, `RAG_ALLOW_GEMINI`, API keys via `GOOGLE_API_KEY`/`GEMINI_API_KEY` (used by GeminiGenerator); preserved.
9. **Generation-mode finding:** `RAG_GENERATION_MODE=extractive` sets default request mode to extractive. `RAG_ALLOW_GEMINI=true` only permits Gemini when a request explicitly asks for `mode=gemini`; it does not make extractive requests call Gemini. Therefore the reported combination is compatible/config-consistent, but Gemini will not be used unless requested per query. This must be reflected in deployment/operator expectations.
10. CORS: `_parse_cors_origins` defaults to local dev origins when unset/empty; explicit `*` only when configured; credentials disabled for wildcard. `render.yaml` includes `RAG_CORS_ORIGINS` placeholder to be replaced with actual Vercel origin.
11. `RAG_INGEST_TOKEN` unset by default; `/ingest` fail-closed (403) unless token configured and matched. In Weaviate runtime image, /ingest would be inappropriate/heavy (Chroma-oriented) — keep disabled.
12. `/healthz` lightweight. `/health` uses `RAGService.health()` which is lightweight and does not trigger SentenceTransformer/Chroma init/recovery/embedding generation; Weaviate backend reports local index unknown/not initialized.

**Runtime dependency/import findings:**
- `requirements.weaviate.runtime.txt` contains fastapi, uvicorn, pydantic, python-dotenv, httpx, weaviate-client, google-generativeai, langfuse, pypdf; no torch/sentence-transformers/transformers/chromadb/rank_bm25 package entries (comment text mentions them, but packages not required).
- Import-isolation tests passed in existing suite (subprocess checks for Weaviate backend path). Numpy may still be imported by third-party deps depending on env.
- **Not measured:** actual Render 512 MB RSS, cold-start memory peak, container image size in Render build environment.

**Deployment configuration findings:** Render build/start commands match FastAPI app (`uvicorn backend.main:app --host 0.0.0.0 --port $PORT`, `/healthz`). Dockerfile uses same lightweight runtime. Frontend has production guard requiring `VITE_API_BASE_URL` at build time; `.env.example` present. `docker-compose.yml` is legacy/dev-oriented and may not reflect Weaviate production split; treat as non-canonical for Render deployment.

**Confirmed blockers:** no explicit deploy approval; no Render/Vercel credentials/access; actual Vercel production origin unknown for CORS; 512 MB fit unproven on Render; Weaviate gRPC reliability under Render networking unproven.

**Non-blocking risks / post-deploy quality:** Weaviate retrieval-quality trade-offs vs Chroma (Task 34); UNANS-11 abstention regression in BM25/hybrid (Tasks 34/35); Q07/Q12/Q14 citation-selection issues; Q08 multi-evidence coverage gap; Task 36 experimental abstention disabled and uncalibrated; cold starts/spin-down; Gemini availability/quota when requested.

**Exact outstanding owner decisions:** approve deploy; provide Render/Vercel access; provide/confirm Vercel origin for CORS; confirm whether Gemini should be used by default in production (currently extractive default) or remain optional; confirm if /ingest must remain disabled.

**Test commands/results:**
- `PYTHONPATH=. python -m pytest tests/test_weaviate_retriever.py tests/test_api.py -q` -> **63 passed, 1 failed, 4 warnings**. Failure: `tests/test_weaviate_retriever.py::test_missing_weaviate_credentials_fail_clearly` due to ambient `.env`/dotenv credentials in workspace; passes when Weaviate env vars cleared (`1 passed`). Treated as environment-sensitive test, not a Task 38/39 regression. Warnings: existing Pydantic/Chroma deprecation.
- Static check: `requirements.weaviate.runtime.txt` contains no heavy runtime package entries (torch/sentence-transformers/transformers/chromadb/rank_bm25).

**Explicit confirmation:** No application code was modified. No Python/JS/Dockerfile/requirements/render.yaml/app config files modified by Task 39. Documentation updates only (`docs/ENGINEERING_LOG.md`, `docs/DEPLOYMENT.md`). No git add/stage/commit/push/deploy performed.

---

### Exact Weaviate production publish manifest (2026-11-26, Task 40, manifest-only/no-stage-no-deploy)

**Status/constraints.** Manifest preparation only. No app code/config changes. No staging/commit/push/deploy. No git add ., no reset/checkout/clean/stash/revert. No secret exposure. Existing dirty/untracked work preserved.

**Repo state:** branch `main`, HEAD `e478a1b` ("Harden deployment: protect ingest, expose boost flag, lock down CORS"), origin `https://github.com/Sivasindhuja/RAG` (fetch/push). 207 tracked changes + 28 untracked files at audit time; no cached staged changes.

**Publish-dependency trace (Weaviate production path):** `backend/main.py` -> `src/rag_app/{config,service,schemas,metrics,observability,langfuse_tracing,prompts,__init__}` -> `src/rag_core/{retrieval_types,weaviate_retriever,pipeline,generation,index,retrieval,chunker,document,evaluation,__init__}` plus runtime/deploy files `requirements.weaviate.runtime.txt`, `render.yaml`, `Dockerfile`, frontend `frontend/src/App.jsx`, `frontend/.env.example`, and validation tests/fixtures. Unchanged tracked files already at HEAD are not in publish diff.

**Proposed exact publish allowlist (dirty/untracked files only; review before staging individually):**
- INCLUDE: `.dockerignore`, `.gitignore`, `Dockerfile`, `docker-compose.yml`, `frontend/src/App.jsx`, `render.yaml`, `src/rag_app/config.py`, `src/rag_app/schemas.py`, `src/rag_app/service.py`, `src/rag_core/__init__.py`, `src/rag_core/evaluation.py`, `src/rag_core/index.py`, `src/rag_core/pipeline.py`, `src/rag_core/retrieval.py`, `frontend/.env.example`, `requirements.weaviate.runtime.txt`, `src/rag_core/retrieval_types.py`, `src/rag_core/weaviate_retriever.py`.
- INCLUDE-DOCS: `README.md`, `docs/DEPLOYMENT.md`, `docs/ENGINEERING_LOG.md`, `docs/RUN_LOCAL.md`.
- INCLUDE-TEST: `tests/test_api.py`, `tests/test_citation_metadata.py`, `tests/test_fixed_chunker.py`, `tests/test_rag_evaluation.py`, `tests/fixtures/exp001/*`, `tests/test_weaviate_retriever.py`.

**Explicit exclusions/reviews:** `.env`/secrets/local data EXCLUDE-LOCAL; `.agents/` EXCLUDE-LOCAL; `evaluation/drafts/` REVIEW (Task 37 drafts); `evaluation/weaviate_eval.py`, `scripts/weaviate_ingest.py`, `requirements.weaviate.txt` REVIEW (useful ops/experiment, not minimal runtime); `requirements.weaviate.runtime.proposed.txt` EXCLUDE-HISTORICAL; all pre-existing deletions/dirty docs/tests/src unrelated to Weaviate runtime REVIEW; `todo.txt` deletion preserved.

**Validation:** `pytest tests/test_weaviate_retriever.py tests/test_api.py -q` => 63 passed, 1 failed, 4 warnings; failure `test_missing_weaviate_credentials_fail_clearly` is environment-sensitive with workspace `.env` present (passes when Weaviate env vars cleared). No ingestion/deletion/live eval/Weaviate mutation performed.

**Blockers/owner actions:** explicit approval to stage exact allowlist; confirm Vercel origin for `RAG_CORS_ORIGINS`; confirm Render/Vercel credentials/access; confirm Gemini default-mode expectation; confirm `/ingest` remains disabled; confirm handling of REVIEW files/deletions before publish.

**Confirmation:** No files staged, committed, pushed, or deployed. No app code/config modified by Task 40.


---

### Task 41 - Publish manifest resolution (2026-11-26, manifest-only/no-stage-no-deploy)

**Status/constraints.** Audit/resolution only. No app code/config behavior changes. No staging/commit/push/deploy. No git add ., no reset/checkout/clean/stash/revert. No secret exposure. All existing work preserved.

**Inventory:** 194 tracked changes + 28 untracked files (207+28). No staged changes. HEAD `e478a1b` on `main`, origin `https://github.com/Sivasindhuja/RAG`.

**Resolved allowlist (smallest complete Weaviate production publish; dirty/untracked only; unchanged HEAD runtime files not listed):**
- **INCLUDE (runtime/deploy/config):** `Dockerfile`, `render.yaml`, `requirements.weaviate.runtime.txt`, `frontend/src/App.jsx`, `frontend/.env.example`, `src/rag_app/config.py`, `src/rag_app/schemas.py`, `src/rag_app/service.py`, `src/rag_core/__init__.py`, `src/rag_core/retrieval_types.py`, `src/rag_core/weaviate_retriever.py`, `src/rag_core/pipeline.py`, `src/rag_core/index.py`, `src/rag_core/retrieval.py`, `src/rag_core/evaluation.py`.
- **INCLUDE (reviewed hygiene/docs):** `.dockerignore`, `.gitignore`, `.env.example` (template only; no secrets), `docker-compose.yml` (dev parity; thread env vars only), `README.md`, `docs/DEPLOYMENT.md`, `docs/RUN_LOCAL.md`, `docs/ENGINEERING_LOG.md`.
- **INCLUDE-TEST:** `tests/test_api.py`, `tests/test_citation_metadata.py`, `tests/test_fixed_chunker.py`, `tests/test_rag_evaluation.py`, `tests/test_weaviate_retriever.py`, `tests/fixtures/exp001/chunks.json`, `tests/fixtures/exp001/exp001_ground_truth.json`, `tests/fixtures/exp001/exp001_questions.csv`.

**Resolved exclusions (do not stage in this publish):** `.agents/**`, `evaluation/drafts/*`, `evaluation/weaviate_eval.py`, `scripts/weaviate_ingest.py`, `requirements.weaviate.txt`, `requirements.weaviate.runtime.proposed.txt`, `todo.txt` deletion, all `experiments/results/**` and historical experiment file deletions, `chroma_db*` local data, caches/build outputs, pre-existing legacy deletions under `src/chunking`, `src/config`, `src/generation/models.py`, `src/memory`, `src/metrics`, `src/observability`, `src/restart.py`, `src/retrieval/*`, `api/index.py`, `.github/workflows/rag_eval.yml`, `tests/eval_pipeline.py`, `tests/golden_dataset.csv`, `tests/test_exp001.py`, and dirty docs not required for production (`docs/BACKLOG.md`, `CHUNKING_STRATEGY.md`, `CI_EVALUATION.md`, `EXP-*`, `M4_*`, `PRODUCTION_SLICE.md`, `PROJECT_BRIEF.md`) and `evaluation/ci_eval.py` modification - keep local, owner review for separate cleanup/CI PR.

**Test diagnosis:** `test_missing_weaviate_credentials_fail_clearly` fails when workspace `.env` provides `WEAVIATE_URL/WEAVIATE_API_KEY/WEAVIATE_COLLECTION` because `WeaviateRetrieverConfig.from_env()` reads `os.getenv` and test does not isolate env. Minimal fix (not implemented): add `monkeypatch` and `monkeypatch.delenv`/`setenv(..., "")` for `WEAVIATE_URL`, `WEAVIATE_API_KEY`, `WEAVIATE_COLLECTION`, `RAG_WEAVIATE_URL`, `RAG_WEAVIATE_API_KEY`, `RAG_WEAVIATE_COLLECTION` before calling `from_env()`. No secrets exposed.

**Blockers unchanged:** owner approval for exact staging; Vercel origin for `RAG_CORS_ORIGINS`; Render/Vercel secrets/access; Gemini default-mode confirmation; `/ingest` disabled; Render Free 512MB unverified; Weaviate gRPC/quality trade-offs known.


---

### Task 42 - Weaviate credential test isolation and manifest refinement (2026-11-26, no-stage/no-deploy)

**Status/constraints.** Targeted validation only. No staging/commit/push/deploy. No `git add .`/`-A`, no reset/clean/stash/revert. No secrets exposed. No production behavior changes.

**Test isolation:** `tests/test_weaviate_retriever.py::test_missing_weaviate_credentials_fail_clearly` now uses `monkeypatch` to clear `WEAVIATE_URL`, `WEAVIATE_API_KEY`, `WEAVIATE_COLLECTION`, `RAG_WEAVIATE_URL`, `RAG_WEAVIATE_API_KEY`, `RAG_WEAVIATE_COLLECTION` before calling `WeaviateRetrieverConfig.from_env()`. Assertion unchanged.

**Validation:** `PYTHONPATH=. python -m pytest tests/test_weaviate_retriever.py tests/test_api.py -q` => **64 passed, 0 failed, 0 skipped, 4 warnings** (62.71s). Warnings unchanged: PydanticDeprecatedSince211 from `chromadb/types.py` in real-index smoke test.

**Manifest refinement:** Removed `docker-compose.yml` from minimal production publish allowlist (local dev convenience; no concrete Render/Vercel production dependency). Inspected diffs: `src/rag_core/index.py` (lazy chromadb + retrieval_types; keep), `src/rag_core/retrieval.py` (lazy rank_bm25 + retrieval_types; keep), `src/rag_core/evaluation.py` (retrieval_types import; keep), `.gitignore` (hygiene; keep), `.dockerignore` (deployment build-context hygiene; keep).

**Refined allowlist (31 files):** `.dockerignore`, `.gitignore`, `.env.example`, `Dockerfile`, `render.yaml`, `requirements.weaviate.runtime.txt`, `frontend/src/App.jsx`, `frontend/.env.example`, `src/rag_app/config.py`, `src/rag_app/schemas.py`, `src/rag_app/service.py`, `src/rag_core/__init__.py`, `src/rag_core/retrieval_types.py`, `src/rag_core/weaviate_retriever.py`, `src/rag_core/pipeline.py`, `src/rag_core/index.py`, `src/rag_core/retrieval.py`, `src/rag_core/evaluation.py`, `tests/test_api.py`, `tests/test_citation_metadata.py`, `tests/test_fixed_chunker.py`, `tests/test_rag_evaluation.py`, `tests/test_weaviate_retriever.py`, `tests/fixtures/exp001/*` (3 files), `README.md`, `docs/DEPLOYMENT.md`, `docs/RUN_LOCAL.md`, `docs/ENGINEERING_LOG.md`.

**Blockers unchanged:** owner approval for exact staging; actual Vercel origin for `RAG_CORS_ORIGINS`; Render/Vercel access/secrets; Gemini default-mode confirmation; `RAG_INGEST_TOKEN` unset; Render Free 512MB fit and Weaviate gRPC reliability unverified until smoke/memory evidence.


---

### Task 43 - Gemini default grounded answer generation (2026-11-26, no-stage/no-deploy)

**Status/constraints.** Local implementation only. No staging/commit/push/deploy. No `git add .`/`-A`, no reset/clean/stash/revert. No secrets exposed. No Weaviate retrieval ranking/chunking/threshold changes.

**Request flow (after change):** API request -> `QueryRequest.mode` defaults to `gemini` (or explicit `extractive`/`gemini`) -> `RAGService.query` defaults to config `RAG_GENERATION_MODE` (`gemini`) -> retrieve evidence via configured backend (Weaviate/Chroma) -> select/grounded evidence and abstention guard in `GroundedPipeline` -> if insufficient evidence, abstain -> if mode gemini, `GeminiGenerator.generate` receives question + selected statements/citations only; app validates claim source IDs and resolves citations to retrieved chunks -> on Gemini disabled/unavailable/error, explicit grounded extractive fallback (`fallback_to_extractive=true`, status/errors recorded), never ungrounded/fabricated answer.

**Files changed:** `src/rag_app/config.py`, `src/rag_app/schemas.py`, `src/rag_app/service.py`, `tests/test_api.py`, `.env.example`, `render.yaml`, `README.md`, `docs/DEPLOYMENT.md`, `docs/ENGINEERING_LOG.md`.

**Validation:** `PYTHONPATH=. python -m pytest tests/test_weaviate_retriever.py tests/test_api.py tests/test_citation_metadata.py tests/test_rag_evaluation.py -q` => **86 passed, 0 failed, 0 skipped, 31 warnings** (28.19s). Existing chromadb Pydantic deprecation warnings unchanged. During first run, one new Task 43 test was corrected for lexical-support guard (no assertion weakening; test data adjusted to be valid).

**Remaining limitations/blockers:** real Gemini API key/quota/model not exercised in tests (mocked); Render/Vercel not deployed; Render Free 512MB fit and Weaviate gRPC reliability unverified; owner must set Vercel origin, secrets/access, and confirm deploy.


---

### Task 44 - Final implementation review and publish manifest reconciliation (2026-11-26, no-stage/no-deploy)

**Status/constraints.** Review/reconciliation only. No app code behavior changes. No staging/commit/push/deploy. No git add/reset/clean/stash/revert. No secrets exposed.

**Task 43 code review:** inspected diffs in `src/rag_app/config.py`, `src/rag_app/schemas.py`, `src/rag_app/service.py`, `tests/test_api.py`, `.env.example`, `render.yaml`, `README.md`, `docs/DEPLOYMENT.md`, `docs/ENGINEERING_LOG.md` and relevant implementation (`RAGService.query/_apply_gemini`, `GeminiGenerator.generate`, `assemble_generated_answer`). Confirmed Gemini default for omitted mode, explicit extractive supported, schema Literal rejects invalid modes, Gemini receives selected evidence only, citations app-resolved against retrieved chunks, insufficient evidence abstains, Gemini disabled/unavailable/error falls back safely to grounded extractive with explicit status/errors. No concrete production defect found.

**Manifest reconciliation:** Task 42 text reported 31 files but enumerated 30; corrected exact publish allowlist count is 30. `docker-compose.yml` remains excluded. All 30 files verified present.

**Validation:** not re-run (no code changes). Last required result: 86 passed, 0 failed, 0 skipped, 31 warnings.

**Repo state:** branch `main`, HEAD `e478a1b`, 207 tracked changes, 28 untracked, 0 staged. Protected/local artifacts preserved (`.env`, `.agents/`, `tests/fixtures/`, `chroma_db_production/`, `todo.txt` deletion).
