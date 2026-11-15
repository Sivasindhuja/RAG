# SATCOM NGP RAG — Project Brief (Authoritative)

**Status:** living, authoritative project direction. Last updated 2026-11-15 (strategy pivot: prioritize end-to-end application completion; component experiments paused).  
**Owner:** engineering.  
**Companion documents:** `docs/ENGINEERING_LOG.md`, `docs/BACKLOG.md`,
`docs/EXP-001.md`–`docs/EXP-008.md`, `docs/EXP-008-citation-metadata-correction.md`,
`docs/PRODUCTION_SLICE.md`, `docs/CHUNKING_STRATEGY.md`.

This document defines the fixed destination for the project. Experiments and
engineering decisions must help us reach it. We do not add complexity without a
measurable reason, and we do not experiment indefinitely.

---

## 1. Ultimate objective

Build a **production-grade, domain-specific RAG application for the ISRO SATCOM
NGP policy document**.

The objective is not a chatbot demo and not a collection of fashionable RAG
techniques. The system must retrieve relevant policy evidence, generate accurate
answers grounded in that evidence, provide verifiable citations, and know when it
cannot answer.

The project has three major phases.

### Phase 1 — Functional RAG application

Deliver a working end-to-end application with:

- PDF ingestion and document processing.
- Appropriate chunking.
- Embedding generation.
- Vector database and indexing.
- Retrieval pipeline.
- Answer generation using **Gemini**.
- Accurate citations linked to the original PDF.
- Citation enforcement and abstention when evidence is insufficient.
- A functional user interface.

A user must be able to ask a question and inspect the exact policy evidence used
to answer it.

### Phase 2 — Production-quality RAG

Improve the system based on observed limitations.

Potential techniques include hybrid retrieval (BM25 + dense), cross-encoder
reranking, better context construction, citation enforcement, improved
abstention, and versioned prompts/configurations.

**These are potential solutions, not mandatory components.** Do not implement
BM25, reranking, query expansion or other techniques because they are popular.
First identify an actual problem, form a hypothesis, design a controlled
experiment, measure the outcome, and introduce the technique only if results
justify it.

### Phase 3 — Evaluation and quality assurance

Build a manually verified golden evaluation dataset of approximately **50–200**
question–answer pairs.

Implement offline evaluation for retrieval quality, answer correctness,
faithfulness, completeness, citation correctness, citation coverage, and
abstention quality.

Integrate evaluation into CI so important quality regressions can prevent changes
from being accepted.

Define thresholds using actual baseline results and evaluation evidence. **Do not
invent arbitrary thresholds.**

### Observability and monitoring

Implement an appropriate observability layer. Track:

- Individual request traces.
- Retrieved chunks, ranking and selection decisions.
- Prompts and model responses (with appropriate secret/data handling).
- Token usage and cost per request.
- P50 and P95 latency.
- Citation coverage.
- Failure rates and quality degradation.

Provide useful monitoring and a way to investigate why a request failed.

---

## 2. Fixed project constraints

These are project-level decisions (recorded as DR-013 in `docs/ENGINEERING_LOG.md`).

| Constraint | Decision |
|---|---|
| **Corpus** | Exactly one source document for the initial application: `data/satcom-ngp.pdf` (ISRO SATCOM NGP policy). Do not expand to multiple PDFs or arbitrary user uploads. |
| **LLM** | **Google Gemini** via the Gemini API. Select an appropriate model based on answer quality, availability, cost and latency; document the selected model and configuration. The local Qwen/Ollama generator is experimental only and must not become the production default (EXP-004 rejected). |
| **Deployment** | **Vercel Hobby plan**. Target a publicly accessible application within Hobby limitations. Account for serverless execution limits, cold starts, function bundle size, and no persistent local filesystem/vector DB across invocations. Investigate a suitable free-tier/Hobby-compatible storage solution. Do not introduce paid infrastructure without explicitly documenting the need and obtaining approval. |
| **Cost** | Prefer free or free-tier services. Track actual usage and costs. Do not compromise correctness merely to remain within a free tier; document constraints and trade-offs. |
| **Application** | Retain a clean frontend/backend separation. Use the existing React frontend and Python/FastAPI RAG core where appropriate. Replace legacy implementation details if a cleaner architecture is justified. |

---

## 3. Engineering philosophy

We build a production-quality system by identifying real problems and solving them
with measurable engineering improvements.

- Exact terminology missed → investigate sparse retrieval/BM25.
- Semantically relevant evidence missed → investigate embedding/dense retrieval.
- Irrelevant chunks rank too highly → investigate reranking.
- Correct evidence retrieved but omitted → investigate evidence selection/context.
- Unsupported claims → investigate generation constraints/citation enforcement.
- Quality degrades silently → implement evaluation gates and monitoring.
- Failures hard to diagnose → improve tracing/observability.

These are examples of problem-to-solution reasoning, not a mandatory sequence.
Every additional component must have a clear justification.

### 3.1 Source-of-truth and integrity rules

1. **No fabricated policy.** Never invent clauses, answers, clause IDs, pages or citations.
2. **No silent correction.** Never rewrite source text or ground truth to improve metrics.
3. **One shared core.** Experiments and production must use the same canonical document/chunking/retrieval/metrics/answering code (`src/rag_core/`). `experiments/exp001.py` is the single frozen historical exception.
4. **Controlled experiments.** Change one variable at a time; hold PDF, questions, ground truth, embedding, retrieval, top-K and metrics constant unless the experiment is about that variable. Pre-register success criteria.
5. **Honest reporting.** Do not claim a test passed unless executed. Do not hide failed/unfavourable results. Rejected experiments remain documented.
6. **Preserve user work.** Do not rewrite commits or discard unrelated changes.

### 3.2 Experimentation rules

For every experiment: establish baseline; identify observed problem; form
hypothesis; define success criteria before running; identify fixed/changed
variables; run; inspect aggregate and per-question results; analyse regressions;
accept/reject/defer/inconclusive; document decision. Never modify an approved
evaluation dataset merely to improve metrics. Preserve historical experiment
results.

---

## 4. Definition of project completion

The project is complete when:

1. The SATCOM NGP PDF is ingested reproducibly.
2. The RAG pipeline works end to end.
3. Answers are grounded in retrieved policy evidence.
4. Citations are accurate and inspectable.
5. Unsupported questions are handled through appropriate abstention.
6. Retrieval and generation quality are evaluated.
7. A manually verified golden dataset of approximately 50–200 questions exists.
8. CI runs quality evaluations and detects important regressions.
9. The application is deployed on Vercel Hobby.
10. Observability captures the important pipeline stages.
11. Latency, cost, citation coverage and failure rates are measured.
12. The architecture is clean and understandable.
13. Engineering decisions, bugs, experiments and fixes are documented.

A running chatbot alone does not satisfy the definition of done.

---

## 5. Current implementation status (reconciled 2026-11-14)

Reconciled against the repository and Git history (not prior agent reports).
Unrelated existing working-tree changes are preserved and not treated as project
progress: deleted `.github/workflows/rag_eval.yml`, `experiments/rag.py`,
`experiments/results/eval_report.json`, `src/memory/working.py`,
`src/metrics/citation.py`, `src/restart.py`, `tests/eval_pipeline.py`,
`todo.txt`; modified `backend/main.py`.

### 5.1 Completed / validated

| Area | Status | Evidence |
|---|---|---|
| Source of truth | Fixed | `data/satcom-ngp.pdf` (15 pages, 35,407 chars). Approved 15 questions / 20 evidence units frozen. |
| Canonical core | **Done** | `src/rag_core/`: `document.py`, `chunker.py`, `index.py`, `evaluation.py`, `pipeline.py`, `generation.py` (experimental). Shared by experiments; production slice uses it. |
| Document integrity | **Done** | Page-aware heading scan, conservative clause-ID normalization, strict coverage audit (100% non-whitespace, fidelity/bounds/page checks). |
| Chunking baseline | **Accepted** | EXP-001 v3 fixed-character 1000/200/800, 49 chunks. Reproduced by `FixedCharacterChunker`. |
| Retrieval baseline | **Accepted** | Dense MiniLM `all-MiniLM-L6-v2`, top-5. Recall@5 0.8889, Precision@5 0.2533, MRR@5 0.7944 (frozen). |
| Answer selection | **Accepted (extractive)** | EXP-003 sibling-clause selection (ratio 0.5). Statement support 15/45 (0.3333), citation support 16/31 (0.5161), 12/15 answers contain ≥1 supported statement. Q13 3/3. |
| Citations/provenance | **Done (extractive)** | Verbatim statements with exact global offsets; citations only from contributing chunks; page/section/source traceability. |
| Abstention | **Smoke-tested** | 12/12 out-of-corpus abstained (8 calibration + 4 held-out). Not a validated answerability classifier; does not catch in-corpus retrieval failure (Q05). |
| Experiments | **Recorded** | EXP-001 accepted; EXP-002 rejected (structure-aware default); EXP-003 accepted; EXP-004 rejected (constrained local generator, not production); EXP-005 rejected (depth≥3 clause-body units insufficient for Q10). |
| Tests | **Executed** | `91 passed, 25 warnings` (`python -m pytest tests -q`, 2026-11-14). Warnings are existing Chroma/Pydantic deprecation notices. |
| Engineering history | **Maintained** | `docs/PROJECT_BRIEF.md`, `docs/ENGINEERING_LOG.md`, `docs/BACKLOG.md`, EXP reports, PROD-001 report. |

### 5.2 Partial / legacy / not production-ready

| Area | Status | Notes |
|---|---|---|
| End-to-end app | **Not complete** | No validated UI→API→validated RAG→Gemini→cited answer flow. `experiments/production_slice.py` demonstrates the extractive slice, not a deployed service. |
| Gemini generation | **Integrated as controlled optional layer (M1 not accepted)** | `src/rag_core/generation.py` (`GeminiGenerator`, m1-v1) wired via `RAGService`/`/query` with sufficiency gate, app-side citation ID validation/resolution, and safe fallback to extractive grounded evidence; extractive remains default. Live M1 quality/correctness/completeness insufficient; model/config/cost not production-accepted. Legacy `src/generation/models.py` not used. |
| Citation enforcement (LLM) | **Not done** | Extractive path is grounded by construction; LLM citation validity/enforcement is unimplemented/unvalidated. |
| Frontend | **Integrated to `/query` (local, unverified build here)** | `frontend/src/App.jsx` rewritten to call validated backend (`/health`, `/query`, extractive default), render answer + inspectable app-resolved citations/offsets, abstention reasons, errors/loading and diagnostics. Legacy `/chat`/`/metrics` dependency removed from active UI. `npm run build` not executed in this environment (vite deps missing); manual local run/validation pending. |
| Backend API | **WP1 implemented (local, extractive-first)** | `backend/main.py` rewritten as thin FastAPI app over `src/rag_app/RAGService` + validated `src/rag_core` (corrected production index, EXP-003 default, inspectable app-resolved citations, explicit abstention, lightweight tracing). KNOWN-001 legacy behavior removed from active path; Gemini optional/controlled with fallback (M1 not accepted). Frontend contract/integration pending. |
| Vector DB for production | **Not selected for Vercel** | Validated experiments use local persistent Chroma collections. No Hobby-compatible persistent store chosen/integrated. |
| Golden dataset (50–200) | **Not started** | Only 15 approved questions / 20 evidence units + 12 out-of-corpus abstention questions exist. |
| Offline evaluation (Phase 3) | **Not started** | Retrieval metrics exist for the small set; answer correctness/faithfulness/completeness/citation correctness/coverage/abstention quality for LLM answers are not implemented. |
| CI quality gates | **Not present** | No quality-evaluation CI workflow preventing regressions. (Unrelated deleted workflow exists in working tree; not project progress.) |
| Observability/monitoring | **Not started** | No request tracing, prompt/response logging with safeguards, token/cost tracking, latency percentiles, citation/failure dashboards, or investigation workflow. |
| Deployment | **Not started** | No Vercel configuration, no deployed app, no Hobby feasibility validation. |

### 5.3 Known unresolved answer-quality problems

| ID | Problem | Type | Evidence |
|---|---|---|---|
| Q05 / BL-001 | Evidence not retrieved | Retrieval miss | Recall@5 0.0 (`2.5.6`); answered without support. |
| Q08 / BL-002 | Required child clauses incomplete | Retrieval miss | Recall@5 0.3333 (1/3 clauses). |
| Q10 / BL-003 | Evidence retrieved but not selected | Selection/ranking | Recall@5 1.0, 0/3 supported; EXP-003 did not apply, EXP-005 formed 3.6.8 unit but lexical/context ranking still preferred 3.6.7(a)/(b) list and 3.6.10. |
| Q04 / BL-004 | Heading-only selected evidence | Answer unit | Selector returns heading, not following list. |
| Q12 / BL-005 | Heading-only selected evidence | Answer unit | Selector returns classification heading, not definition. |
| Q14 / note | Two statements overlapped one evidence unit | Answer completeness | Known from EXP-003/PROD-001; EXP-005 changed granularity but did not solve completeness. |
| BL-006 | LLM generator faithfulness/citations | Generation | EXP-004 rejected constrained local synthesis; Gemini path unvalidated. |
| BL-007 | Abstention misses in-corpus retrieval failure | Abstention | Q05 answered despite missing evidence. |

Details and prioritization: `docs/BACKLOG.md`.

---

## 6. Milestone-based roadmap

Roadmap is ordered by dependency and by the fixed destination. Each milestone
must define success criteria and measured evidence before closure.

| Milestone | Objective | Key outputs | Exit evidence |
|---|---|---|---|
| **M0 — Direction & backlog (this step)** | Fix authoritative objective, constraints, roadmap, backlog | Updated brief/log, `docs/BACKLOG.md` | Documented and reconciled; no new RAG technique added |
| **M1 — Validated Gemini answer path (local/integration)** | Replace/augment extractive-only user answers with Gemini grounded answers on the validated core | Gemini model/config documented, versioned prompt contract, retrieval→context→Gemini flow, citation format linked to PDF evidence, abstention/citation enforcement rules, faithfulness checks | Controlled evaluation on current set shows grounded, inspectable citations; failures recorded; no silent regression vs extractive evidence traceability |
| **M2 — End-to-end application integration** | Connect React UI + FastAPI API + validated RAG/Gemini pipeline | Clean API contract, UI evidence/citation display, local reproducible run, KNOWN-001/legacy path isolated or fixed as planned | User can ask and inspect exact evidence/citations locally; API/integration tests pass |
| **M3 — Deployment-ready architecture (Vercel Hobby)** | Make system deployable within Hobby constraints | Storage/index strategy (free-tier/Hobby-compatible), serverless-safe packaging, env/config/secret handling, cold-start/latency/cost measurements, deployment runbook | Publicly accessible app or documented blocking constraint with approved alternative; measured limits recorded |
| **M4 — Golden dataset & offline evaluation** | Build ~50–200 verified Q/A set and evaluation harness | Dataset with evidence/citations/abstention labels, retrieval/answer/faithfulness/completeness/citation/abstention metrics, baseline report | Reproducible evaluation run with documented thresholds derived from baseline evidence |
| **M5 — Quality gates & observability** | Prevent regressions and make failures diagnosable | CI quality checks, tracing, prompt/response/token/cost/latency/citation/failure instrumentation, monitoring/investigation path | CI blocks important regressions per defined thresholds; traces explain request outcomes |
| **M6 — Targeted quality improvements** | Address highest-impact observed gaps | Only justified experiments/fixes (e.g. Q05/Q08 retrieval, Q10 reference-aware selection, Q04/Q12 units, abstention, reranking if evidenced) | Each change accepted/rejected by controlled evidence; backlog updated |

Milestones may overlap in planning but must not be declared complete without
evidence. M6 items are backlog-driven, not a license for indefinite tuning.

---

## 7. Known issues and deferred improvements

Maintained in `docs/BACKLOG.md`. Rules:

1. Document newly discovered important problems.
2. Assess impact and urgency.
3. Add to backlog with owner/status/target milestone.
4. Decide when to address; do not derail current milestone for every issue.
5. Do not silently discard problems. Rejected experiments remain documented.

---

## 8. Validated core and metrics (reference)

### 8.1 Canonical core — `src/rag_core/`

| Module | Responsibility |
|---|---|
| `document.py` | PDF extraction, pages, global character offsets, `page_for_offset`. |
| `chunker.py` | `StructureAwareChunker`, `FixedCharacterChunker`, heading detection, clause-ID normalization, unique IDs, `audit_source_coverage`. |
| `index.py` | Dense Chroma index, config, retrieved chunks, metadata. |
| `evaluation.py` | Verified evidence loading, retrieval metrics, answer-support evaluation. |
| `pipeline.py` | `GroundedPipeline`: extractive answers, EXP-003 sibling selection, provenance, contributing-chunk citations, abstention policy; experimental `clause_body_units` (off by default); optional unevaluated generator hook. |
| `generation.py` | Experimental selected-evidence generator (EXP-004), not wired into production. |

### 8.2 Retrieval metrics (frozen definitions)

- **Recall@K** = distinct required evidence units overlapped by top-K / total required.
- **Precision@K** = relevant top-K chunks / K.
- **MRR@K** = reciprocal rank of first relevant chunk (0 if none).

Relevance requires overlap with an actual supporting evidence span (character-level).

### 8.3 Experiment registry

| ID | Variable | Status | Key result |
|---|---|---|---|
| EXP-001 v3 | baseline | **Accepted** | Recall@5 0.8889, Precision@5 0.2533, MRR@5 0.7944 |
| EXP-002 | chunking → structure-aware | **Rejected** default | 0.8833 / 0.1867 / 0.6244 |
| PROD-001 | production extractive slice | **Accepted retrieval, limited answers** | retrieval parity; statement 15/45, citation 16/31, 12/15 supported |
| EXP-003 | answer selection → sibling clauses | **Accepted** | Q13 1/3→3/3; statement 0.2889→0.3333 |
| EXP-004 | selected-evidence local generator | **Rejected** production | gate 0.875<0.90; manual audit 5/6/4, 4 incorrect same as extractive |
| EXP-005 | depth≥3 clause-body units (Q10) | **Rejected** | Q10 0/3; 3.6.8 unit formed but outranked; Q14 statement -1 |

---

## 9. Commands

```bash
# Full test suite
python -m pytest tests -q

# Reproduce accepted baseline / experiments
python -m experiments.production_slice
python -m experiments.exp003
python -m experiments.exp004   # requires local Ollama + qwen2.5:1.5b
python -m experiments.exp004_audit
python -m experiments.exp005
```

EXP-001 v3 is reproduced only from its frozen runner/artifacts.

---

## 10. Documentation index

| Document | Purpose |
|---|---|
| `docs/PROJECT_BRIEF.md` | Authoritative objective, constraints, roadmap, status. |
| `docs/ENGINEERING_LOG.md` | Decisions, defects, experiments, verification, risks, history. |
| `docs/BACKLOG.md` | Known issues and deferred improvements. |
| `docs/EXP-001.md` … `docs/EXP-005.md` | Experiment reports. |
| `docs/PRODUCTION_SLICE.md` | Validated extractive slice. |
| `docs/CHUNKING_STRATEGY.md` | Historical design analysis (context only). |## Current Status (2026-11-15, zero-cost deployment constraints applied)
- Baseline (M0, DR-013): , DR-013, and  established ().
- M1 executed but **not accepted**: Gemini evidence-only path implemented and validated (, DR-014/DR-015, , corrected prod index), but grounded correctness/completeness and cost criteria not met on current 15q set.
- EXP-006/007/008 recorded as separate opt-ins/candidates (not production default/combined); citation metadata correction applied and production index rebuilt (, 49 chunks) while preserving artifacts/history.
- WP0/WP1/WP1.1 completed locally: single backend service over  with corrected index, EXP-003 default, explicit abstention, inspectable app-validated citations, request tracing, and conservative answerability guard (Q05 abstains with traceable evidence; Q12 valid on 15q regression). WP2 frontend integrated to +; WP2/WP3 local run/build/tests verified (,  pass) with DOM-automation limitation noted.
- **M3 deployment feasibility assessed ():** Vercel Hobby not approved/compatible-claimed; Hugging Face Docker Spaces rejected under strict zero-cost for personal Docker compute; Render free likely incompatible. **Oracle Always Free investigated separately (, no provisioning):** plausibly closest VM path but **not confidently guaranteed /usr/bin/bash/obtainable** without tenant-specific verification (evidence-limited in-session; UNVERIFIED items noted). **No consequential infrastructure changes made.**
- **M4 foundation (parallel, 2026-11-15):** eval plan + reproducible offline harness + measured baseline on preserved sets (, , , ) — 15 answerable (recall@5 0.8889, supported 0.80, stmt/cit support 0.3333/0.5161, Q05 abstained) and 12 unanswerable (abstention accuracy 1.00); thresholds pending measured expanded baselines. No production behavior changes.
- **M5 observability (parallel, 2026-11-15):** lightweight $0 tracing/metrics implemented (, service instrumentation, , );  8 passed,  108 passed.
- **Next (pending user-approved direction):** manual browser UI pass; manual verification/curation of expanded M4 dataset (50–100 initial) and measured thresholds/CI gates; any Oracle pre-provision verification/resource measurement only if explicitly approved. Further retrieval/selection work (BL-001/002, EXP-006/008 promotion, EXP-009) deferred pending authorization/evidence. Local app remains validated source of truth.

# SATCOM NGP RAG — Project Brief (Authoritative)

**Status:** living, authoritative project direction. Last updated 2026-11-15 (strategy pivot: prioritize end-to-end application completion; component experiments paused).  
**Owner:** engineering.  
**Companion documents:** `docs/ENGINEERING_LOG.md`, `docs/BACKLOG.md`,
`docs/EXP-001.md`–`docs/EXP-008.md`, `docs/EXP-008-citation-metadata-correction.md`,
`docs/PRODUCTION_SLICE.md`, `docs/CHUNKING_STRATEGY.md`.

This document defines the fixed destination for the project. Experiments and
engineering decisions must help us reach it. We do not add complexity without a
measurable reason, and we do not experiment indefinitely.

---

## 1. Ultimate objective

Build a **production-grade, domain-specific RAG application for the ISRO SATCOM
NGP policy document**.

The objective is not a chatbot demo and not a collection of fashionable RAG
techniques. The system must retrieve relevant policy evidence, generate accurate
answers grounded in that evidence, provide verifiable citations, and know when it
cannot answer.

The project has three major phases.

### Phase 1 — Functional RAG application

Deliver a working end-to-end application with:

- PDF ingestion and document processing.
- Appropriate chunking.
- Embedding generation.
- Vector database and indexing.
- Retrieval pipeline.
- Answer generation using **Gemini**.
- Accurate citations linked to the original PDF.
- Citation enforcement and abstention when evidence is insufficient.
- A functional user interface.

A user must be able to ask a question and inspect the exact policy evidence used
to answer it.

### Phase 2 — Production-quality RAG

Improve the system based on observed limitations.

Potential techniques include hybrid retrieval (BM25 + dense), cross-encoder
reranking, better context construction, citation enforcement, improved
abstention, and versioned prompts/configurations.

**These are potential solutions, not mandatory components.** Do not implement
BM25, reranking, query expansion or other techniques because they are popular.
First identify an actual problem, form a hypothesis, design a controlled
experiment, measure the outcome, and introduce the technique only if results
justify it.

### Phase 3 — Evaluation and quality assurance

Build a manually verified golden evaluation dataset of approximately **50–200**
question–answer pairs.

Implement offline evaluation for retrieval quality, answer correctness,
faithfulness, completeness, citation correctness, citation coverage, and
abstention quality.

Integrate evaluation into CI so important quality regressions can prevent changes
from being accepted.

Define thresholds using actual baseline results and evaluation evidence. **Do not
invent arbitrary thresholds.**

### Observability and monitoring

Implement an appropriate observability layer. Track:

- Individual request traces.
- Retrieved chunks, ranking and selection decisions.
- Prompts and model responses (with appropriate secret/data handling).
- Token usage and cost per request.
- P50 and P95 latency.
- Citation coverage.
- Failure rates and quality degradation.

Provide useful monitoring and a way to investigate why a request failed.

---

## 2. Fixed project constraints

These are project-level decisions (recorded as DR-013 in `docs/ENGINEERING_LOG.md`).

| Constraint | Decision |
|---|---|
| **Corpus** | Exactly one source document for the initial application: `data/satcom-ngp.pdf` (ISRO SATCOM NGP policy). Do not expand to multiple PDFs or arbitrary user uploads. |
| **LLM** | **Google Gemini** via the Gemini API. Select an appropriate model based on answer quality, availability, cost and latency; document the selected model and configuration. The local Qwen/Ollama generator is experimental only and must not become the production default (EXP-004 rejected). |
| **Deployment** | **Vercel Hobby plan**. Target a publicly accessible application within Hobby limitations. Account for serverless execution limits, cold starts, function bundle size, and no persistent local filesystem/vector DB across invocations. Investigate a suitable free-tier/Hobby-compatible storage solution. Do not introduce paid infrastructure without explicitly documenting the need and obtaining approval. |
| **Cost** | Prefer free or free-tier services. Track actual usage and costs. Do not compromise correctness merely to remain within a free tier; document constraints and trade-offs. |
| **Application** | Retain a clean frontend/backend separation. Use the existing React frontend and Python/FastAPI RAG core where appropriate. Replace legacy implementation details if a cleaner architecture is justified. |

---

## 3. Engineering philosophy

We build a production-quality system by identifying real problems and solving them
with measurable engineering improvements.

- Exact terminology missed → investigate sparse retrieval/BM25.
- Semantically relevant evidence missed → investigate embedding/dense retrieval.
- Irrelevant chunks rank too highly → investigate reranking.
- Correct evidence retrieved but omitted → investigate evidence selection/context.
- Unsupported claims → investigate generation constraints/citation enforcement.
- Quality degrades silently → implement evaluation gates and monitoring.
- Failures hard to diagnose → improve tracing/observability.

These are examples of problem-to-solution reasoning, not a mandatory sequence.
Every additional component must have a clear justification.

### 3.1 Source-of-truth and integrity rules

1. **No fabricated policy.** Never invent clauses, answers, clause IDs, pages or citations.
2. **No silent correction.** Never rewrite source text or ground truth to improve metrics.
3. **One shared core.** Experiments and production must use the same canonical document/chunking/retrieval/metrics/answering code (`src/rag_core/`). `experiments/exp001.py` is the single frozen historical exception.
4. **Controlled experiments.** Change one variable at a time; hold PDF, questions, ground truth, embedding, retrieval, top-K and metrics constant unless the experiment is about that variable. Pre-register success criteria.
5. **Honest reporting.** Do not claim a test passed unless executed. Do not hide failed/unfavourable results. Rejected experiments remain documented.
6. **Preserve user work.** Do not rewrite commits or discard unrelated changes.

### 3.2 Experimentation rules

For every experiment: establish baseline; identify observed problem; form
hypothesis; define success criteria before running; identify fixed/changed
variables; run; inspect aggregate and per-question results; analyse regressions;
accept/reject/defer/inconclusive; document decision. Never modify an approved
evaluation dataset merely to improve metrics. Preserve historical experiment
results.

---

## 4. Definition of project completion

The project is complete when:

1. The SATCOM NGP PDF is ingested reproducibly.
2. The RAG pipeline works end to end.
3. Answers are grounded in retrieved policy evidence.
4. Citations are accurate and inspectable.
5. Unsupported questions are handled through appropriate abstention.
6. Retrieval and generation quality are evaluated.
7. A manually verified golden dataset of approximately 50–200 questions exists.
8. CI runs quality evaluations and detects important regressions.
9. The application is deployed on Vercel Hobby.
10. Observability captures the important pipeline stages.
11. Latency, cost, citation coverage and failure rates are measured.
12. The architecture is clean and understandable.
13. Engineering decisions, bugs, experiments and fixes are documented.

A running chatbot alone does not satisfy the definition of done.

---

## 5. Current implementation status (reconciled 2026-11-14)

Reconciled against the repository and Git history (not prior agent reports).
Unrelated existing working-tree changes are preserved and not treated as project
progress: deleted `.github/workflows/rag_eval.yml`, `experiments/rag.py`,
`experiments/results/eval_report.json`, `src/memory/working.py`,
`src/metrics/citation.py`, `src/restart.py`, `tests/eval_pipeline.py`,
`todo.txt`; modified `backend/main.py`.

### 5.1 Completed / validated

| Area | Status | Evidence |
|---|---|---|
| Source of truth | Fixed | `data/satcom-ngp.pdf` (15 pages, 35,407 chars). Approved 15 questions / 20 evidence units frozen. |
| Canonical core | **Done** | `src/rag_core/`: `document.py`, `chunker.py`, `index.py`, `evaluation.py`, `pipeline.py`, `generation.py` (experimental). Shared by experiments; production slice uses it. |
| Document integrity | **Done** | Page-aware heading scan, conservative clause-ID normalization, strict coverage audit (100% non-whitespace, fidelity/bounds/page checks). |
| Chunking baseline | **Accepted** | EXP-001 v3 fixed-character 1000/200/800, 49 chunks. Reproduced by `FixedCharacterChunker`. |
| Retrieval baseline | **Accepted** | Dense MiniLM `all-MiniLM-L6-v2`, top-5. Recall@5 0.8889, Precision@5 0.2533, MRR@5 0.7944 (frozen). |
| Answer selection | **Accepted (extractive)** | EXP-003 sibling-clause selection (ratio 0.5). Statement support 15/45 (0.3333), citation support 16/31 (0.5161), 12/15 answers contain ≥1 supported statement. Q13 3/3. |
| Citations/provenance | **Done (extractive)** | Verbatim statements with exact global offsets; citations only from contributing chunks; page/section/source traceability. |
| Abstention | **Smoke-tested** | 12/12 out-of-corpus abstained (8 calibration + 4 held-out). Not a validated answerability classifier; does not catch in-corpus retrieval failure (Q05). |
| Experiments | **Recorded** | EXP-001 accepted; EXP-002 rejected (structure-aware default); EXP-003 accepted; EXP-004 rejected (constrained local generator, not production); EXP-005 rejected (depth≥3 clause-body units insufficient for Q10). |
| Tests | **Executed** | `91 passed, 25 warnings` (`python -m pytest tests -q`, 2026-11-14). Warnings are existing Chroma/Pydantic deprecation notices. |
| Engineering history | **Maintained** | `docs/PROJECT_BRIEF.md`, `docs/ENGINEERING_LOG.md`, `docs/BACKLOG.md`, EXP reports, PROD-001 report. |

### 5.2 Partial / legacy / not production-ready

| Area | Status | Notes |
|---|---|---|
| End-to-end app | **Not complete** | No validated UI→API→validated RAG→Gemini→cited answer flow. `experiments/production_slice.py` demonstrates the extractive slice, not a deployed service. |
| Gemini generation | **Integrated as controlled optional layer (M1 not accepted)** | `src/rag_core/generation.py` (`GeminiGenerator`, m1-v1) wired via `RAGService`/`/query` with sufficiency gate, app-side citation ID validation/resolution, and safe fallback to extractive grounded evidence; extractive remains default. Live M1 quality/correctness/completeness insufficient; model/config/cost not production-accepted. Legacy `src/generation/models.py` not used. |
| Citation enforcement (LLM) | **Not done** | Extractive path is grounded by construction; LLM citation validity/enforcement is unimplemented/unvalidated. |
| Frontend | **Integrated to `/query` (local, unverified build here)** | `frontend/src/App.jsx` rewritten to call validated backend (`/health`, `/query`, extractive default), render answer + inspectable app-resolved citations/offsets, abstention reasons, errors/loading and diagnostics. Legacy `/chat`/`/metrics` dependency removed from active UI. `npm run build` not executed in this environment (vite deps missing); manual local run/validation pending. |
| Backend API | **WP1 implemented (local, extractive-first)** | `backend/main.py` rewritten as thin FastAPI app over `src/rag_app/RAGService` + validated `src/rag_core` (corrected production index, EXP-003 default, inspectable app-resolved citations, explicit abstention, lightweight tracing). KNOWN-001 legacy behavior removed from active path; Gemini optional/controlled with fallback (M1 not accepted). Frontend contract/integration pending. |
| Vector DB for production | **Not selected for Vercel** | Validated experiments use local persistent Chroma collections. No Hobby-compatible persistent store chosen/integrated. |
| Golden dataset (50–200) | **Not started** | Only 15 approved questions / 20 evidence units + 12 out-of-corpus abstention questions exist. |
| Offline evaluation (Phase 3) | **Not started** | Retrieval metrics exist for the small set; answer correctness/faithfulness/completeness/citation correctness/coverage/abstention quality for LLM answers are not implemented. |
| CI quality gates | **Not present** | No quality-evaluation CI workflow preventing regressions. (Unrelated deleted workflow exists in working tree; not project progress.) |
| Observability/monitoring | **Not started** | No request tracing, prompt/response logging with safeguards, token/cost tracking, latency percentiles, citation/failure dashboards, or investigation workflow. |
| Deployment | **Not started** | No Vercel configuration, no deployed app, no Hobby feasibility validation. |

### 5.3 Known unresolved answer-quality problems

| ID | Problem | Type | Evidence |
|---|---|---|---|
| Q05 / BL-001 | Evidence not retrieved | Retrieval miss | Recall@5 0.0 (`2.5.6`); answered without support. |
| Q08 / BL-002 | Required child clauses incomplete | Retrieval miss | Recall@5 0.3333 (1/3 clauses). |
| Q10 / BL-003 | Evidence retrieved but not selected | Selection/ranking | Recall@5 1.0, 0/3 supported; EXP-003 did not apply, EXP-005 formed 3.6.8 unit but lexical/context ranking still preferred 3.6.7(a)/(b) list and 3.6.10. |
| Q04 / BL-004 | Heading-only selected evidence | Answer unit | Selector returns heading, not following list. |
| Q12 / BL-005 | Heading-only selected evidence | Answer unit | Selector returns classification heading, not definition. |
| Q14 / note | Two statements overlapped one evidence unit | Answer completeness | Known from EXP-003/PROD-001; EXP-005 changed granularity but did not solve completeness. |
| BL-006 | LLM generator faithfulness/citations | Generation | EXP-004 rejected constrained local synthesis; Gemini path unvalidated. |
| BL-007 | Abstention misses in-corpus retrieval failure | Abstention | Q05 answered despite missing evidence. |

Details and prioritization: `docs/BACKLOG.md`.

---

## 6. Milestone-based roadmap

Roadmap is ordered by dependency and by the fixed destination. Each milestone
must define success criteria and measured evidence before closure.

| Milestone | Objective | Key outputs | Exit evidence |
|---|---|---|---|
| **M0 — Direction & backlog (this step)** | Fix authoritative objective, constraints, roadmap, backlog | Updated brief/log, `docs/BACKLOG.md` | Documented and reconciled; no new RAG technique added |
| **M1 — Validated Gemini answer path (local/integration)** | Replace/augment extractive-only user answers with Gemini grounded answers on the validated core | Gemini model/config documented, versioned prompt contract, retrieval→context→Gemini flow, citation format linked to PDF evidence, abstention/citation enforcement rules, faithfulness checks | Controlled evaluation on current set shows grounded, inspectable citations; failures recorded; no silent regression vs extractive evidence traceability |
| **M2 — End-to-end application integration** | Connect React UI + FastAPI API + validated RAG/Gemini pipeline | Clean API contract, UI evidence/citation display, local reproducible run, KNOWN-001/legacy path isolated or fixed as planned | User can ask and inspect exact evidence/citations locally; API/integration tests pass |
| **M3 — Deployment-ready architecture (Vercel Hobby)** | Make system deployable within Hobby constraints | Storage/index strategy (free-tier/Hobby-compatible), serverless-safe packaging, env/config/secret handling, cold-start/latency/cost measurements, deployment runbook | Publicly accessible app or documented blocking constraint with approved alternative; measured limits recorded |
| **M4 — Golden dataset & offline evaluation** | Build ~50–200 verified Q/A set and evaluation harness | Dataset with evidence/citations/abstention labels, retrieval/answer/faithfulness/completeness/citation/abstention metrics, baseline report | Reproducible evaluation run with documented thresholds derived from baseline evidence |
| **M5 — Quality gates & observability** | Prevent regressions and make failures diagnosable | CI quality checks, tracing, prompt/response/token/cost/latency/citation/failure instrumentation, monitoring/investigation path | CI blocks important regressions per defined thresholds; traces explain request outcomes |
| **M6 — Targeted quality improvements** | Address highest-impact observed gaps | Only justified experiments/fixes (e.g. Q05/Q08 retrieval, Q10 reference-aware selection, Q04/Q12 units, abstention, reranking if evidenced) | Each change accepted/rejected by controlled evidence; backlog updated |

Milestones may overlap in planning but must not be declared complete without
evidence. M6 items are backlog-driven, not a license for indefinite tuning.

---

## 7. Known issues and deferred improvements

Maintained in `docs/BACKLOG.md`. Rules:

1. Document newly discovered important problems.
2. Assess impact and urgency.
3. Add to backlog with owner/status/target milestone.
4. Decide when to address; do not derail current milestone for every issue.
5. Do not silently discard problems. Rejected experiments remain documented.

---

## 8. Validated core and metrics (reference)

### 8.1 Canonical core — `src/rag_core/`

| Module | Responsibility |
|---|---|
| `document.py` | PDF extraction, pages, global character offsets, `page_for_offset`. |
| `chunker.py` | `StructureAwareChunker`, `FixedCharacterChunker`, heading detection, clause-ID normalization, unique IDs, `audit_source_coverage`. |
| `index.py` | Dense Chroma index, config, retrieved chunks, metadata. |
| `evaluation.py` | Verified evidence loading, retrieval metrics, answer-support evaluation. |
| `pipeline.py` | `GroundedPipeline`: extractive answers, EXP-003 sibling selection, provenance, contributing-chunk citations, abstention policy; experimental `clause_body_units` (off by default); optional unevaluated generator hook. |
| `generation.py` | Experimental selected-evidence generator (EXP-004), not wired into production. |

### 8.2 Retrieval metrics (frozen definitions)

- **Recall@K** = distinct required evidence units overlapped by top-K / total required.
- **Precision@K** = relevant top-K chunks / K.
- **MRR@K** = reciprocal rank of first relevant chunk (0 if none).

Relevance requires overlap with an actual supporting evidence span (character-level).

### 8.3 Experiment registry

| ID | Variable | Status | Key result |
|---|---|---|---|
| EXP-001 v3 | baseline | **Accepted** | Recall@5 0.8889, Precision@5 0.2533, MRR@5 0.7944 |
| EXP-002 | chunking → structure-aware | **Rejected** default | 0.8833 / 0.1867 / 0.6244 |
| PROD-001 | production extractive slice | **Accepted retrieval, limited answers** | retrieval parity; statement 15/45, citation 16/31, 12/15 supported |
| EXP-003 | answer selection → sibling clauses | **Accepted** | Q13 1/3→3/3; statement 0.2889→0.3333 |
| EXP-004 | selected-evidence local generator | **Rejected** production | gate 0.875<0.90; manual audit 5/6/4, 4 incorrect same as extractive |
| EXP-005 | depth≥3 clause-body units (Q10) | **Rejected** | Q10 0/3; 3.6.8 unit formed but outranked; Q14 statement -1 |

---

## 9. Commands

```bash
# Full test suite
python -m pytest tests -q

# Reproduce accepted baseline / experiments
python -m experiments.production_slice
python -m experiments.exp003
python -m experiments.exp004   # requires local Ollama + qwen2.5:1.5b
python -m experiments.exp004_audit
python -m experiments.exp005
```

EXP-001 v3 is reproduced only from its frozen runner/artifacts.

---

## 10. Documentation index

| Document | Purpose |
|---|---|
| `docs/PROJECT_BRIEF.md` | Authoritative objective, constraints, roadmap, status. |
| `docs/ENGINEERING_LOG.md` | Decisions, defects, experiments, verification, risks, history. |
| `docs/BACKLOG.md` | Known issues and deferred improvements. |
| `docs/EXP-001.md` … `docs/EXP-005.md` | Experiment reports. |
| `docs/PRODUCTION_SLICE.md` | Validated extractive slice. |
| `docs/CHUNKING_STRATEGY.md` | Historical design analysis (context only). |
