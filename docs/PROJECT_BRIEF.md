# SATCOM NGP RAG — Project Brief

Status: living document. Last updated 2026-11-14.
Owner: engineering. Companion documents: `docs/ENGINEERING_LOG.md`,
`docs/EXP-001.md`, `docs/EXP-002.md`, `docs/CHUNKING_STRATEGY.md`.

---

## 1. Purpose

Build a production-quality, evidence-grounded Retrieval-Augmented Generation
(RAG) application over the ISRO **Satellite Communications Norm, Guidelines and
Procedures (SATCOM NGP)** policy PDF.

The application must:

- answer questions **only** from the policy document;
- attach **traceable citations** (chunk → exact source offsets → PDF page);
- **abstain** when the retrieved evidence does not support an answer;
- correctly handle **multi-clause** questions that require several evidence
  units, not just one passage;
- be **measurably** better than, or at least not silently worse than, the
  accepted baseline;
- be **reproducible** from committed code, configuration and recorded results;
- be covered by **executed** tests and honest engineering documentation.

This is an engineering project, not a demo. A favourable-looking answer without
source evidence is a defect, not a success.

---

## 2. Source of truth

| Item | Location | Rule |
|---|---|---|
| Policy PDF | `data/satcom-ngp.pdf` | Immutable input. Never edit policy text. |
| Extracted corpus | `src/rag_core/document.py` | 15 pages, 35,407 extracted characters. Stored as-is. |
| Approved questions | `experiments/exp001_questions.csv` | 15 questions (`EXP001-Q01`…`EXP001-Q15`). Do not modify. |
| Verified ground truth | `experiments/exp001_ground_truth.json` | 20 evidence units with exact excerpts + offsets. Do not modify. |
| Accepted baseline run | `experiments/results/exp001_v3/` | Frozen historical baseline. Do not regenerate or overwrite. |
| Frozen baseline code | `experiments/exp001.py` | Historical implementation, kept for reproducibility. Do not refactor. |

Any apparent typo or oddity in the PDF (for example `Satellites�`,
`Normatves`, `procedurs`) is **source text**, not something the system may
silently "fix". Ground-truth excerpts must match the extracted bytes exactly.

---

## 3. Non-negotiable rules

1. **No fabricated policy.** Never invent clauses, answers, clause IDs, page
   numbers or citations.
2. **No silent correction.** Never rewrite source text or ground truth to make a
   metric look better. Corrections are explicit, recorded and re-verified.
3. **One shared core.** Experiments and production must use the same canonical
   document/chunking/retrieval/metrics code. Duplicate implementations are not
   allowed (`experiments/exp001.py` is the single frozen exception, retained as a
   historical baseline).
4. **Controlled experiments.** Change one variable at a time. Hold the PDF,
   questions, ground truth, embedding model, retrieval algorithm, top-K and
   metric definitions constant unless the experiment is explicitly about that
   variable.
5. **Evidence before complexity.** Do not add hybrid retrieval, query expansion,
   reranking, parent–child retrieval, semantic chunking or generation changes
   without an observed failure and a controlled experiment that justifies it.
6. **Honest reporting.** Do not claim a test passed unless it was executed. Do
   not hide failed or unfavourable results.
7. **Preserve user work.** Do not rewrite commits or discard unrelated changes.

---

## 4. Definition of quality (acceptance criteria)

The system is acceptable only when all of the following are demonstrated:

### 4.1 Retrieval
- Evidence-level Recall@K, Precision@K and MRR@K are reported against the verified
  20-unit ground truth.
- No regression against the accepted baseline without a recorded, justified
  trade-off.

### 4.2 Grounding and citation
- Every substantive answer claim can be traced to a retrieved chunk, and the
  chunk can be traced to an exact character range and page in
  `data/satcom-ngp.pdf`.
- Citations are derived from metadata, not guessed from model output.

### 4.3 Abstention
- When no retrieved evidence supports the question, the system says so and does
  **not** answer from parametric memory.

### 4.4 Multi-clause handling
- Questions whose answer spans several clauses (e.g. `EXP001-Q08`: 3.4.1–3.4.3;
  `EXP001-Q13`: 4.3.1–4.3.4) are answered with all required evidence, and missing
  units are visible in evaluation.

### 4.5 Document integrity
Every chunk set must satisfy the audit in `src/rag_core/chunker.py`
(`audit_source_coverage`):

- each chunk is an exact slice of the extracted source;
- offsets are within bounds;
- chunk ranges do not overlap;
- **100% of non-whitespace source characters are covered**;
- chunk IDs are unique and stable across runs;
- `chunk_sequence` is contiguous and in source order;
- page metadata is derivable from offsets;
- no empty chunks and no chunks larger than the configured maximum.

### 4.6 Reproducibility
- Configuration, chunks, evidence mapping, retrieval results, metrics and
  comparison are written to `experiments/results/<experiment>/`.
- The full test suite passes from a clean checkout of the committed tree.

---

## 5. Architecture

### 5.1 Canonical core — `src/rag_core/`

| Module | Responsibility |
|---|---|
| `document.py` | `Page` / `SourceDocument`; PDF extraction; global character offsets; `page_for_offset`. |
| `chunker.py` | `StructureAwareChunker`; page-aware heading detection; conservative clause-ID normalisation; unique chunk IDs; `audit_source_coverage`. |
| `index.py` | `DenseIndex` over Chroma; `DenseIndexConfig`; `RetrievedChunk`; `chunk_to_metadata`. |
| `evaluation.py` | `Question` / `EvidenceUnit`; verified EXP-001 manifest loading into global offsets; metric functions; coverage summary. |

Design constraints:

- Extraction is conservative: no stripping or normalisation is stored as source
  text.
- Offsets are half-open (`start <= offset < end`) in global document
  coordinates.
- Heading blocks are scanned **per page** so a page footer cannot fuse with the
  next page's first clause (`2` + `2.4` must not become `22.4`).
- Clause-ID normalisation is conservative and article-aware; it never strips a
  legitimate ID such as `12.4.1` inside `ARTICLE-12`.
- Unrecognised or malformed headings never cause content loss; unlabelled body
  text becomes an unsectioned chunk.

### 5.2 Experiments

```
experiments/
  exp001.py                  # frozen accepted fixed-character baseline
  exp001_questions.csv       # approved 15-question dataset (immutable)
  exp001_ground_truth.json   # 20 verified evidence units (immutable)
  exp002.py                  # structure-aware chunking experiment
  results/
    exp001/                  # historical, invalidated original run (do not use)
    exp001_v2/               # corrected intermediate run
    exp001_v3/               # ACCEPTED fixed-character baseline
    exp002/                  # structure-aware run + comparison + failure analysis
```

### 5.3 Production application (current state)

- `backend/main.py` and `frontend/` contain the legacy application shell.
- The backend still points at legacy collections and an unvalidated advanced
  pipeline. It is **not yet** the validated production path and must not be
  presented as the outcome of this work.
- The production ingestion path will call `src/rag_core`; generation, citation
  rendering and abstention are **not yet implemented/evaluated** in the validated
  pipeline.

---

## 6. Metrics (must remain identical across experiments)

Given a question with a set of required verified evidence units:

- **Recall@K** = distinct required evidence units overlapped by the top-K
  retrieved chunks / total required units.
- **Precision@K** = retrieved chunks in the top-K that overlap at least one
  required evidence unit / K.
- **MRR@K** = reciprocal rank of the first retrieved chunk overlapping any
  required evidence unit; 0 if none.

A chunk is relevant only when it overlaps an **actual supporting evidence span**
(character-level, same source). Headings are never ground truth.

---

## 7. Experiment protocol

1. State a hypothesis and the single changed variable.
2. Hold everything else constant (see §3.4 and the experiment's `config.json`).
3. Run integrity checks **before** measuring retrieval.
4. Record per-question and aggregate metrics.
5. Manually inspect failures and regressions; aggregate numbers alone are not
   sufficient on a 15-question dataset.
6. Record outcome as **accepted**, **rejected** or **inconclusive**, with the
   reasoning, in `docs/ENGINEERING_LOG.md`.
7. Keep outputs isolated per experiment.

### Experiment registry

| ID | Variable changed | Status | Aggregate Recall@5 / Precision@5 / MRR@5 |
|---|---|---|---|
| EXP-001 v3 | none (baseline) | **Accepted** | 0.8889 / 0.2533 / 0.7944 |
| EXP-002 | chunking → structure-aware | **Rejected** as default | 0.8833 / 0.1867 / 0.6244 |

Details: `docs/EXP-001.md`, `docs/EXP-002.md`, `docs/ENGINEERING_LOG.md`.

---

## 8. Current status and roadmap

### Done
- Independent repository audit and identification of document-integrity defects.
- Phase 2: canonical document model, structure-aware chunker, strict integrity
  audit, shared dense index, shared evidence/metrics core.
- Phase 3: EXP-002 controlled structure-aware chunking experiment, including
  failure analysis and a documented **rejection** of this configuration as the
  production default.
- Test suite: **57 passed** (`python -m pytest tests -q`), including 35 new
  core/evaluation tests.

### Next
- Choose the next controlled experiment based on the observed compound-question
  failure mode (clause-level chunks lose parent/child coverage). Candidates must
  be justified, not assumed; parent-context/sibling expansion is the leading
  hypothesis but has **not** been validated.
- Build the validated production ingestion path on `src/rag_core`.
- Add generation with citation rendering and abstention, then evaluate
  faithfulness, answer correctness and citation validity.
- Fix the legacy backend defect and retire or isolate unvalidated legacy paths
  before exposing the app.

---

## 9. Known limitations and risks

- **Small evaluation set.** 15 questions / 20 evidence units. Treat small metric
  differences as noisy; require manual failure inspection.
- **Clause granularity trade-off.** Clause-level chunks help pinpoint answers but
  can lose the broader context needed for compound questions (observed in
  EXP-002: Q08 and Q13).
- **Very small chunks.** EXP-002 produced 16 chunks under 100 characters, which
  can carry weak embedding signal.
- **Legacy backend defect (open).** `backend/main.py` contains
  `raise HTTPException(status_status=429, ...)` (invalid keyword) and still uses
  legacy collections/pipeline. Not used by the validated experiments.
- **Generation is not validated.** No citation-faithfulness or abstention
  evaluation exists yet. README performance claims predate and are not produced
  by the validated pipeline in this brief.
- **Chroma telemetry warnings.** Harmless
  `Failed to send telemetry event … capture() takes 1 positional argument`
  messages may appear; they do not affect results.
- **Environment.** Python 3.11, Chroma 0.5.23, sentence-transformers 3.0.1,
  `sentence-transformers/all-MiniLM-L6-v2`.

---

## 10. Commands

```bash
# Full test suite
python -m pytest tests -q

# Reproduce EXP-002 (writes to experiments/results/exp002/)
python -m experiments.exp002
```

The accepted EXP-001 baseline is reproduced only from its frozen runner and
artifacts; it is not re-run as part of normal development.

---

## 11. Documentation index

| Document | Purpose |
|---|---|
| `docs/PROJECT_BRIEF.md` | This specification and acceptance criteria. |
| `docs/ENGINEERING_LOG.md` | Decisions, defects, experiments, rejected approaches, risks, test evidence. |
| `docs/EXP-001.md` | Accepted fixed-character baseline. |
| `docs/EXP-002.md` | Structure-aware chunking experiment report. |
| `docs/CHUNKING_STRATEGY.md` | Earlier chunking design analysis (context only; EXP-002 supersedes its unvalidated assumptions). |
