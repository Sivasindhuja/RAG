# PROD-001 — Production Slice

Status: implemented and evaluated. Retrieval is accepted; the extractive answer
layer is a validated traceability baseline, **not** a complete answer engine.

Last updated: 2026-11-14.

Companion documents: `docs/PROJECT_BRIEF.md`, `docs/ENGINEERING_LOG.md`,
`docs/EXP-001.md`, `docs/EXP-002.md`.

---

## 1. Purpose and scope

This slice takes the **accepted EXP-001 retrieval configuration** out of the
frozen experiment script and runs it through the shared production core:

```
data/satcom-ngp.pdf
  → extract_source_document            (15 pages, 35,407 chars)
  → FixedCharacterChunker(1000, 200)   (49 chunks)
  → audit_source_coverage(...)         (release gate)
  → DenseIndex (Chroma + MiniLM)       (dense-only, top-5)
  → GroundedPipeline                   (extractive answer, citations, abstention)
```

Scope in this slice:

- production ingestion with the accepted chunking strategy;
- traceable chunk/page/offset citations;
- extractive verbatim answers;
- a configurable abstention policy;
- reproducible evaluation artifacts.

Out of scope: LLM generation, answer fluency, multi-hop reasoning and a
production web application. The optional generator hook exists but is not
evaluated.

---

## 2. Configuration

| Parameter | Value |
|---|---|
| Source | `data/satcom-ngp.pdf` |
| Chunking | `fixed_character`, 1000 chars, 200 overlap (step 800), per page |
| Chunk count | 49 |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | Chroma collection `satcom_ngp_production_fixed_char` |
| Retrieval | dense-only, top-5 |
| Answer mode | `extractive_verbatim` |
| Max answer sentences | 3 |
| Max chunks considered for selection | 5 |
| Citation quote cap | 500 chars |

The abstention policy is a separate calibration artifact:

```json
{
  "hard_max_distance": 1.30,
  "soft_max_distance": 1.10,
  "min_token_support": 0.50,
  "strict_min_token_support": 0.70
}
```

Decision order in `GroundedPipeline.answer`:

1. empty question → abstain;
2. no retrieval results → abstain;
3. best distance > `hard_max_distance` → abstain;
4. question-token lexical support < `min_token_support` → abstain;
5. distance > `soft_max_distance` **and** support <
   `strict_min_token_support` → abstain;
6. otherwise answer.

These thresholds are heuristics tied to this embedding model and index. They
must be recalibrated if either changes.

---

## 3. Document integrity

`audit_source_coverage(..., allow_overlap=True)` passed before retrieval:

| Check | Result |
|---|---|
| Chunks | 49 |
| Non-whitespace coverage | 100.0000% (29,250 / 29,250 chars) |
| Intentional overlap | 6,800 chars across 34 adjacent windows |
| Fidelity errors | 0 |
| Bounds errors | 0 |
| Duplicate IDs | 0 |
| Page-mapping errors | 0 |
| Empty / oversized chunks | 0 / 0 |
| Sequence and source order | OK |

`allow_overlap=True` is used **only** because the accepted fixed-character
strategy intentionally overlaps windows. The audit still reports every overlap
and still requires complete non-whitespace coverage.

The 49 chunks reproduce the accepted EXP-001 v3 chunk texts and page assignments
exactly; this equivalence is covered by `tests/test_fixed_chunker.py`.

---

## 4. Retrieval result

Retrieval is unchanged from the accepted baseline and reproduces it exactly:

| Metric | PROD-001 | EXP-001 v3 | Delta |
|---|---:|---:|---:|
| Recall@5 | 0.8889 | 0.8889 | +0.0000 |
| Precision@5 | 0.2533 | 0.2533 | +0.0000 |
| MRR@5 | 0.7944 | 0.7944 | +0.0000 |

Observed build and query latency (single local run, not a benchmark):

- index build: 1,927.89 ms for 49 chunks;
- pipeline query (retrieval + answer assembly): mean 32.44 ms, median 32.50 ms,
  max 34.67 ms over 15 questions.

Latency varies by machine and Chroma state; only the correctness numbers are
release-gating.

---

## 5. Grounded answer behaviour

Every answered sentence is a **verbatim slice of the retrieved source text**.
For each statement the pipeline records:

```text
text, marker, chunk_id, global start_offset, global end_offset
```

Each citation records:

```text
marker, chunk_id, source, article, section, heading,
start_page, end_page, global start_offset, end_offset, quote, distance
```

`source[start_offset:end_offset]` is the exact statement text. A citation is
emitted only for a chunk that actually contributed at least one selected
sentence; retrieved-but-unused chunks are omitted. Markers are numbered `[1]`,
`[2]`, … by first contribution.

This guarantees **faithfulness by construction** for the extractive path: the
system cannot emit an invented sentence. It does **not** guarantee relevance:
verbatim policy text can be true but fail to answer the question.

### 5.1 Measured answer-layer quality

Support is measured against the 20 verified evidence units: a statement or
citation is "supported" when its source span overlaps an evidence unit for that
question.

| Measure | Value |
|---|---:|
| Answered | 15 / 15 |
| Answers with ≥1 evidence-supported statement | 12 / 15 |
| Statements supported | 13 / 45 (0.2889) |
| Citations supported | 15 / 32 (0.4688) |
| Mean statements per answer | 3.00 |
| Mean citations per answer | 2.13 |

Per-question detail:

| ID | Recall@5 | Supported statements | Supported citations |
|---|---:|---:|---:|
| EXP001-Q01 | 1.0000 | 1/3 | 1/2 |
| EXP001-Q02 | 1.0000 | 1/3 | 1/3 |
| EXP001-Q03 | 1.0000 | 1/3 | 1/2 |
| EXP001-Q04 | 1.0000 | 1/3 | 1/2 |
| EXP001-Q05 | 0.0000 | 0/3 | 0/2 |
| EXP001-Q06 | 1.0000 | 1/3 | 1/2 |
| EXP001-Q07 | 1.0000 | 1/3 | 1/2 |
| EXP001-Q08 | 0.3333 | 0/3 | 1/2 |
| EXP001-Q09 | 1.0000 | 1/3 | 2/2 |
| EXP001-Q10 | 1.0000 | 0/3 | 0/2 |
| EXP001-Q11 | 1.0000 | 1/3 | 1/1 |
| EXP001-Q12 | 1.0000 | 1/3 | 1/2 |
| EXP001-Q13 | 1.0000 | 1/3 | 1/2 |
| EXP001-Q14 | 1.0000 | 2/3 | 2/3 |
| EXP001-Q15 | 1.0000 | 1/3 | 1/3 |

### 5.2 Failure analysis

Three distinct failure modes are visible:

1. **Retrieval miss — Q05 (Recall@5 = 0.0).** The required clause `2.5.6` is not
   in the top-5, so no answer layer can recover it. The pipeline still answers
   because the retrieved chunks share question vocabulary and pass the abstention
   policy. Abstention currently detects *out-of-corpus* questions, not *retrieved
   but irrelevant in-corpus* evidence.
2. **Partial retrieval — Q08 (Recall@5 = 0.3333).** Only one of the three required
   authorisation clauses is retrieved. The answer cannot cover `3.4.1`–`3.4.3`.
3. **Selection miss — Q10 and Q13 (Recall@5 = 1.0, answers weak).** The required
   evidence is in the top-5, but the lexical sentence selector chooses other
   sentences. Q10 selects a long list block and later clauses instead of the
   first-come-first-served rule. Q13 selects classification preamble and `4.3.4`
   but not sibling categories `4.3.2`/`4.3.3`.

Additional observed weakness: sentence splitting only breaks on sentence-ending
punctuation and blank lines. OCR line breaks (including intra-word breaks such as
`op erations`) and sibling clauses remain inside long sentences, which hurts both
readability and selection.

The statement-support metric is deliberately demanding and is **not** an
answer-correctness score: an unsupported statement may still be true policy text
that is merely irrelevant to the annotated evidence. Conversely, citation support
is chunk-level and can be positive even when the selected sentence inside that
chunk is not the needed one.

---

## 6. Abstention

The abstention set contains questions that cannot be answered from the SATCOM NGP
document.

| Split | Questions | Abstained | Rate |
|---|---:|---:|---:|
| Calibration | 8 | 8 | 1.0000 |
| Held-out | 4 | 4 | 1.0000 |
| Overall | 12 | 12 | 1.0000 |

All 15 in-corpus questions were answered, so no false abstention was observed on
the accepted question set.

Caveats:

- The calibration and held-out sets are small and hand-written, not a
  representative sample of real user questions.
- The held-out split is a smoke check, not an unbiased validation set.
- No adversarial paraphrase, near-miss or partially answerable question set has
  been evaluated.
- The policy is lexical/distance-based; it is not an entailment or
  answerability classifier.

---

## 7. Extractive versus generator path

`GroundedPipeline` accepts an optional generator hook:

```python
Generator = Callable[[str, Sequence[RetrievedChunk], Sequence[Citation]], str]
```

When supplied, the generator receives the retrieved chunks and citations and may
synthesise prose. If it returns an empty string, the extractive path is used.
Generated text has no statement-level provenance and is **not** validated for
faithfulness, citation correctness or abstention. Only the extractive path is
covered by tests and the metrics above.

Any claim about answer fluency, synthesis quality or LLM faithfulness requires a
new, controlled generation experiment. None is made here.

---

## 8. Reproduce

```bash
# Grounded answer + abstention tests
python -m pytest tests/test_pipeline.py tests/test_fixed_chunker.py -q

# Full suite (last executed: 78 passed, 25 Chroma/Pydantic warnings)
python -m pytest tests -q

# Rebuild the production index and write all artifacts
python -m experiments.production_slice
```

Outputs under `experiments/results/production_slice/`:

| File | Contents |
|---|---|
| `config.json` | Full slice configuration |
| `chunks.json` | 49 chunks with offsets, pages and text |
| `integrity_audit.json` | Coverage/fidelity/overlap audit |
| `retrieval_results.json` | Baseline comparison and top-5 per question |
| `answers.json` | Statements, citations and support checks |
| `abstention_results.json` | Calibration/held-out outcomes |
| `per_question_metrics.csv` | Compact retrieval + answer table |
| `summary.md` | Human-readable result summary |

Chroma telemetry warnings (`capture() takes 1 positional argument`) are harmless
and do not affect results.

---

## 9. Acceptance status

| Criterion | Status |
|---|---|
| Production ingestion uses accepted EXP-001 chunking | Met |
| Document integrity audit | Met |
| Retrieval reproduces accepted baseline | Met |
| Every answer statement traceable to source text | Met (extractive path) |
| Citations only from contributing chunks | Met |
| Abstention on out-of-corpus questions | Smoke met (12/12, small set) |
| Multi-clause completeness (Q08, Q13) | **Not met** |
| Answer relevance / correctness | **Not met** as a quality bar |
| Generation faithfulness | Not evaluated |

The slice is suitable as the validated retrieval-and-provenance foundation. It is
not yet a user-facing answer system.

---

## 10. Next work

1. A controlled answer-layer experiment on multi-clause selection, holding
   retrieval fixed. Observed justification: Q10 and Q13 have the evidence in the
   top-5 but do not select it.
2. Decide whether retrieval expansion is justified for Q05/Q08, with the same
   PDF, questions, ground truth, embedding, top-K and metrics held constant.
3. Expand the verified question/evidence set before making broader quality claims.
4. Evaluate any generation path separately for faithfulness and citation validity.
5. Keep the legacy `backend/` path isolated until KNOWN-001 is fixed.
