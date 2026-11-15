# EXP-008 Citation Metadata Correction (separate from experiment results)

**Date:** 2026-11-15  
**Context:** Post-review audit of EXP-008 Q12 citations (requested before proposing another experiment).  
**EXP-008 experiment artifacts/results preserved unchanged:** `experiments/results/exp008/*` remain as recorded history of the executed run. This document records the separate correction.

## Audit

### Q12 selected evidence (EXP-008, frozen selection result)
- Statement 1: `satcom-ngp::page_13::chunk_1::off30978`, offsets **31006–31254** (global), supported by `EXP001-Q12-E01`
  - Text: International Communications Services include PSTN, GMPCS, Value Added Services, Programme Exchange, foreign-country link termination/origination.
- Statement 2: `satcom-ngp::page_13::chunk_1::off30978`, offsets **31255–31506**, supported by `EXP001-Q12-E01`
  - Text: MMSS and AMSS may be treated as International communications (domestic-traffic element possible) for operations with foreign satellites.
- Statement 3: `satcom-ngp::page_13::chunk_0::off30178`, offsets **31006–31178**, supported by `EXP001-Q12-E01`
  - Text: partial definition-start sentence (truncated at chunk boundary) overlapping statement 1 content.

### Verification against original PDF/source (`data/satcom-ngp.pdf`, extracted text offsets)
- All statement offsets and chunk ranges are within page 13 and map to verbatim 4.2 definition content.
- Clause boundaries (document): `4.2 INTERNATIONAL COMMUNICATION SERVICES` starts at global offset **30381**; `4.3 DOMESTIC COMMUNICATION  SERVICES CLASSIFICATION` starts at **31512**.
- Therefore offsets 31006–31506 are within **clause 4.2** (before 4.3), not 4.1/4.3.
- Pages: 13 (correct). Exact source text/offsets preserved.

### Recorded EXP-008 citation metadata (historical run artifacts)
- `[1] satcom-ngp::page_13::chunk_1::off30978` labeled **section 4.3 / heading “4.3 DOMESTIC COMMUNICATION  SERVICES CLASSIFICATION”** (chunk 30978–31978, page 13)
- `[2] satcom-ngp::page_13::chunk_0::off30178` labeled **section 4.1 / heading “4.1 SERVICE CLASSIFICATION”** (chunk 30178–31178, page 13)

### Root cause (classified)
**Chunk metadata assignment issue interacting with fixed-character windowing (source-structure boundary).**
- Fixed-character chunks span multiple clauses and may start/end mid-clause.
- `FixedCharacterChunker._metadata` previously selected the first heading *inside* the chunk window, ignoring the governing clause at chunk start when a clause began before the chunk. Chunk `...off30978` (30978–31978) starts inside 4.2 but contains 4.3 heading at 31512, so it was mislabeled 4.3. Chunk `...off30178` starts at 4.1 and contains 4.2; chunk-level start-governing metadata is 4.1 but the cited statement at 31006 is in 4.2.
- Citations reported chunk-level `section`/`heading`, so statement-level clause was not reflected. **Not a ground-truth issue; not a PDF text/offset issue.**

## Smallest justified correction (implemented)

1. **Chunker (`src/rag_core/chunker.py`, `FixedCharacterChunker._metadata`):** choose governing metadata from the last heading at or before chunk start (start-governing) when available, avoiding labeling by a later in-chunk clause. Future chunk builds/indexes assign chunk `...off30978` as 4.2 (start-governing) and `...off30178` as 4.1.
2. **Citation resolution (`src/rag_core/pipeline.py`):** resolve citation `article/section/heading` from the governing clause of the first selected statement offset in each cited chunk (nearest clause/article heading at/before offset inside chunk text; if statement is before first in-chunk clause heading, retain chunk start-governing metadata instead of inheriting a later clause). Citation chunk range/page/offset/quote remain chunk-level and traceable.
3. **Production index rebuilt** with corrected chunk metadata: `chroma_db_production/satcom_ngp_production_fixed_char` (49 chunks). This affects runtime metadata, not stored experiment result files.

## Post-correction verification (Q12, EXP-008 selection settings)
- Statement offsets unchanged and correct: 31006–31254, 31255–31506, 31006–31178 (page 13).
- Citations now resolve to **4.2 / “4.2 INTERNATIONAL COMMUNICATION SERVICES”** for both cited chunks’ selected statements (verified via pipeline run after correction). Chunk ranges remain 30978–31978 and 30178–31178.

## Tests
- Added regression coverage for start-governing chunk metadata and statement-governing citation metadata (see `tests/test_citation_metadata.py`). Full suite recorded separately in engineering log.

## Limitations preserved
- EXP-008 definition-intent boost remains validated only on the current 15-question evaluation set.
- Fixed-character chunks can still span multiple clauses; citation now reflects governing clause of first selected statement per chunk, not every possible subspan label. Source offsets/pages/text remain the traceability anchor.
