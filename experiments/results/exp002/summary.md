# EXP-002 — Structure-aware chunking

- Experiment ID: `EXP-002`
- Embedding: `sentence-transformers/all-MiniLM-L6-v2`
- Chunking: structure-aware, max 1000 characters
- Retrieval: dense-only, top-K = 5
- Dataset: 15 approved questions / 20 verified evidence units

## Aggregate metrics

| Metric | EXP-002 | EXP-001 v3 | Delta |
|---|---:|---:|---:|
| recall@5 | 0.8833 | 0.8889 | -0.0056 |
| precision@5 | 0.1867 | 0.2533 | -0.0666 |
| mrr@5 | 0.6244 | 0.7944 | -0.1700 |

## Chunk statistics

- Chunks: 107
- Characters: min 22, median 256.0, p90 761.0, max 1000
- Oversized chunks: 0
- Small chunks (<100 chars): 16
- Multi-page chunks: 7
- Clauses split into multiple parts: 5
- Source coverage: 100.0000% non-whitespace, overlaps=0, integrity ok=True

## Per-question comparison

| Question | Recall@5 | Precision@5 | MRR@5 | ΔRecall | ΔMRR |
|---|---:|---:|---:|---:|---:|
| EXP001-Q01 | 1.0000 | 0.2000 | 0.2000 | +0.0000 | -0.0500 |
| EXP001-Q02 | 1.0000 | 0.2000 | 1.0000 | +0.0000 | +0.0000 |
| EXP001-Q03 | 1.0000 | 0.2000 | 1.0000 | +0.0000 | +0.0000 |
| EXP001-Q04 | 1.0000 | 0.2000 | 1.0000 | +0.0000 | +0.0000 |
| EXP001-Q05 | 1.0000 | 0.2000 | 1.0000 | +1.0000 | +1.0000 |
| EXP001-Q06 | 1.0000 | 0.2000 | 0.5000 | +0.0000 | -0.5000 |
| EXP001-Q07 | 1.0000 | 0.2000 | 0.5000 | +0.0000 | -0.5000 |
| EXP001-Q08 | 0.0000 | 0.0000 | 0.0000 | -0.3333 | -0.3333 |
| EXP001-Q09 | 1.0000 | 0.2000 | 1.0000 | +0.0000 | +0.0000 |
| EXP001-Q10 | 1.0000 | 0.2000 | 0.3333 | +0.0000 | +0.0000 |
| EXP001-Q11 | 1.0000 | 0.2000 | 1.0000 | +0.0000 | +0.0000 |
| EXP001-Q12 | 1.0000 | 0.2000 | 0.5000 | +0.0000 | -0.5000 |
| EXP001-Q13 | 0.2500 | 0.2000 | 0.3333 | -0.7500 | -0.6667 |
| EXP001-Q14 | 1.0000 | 0.2000 | 0.5000 | +0.0000 | -0.5000 |
| EXP001-Q15 | 1.0000 | 0.2000 | 0.5000 | +0.0000 | -0.5000 |

## Failure inspection

### EXP001-Q05 — Recall@5 1.0000, MRR@5 1.0000

- Question: Who is responsible for obtaining licenses to offer services in a territory?
- Required clauses: 2.5.6
- Missing evidence: none

  1. `2.5.6` p3-3 relevant=True overlap=203 — 2.5.6 The responsibility for obtaining the necessary licenses to offer a service in a  particular territory (in India or
  2. `3.5` p6-6 relevant=False overlap=0 — 3.5 RESPONSIBLE AUTHORITY FOR ISSUE OF LICENSES 
  3. `3.6` p6-6 relevant=False overlap=0 — 3.6 NORMS, GUIDELINES AND PROCEDURES TO BE FOLLOWED IN  ISSUING THE LICENSES    In case of satellite systems of Privat e
  4. `3.7.2.2` p10-10 relevant=False overlap=0 — 3.7.2.2 Department of Space may charge a re asonable fee for issuing a license for  establishing and operating an Indian
  5. `3.7.4.7` p11-11 relevant=False overlap=0 — 3.7.4.7 DOS shall have the right to take over th e system, equipment and networks of the  licensee in part or in whole i

### EXP001-Q08 — Recall@5 0.0000, MRR@5 0.0000

- Question: What authorisations or licenses are required to establish an Indian Satellite System?
- Required clauses: 3.4.1, 3.4.2, 3.4.3
- Missing evidence: EXP001-Q08-E01, EXP001-Q08-E02, EXP001-Q08-E03

  1. `3.4` p5-5 relevant=False overlap=0 — 3.4 LICENSES REQUIRED    For establishing Indian Satellite Systems three distinct authorisations/licenses will  be requi
  2. `3.5` p6-6 relevant=False overlap=0 — In so far as the first two authorisati ons/licenses of 3.4 a bove are concerned a  committee consisting of Secretaries t
  3. `3.6.9` p8-8 relevant=False overlap=0 — 3.6.9 For 3.6.7(c) and 3.6.7(d) above, where these are linked to particular service (e.g.,  Direct Broadcast Satellite S
  4. `4.4.3` p14-14 relevant=False overlap=0 — 4.4.3 For services listed from 4.4.2 to 4.4.4 wherever capacity and capability exists in  the Indian Satellites, the ser
  5. `-` p12-12 relevant=False overlap=0 — (c) Satellite broadcasting includi ng Direct to Home (DTH) TV  broadcasting, may be licensed by the Licensing Authority 

### EXP001-Q13 — Recall@5 0.2500, MRR@5 0.3333

- Question: What are the categories of domestic satellite communication services?
- Required clauses: 4.3.1, 4.3.2, 4.3.3, 4.3.4
- Missing evidence: EXP001-Q13-E01, EXP001-Q13-E02, EXP001-Q13-E03

  1. `4.1` p13-13 relevant=False overlap=0 — 4.1 SERVICE CLASSIFICATION    The Satellite Communication Services could be broadly classified as    1. International (i
  2. `4.3` p13-13 relevant=False overlap=0 — 4.3 DOMESTIC COMMUNICATION  SERVICES CLASSIFICATION    In so far as Domestic Satellite Co mmunications Services are conc
  3. `4.3.4` p14-14 relevant=True overlap=182 — 4.3.4 Land Mobile Satellite Services (Domestic and Coastal Waters):    This includes communications to and from moving v
  4. `4.2` p13-13 relevant=False overlap=0 —  element of  domestic traffic, in respect for operations with foreign satellites.
  5. `4.2` p13-13 relevant=False overlap=0 — In so far as International communication is concerned the existing practice of  using International Inter-Governmental S

## Decision

See `docs/ENGINEERING_LOG.md` for the accepted/rejected/inconclusive decision and the
manual inspection that accompanies these metrics. The aggregate numbers alone are not
sufficient evidence because the dataset has only 15 questions.
