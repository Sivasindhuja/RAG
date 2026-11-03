# EXP-001 — Original Run (Historical, INVALIDATED)

This directory contains the **original** EXP-001 retrieval outputs.

> **⚠ INVALIDATED — DO NOT USE AS BASELINE**
>
> The original EXP-001 run is **invalidated** and must not be treated as a
> trustworthy retrieval baseline. Two ground-truth defects were identified and
> corrected under `experiments/results/exp001_v2/`:
>
> 1. The evaluation dataset did not match the approved 15-question set
>    (`experiments/exp001_questions.csv`), and
> 2. Ground-truth spans used heading-only detection instead of the actual source
>    passages supporting each answer.
>
> The original Recall@5 / Precision@5 / MRR@5 figures presented here are retained
> for historical comparison only. They are superseded by the corrected run in
> `experiments/results/exp001_v2/` and must not be quoted as official baseline
> metrics.

## Contents

- `chunks.json` — chunk metadata produced by the original run.
- `retrieval_results.json` — ranked retrieval results and original metrics.

## Corrected run

See `experiments/results/exp001_v2/` and `docs/EXP-001.md` for the corrected
dataset, verified ground truth and corrected experiment results.
