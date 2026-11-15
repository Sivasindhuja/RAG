# M4 Evaluation Baseline (extractive)

## Summary
```json
{
  "config": {
    "mode": "extractive",
    "k": 5,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "collection": "satcom_ngp_production_fixed_char",
    "defaults": {
      "exp003_sibling_ratio": 0.5,
      "answerability_threshold": 0.5,
      "cross_reference_selection": false,
      "associated_heading_units": false,
      "definition_intent_boost": false
    }
  },
  "answerable": {
    "count": 15,
    "answered": 14,
    "abstained": 1,
    "supported_answers": 12,
    "supported_answer_rate": 0.8,
    "statement_support_rate": 0.3333,
    "citation_support_rate": 0.5161,
    "retrieval": {
      "mean_recall@5": 0.8889,
      "mean_precision@5": 0.2533,
      "mean_mrr@5": 0.7944
    }
  },
  "unanswerable": {
    "count": 12,
    "correct_abstentions": 12,
    "abstention_accuracy": 1.0
  },
  "latency_ms": {
    "total": {
      "count": 27,
      "mean": 658.412,
      "p50": 19.704,
      "p95": 22.648,
      "max": 17265.834
    },
    "retrieval": {
      "count": 15,
      "mean": 1168.481,
      "p50": 18.911,
      "p95": 5194.283,
      "max": 17263.756
    },
    "selection_and_extractive": {
      "count": 15,
      "mean": 1.655,
      "p50": 1.606,
      "p95": 2.003,
      "max": 2.078
    }
  },
  "notes": "Baseline measured on preserved datasets; not a production-quality claim. Thresholds must be derived from measured baselines, not set arbitrarily."
}
```

## Answerable per-question
| id | status | recall@5 | supported | stmt sup/total | cit sup/total | total_ms |
|---|---|---:|---:|---:|---:|---:|
| EXP001-Q01 | answered | 1.0 | True | 1/3 | 1/2 | 17265.834 |
| EXP001-Q02 | answered | 1.0 | True | 1/3 | 1/3 | 20.517 |
| EXP001-Q03 | answered | 1.0 | True | 1/3 | 1/2 | 19.216 |
| EXP001-Q04 | answered | 1.0 | True | 1/3 | 1/2 | 20.086 |
| EXP001-Q05 | abstained | 0.0 | False | 0/3 | 0/2 | 19.784 |
| EXP001-Q06 | answered | 1.0 | True | 1/3 | 1/2 | 21.233 |
| EXP001-Q07 | answered | 1.0 | True | 1/3 | 1/2 | 18.781 |
| EXP001-Q08 | answered | 0.3333 | False | 0/3 | 1/2 | 19.677 |
| EXP001-Q09 | answered | 1.0 | True | 1/3 | 2/2 | 19.687 |
| EXP001-Q10 | answered | 1.0 | False | 0/3 | 0/2 | 23.028 |
| EXP001-Q11 | answered | 1.0 | True | 1/3 | 1/1 | 20.619 |
| EXP001-Q12 | answered | 1.0 | True | 1/3 | 1/2 | 19.753 |
| EXP001-Q13 | answered | 1.0 | True | 3/3 | 2/2 | 20.738 |
| EXP001-Q14 | answered | 1.0 | True | 2/3 | 2/3 | 21.325 |
| EXP001-Q15 | answered | 1.0 | True | 1/3 | 1/2 | 21.76 |

## Unanswerable
| id | expected abstain | abstained | correct | reason |
|---|---|---|---|---|
| UNANS-01 | true | True | True | best distance 1.7416 exceeds hard limit 1.3 |
| UNANS-02 | true | True | True | best distance 1.7808 exceeds hard limit 1.3 |
| UNANS-03 | true | True | True | lexical support 0.2000 is below 0.5 |
| UNANS-04 | true | True | True | best distance 1.1968 needs support >= 0.7 (got 0.6667) |
| UNANS-05 | true | True | True | best distance 1.4218 exceeds hard limit 1.3 |
| UNANS-06 | true | True | True | best distance 1.8221 exceeds hard limit 1.3 |
| UNANS-07 | true | True | True | best distance 1.7089 exceeds hard limit 1.3 |
| UNANS-08 | true | True | True | best distance 1.2386 needs support >= 0.7 (got 0.5000) |
| UNANS-09 | true | True | True | best distance 1.7956 exceeds hard limit 1.3 |
| UNANS-10 | true | True | True | best distance 1.6844 exceeds hard limit 1.3 |
| UNANS-11 | true | True | True | lexical support 0.2500 is below 0.5 |
| UNANS-12 | true | True | True | best distance 1.4505 exceeds hard limit 1.3 |