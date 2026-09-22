# V6 vs V7 vs V8.4 Comparison

## Competency Mapping

| Metric | V6 | V7 | V8.4 |
|---|---:|---:|---:|
| Competency Micro Precision | 37.50% | 37.50% | 76.92% |
| Competency Micro Recall | 18.75% | 18.75% | 25.64% |
| Competency Micro F1 | 25.00% | 25.00% | 38.46% |
| Competency Macro F1 | 25.56% | 25.56% | 39.17% |
| Exact Match | 0.00% | 0.00% | Not reported |
| Difficulty Accuracy | 58.33% | 58.33% | Not reported |
| Recall@10 | Not reported | Not reported | 31.25% |
| Hit Rate@10 | Not reported | Not reported | 91.67% |

## Tagging

| Metric | V6 | V7 | V8.4 |
|---|---:|---:|---:|
| Tag Micro Precision | 9.09% | 9.09% | Not reported |
| Tag Micro Recall | 4.21% | 4.21% | Not reported |
| Tag Micro F1 | 5.76% | 5.76% | Not reported |
| Tag Macro F1 | 5.53% | 5.53% | Not reported |

## Architecture Evolution

### V6
Initial Qwen-based competency mapping pipeline.

### V7
Intermediate Qwen-based prediction and evaluation stage.

### V8.4
Qwen topic tagging
→ semantic summary
→ E5 + lexical hybrid retrieval
→ cross-chunk aggregation
→ Qwen reranking
→ final competencies.

## V8.4 Final Quality

- Macro Precision: 75.00%
- Macro Recall: 27.08%
- Macro F1: 39.17%
- Micro Precision: 76.92%
- Micro Recall: 25.64%
- Micro F1: 38.46%
- Recall@10: 31.25%
- Hit Rate@10: 91.67%

## Deployed E2E Baseline

- Requests: 12
- Success: 12/12
- Error Rate: 0.0%
- Average Latency: 16.385 s
- P50: 11.723 s
- P95: 37.040 s
- P99: 39.267 s
- Throughput: 0.061 req/s

Note: deployed latency/throughput results are infrastructure measurements and are kept separate from offline model-quality evaluation.
