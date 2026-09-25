# Context-Length Benchmark — Qwen2.5-7B-Instruct-AWQ (BeamData V8.4)

**Status:** cache-on run **completed** (2026-09-24). Cache-off run **pending** (see §7).

## 1. Prior evidence in this repo

| Source | Finding |
|---|---|
| `Infrastructure_Benchmark/model-lock.md` | `--max-model-len 8192` rejected the two longest course files (~9,800 and ~8,900 input tokens): 15% error rate on the 12-file set. At **16,384** there were no errors. |
| `Infrastructure_Benchmark/Infrastructure_Benchmark_Report.md` | Real course files average ~3,900 input tokens (range ~1,500–~9,800); 0% errors at 16,384 for concurrency 1/2/5/10. |
| `lama/Infrastructure_Benchmark_Report_Qwen2.5-7B-AWQ_T4.md` | Same conclusion on T4; that run had prefix caching **on**, so it is an optimistic bound. |

Those runs varied **concurrency**. This benchmark isolates **input length**.

## 2. Method

Script: `Infrastructure_Benchmark/context_length_benchmark.py`.

- Data: the 12 evaluation files in `Model/evaluation_data/` (`README_STUDENTS.md` and `model_output_template.csv` are excluded — they are not part of the evaluation set).
- Each file is text-extracted with the same logic as `Model/v8/api.py` (python-pptx / openpyxl / notebook cells) and counted with the `Qwen/Qwen2.5-7B-Instruct` tokenizer, then assigned to a bucket.
- Requests are sent **sequentially** (concurrency 1) to `POST /predict`, **3 repeats per file**, 36 requests total.
- Backend: the deployed V8.4 stack (`Infrastructure/k8s/vllm-7b.yaml`): AWQ 4-bit, `--max-model-len 16384`, `--enable-prefix-caching` (**cache ON**).

```bash
python3 Infrastructure_Benchmark/context_length_benchmark.py \
  --base-url http://127.0.0.1:8001 \
  --data-dir Model/evaluation_data --repeat 3 \
  --out Infrastructure_Benchmark/results/context_len_cache_on.json
```

## 3. Results — by input-length bucket (cache ON)

| Bucket | Files | Requests | Errors | Avg latency | p95 latency |
|---|---:|---:|---:|---:|---:|
| short (<2k tokens) | 4 | 12 | 0 | 5.54 s | 7.40 s |
| medium (2k–6k) | 6 | 18 | 0 | 7.94 s | 10.49 s |
| long (6k–12k) | 2 | 6 | 0 | 22.27 s | 22.45 s |
| very_long (≥12k) | 0 | 0 | — | — | — |

**Overall: 36/36 requests succeeded (0% error rate).** No file in the evaluation set reaches the very_long bucket, so behaviour above ~9k input tokens is **not tested**. p95 values are computed from 6–18 samples per bucket and should be read as approximate.

## 4. Results — by file

| File | Est. input tokens | Chunks* | Avg latency (3 runs) |
|---|---:|---:|---:|
| WK4_D2_Quiz2_EnsembleLearningScenarios_Afternoon.xlsx | 872 | 1 | 3.78 s |
| WK2_D1_Quiz1_IntroToML_LectureNotebook.xlsx | 986 | 1 | 4.06 s |
| SQL_Foundations_for_Data_Science.pptx | 2,757 | 2 | 6.76 s |
| HandsOn1_Window_Functions.md | 2,370 | 2 | 6.85 s |
| Introduction to Pandas_.pptx | 1,958 | 2 | 6.98 s |
| Lecture - Pandas Basics.ipynb | 2,156 | 2 | 7.07 s |
| ML Workflow Introduction.pptx | 3,046 | 2 | 7.17 s |
| Decision Trees Revised.pptx | 1,756 | 2 | 7.32 s |
| Lecture - Decision Trees.ipynb | 3,788 | 3 | 9.36 s |
| Lecture_ML_Workflow.ipynb | 3,680 | 3 | 10.42 s |
| Prompt_Engineering.ipynb | 8,972 | 7 | 22.25 s |
| Retrieval_Augmented_Generation.ipynb | 8,380 | 7 | 22.28 s |

\*Chunks = `split_content()` in `Model/v8/model_v8.py` (6,000 characters, 500 overlap) applied to the text extracted by `api.py`.

## 5. Findings

1. **No context-length failures** at `--max-model-len 16384` across all 12 files, including the two longest (~9k tokens).
2. **Latency tracks the number of chunks, not raw token count.** Mean latency by chunk count: 1 chunk ≈ 3.9 s, 2 chunks ≈ 7.0 s, 3 chunks ≈ 9.9 s, 7 chunks ≈ 22.3 s — roughly **3.3 s per chunk**, close to linear. This is consistent with the pipeline design: `process_document()` runs the tagger (and a semantic-summary call) on each chunk **sequentially**, followed by one reranker call.
3. **No visible prefix-cache benefit on repeats.** Runs 1, 2 and 3 of the same file have near-identical latency (e.g. Prompt_Engineering: 22.4 / 22.4 / 21.9 s). Total time appears dominated by sequential generation calls rather than prompt prefill, so cache-on vs cache-off is expected to matter little for this pipeline — this must be confirmed by the cache-off run (§7).
4. **Token-count buckets are a weak predictor here.** Because chunks are processed independently, chunk count is the more informative x-axis; the bucket table is kept for comparability with the other benchmarks.

## 6. Limitations

- **Bucket assignment uses whole-file token counts**, not the prompts actually sent. Reading `model_v8.py`, no single LLM call appears to receive the whole file: the tagger and summary calls receive one chunk (≤6,000 characters) and the reranker receives the first 1,800 characters of each chunk plus retrieval candidates. The original `--max-model-len 8192` failures (§1) may therefore come from the earlier benchmark harness, which sent whole files. **This has not been measured for V8.4**; to confirm, record the maximum `prompt_tokens` seen in the vLLM logs during a run. Do not lower `--max-model-len` based on this note alone.
- Only 12 files; the very_long bucket is empty; 3 repeats per file; concurrency 1 only.
- Latency here includes the API's E5 retrieval and extraction time, not only vLLM inference.
- Cache-on is the optimistic case for repeated content; see §7.

## 7. Pending: cache-off run

Change the vLLM arg in `Infrastructure/k8s/vllm-7b.yaml` from `--enable-prefix-caching` to `--no-enable-prefix-caching` (the flag must be explicit — prefix caching is on by default), wait for the pod to become ready (2–5 min), then re-run with `--out Infrastructure_Benchmark/results/context_len_cache_off.json`. Restore the original setting afterwards. Add a second results table and a cache-on vs cache-off comparison to §3 and §5.
