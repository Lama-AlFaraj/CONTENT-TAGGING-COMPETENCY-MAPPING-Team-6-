# Infrastructure Benchmark Report
## Content Tagging & Competency Mapping: serving Qwen2.5-7B-Instruct-AWQ on one T4

**Status:** draft. Sections marked **TBD** are filled after the W3D5 lab run (Part A) and the real-content run with the prefix cache off (Part B).

## 1. Summary

- The model runs on a single Tesla T4 (15 GB) with AWQ 4-bit quantization and vLLM 0.29.0. All requests in the preliminary runs succeeded once the context limit was raised to 16,384 tokens.
- On real course files, throughput saturates at about 1 request/s (about 100 tokens/s) at concurrency 10, where average latency is about 9.5 s.
- Compute cost per course is small either way. The preliminary estimate is about $0.006 per course when the prompt cache is hot, and about $0.05 per course when every file is new (assuming 12 files per course and $1.80/hour). That is about $1 to $10 per month for 200 courses if the GPU is billed only while working, versus about $1,296 per month if it runs 24/7. The billing model matters far more than the per-request figure.
- Output quality has not been measured in this report (see section 6).

## 2. Setup

| Parameter | Value |
|---|---|
| Model | Qwen/Qwen2.5-7B-Instruct-AWQ |
| Engine | vLLM 0.29.0, OpenAI-compatible server |
| GPU | 1x Tesla T4, 15,360 MB VRAM (about 12.1 GB in use) |
| dtype / max length | float16 / 16,384 tokens |
| GPU memory utilization | 0.85 |
| Attention backend | TRITON_ATTN (FlashAttention 2 unsupported on T4) |
| Harness | `bench.py` (lab harness, unmodified): streaming requests, one warm-up request per level (excluded), nearest-rank percentiles |
| Concurrency levels | 1, 2, 4, 8, 16 (lab); 1, 2, 5, 10 (preliminary custom script) |
| Requests per level | 20 |

Full flags and versions: `model-lock.md`.

**Context length.** With `--max-model-len 8192`, the two longest files (about 9,800 and 8,900 input tokens) were rejected, giving a 15% error rate. At 16,384 there were no errors.

## 3. Lab results (W3D5, `prompts.txt`, prefix cache on)

**TBD.** Insert the `bench.py` table (`concurrency, tokens/s, ttft_p95, latency_p95, errors`), the SLO, the knee concurrency, tokens/s at the knee, and the max sustainable request rate. Report the knee at the SLO, not the peak throughput.

## 4. Real course content (12 files, about 3,900 input tokens on average)

### 4.1 Preliminary results (custom script, prefix cache on)

| Concurrency | Avg latency | p95 | Throughput | Tokens/s | Errors |
|---|---|---|---|---|---|
| 1 | 8.55 s | 33.9 s | 0.117 req/s | 12.3 | 0% |
| 2 | 4.07 s | 5.7 s | 0.473 req/s | 50.0 | 0% |
| 5 | 6.05 s | 8.4 s | 0.792 req/s | 83.4 | 0% |
| 10 | 9.52 s | 11.7 s | 0.971 req/s | 102.0 | 0% |

**How to read this.** The sweep sends 20 requests cycling through 12 files. At concurrency 1 the first 12 requests each process a new file from scratch, and the last 8 repeat files. Later levels then re-read prompts already in the prefix cache. The concurrency-1 row is therefore the closest to production (new file every time), and the other rows are optimistic. This is an interpretation of the numbers and of the server configuration (prefix caching was enabled); it has not been isolated with a cache-off run.

Throughput flattens between concurrency 5 and 10 (+23%) while latency rises by more than 50%, so higher concurrency will not help on one T4.

### 4.2 Cache-off run

**TBD.** Insert the `bench_report_real.json` table from Part B (`--no-enable-prefix-caching`, `--max-tokens 400`). This becomes the primary result; section 4.1 becomes the optimistic bound.

Note: `bench.py` reads one prompt per line, so each file's text was collapsed onto a single line, which removes line breaks inside the files.

## 5. Cost

Assumptions (confirm with the use-case owner): 12 files per course, 200 courses per month, $1.80 per GPU-hour (cloud-equivalent rate).

| Scenario | Throughput | Cost per request | Cost per course | Monthly, pay-per-use |
|---|---|---|---|---|
| Cache hot (best level, concurrency 10) | 0.971 req/s | about $0.0005 | about $0.0062 | about $1.24 |
| New files (concurrency 1 as proxy) | 0.117 req/s | about $0.0043 | about $0.051 | about $10.3 |
| GPU billed 24 h/day | n/a | n/a | n/a | about $1,296 |

The pay-per-use rows assume the GPU is fully busy while billed and idle time is free. Replace the second row with the Part B result when available.

## 6. Output quality

**TBD.** The repository contains evaluation files for the tagging model (`Model/qwen7b_*_evaluation_v7.csv`). State which model and serving setup produced them, and summarize the tag and competency scores here. If they came from a different setup (for example unquantized weights), say so, because 4-bit AWQ may score differently.

## 7. Limitations

1. Prefix caching inflated part of the preliminary results (section 4.1).
2. Outputs were short in the preliminary runs (about 100 tokens per request). Longer structured output lowers throughput.
3. `bench.py` counts streamed content chunks as tokens, so tokens/s is approximate.
4. Small samples: 20 requests per level, so p95 is indicative only.
5. One T4 with AWQ only. Results do not transfer to faster GPUs, which need their own measurement and price.
6. Files per course and courses per month are assumptions.
7. Packages were installed unpinned (latest at the time); versions used are listed in `requirements.txt`.

## 8. Recommendations

- Decide the billing model first: pay-per-use versus always-on.
- Use the cache-off numbers for cost planning.
- Choose concurrency from the knee at the agreed SLO, not from peak throughput.
- Validate output quality on a sample before adopting the quantized model.
