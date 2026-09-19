# Infrastructure Benchmark Report
## Content Tagging & Competency Mapping Model

**Model:** Qwen/Qwen2.5-7B-Instruct-AWQ (4-bit AWQ quantization)
**Serving engine:** vLLM 0.29.0 (OpenAI-compatible API)
**Hardware:** 1x NVIDIA Tesla T4 (15 GB VRAM), Google Colab
**Date:** September 18, 2026

---

## 1. Objective

Measure the latency, throughput, and cost of running the content tagging
model on a single self-hosted GPU, under concurrent load, using real course
content as input.

## 2. Test Setup

| Parameter | Value |
|---|---|
| Model | Qwen2.5-7B-Instruct-AWQ |
| Engine | vLLM 0.29.0 |
| GPU | Tesla T4, 15,360 MB VRAM |
| VRAM in use during test | ~12,123 MB |
| Data type | float16 |
| Max model length | 16,384 tokens |
| GPU memory utilization | 0.85 |
| Attention backend | TRITON_ATTN (FlashAttention 2 is not supported on T4) |
| Prefix caching | Enabled (default) |
| Payloads | 12 real content files (avg ~3,914 input tokens; range ~1,500 to ~9,800) |
| Requests per concurrency level | 20 (payloads cycled in order) |
| Concurrency levels | 1, 2, 5, 10 |
| max_tokens | 400 (observed output ~100 tokens per request) |
| Temperature | 0 |
| Streaming | Yes (TTFT measured from first streamed chunk) |
| Warm-up | Short and long synthetic prompts (up to ~11k tokens) before measurement |

**Note on context length:** an initial run with a max model length of 8,192
tokens rejected the two longest files (~9,800 and ~8,900 tokens), producing
a 15% error rate. Raising the limit to 16,384 removed all errors.

## 3. Results

| Concurrency | Avg latency | p95 latency | Avg TTFT | Throughput | Tokens/s | Errors |
|---|---|---|---|---|---|---|
| 1 | 3.47 s | 4.52 s | 0.11 s | 0.288 req/s | 30.6 | 0% |
| 2 | 4.12 s | 5.69 s | 0.15 s | 0.468 req/s | 49.4 | 0% |
| 5 | 6.21 s | 8.55 s | 0.23 s | 0.772 req/s | 81.3 | 0% |
| 10 | 9.61 s | 11.72 s | 0.44 s | 0.959 req/s | 99.7 | 0% |

GPU utilization was 100% in every run.

### Observations

- **Throughput saturates around concurrency 5 to 10.** Going from 5 to 10
  increased throughput by about 24% while average latency rose by about 55%.
  Higher concurrency is unlikely to help on a single T4.
- **Latency vs. throughput trade-off:** concurrency 1 to 2 gives the best
  per-request latency (3.5 to 4.1 s). Concurrency 10 gives the best
  throughput but roughly 9.6 s average latency.
- **No errors** at any concurrency level once the context limit was raised.
- Run-to-run variation was small (throughput at concurrency 10 was 0.953 and
  0.959 across two runs), but each level used only 20 requests, so figures
  should be treated as approximate.

## 4. Cost Analysis

Cost is based on the highest measured throughput (concurrency 10).

| Item | Value |
|---|---|
| Assumed GPU rate | $1.80 / hour (cloud-equivalent rate) |
| Best throughput | 0.959 req/s (~3,452 requests/hour) |
| Cost per request | ~$0.00052 |
| Files per course (assumption) | 12 |
| Cost per course | ~$0.0063 |
| Courses per month (assumption) | 200 |
| Monthly cost, pay-per-use | ~$1.25 |
| Monthly cost, GPU billed 24 h/day | ~$1,296 |

**Interpretation:** the compute cost per course is negligible if the GPU is
billed only while processing. The real cost driver is the billing model:
an always-on instance costs about $1,296 per month regardless of how many
courses are processed. The pay-per-use figure assumes the GPU is fully
utilized while running and idle time is not billed.

## 5. Limitations and Assumptions

1. **Prefix caching was enabled.** The benchmark cycles through the same 12
   payloads, so repeated runs likely benefited from cached prefill. The very
   low TTFT (~0.11 s for ~3,900-token prompts on a T4) suggests this. In
   production, new files will not benefit from the cache, so real latency is
   likely higher and real throughput lower. A run with prefix caching
   disabled is recommended to establish a conservative bound.
2. **Short outputs.** Observed output was ~100 tokens per request. If the
   production prompt requires longer structured output, decode time will
   increase and throughput will decrease.
3. **Output quality was not evaluated.** This benchmark measures speed only.
   The validity of the generated JSON and the correctness of the assigned
   tags against the skills taxonomy have not been verified. 4-bit
   quantization of a 7B model may reduce accuracy compared with a larger or
   unquantized model.
4. **Business assumptions must be confirmed.** Files per course (12) and
   courses per month (200) are assumptions and should be confirmed with the
   use-case owner. Cost per course scales linearly with files per course.
5. **Hardware.** All results are from a single Tesla T4 with AWQ
   quantization. They do not represent the performance or cost of more
   powerful GPUs (e.g., L4 or A100), which need separate measurement at
   their own hourly rates.
6. **Small sample.** Each concurrency level used 20 requests from 12
   payloads. Percentile figures (p95) are indicative only.

## 6. Recommendations

- Confirm the billing model (pay-per-use vs. always-on) before committing to
  a cost figure.
- Re-run the benchmark with prefix caching disabled to obtain a conservative
  cost estimate.
- Validate output quality on a sample of 20 to 30 files (JSON validity and
  tag membership in the skills taxonomy) before adopting this model.
- Use a concurrency of about 5 for interactive use, and 10 for batch
  processing where latency is less important.
