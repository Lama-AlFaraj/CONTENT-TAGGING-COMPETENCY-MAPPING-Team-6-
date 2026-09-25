# Context-Length Benchmark — Qwen2.5-7B-Instruct-AWQ (BeamData V8.4)

**Status:** partially answered from existing benchmark runs; the dedicated per-bucket run below still needs to be executed against the live API (see "How to run" — this needs a GPU + the deployed API, which isn't available in this environment).

## 1. What we already know (from existing runs in this repo)

| Source | Finding |
|---|---|
| `Infrastructure_Benchmark/model-lock.md` | `--max-model-len 8192` **rejected the two longest course files** (~9,800 and ~8,900 input tokens) — a 15% error rate on the 12-file set. Raising `--max-model-len` to **16,384** removed all errors. |
| `Infrastructure_Benchmark/Infrastructure_Benchmark_Report.md` | Real course files average ~3,900 input tokens (range ~1,500–~9,800). At `max-model-len=16384`, all 20-request runs at concurrency 1/2/5/10 completed with 0% errors. |
| `lama/Infrastructure_Benchmark_Report_Qwen2.5-7B-AWQ_T4.md` | Same conclusion, T4 hardware: warm-up prompts up to ~11k tokens succeeded at `max-model-len=16384`; TTFT stayed low (~0.11s at concurrency 1) but that run had prefix caching **on**, so it is an optimistic bound, not a worst case. |

**So the headline result is already established:** 8,192 tokens is *not* enough context for this corpus; 16,384 is. What is **not** yet established is *how latency scales specifically with input length* (as opposed to concurrency) and whether that holds with prefix caching **off** — which is what the script below adds.

## 2. What this benchmark adds

`Infrastructure_Benchmark/context_length_benchmark.py` (new script) buckets the same 12 evaluation files by estimated input token count and reports per-bucket latency and error rate:

| Bucket | Token range |
|---|---|
| short | < 2,000 tokens |
| medium | 2,000–6,000 tokens |
| long | 6,000–12,000 tokens |
| very_long | ≥ 12,000 tokens |

This isolates the effect of prompt length from the effect of concurrency (which the existing `bench.py` / `v84_api_e2e.py` already cover well).

## 3. How to run it

```bash
# from the repo root, with the V8.4 API reachable (port-forward or in-cluster)
python Infrastructure_Benchmark/context_length_benchmark.py \
    --base-url http://127.0.0.1:8002 \
    --data-dir Model/evaluation_data \
    --out Infrastructure_Benchmark/results/context_length_benchmark.json
```

Run it twice if possible: once with prefix caching on (current default) and once with `--no-enable-prefix-caching` on the vLLM launch command from `model-lock.md`, so the "long file, cold cache" case — the realistic production case — is captured, not just the optimistic cached case.

## 4. Results

**TBD — fill in after running the script against the live API.** Paste the `by_bucket` section of the JSON output here, e.g.:

| Bucket | n files | errors | avg latency | p95 latency |
|---|---:|---:|---:|---:|
| short (<2k) | ? | ? | ?s | ?s |
| medium (2k-6k) | ? | ? | ?s | ?s |
| long (6k-12k) | ? | ? | ?s | ?s |
| very_long (>=12k) | ? | ? | ?s | ?s |

## 5. What to look for once you have numbers

- **Is latency roughly linear in token count, or does it jump at a threshold?** vLLM's prefill cost scales with prompt length; a smooth increase across buckets is expected. A sharp jump at one bucket boundary (e.g. chunking kicking in above ~6,000 characters per `Model_Pipeline/README.md`'s chunk size) would be worth flagging.
- **Any errors in the very_long bucket** even at `max-model-len=16384` — if the two longest files (~9,800 tokens) plus the pipeline's own prompt overhead (instructions, taxonomy context, chunk headers) pushes the *effective* prompt over 16,384, you'd see failures here specifically, which the existing "no errors at 16384" claim doesn't rule out (it was checked with the same short synthetic prompts, not necessarily headroom for prompt-plus-taxonomy overhead — worth double-checking `model_v8.py`'s prompt template length).
- **Compare cache-on vs cache-off** for the long/very_long buckets specifically — that's where a cold cache should hurt the most (largest prefill), and it's the gap the existing infra report explicitly leaves as "TBD" (`Infrastructure_Benchmark_Report.md`, §4.2).

## 6. Limitations

- Token counts are estimated with the Qwen tokenizer over raw extracted text, not the exact prompt the pipeline sends (which adds instructions, taxonomy snippets and chunk headers) — treat bucket boundaries as approximate.
- Only 12 files exist in the evaluation set, so some buckets may have very few (or zero) files — check the printed per-file token counts before trusting a bucket's average.
- `.pptx`/`.xlsx` token counts fall back to a byte-length proxy in the script if `python-pptx`/`openpyxl` aren't available in the environment running the script (they typically are, since the API itself depends on them) — verify the printed token estimates look sane before relying on the bucket assignment.
