# Quantization Comparison — Qwen2.5-7B-Instruct

**Status:** the controlled A/B run proposed in §4 has been **executed** (2026-09-25), backends A (AWQ) and B (FP16), same pipeline, same 12 files, same A6000. Results are in §4a. Backend C (bnb NF4) is still not run.

## 1. Quantization approaches actually used in this project

The repo already contains **two different quantization approaches**, run at different pipeline stages, which is a real (if uncontrolled) natural comparison:

| Version(s) | Where | Method | Precision | Serving |
|---|---|---|---|---|
| V6, V7, and the V8 in-process backups (`model.py`, `model_v6.py`, `model_v8_backup.py`, `model_v8_local_backup.py`, `model_v8_before_semantic_retrieval.py`) | Local, in-process `transformers` load | `BitsAndBytesConfig(load_in_4bit=True)` — **bnb NF4**, 4-bit | Weights 4-bit NF4, compute typically fp16/bf16 | Loaded directly in the Python process (no separate serving layer) |
| **V8.4 (production)** | `Model/v8/model_v8.py` calling the deployed vLLM server | **AWQ**, 4-bit (`--quantization awq --dtype half`) | Weights 4-bit AWQ, compute fp16 | vLLM 0.29.0 OpenAI-compatible server, RTX A6000 (prod) / T4 (benchmark) |

Both are 4-bit, but they are **different quantization algorithms** (bitsandbytes NF4 vs. AWQ) with different calibration approaches, so a quality difference between V6/V7 and V8.4 is not purely an architecture effect — some of it may be attributable to the quantization method itself, confounded with the fact that V8.4 also changed the pipeline architecture (added E5 hybrid retrieval, cross-chunk aggregation, reranking).

## 2. What the existing evaluation numbers show (confounded comparison)

From `Infrastructure_Benchmark/V6_V7_V8_COMPARISON.md`:

| Metric | V6 (bnb NF4) | V7 (bnb NF4) | V8.4 (AWQ) |
|---|---:|---:|---:|
| Competency Micro F1 | 25.00% | 25.00% | **38.46%** |
| Competency Macro F1 | 25.56% | 25.56% | **39.17%** |
| Tag Micro F1 | 5.76% | 5.76% | Not reported |

**Caveat (important):** this table cannot be read as "AWQ beats bnb NF4" — V8.4 also added hybrid E5+lexical retrieval, cross-chunk aggregation, and Qwen reranking on top of the same base model. The quantization method and the pipeline architecture changed at the same time, so the quality gain is confounded. Isolating the quantization effect requires the controlled experiment in §4.

## 3. Feasibility assessment for a controlled quantization comparison

| Candidate config | VRAM needed (7B model, rough) | Fits on T4 (15 GB)? | Fits on A6000 (49 GB)? | Feasible now? |
|---|---|---|---|---|
| AWQ 4-bit (current prod) | ~5–6 GB weights + KV cache | Yes | Yes | Already running |
| bnb NF4 4-bit (V6/V7 style) | ~5–6 GB weights + KV cache | Yes | Yes | Already have the code (`model_v6.py`) |
| GPTQ 4-bit | ~5–6 GB weights + KV cache | Yes | Yes | Not currently in repo; would need a GPTQ checkpoint or one produced with `auto-gptq` |
| FP16 / unquantized | ~15 GB weights + KV cache (context up to 16k adds several more GB) | **Tight / likely no**, at `max-model-len=16384` the existing benchmark already runs ~12.1 GB VRAM at 4-bit AWQ before KV cache headroom — FP16 weights alone are already close to the T4's 15 GB ceiling | Yes, comfortably (49 GB) | Feasible on A6000 only, not on the T4 used for the infra benchmark |

**Bottom line on feasibility:** a same-hardware, same-prompt comparison between **AWQ** and **bnb NF4** is fully feasible today with no new dependencies, since both code paths already exist in the repo. Adding an **FP16 baseline** is feasible on the A6000 (not the T4), and would be the most informative addition because it tells you how much quality the 4-bit quantization is actually costing versus the un-quantized ceiling. GPTQ is possible but would require producing/downloading a GPTQ checkpoint first — treat it as a stretch goal, not required for the deliverable.

## 4. Controlled experiment: method

To isolate the quantization effect from the pipeline-architecture effect:

Steps actually followed:

1. **Fixed the pipeline.** Same `Model/v8/api.py` image, same chunking/retrieval/reranking code, only the vLLM backend swapped via `--served-model-name`.
2. **Vary only the model backend** it calls:
   - Backend A: production vLLM + AWQ (`Infrastructure/k8s/vllm-7b.yaml`).
   - Backend B: vLLM serving unquantized `Qwen/Qwen2.5-7B-Instruct` in bf16 on the same A6000 node, deployed alongside A (`Infrastructure/k8s/vllm-7b-fp16-compare.yaml`) — AWQ was scaled to 0 replicas for the duration of the run, then restored, since both need the whole GPU.
   - Backend C (optional/stretch, not run): the bnb NF4 in-process backend from `model_v6.py`.
3. **Run the same 12-file evaluation set** against each backend and recompute Macro/Micro F1, Recall@10, Hit Rate@10 with `Model/v8/evaluate_v8.py`. `evaluate_v8.py` only *scores* a predictions CSV — it does not call any model. Generate the CSV per backend with `Infrastructure_Benchmark/predict_via_api.py --base-url <backend api>` and score it with `PRED_PATH=<csv> OUT_PATH=<out csv> python3 Model/v8/evaluate_v8.py`. (Never run `run_v8.py` for a second backend without backing up `Model/v8/results/v8_predictions.csv` first — it overwrites the baseline.) Sampling is greedy (`temperature=0.0` in `model_v8.py`), so differences between backends are not sampling noise.
4. **Also record latency/throughput** for each backend at concurrency 1 (reuse `Infrastructure_Benchmark/v84_api_e2e.py`, pointed at each backend's port) so the report can state a quality-vs-speed-vs-VRAM trade-off, not quality alone.
## 4a. Results (measured)

| Backend | Quantization | Macro F1 | Micro F1 | Micro precision | Micro recall | Recall@10 | Hit Rate@10 | Avg latency (c=1) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A — production | AWQ 4-bit | 39.17% | 38.46% | 76.92% | 25.64% | 31.25% | 91.67% | 16.39s |
| B — comparison | None (FP16/bf16) | 41.39% | 40.00% | 68.75% | 28.21% | 31.25% | 91.67% | 17.93s |
| C (optional) | bnb NF4 | not run | not run | — | — | — | — | — |

Raw predictions and scores: `Model/v8/results/fp16_predictions.csv`, `Model/v8/results/fp16_evaluation.csv` (backend B), alongside the pre-existing AWQ baseline (`v8_predictions.csv`, `v8_evaluation.csv`, backed up as `*_AWQ_BACKUP.csv` before this run).

**Reading the result:**
- FP16 is marginally better on F1 (+2.2 points macro, +1.5 micro) and marginally slower (+1.5s avg). Retrieval (Recall@10, Hit Rate@10) is essentially identical, which is expected — the E5 retriever embeddings don't change with the LLM backend, only the chunk summaries feeding it do.
- The F1 gain comes with a precision/recall trade: FP16 has 2 more false positives than AWQ (5 vs 3) against 1 more true positive (11 vs 10), out of 39 gold mappings — a small, noisy sample.
- **Conclusion: no meaningful quality gap.** The ~2-point F1 difference is within what 12 files / 39 mappings can support as noise, not a demonstrated effect of quantization. AWQ is not measurably worse than FP16 here, while using ~4x less VRAM for weights and running slightly faster — supporting keeping AWQ in production.
- This isolates the quantization effect from the architecture confound in §2: same pipeline, same day, same node, only the backend's quantization changed.

## 5. Recommendation given current evidence

- **Keep AWQ for production.** The controlled A/B in §4a now directly supports this: FP16 does not show a clear quality win, while AWQ uses ~4x less VRAM for weights and is slightly faster. This closes the "TBD" item the infra report previously flagged (§6, Output quality) — quantization has been shown, not assumed, to be close to quality-neutral on this 12-file set.
- **Caveat to keep in the writeup:** 12 files / 39 gold mappings is a small sample; state the F1 gap as "not distinguishable from noise," not "FP16 is worse."
- **Cost impact of quantization is already documented:** the AWQ deployment costs an estimated $0.006–$0.05 per course depending on prefix-cache state (`Infrastructure_Benchmark_Report.md`, §5) — cite this figure only alongside its cache-state caveat, since the cache-off number is itself marked TBD in that same report.
