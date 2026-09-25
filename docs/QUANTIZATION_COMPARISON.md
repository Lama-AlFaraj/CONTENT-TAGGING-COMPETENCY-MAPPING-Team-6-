# Quantization Comparison — Qwen2.5-7B-Instruct

**Status:** assessed from what's already in the repo (feasible today); a controlled same-hardware / same-prompt A/B run is proposed but not yet executed (needs GPU access — see §4).

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

## 4. Proposed controlled experiment (not yet run)

To isolate the quantization effect from the pipeline-architecture effect:

1. **Fix the pipeline.** Take the final V8.4 pipeline logic (`retrieval_v8.py`, chunking, reranking) unchanged.
2. **Vary only the model backend** it calls:
   - Backend A: current vLLM + AWQ (already deployed).
   - Backend B: vLLM serving the **unquantized** `Qwen/Qwen2.5-7B-Instruct` on the A6000 (`--quantization` flag omitted, `--dtype bfloat16` or `half`).
   - Backend C (optional/stretch): the bnb NF4 in-process backend from `model_v6.py`, wrapped behind the same `/predict` interface so the rest of the pipeline is untouched.
3. **Run the same 12-file evaluation set** against each backend and recompute Macro/Micro F1, Recall@10, Hit Rate@10 exactly as in `Model/v8/evaluate_v8.py`.
4. **Also record latency/throughput** for each backend at concurrency 1 (reuse `Infrastructure_Benchmark/v84_api_e2e.py`, pointed at each backend's port) so the report can state a quality-vs-speed-vs-VRAM trade-off, not quality alone.
5. **Report as a table:**

   | Backend | Quantization | Macro F1 | Micro F1 | Recall@10 | Avg latency (c=1) | VRAM used |
   |---|---|---:|---:|---:|---:|---:|
   | A | AWQ 4-bit | (from existing V8.4 run) | (from existing V8.4 run) | (from existing V8.4 run) | 16.39s | ~12.1 GB (T4 run) / TBD on A6000 |
   | B | None (FP16) | TBD | TBD | TBD | TBD | TBD |
   | C (optional) | bnb NF4 | TBD | TBD | TBD | TBD | TBD |

This isn't runnable in this sandbox (no GPU, no network egress here), but it only needs infrastructure the team already has — the A6000 deployment and the existing evaluation script — so it's a same-day run once someone stands up backend B.

## 5. Recommendation given current evidence

- **Keep AWQ for production.** It is already deployed, benchmarked, within budget on both T4 and A6000, and the existing quality numbers (even if confounded with the architecture change) are the best measured result in the project.
- **Before claiming "quantization has no quality cost," run backend B (FP16) at least once.** Right now the project has *not* demonstrated that 4-bit quantization is quality-neutral — it has only compared two different pipeline versions that also happened to use different quantization. This is explicitly one of the "TBD" items the infra report itself flags (§6, Output quality) and is worth closing before the final presentation.
- **Cost impact of quantization is already documented:** the AWQ deployment costs an estimated $0.006–$0.05 per course depending on prefix-cache state (`Infrastructure_Benchmark_Report.md`, §5) — cite this figure only alongside its cache-state caveat, since the cache-off number is itself marked TBD in that same report.
