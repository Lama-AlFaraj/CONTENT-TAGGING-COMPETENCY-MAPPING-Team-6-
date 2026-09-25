# Model Comparison — Commercial vs. Open-Weight, for Content Tagging & Competency Mapping

**Status:** §1 is measured (already in the repo). §2–3 are desk research to widen the comparison beyond OpenAI, clearly separated from measured numbers — no new API calls were run to produce them.

## 1. Measured comparison: OpenAI vs. deployed V8.4 (already in this repo)

From `Infrastructure_Benchmark/OPENAI_VS_V84_COMPARISON.md` (`compare_openai_v84.py`, model `OPENAI_MODEL` env var, default `gpt-5.6-luna` per `openai_evaluation.py`):

- 12 files compared, **no gold labels available for this run**, so it is an agreement study, not an accuracy study.
- Exact competency-set agreement: 3/12 (25.0%). Average Jaccard similarity: 0.465.
- Difficulty-label agreement: 10/12 (83.3%).
- Latency: OpenAI avg 19.24s vs. V8.4 avg 16.39s (V8.4 faster by ~2.9s on average, on this sample).

**Read this carefully:** low competency-set agreement does not tell you which system is *right* — it only tells you the two systems often propose different (but sometimes overlapping, Jaccard 0.465) competency sets for the same file. Closing this gap requires gold labels for the same 12 files scored against both systems, which does not yet exist for the OpenAI side.

## 2. Widening the commercial comparison (desk research — not run against this pipeline)

The project currently only measured one commercial model. For the "add commercial comparisons" deliverable, here is the broader landscape a reviewer would expect to see considered, with the caveat that **only the OpenAI row above is empirically measured on this task** — the rest is general-market pricing/context/positioning as of Sept 2026, useful for justifying *why* OpenAI was the commercial baseline chosen, or for planning a follow-up run.

| Model (commercial, hosted API) | Approx. context window | Approx. pricing (input / output, per 1M tokens) | Positioning relevant to this task |
|---|---|---|---|
| OpenAI (model used in this repo's comparison: `gpt-5.6-luna`, i.e. whatever `OPENAI_MODEL` resolves to) | — (see repo's own run) | — | Already measured above; the only commercial baseline with real numbers on this dataset. |
| GPT-4o-mini class | 128K | roughly $0.15 / $0.60 | Small, cheap, commonly used as the "budget commercial" comparison point for structured extraction tasks; a reasonable second commercial baseline since it's priced close to self-hosting a 7B model. |
| Gemini 2.5/1.5 Flash class | up to 1M | roughly $0.15 / $0.60 (varies by exact model/tier) | Much larger context window than the deployed pipeline needs (files here run 1.5k–9.8k tokens) — the differentiator would be simplicity (no chunking needed) rather than cost. |
| Claude Haiku (3.5/4.5 class) | 200K | roughly $0.25–$0.80 / $1.25–$4.00 (varies by version) | Positioned as a quality-per-dollar option; worth a spot-check run since instruction-following on structured JSON output is a common differentiator among "mini" tiers. |

**Recommendation for the deliverable:** add **one** more commercial data point empirically (not just desk research) — GPT-4o-mini-class is the natural second point since `compare_openai_v84.py` already exists and only needs `OPENAI_MODEL` changed and the script re-run. That turns this table from "desk research" into "measured" with minimal new engineering.

## 3. Open-weight alternatives to Qwen2.5-7B-Instruct (desk research)

The pipeline currently locks to Qwen2.5-7B-Instruct (served AWQ-quantized). For the "add open-weight comparisons" deliverable, the realistic 7–9B-class alternatives are:

| Model | Context window | Relative strengths (general benchmarks, not this task) | Relevant trade-off for this pipeline |
|---|---|---|---|
| **Qwen2.5-7B-Instruct** (current) | 128K | Generally leads same-size Llama/Mistral models on knowledge/reasoning benchmarks (e.g. MMLU-Pro-style comparisons); large context window comfortably covers this project's longest files (~9.8K tokens) with room to spare. | Already integrated, already quantized (AWQ) and benchmarked end-to-end — switching cost is the main argument against changing. |
| **Llama 3.1 8B Instruct** | 128K | Broadest hosting/provider support if the team ever wants a hosted-API fallback instead of self-hosting; competitive on classification/JSON-structured tasks. | Slightly larger than Qwen 7B; benchmarked slightly behind Qwen2.5-7B on several general leaderboards, but the gap is task-dependent and not measured for competency mapping specifically. |
| **Mistral 7B Instruct (v0.3)** | 32K (older Mistral models have smaller windows than Qwen/Llama) | Fastest and cheapest of the three at the same 4-bit precision; simplest license terms. | Smaller context window is the practical risk — the two longest files in the evaluation set (~9.8K, ~8.9K tokens) already needed 16K max-length headroom; a 32K-context model has margin, but it's tighter than Qwen's 128K, and pre-Mistral-Nemo 7B models generally score lower on multi-step instruction-following, which matters for this pipeline's multi-stage tagging→retrieval→reranking prompts. |

**None of these have been run against the 12-file evaluation set.** The fastest way to make this a measured comparison rather than a desk table: point `Model/v8/model_v8.py`'s HTTP call at a second vLLM instance serving Llama-3.1-8B-Instruct-AWQ (same serving stack, same k8s pattern as `Infrastructure/k8s/vllm-7b.yaml`, just a different `--model`), and re-run `evaluate_v8.py` against it. That reuses 100% of the existing evaluation harness.

## 4. Suggested final comparison table shape (to fill in as more runs happen)

| System | Type | Macro F1 | Micro F1 | Recall@10 | Avg latency | Cost per course (est.) |
|---|---|---:|---:|---:|---:|---:|
| Qwen2.5-7B-Instruct-AWQ (V8.4, deployed) | Open-weight, self-hosted | 39.17% | 38.46% | 31.25% | 16.39s | ~$0.006–$0.05 |
| OpenAI (`gpt-5.6-luna`) | Commercial | not scored (no gold labels in this run) | — | — | 19.24s | not computed in this repo |
| GPT-4o-mini class | Commercial | TBD (re-run `compare_openai_v84.py` with `OPENAI_MODEL` changed) | TBD | TBD | TBD | ~usage-based, see §2 pricing |
| Llama 3.1 8B Instruct (AWQ) | Open-weight, self-hosted | TBD | TBD | TBD | TBD | similar to Qwen row, same GPU |

This table is the natural artifact to present at the final review — it currently has one fully measured row (V8.4) and one partially measured row (OpenAI, latency only); the rest are clearly marked TBD rather than invented, which matches the project's actual current state per `Roadmap.md`'s "OpenAI Comparison: PENDING" item.
