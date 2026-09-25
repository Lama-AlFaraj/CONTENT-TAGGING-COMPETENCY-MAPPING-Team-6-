# Model Card — BeamData Content Tagging & Competency Mapping (V8.4)

## 1. Model details

| Field | Value |
|---|---|
| System name | Content Tagging & Competency Mapping — BeamData track |
| Pipeline version | V8.4 (final checkpoint) |
| Base LLM | `Qwen/Qwen2.5-7B-Instruct-AWQ` |
| Quantization | AWQ, 4-bit, `dtype=float16` |
| Serving engine | vLLM 0.29.0 (OpenAI-compatible server) |
| Retrieval model | `intfloat/multilingual-e5-base` (semantic) + lexical retrieval, hybrid weights 0.6 / 0.4 |
| Deployment | Docker → Kubernetes (namespace `shahad`, node `aidc-t12`) |
| Docker image | `shahad09/beamdata-v8-api:v8.4` |
| API | FastAPI, `Model/v8/api.py` — `GET /health`, `POST /predict` |
| Production GPU | NVIDIA RTX A6000 (~49 GB VRAM) |
| Benchmark GPU | NVIDIA Tesla T4 (15,360 MB VRAM, Google Colab) |
| Developed by | Team 6 — AI Data Center Bootcamp (Aug 2026 cohort), track owners Salman and Emad |
| Card last updated | 2026-09-24 |

Qwen2.5-7B-Instruct is not fine-tuned for this task. All task adaptation is done through prompting, chunking, retrieval, and reranking logic in the pipeline — the model card therefore documents a **system** (pipeline + model), not a fine-tuned checkpoint.

## 2. Intended use

**Primary use case.** Given a piece of learning content (slide deck, notebook, spreadsheet, markdown/text/CSV), produce:
- topic tags,
- one or more competencies from the **Saudi Skills Taxonomy** (134 rows),
- a difficulty level,
- a confidence score and free-text notes,
- the retrieval candidate set and chunk count used to reach the final mapping.

**Intended users.** Internal BeamData / AIDC bootcamp reviewers and downstream integrators (e.g. AI Hub) who need automatic competency mapping for course content, as a first pass that a human can review — not as an unsupervised authority on curriculum classification.

**Out of scope.**
- Not intended to grade learners or make pass/fail decisions.
- Not validated on content types outside `.pptx`, `.ipynb`, `.xlsx`, `.md`, `.txt`, `.csv`.
- Not validated on non-technical / non-data-science course content (the evaluation set is data science/ML-oriented).
- Not intended to operate as the sole source of truth for taxonomy compliance — see recall limitations below.

## 3. Pipeline architecture

```
Learning Content
   → Content Extraction (per file type)
   → Qwen Topic/Competency Tagging
   → Semantic Summarization
   → Hybrid Retrieval (E5 semantic 0.6 + Lexical 0.4) against the taxonomy
   → Cross-Chunk Aggregation
   → Qwen Reranking (selects only from retrieved candidates)
   → Final Competency Mapping + Difficulty + Confidence + Notes
```

A key architectural property: **the reranker can only choose from the retrieved candidate set.** If the correct competency is not retrieved, reranking cannot recover it — this is the main source of recall loss (see §5).

## 4. Training / adaptation data

- No fine-tuning was performed; the base Qwen2.5-7B-Instruct weights are used as-is (AWQ-quantized for serving).
- The **Saudi Skills Taxonomy** (`saudi_skills_taxonomy_v1_final.csv`, 134 competency rows: `skill_name_en`, `description_en`, `subsector_en`, `related_job_families_en`, `needs_review`) is used at inference time as the retrieval/mapping reference, not as training data.
- Human-reviewed gold labels exist only for the **12-file evaluation set** and are used strictly for evaluation, never fed into the inference pipeline.

## 5. Evaluation results (final, V8.4, 12 files / 39 gold competency mappings)

| Metric | Value |
|---|---:|
| Macro Precision | 75.00% |
| Macro Recall | 27.08% |
| Macro F1 | 39.17% |
| Micro Precision | 76.92% |
| Micro Recall | 25.64% |
| Micro F1 | 38.46% |
| Recall@1 | 4.86% |
| Recall@3 | 11.81% |
| Recall@5 | 16.67% |
| Recall@10 | 31.25% |
| Hit Rate@10 | 91.67% |
| True / False Positives / False Negatives | 10 / 3 / 29 |

Tag-level (earlier evaluation): precision ≈ 100%, recall ≈ 72.7%.

**Reading the numbers.** Hit Rate@10 (91.7%) means the correct competency is almost always *somewhere* in the top-10 retrieved candidates. Recall@10 (31.3%) and Micro F1 (38.5%) are much lower, which means the system is fairly conservative/precise about which of those candidates it finally commits to (Macro/Micro Precision ≈ 75–77%) at the cost of missing some correct competencies (29 false negatives vs. 10 true positives). Retrieval coverage and final-selection recall are both flagged in the project README as the main areas for future improvement.

## 6. Serving performance (production, RTX A6000, from `Benchmark/final_benchmark_report.csv`)

| Concurrency | Avg latency | P50 | P95 | P99 | Throughput | Error rate |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16.39s | 11.72s | 37.04s | 39.27s | 0.061 req/s | 0% |
| 2 | 25.61s | 21.83s | 50.98s | 50.98s | 0.102 req/s | 41.7%* |
| 4 | 20.50s | 17.89s | 40.38s | 45.30s | 0.170 req/s | 0% |
| 8 | 30.80s | 30.19s | 38.82s | 61.33s | 0.178 req/s | 0% |

\*The concurrency‑2 error rate looks anomalous relative to the neighboring rows and has not been explained in the project's benchmark notes — flag this before citing the row as production-representative.

Generation throughput: ~83.7 tokens/s, ~76 generated tokens/request, average TTFT ≈ 29.2 ms, average inter-token latency ≈ 8.1 ms. GPU: 100% max utilization, 42,821 MiB max VRAM, 297.1 W max power (RTX A6000, controlled monitoring run at concurrency 4).

## 7. Known limitations

1. **Recall ceiling from retrieval.** ~30% of gold competencies are never retrieved into the candidate set, so no amount of reranking improvement can fix them; retrieval quality is the primary lever for improving Recall@10.
2. **Small, homogeneous evaluation set.** 12 files / 39 gold mappings, all data-science/ML course content — metrics may not generalize to other subjects or to non-English content structures.
3. **Quantization not isolated.** AWQ 4-bit is used in production, but the V6 baseline used a different quantization (bitsandbytes NF4) — no controlled same-hardware, same-prompt comparison of quantized vs. unquantized quality exists yet (see `docs/QUANTIZATION_COMPARISON.md`).
4. **Latency is high for interactive use.** 11–40s p50/p95 per file at low concurrency; the system is better suited to batch/background processing than real-time interaction, per the infrastructure benchmark's own recommendation.
5. **No adversarial or out-of-distribution testing.** Content that is malformed, in a language other than English/Arabic mixed technical content, or adversarially crafted has not been evaluated.
6. **AI Hub integration status.** Per the README, integration with AI Hub is "not yet verified" as of the last documentation update — treat any AI-Hub-facing behavior as unconfirmed until re-checked.

## 8. Ethical / responsible-use considerations

- Competency-mapping output should be treated as a **draft suggestion for human review**, especially given the ~38% Micro F1 — using it as an automatic, unreviewed classifier for learner or curriculum decisions risks systematic mis-tagging (roughly 1 in 4 predicted competencies is currently a false positive, and roughly 3 in 4 correct competencies are missed per file on average given the recall figures above).
- The taxonomy itself (Saudi Skills Taxonomy v1) is externally defined; this system does not audit the taxonomy for bias or coverage gaps — it only maps content onto whatever categories the taxonomy provides.
- No personally identifiable learner data is processed by this pipeline; inputs are course/content files, not learner records.

## 9. How to cite / reference

Pipeline: BeamData Content Tagging & Competency Mapping, V8.4, Team 6, AI Data Center Bootcamp (Aug 2026 cohort). Base model: Qwen2.5-7B-Instruct-AWQ (Alibaba Cloud / Qwen team), served via vLLM.
