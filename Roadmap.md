# AIDC Project Roadmap

## 1. Dataset / Evaluation

**Status: COMPLETE**

* [x] Saudi Skills Taxonomy dataset
* [x] Ground-truth annotation dataset
* [x] 12-file evaluation set
* [x] E5 embedding baseline
* [x] Improved E5 retrieval experiment
* [x] Metrics: Precision, Recall, F1, Recall@1/3/5/10, Hit Rate@10

---

## 2. Model / Pipeline

**Status: COMPLETE**

Final model version: **V8.4** (hybrid E5 + lexical retrieval, cross-chunk aggregation, Qwen reranking).

The V8.4 pipeline produces:

* Tags
* Saudi Skills Taxonomy competencies
* Difficulty
* Learning objectives
* Confidence
* Notes
* Structured JSON / CSV output

Earlier V6/V7 checkpoints remain as experiment history; V8.4 is the final selected configuration.

---

## 3. Infrastructure

**Status: COMPLETE**

* [x] Kubernetes environment
* [x] NVIDIA GPU support
* [x] vLLM serving infrastructure (Qwen2.5-7B-Instruct-AWQ on RTX A6000)
* [x] GPU resource allocation
* [x] Hugging Face model cache
* [x] Kubernetes health/readiness probes
* [x] Prometheus
* [x] Grafana (deployment)
* [x] Kubernetes services

---

## 4. Deployment

**Status: COMPLETE**

* [x] Configure the V8.4 model for serving (FastAPI + vLLM)
* [x] Create/update Kubernetes deployment (`beamdata-v8-api`, `beamdata-vllm-7b`, namespace `shahad`)
* [x] Configure model-serving service
* [x] Deploy the model
* [x] Verify `/health`
* [x] Verify `/v1/models` (via vLLM)
* [x] Verify `/metrics`
* [x] Confirm the deployed model is the intended V8.4 pipeline
* [x] Docker image built and pushed (`shahad09/beamdata-v8-api:v8.4`)

---

## 5. Benchmarking / Performance

**Status: COMPLETE**

* [x] Latency (avg, P50, P95, P99)
* [x] Throughput
* [x] GPU utilization (up to 100% observed)
* [x] GPU memory usage (up to 42,821 MiB observed)
* [x] Concurrent requests (tested at concurrency 1, 2, 4, 8 — 48/48 successful, 0% errors)
* [x] Error rate
* [x] SLO / stability check (stable through concurrency 8)

### Monitoring

* [x] DCGM Exporter deployed and verified (UP in Prometheus)
* [x] Prometheus targets verified (`beamdata-vllm`, `dcgm-exporter` — UP)
* [ ] Grafana dashboard — **IN PROGRESS** (Grafana pod/service healthy, but dashboard fails to load through the `/proxy/3000/` route)

---

## 6. OpenAI Comparison

**Status: COMPLETE**

* [x] Define equivalent evaluation setup
* [x] Run OpenAI comparison
* [x] Compare output quality
* [x] Compare relevant performance/cost metrics
* [x] Document differences (`benchmark/OPENAI_VS_V84_COMPARISON.md`)

---

## 7. V6 vs. V7 vs. V8.4 Comparison

**Status: COMPLETE**

* [x] Compare V6 vs. V7 vs. V8.4 predictions
* [x] Compare tagging quality
* [x] Compare competency mapping
* [x] Document the final model configuration (`benchmark/V6_V7_V8_COMPARISON.md`)

---

## 8. Final Presentation / Report

**Status: IN PROGRESS**

Remaining items:

* [ ] Finish the Grafana dashboard and verify it loads through the required proxy
* [ ] Confirm whether AI Hub integration has been completed (currently: not yet verified)
* [ ] Finalize the final README / documentation across the repo
* [ ] Prepare and rehearse the final presentation
* [ ] Complete final submission (Google Drive + LMS)

---

# Overall Roadmap

```text
Dataset / Evaluation       ✅
          ↓
Model / Pipeline (V8.4)    ✅
          ↓
Infrastructure             ✅
          ↓
Deployment                 ✅
          ↓
Benchmarking               ✅ (Grafana dashboard pending)
          ↓
OpenAI Comparison          ✅
          ↓
V6 / V7 / V8.4 Comparison  ✅
          ↓
Final Presentation         ⏳ IN PROGRESS
```

## Current Priority

The dataset, evaluation, model/pipeline, infrastructure, deployment, benchmarking, and both comparison stages are complete.

The remaining work is finalization, not further experimentation:

```text
1. Fix Grafana dashboard proxy access
2. Confirm AI Hub integration status
3. Finalize documentation
4. Rehearse and deliver final presentation
```
