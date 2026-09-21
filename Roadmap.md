# AIDC Project Roadmap

## 1. Dataset / Evaluation

### Current Stage

You have:

* [x] Saudi Skills Taxonomy dataset
* [x] Ground-truth annotation dataset
* [x] 12-file evaluation set
* [x] E5 embedding baseline
* [x] Improved E5 retrieval experiment
* [x] Metrics: Precision, Recall, F1, Top-1/3/5, MRR

### Evaluation Work

* [x] Finish the embedding/reranking experiment
* [x] Decide the final evaluation methodology
* [x] Document baseline vs. improved model
* [x] Finalize the evaluation dataset

**Status: COMPLETE**

---

## 2. Model / Pipeline

Build the actual tagging + competency-mapping pipeline.

### Expected Flow

```text
Learning Content
      ↓
Content Extraction
      ↓
Topic Tagging
      ↓
Embedding / Retrieval
      ↓
Saudi Skills Taxonomy
      ↓
Competency Mapping
      ↓
Difficulty
      ↓
Learning Objectives
      ↓
Structured JSON Output
```

The V6 pipeline currently produces:

* Tags
* Saudi Skills Taxonomy competencies
* Difficulty
* Learning objectives
* Confidence
* Notes
* Structured JSON output

OpenAI API can be used later for the required model comparison and/or specific LLM pipeline components.

**Status: COMPLETE**

---

## 3. Infrastructure

The AIDC infrastructure has already been provided/set up by the teammate.

This includes:

```text
Docker / Container Images
        ↓
Kubernetes
        ↓
GPU Support
        ↓
vLLM Model Serving
        ↓
Kubernetes Services
        ↓
Prometheus
        ↓
Grafana
```

### Current Infrastructure

* [x] Kubernetes environment
* [x] NVIDIA GPU support
* [x] vLLM serving infrastructure
* [x] GPU resource allocation
* [x] Hugging Face model cache
* [x] Kubernetes health/readiness probes
* [x] Prometheus
* [x] Grafana
* [x] Kubernetes services

**Status: COMPLETE**

> Grafana infrastructure is complete here. The project-specific Grafana dashboard and benchmark visualization will be done during Benchmarking.

---

## 4. Deployment

This is the actual deployment of the project's V6 model/pipeline into the existing AIDC infrastructure.

### Remaining

* [ ] Configure the V6 model for serving
* [ ] Create/update Kubernetes deployment
* [ ] Configure model-serving service
* [ ] Deploy the model
* [ ] Verify `/health`
* [ ] Verify `/v1/models`
* [ ] Verify `/metrics`
* [ ] Confirm the deployed model is the intended V6 model

The existing infrastructure currently contains vLLM configurations for Qwen2.5-1.5B-Instruct-AWQ, while the V6 pipeline uses Qwen2.5-7B-Instruct, so the deployment configuration still needs to be finalized.

**Status: NEXT**

---

## 5. Benchmarking / Performance

After the actual project model is deployed, measure:

* [ ] Latency
* [ ] P95 latency
* [ ] Throughput
* [ ] GPU utilization
* [ ] GPU memory usage
* [ ] Concurrent requests
* [ ] TTFT
* [ ] Tokens/sec
* [ ] Error rate
* [ ] Resource/cost usage
* [ ] SLO compliance

### Monitoring

Create/finalize the project Grafana dashboard to visualize:

```text
Request Rate
Latency
TTFT
Throughput
GPU Utilization
GPU Memory
Errors
```

Then run the benchmark against the actual deployed project model and save the results.

**Status: NOT STARTED FOR THE FINAL V6 DEPLOYMENT**

> The teammate's previous GPU benchmark exists, but it has not been established that it benchmarks the V6 7B model, so it should not yet be treated as the final V6 benchmark.

---

## 6. OpenAI Comparison

Compare the project approach with OpenAI as required by the project guide.

* [ ] Define equivalent evaluation setup
* [ ] Run OpenAI comparison
* [ ] Compare output quality
* [ ] Compare relevant performance/cost metrics
* [ ] Document differences

**Status: PENDING**

---

## 7. V6 vs. V7 Comparison

After deployment and benchmarking:

* [ ] Compare V6 vs. V7 predictions
* [ ] Compare tagging quality
* [ ] Compare competency mapping
* [ ] Compare difficulty
* [ ] Compare overall output quality
* [ ] Compare relevant performance results
* [ ] Document the model configuration used for the final system

**Status: PENDING**

---

## 8. Final Presentation / Report

Finally:

```text
Problem
   ↓
Dataset
   ↓
Method
   ↓
Model / Pipeline
   ↓
Evaluation
   ↓
Infrastructure
   ↓
Deployment
   ↓
Monitoring
   ↓
Benchmarking
   ↓
OpenAI Comparison
   ↓
V6 vs. V7
   ↓
Results
   ↓
Limitations / Future Work
```

---

# Overall Roadmap

```text
Dataset / Evaluation       ✅
          ↓
Model / Pipeline           ✅
          ↓
Infrastructure             ✅
          ↓
Deployment                 ⏳ NEXT
          ↓
Benchmarking               ⏳
          ↓
OpenAI Comparison          ⏳
          ↓
V6 vs. V7 Comparison       ⏳
          ↓
Final Presentation         ⏳
```

## Current Priority

The dataset, evaluation, model/pipeline, and infrastructure stages are complete.

The next step is:

```text
1. Deploy V6
2. Verify the deployed service
3. Connect/verify metrics
4. Create the project Grafana dashboard
5. Run benchmarking
6. Complete OpenAI comparison
7. Compare V6 vs. V7
8. Finalize presentation/report
```

The goal is to demonstrate the **complete end-to-end system**, from learning content to structured AI output, running on the AIDC infrastructure and supported by measurable deployment and performance results.
