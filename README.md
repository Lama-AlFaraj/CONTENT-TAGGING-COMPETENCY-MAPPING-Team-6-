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

### Still Useful Here

* [x] Finish the embedding/reranking experiment
* [x] Decide the final evaluation methodology
* [x] Document baseline vs. improved model
* [x] Finalize the evaluation dataset

---

## 2. Model / Pipeline

Then make the **actual tagging + competency-mapping pipeline**, not just the evaluation script.

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

This is where your **OpenAI API key** may become useful if your final pipeline uses an LLM for:

* Topic tagging
* Learning objective generation
* Difficulty classification
* Structured output generation

---

## 3. Infrastructure

Then deploy the pipeline/model in your AIDC environment.

This includes:

```text
Docker
  ↓
Containerized Application
  ↓
Kubernetes
  ↓
Model / API Serving
  ↓
Service
  ↓
Monitoring
```

You already have substantial experience from the AIDC labs here:

* [x] Docker
* [x] Kubernetes
* [x] Helm
* [x] vLLM
* [x] Prometheus
* [x] Grafana
* [x] HPA
* [x] SLO / TTFT monitoring

So this should **not** be treated as starting from zero.

---

## 4. Deployment

This is the actual **running application/service**.

For example:

```text
User uploads learning content
          ↓
         API
          ↓
     AI Pipeline
          ↓
   Taxonomy Matching
          ↓
   Structured Result
          ↓
       JSON / UI
```

Deployment should demonstrate that the model isn't just working in a notebook — it is accessible as a service.

---

## 5. Benchmarking / Performance

After deployment, measure:

* [ ] Latency
* [ ] Throughput
* [ ] GPU / CPU utilization
* [ ] Memory usage
* [ ] Concurrent requests
* [ ] TTFT if using an LLM
* [ ] Cost / resource usage
* [ ] SLO compliance

This connects directly to the infrastructure work already completed in the AIDC labs.

---

## 6. Final Presentation / Report

Finally:

```text
Problem
   ↓
Dataset
   ↓
Method
   ↓
Model
   ↓
Evaluation
   ↓
Infrastructure
   ↓
Deployment
   ↓
Performance
   ↓
Results
   ↓
Limitations / Future Work
```

---

# Overall Roadmap

```text
Dataset
   ↓
Model / Pipeline
   ↓
Infrastructure
   ↓
Deployment
   ↓
Benchmarking
   ↓
Presentation
```

## Current Priority

Right now, do **not** spend all the remaining time tweaking E5.

First finish enough of the **dataset + evaluation methodology** that you have a defensible model to deploy.

Then move directly to:

```text
1. Finalize Dataset / Evaluation
              ↓
2. Build Actual Pipeline
              ↓
3. Dockerize
              ↓
4. Kubernetes Deployment
              ↓
5. Monitoring / Benchmarking
              ↓
6. Final Slides
```

The goal is to finish the **complete end-to-end system**, not only the embedding evaluation experiment.
