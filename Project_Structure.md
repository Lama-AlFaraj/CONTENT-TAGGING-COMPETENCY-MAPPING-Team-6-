# BeamData — Content Tagging & Competency Mapping

## 1. Project Overview

This project develops an end-to-end AI system for **content tagging and competency mapping**.

The system analyzes learning materials, extracts relevant topics and competencies, maps the content to the **Saudi Skills Taxonomy**, predicts difficulty, and produces structured competency-mapping results.

The project covers the complete workflow:

```text
Learning Content
      ↓
Content Extraction
      ↓
Topic / Competency Tagging
      ↓
Semantic Representation
      ↓
Hybrid Retrieval
(E5 + Lexical)
      ↓
Cross-Chunk Aggregation
      ↓
Qwen Reranking
      ↓
Final Competency Mapping
      ↓
Difficulty / Confidence / Notes
      ↓
Structured Output
      ↓
Evaluation
      ↓
Docker + Kubernetes Deployment
      ↓
Benchmarking + Monitoring
```

The final model checkpoint is **V8.4**.

---

# 2. Project / Group Structure

The project is organized into the following main workstreams:

```text
BeamData Content Tagging & Competency Mapping
│
├── 1. Dataset & Evaluation
│   ├── Evaluation files
│   ├── Human annotations
│   ├── Ground-truth labels
│   ├── Saudi Skills Taxonomy
│   └── Evaluation methodology
│
├── 2. Model / AI Pipeline
│   ├── Content extraction
│   ├── Qwen tagging
│   ├── Semantic summarization
│   ├── E5 retrieval
│   ├── Lexical retrieval
│   ├── Cross-chunk aggregation
│   ├── Qwen reranking
│   └── Final competency mapping
│
├── 3. Infrastructure / Deployment
│   ├── Docker
│   ├── FastAPI
│   ├── Kubernetes
│   ├── vLLM
│   └── GPU deployment
│
├── 4. Benchmarking
│   ├── Model evaluation
│   ├── API benchmarking
│   ├── Concurrency testing
│   └── GPU / serving measurements
│
└── 5. Monitoring
    ├── DCGM Exporter
    ├── Prometheus
    └── Grafana
```

The project was developed as a group, with the work divided across the dataset/evaluation, model/pipeline, infrastructure, monitoring, benchmarking, documentation, and presentation tasks.

Track owners for the BeamData track are **Salman and Emad**.

---

# 3. Repository Structure

Current repository:

```text
CONTENT-TAGGING-COMPETENCY-MAPPING-Team-6-
│
├── Model/
│   ├── v8/
│   │   ├── api.py
│   │   ├── model_v8.py
│   │   ├── model_v8_backup.py
│   │   ├── model_v8_local_backup.py
│   │   ├── model_v8_before_semantic_retrieval.py
│   │   ├── retrieval_v8.py
│   │   ├── run_v8.py
│   │   ├── evaluate_v8.py
│   │   ├── evaluate_retrieval_hybrid.py
│   │   ├── requirements.txt
│   │   ├── hybrid_retrieval_results.csv
│   │   ├── hybrid_retrieval_summary.csv
│   │   └── results/
│   │       ├── v8_predictions.csv
│   │       └── v8_evaluation.csv
│   │
│   ├── evaluation_data/
│   │   └── official evaluation files
│   │
│   └── saudi_skills_taxonomy_v1_final.csv
│
├── Infrastructure/
│   └── k8s/
│       ├── vllm-7b.yaml
│       └── beamdata-v8-api.yaml
│
├── benchmark/
│   ├── bench.py
│   ├── v84_api_e2e.py
│   ├── Infrastructure_Benchmark_Report.md
│   └── results/
│       ├── v84_api_e2e.json
│       ├── v84_api_e2e_c1.json
│       ├── v84_api_e2e_c1_clean.json
│       ├── v84_api_e2e_c2.json
│       └── v84_monitoring_c4.json
│
├── Dockerfile
└── README.md
```

---

# 4. Dataset & Evaluation Work

## 4.1 Evaluation Dataset

A fixed benchmark of **12 official evaluation files** was prepared.

The evaluation set contains different learning-content formats, including:

* PPTX
* Jupyter Notebook
* Markdown
* Other structured learning-content files

Examples include:

```text
Decision Trees Revised.pptx
HandsOn1_Window_Functions.md
Introduction to Pandas_.pptx
Lecture - Decision Trees.ipynb
Lecture - Pandas Basics.ipynb
Lecture_ML_Workflow.ipynb
ML Workflow Introduction.pptx
Prompt Engineering
RAG
SQL Foundations
WK2 IntroToML
WK4 EnsembleLearning
```

---

## 4.2 Human Annotations

Human-reviewed annotations were prepared for the evaluation files.

The annotation structure includes:

```text
file_name
annotator_name
predicted_tags
proposed_competencies
difficulty
confidence
notes
```

Competency names were copied from the valid taxonomy reference to ensure that the gold labels use valid taxonomy terminology.

---

## 4.3 Saudi Skills Taxonomy

The project uses the frozen taxonomy:

```text
saudi_skills_taxonomy_v1_final.csv
```

The taxonomy contains **134 competency rows**.

Main columns:

```text
skill_name_en
description_en
subsector_en
related_job_families_en
needs_review
```

The taxonomy is treated as the standardized competency reference for the mapping stage.

---

# 5. Evaluation Methodology

The evaluation covers both content tagging and competency mapping.

## Tag Evaluation

AI-generated tags were compared with the human-reviewed ground truth using semantic matching.

Previously measured tag performance:

```text
Precision: 100%
Recall: approximately 72.7%
```

The evaluation indicates that the system generally produced relevant tags, while some expected tags were missed.

---

## Competency Evaluation

The competency evaluation uses:

* Macro Precision
* Macro Recall
* Macro F1
* Micro Precision
* Micro Recall
* Micro F1
* Recall@1
* Recall@3
* Recall@5
* Recall@10
* Hit Rate@10

The final evaluation contains:

```text
Evaluation files: 12
Gold competency mappings: 39
```

---

# 6. Model / AI Pipeline

The final model version is **V8.4**.

V8.4 combines:

```text
Qwen
+
Semantic Summarization
+
Multilingual E5
+
Lexical Retrieval
+
Cross-Chunk Aggregation
+
Qwen Reranking
```

---

## 6.1 Content Extraction

The FastAPI application supports:

```text
.pptx
.ipynb
.xlsx
.md
.txt
.csv
```

The content is extracted and converted into a common representation before being passed to the AI pipeline.

---

## 6.2 Qwen Topic / Competency Tagging

Qwen analyzes the learning content and extracts relevant topics and competency-related concepts.

This provides the semantic information required for downstream retrieval.

---

## 6.3 Semantic Summarization

The extracted content is converted into a semantic representation that preserves the concepts important for competency mapping.

This representation is used to improve retrieval queries.

---

## 6.4 Hybrid Retrieval

The retrieval stage combines:

```text
Multilingual E5 semantic retrieval
              +
Lexical retrieval
```

The current retrieval implementation uses:

```text
Model:
intfloat/multilingual-e5-base

Semantic weight:
0.6

Lexical weight:
0.4
```

The taxonomy competency name and description are used during retrieval.

---

## 6.5 Cross-Chunk Aggregation

Long learning materials are divided into chunks.

The system retrieves competency candidates for individual chunks and aggregates the evidence across chunks.

This prevents the final mapping from depending only on a single section of the learning material.

---

## 6.6 Qwen Reranking

The retrieved candidates are passed to Qwen for reranking.

The process is:

```text
Hybrid Retrieval
      ↓
Candidate Set
      ↓
Qwen Reranking
      ↓
Final Competencies
```

An important property of the architecture is:

> The reranker can only select from retrieved candidates.

Therefore, if a correct competency is absent from the candidate set, the reranker cannot recover it.

---

## 6.7 Final Structured Output

The V8.4 pipeline produces:

```text
predicted_tags
proposed_competencies
difficulty_level
confidence
notes
retrieval_candidates
chunk_count
```

---

# 7. V8 Experiment History

Several pipeline versions were evaluated.

## V8.1

Intermediate pipeline configuration used as a baseline for subsequent improvements.

## V8.2

Introduced additional retrieval and mapping changes and was evaluated against the same benchmark.

## V8.3

Tested as an alternative configuration.

The changes were evaluated but ultimately reverted.

## V8.4

V8.4 became the final selected checkpoint.

It was:

* Evaluated on the full benchmark
* Cleaned up
* Syntax validated
* Checked with `git diff --check`
* Committed to Git
* Pushed to the `hybrid-v8` branch

---

# 8. Final V8.4 Model Results

```text
Evaluation files: 12
Gold competency mappings: 39
```

| Metric          |       V8.4 |
| --------------- | ---------: |
| Macro Precision | **75.00%** |
| Macro Recall    | **27.08%** |
| Macro F1        | **39.17%** |
| Micro Precision | **76.92%** |
| Micro Recall    | **25.64%** |
| Micro F1        | **38.46%** |
| Recall@1        |  **4.86%** |
| Recall@3        | **11.81%** |
| Recall@5        | **16.67%** |
| Recall@10       | **31.25%** |
| Hit Rate@10     | **91.67%** |

Confusion counts:

```text
True Positives: 10
False Positives: 3
False Negatives: 29
```

---

# 9. Interpretation of Model Results

The results show an important distinction between candidate retrieval and final competency selection.

The **91.67% Hit Rate@10** means that most evaluation files had at least one relevant competency somewhere in the top-10 candidate set.

However:

```text
Hit Rate@10 = 91.67%
Recall@10   = 31.25%
Micro F1    = 38.46%
```

This indicates that finding a relevant candidate does not automatically mean that the system selects all correct competencies in the final mapping.

The remaining errors occur across both:

1. Candidate retrieval coverage
2. Final competency selection / reranking

Retrieval coverage remains a major area for future improvement.

---

# 10. Docker Deployment

The V8.4 API was containerized using Docker.

Docker image:

```text
shahad09/beamdata-v8-api:v8.4
```

The image contains:

* Python 3.11
* FastAPI
* Uvicorn
* Sentence Transformers
* PyTorch
* Transformers
* Pandas
* NumPy
* python-pptx
* openpyxl
* nbformat

The image was pushed to Docker Hub.

---

# 11. FastAPI Service

The final API is:

```text
Model/v8/api.py
```

Endpoints include:

```text
GET  /health
POST /predict
```

The API was updated to use a thread pool for the synchronous model pipeline:

```python
result = await run_in_threadpool(process_document, content)
```

This allows concurrent API requests without blocking the FastAPI event loop.

The final API image was rebuilt and redeployed after this fix.

---

# 12. Kubernetes Deployment

The project is deployed in:

```text
Namespace: shahad
Node: aidc-t12
```

Main deployments:

```text
beamdata-v8-api
beamdata-vllm-7b
```

Main services:

```text
beamdata-v8-api
beamdata-vllm
```

---

# 13. vLLM / Qwen Deployment

The LLM serving layer uses:

```text
Qwen/Qwen2.5-7B-Instruct-AWQ
```

served using:

```text
vLLM
```

Configuration includes:

```text
AWQ quantization
FP16
max model length: 16384
GPU memory utilization: 0.85
prefix caching enabled
```

The model runs on the NVIDIA RTX A6000.

Observed environment:

```text
GPU:
NVIDIA RTX A6000

VRAM:
~49 GB

Driver:
550.90.12

CUDA:
12.4
```

---

# 14. API Architecture

The deployed architecture is:

```text
User / AI Hub
      ↓
V8.4 FastAPI
      ↓
┌─────────────────────────────┐
│ Content Processing          │
│ Qwen Tagging                │
│ Semantic Representation     │
│ E5 + Lexical Retrieval      │
│ Cross-Chunk Aggregation     │
│ Qwen Reranking              │
└─────────────────────────────┘
      ↓
Saudi Skills Taxonomy
      ↓
Final Competency Mapping
```

The API communicates with the internal vLLM service:

```text
beamdata-v8-api
        ↓
beamdata-vllm:8000
        ↓
Qwen 7B AWQ
        ↓
RTX A6000
```

---

# 15. API Benchmarking

The API was benchmarked using the 12-file evaluation set at multiple concurrency levels.

## Concurrency Results

| Concurrency | Requests | Success | Avg Latency |     P50 |     P95 |     P99 |   Throughput |
| ----------: | -------: | ------: | ----------: | ------: | ------: | ------: | -----------: |
|           1 |       12 |      12 |     10.014s |  7.259s | 22.930s | 25.349s | 0.0999 req/s |
|           2 |       12 |      12 |     13.789s | 10.726s | 34.401s | 35.799s | 0.1420 req/s |
|           4 |       12 |      12 |     21.048s | 18.046s | 42.470s | 46.983s | 0.1654 req/s |
|           8 |       12 |      12 |     33.361s | 31.002s | 43.654s | 66.892s | 0.1681 req/s |

Overall:

```text
Total requests: 48
Successful: 48
Errors: 0
Error rate: 0%
```

The API remained stable through concurrency 8.

Throughput increased as concurrency increased, while latency also increased. Throughput began to plateau between concurrency 4 and 8.

---

# 16. Controlled Monitoring Benchmark

A dedicated monitoring run was executed at concurrency 4:

```text
Requests: 12
Successful: 12
Errors: 0
Wall time: 66.8709s
```

Results:

```text
Average latency: 19.624s
P50:             17.368s
P95:             39.275s
P99:             44.100s
Throughput:      0.1795 req/s
```

This run was used to observe GPU and vLLM behavior through Prometheus.

---

# 17. Monitoring Architecture

The monitoring stack is:

```text
GPU
 ↓
DCGM Exporter
 ↓
Prometheus
 ↓
Grafana
```

The AI serving layer is also monitored:

```text
vLLM
 ↓
Prometheus
 ↓
Grafana
```

Monitoring targets include:

```text
vLLM metrics
GPU metrics
API/application metrics
```

---

# 18. DCGM GPU Monitoring

DCGM Exporter was deployed to expose NVIDIA GPU metrics.

The initial exporter encountered a profiling-module issue.

The configuration was adjusted to use a custom counter set containing:

```text
DCGM_FI_DEV_GPU_UTIL
DCGM_FI_DEV_FB_FREE
DCGM_FI_DEV_FB_USED
DCGM_FI_DEV_GPU_TEMP
DCGM_FI_DEV_POWER_USAGE
```

The exporter was successfully redeployed.

Current state:

```text
dcgm-exporter: Running
Prometheus target: UP
```

---

# 19. Prometheus

Prometheus is running in:

```text
Namespace: team
Service: prometheus
Port: 9090
```

The Prometheus configuration includes the BeamData vLLM and DCGM exporters.

Important active targets:

```text
beamdata-vllm
dcgm-exporter
```

Both were verified as:

```text
UP
```

Older serving targets also exist in the Prometheus configuration, but they belong to older/unrelated services and should not be treated as BeamData V8.4 monitoring targets.

---

# 20. GPU Monitoring Results

During the controlled monitoring benchmark, Prometheus recorded:

| Metric                | Maximum Observed |
| --------------------- | ---------------: |
| GPU Utilization       |         **100%** |
| GPU VRAM Used         |   **42,821 MiB** |
| GPU Power             |    **297.104 W** |
| vLLM Running Requests |            **2** |
| vLLM Waiting Requests |            **0** |
| KV Cache Usage        |       **0.433%** |

These measurements demonstrate that GPU-level monitoring is working and that the benchmark traffic reached the deployed GPU model.

---

# 21. Grafana

Grafana is deployed in:

```text
Namespace: team
Service: grafana
Port: 3000
NodePort: 30300
```

Current Kubernetes state:

```text
Grafana pod: Running
Grafana service: 3000:30300
Grafana endpoint: 10.42.1.63:3000
```

Grafana itself has been verified as healthy internally.

The required dashboard is intended to display:

### API

* Request count
* Errors
* Latency
* Throughput

### vLLM

* Running requests
* Waiting requests
* KV cache usage
* Token-related metrics where available

### GPU

* GPU utilization
* VRAM used/free
* Temperature
* Power usage

### Current Grafana Status

The **Grafana application/dashboard configuration is not yet finalized**.

The Kubernetes Grafana deployment and Prometheus data sources are healthy, but accessing Grafana through the existing code-server `/proxy/3000/` route is currently producing:

```text
Grafana has failed to load its application files
```

Therefore:

```text
Grafana deployment: COMPLETE
Prometheus monitoring: COMPLETE
DCGM monitoring: COMPLETE
Grafana dashboard: IN PROGRESS
```

No additional Grafana networking changes should be considered part of the completed project until the dashboard loads correctly through the required proxy.

---

# 22. Port / Access Structure

The current local port-forward setup used during development and benchmarking is:

```text
8001 → vLLM
8002 → V8.4 FastAPI
9091 → Prometheus
3000 → Grafana
```

The first three ports are independent of Grafana:

```text
8001 = vLLM
8002 = V8.4 API
9091 = Prometheus
3000 = Grafana
```

These ports should not be confused with Kubernetes service ports.

---

# 23. AI Hub

The project architecture allows the V8.4 API to be integrated with AI Hub.

However, **AI Hub integration has not yet been independently verified as completed**.

Therefore it should currently be documented as:

```text
AI Hub integration: Not verified / pending final confirmation
```

The model/API/Kubernetes deployment itself is already completed.

---

# 24. Validation Completed

The final V8.4 implementation was checked using:

```text
Python syntax validation
git diff --check
API health checks
Kubernetes deployment checks
vLLM health checks
Prometheus target checks
DCGM exporter checks
End-to-end API benchmarking
GPU monitoring benchmark
```

Completed status:

```text
Python syntax validation: Passed
git diff --check: Passed
API health: Passed
vLLM deployment: Passed
Kubernetes deployment: Passed
Prometheus target: UP
DCGM target: UP
API benchmark: Passed
GPU monitoring: Passed
```

---

# 25. Final Git Status

Final model checkpoint:

```text
Version: V8.4
Branch: hybrid-v8
Status: Completed and pushed
```

Docker image:

```text
shahad09/beamdata-v8-api:v8.4
```

---

# 26. Overall Project Completion Status

| Workstream             | Status               |
| ---------------------- | -------------------- |
| Dataset preparation    | Complete             |
| Evaluation dataset     | Complete             |
| Human annotations      | Complete             |
| Saudi Skills Taxonomy  | Complete             |
| Evaluation methodology | Complete             |
| Baseline retrieval     | Complete             |
| Improved retrieval     | Complete             |
| V8 experiments         | Complete             |
| Final V8.4 pipeline    | Complete             |
| FastAPI API            | Complete             |
| Docker image           | Complete             |
| Kubernetes deployment  | Complete             |
| Qwen/vLLM deployment   | Complete             |
| API benchmarking       | Complete             |
| GPU monitoring         | Complete             |
| DCGM Exporter          | Complete             |
| Prometheus             | Complete             |
| Grafana deployment     | Complete             |
| Grafana dashboard      | **In progress**      |
| AI Hub integration     | **Not yet verified** |
| Final documentation    | In progress          |
| Final presentation     | In progress          |

---

# 27. Final Technical Architecture

The complete project can be represented as:

```text
                         ┌──────────────────────┐
                         │   Learning Content   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │  Content Extraction  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Qwen Tagging       │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Semantic Summarizing │
                         └──────────┬───────────┘
                                    │
                         ┌──────────┴───────────┐
                         ▼                      ▼
                ┌────────────────┐     ┌────────────────┐
                │ E5 Retrieval   │     │ Lexical Search │
                └───────┬────────┘     └───────┬────────┘
                        │                      │
                        └──────────┬───────────┘
                                   ▼
                         ┌──────────────────────┐
                         │ Candidate Aggregation│
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Qwen Reranking     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Saudi Skills         │
                         │ Taxonomy Mapping     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Structured Results   │
                         └──────────────────────┘


Deployment / Infrastructure:

              Docker
                 │
                 ▼
             Kubernetes
                 │
        ┌────────┴─────────┐
        ▼                  ▼
   FastAPI V8.4         vLLM
        │                  │
        │                  ▼
        │             Qwen 7B AWQ
        │                  │
        └─────────┬────────┘
                  ▼
             RTX A6000


Monitoring:

 RTX A6000 ──► DCGM Exporter ──► Prometheus ──► Grafana
                                      ▲
                                      │
                                    vLLM
                                      ▲
                                      │
                                  V8.4 API
```

---

# 28. Final Results Summary

The project has completed the major technical stages from dataset preparation through AI model development, deployment, benchmarking, and GPU monitoring.

The final V8.4 competency results are:

```text
Macro F1:       39.17%
Micro F1:       38.46%
Recall@10:      31.25%
Hit Rate@10:    91.67%
```

The deployed API achieved:

```text
48 / 48 successful benchmark requests
0% error rate
Stable operation through concurrency 8
```

The monitoring stack successfully captured:

```text
100% maximum GPU utilization
42,821 MiB maximum VRAM used
297.104 W maximum GPU power
vLLM running/waiting request metrics
KV cache metrics
```

---

# 29. Remaining Final Tasks

The remaining work is primarily finalization rather than another model-development cycle:

1. Finish the Grafana dashboard and verify it loads through the required proxy.
2. Confirm whether the AI Hub integration has been completed.
3. Finalize the benchmark/report CSV.
4. Finalize the README and project documentation.
5. Prepare the final presentation and demonstration.

The final evaluated model remains **V8.4**. No additional model version is currently required for the completed benchmark.

---

# 30. Final Project Status

```text
PROJECT: Content Tagging & Competency Mapping

FINAL MODEL:
V8.4

FINAL BRANCH:
hybrid-v8

DATASET / EVALUATION:
Complete

MODEL / PIPELINE:
Complete

DOCKER:
Complete

KUBERNETES:
Complete

vLLM / QWEN:
Complete

API:
Complete

BENCHMARK:
Complete

GPU MONITORING:
Complete

PROMETHEUS:
Complete

GRAFANA:
Dashboard in progress

AI HUB:
Not yet verified

FINAL PRESENTATION:
Pending
```
