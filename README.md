# Content Tagging & Competency Mapping

## 1. Project Overview

This project develops an AI-based pipeline for **learning-content tagging and competency mapping**.

The system processes educational content, identifies relevant topics, and maps the content to competencies from a standardized **Saudi Skills Taxonomy**.

The project covers the development and evaluation of the AI pipeline as well as its deployment and serving infrastructure.

The work includes:

* Learning-content extraction
* AI-based topic tagging
* Semantic representation of content
* Competency retrieval
* Hybrid semantic and lexical retrieval experiments
* Candidate aggregation across content chunks
* Competency ranking and selection experiments
* Automated evaluation against human-reviewed ground truth
* Containerized API deployment
* Kubernetes-based model serving
* GPU monitoring and benchmarking

The project reached a final evaluated model checkpoint of **V8.4** for the end-to-end pipeline.

---

# 2. Problem

Educational learning materials can contain multiple concepts and skills distributed across slides, notebook cells, tables, documents, and other content sections.

Manually identifying the relevant topics and mapping them to a standardized competency taxonomy is:

* Time-consuming
* Difficult to scale
* Potentially inconsistent across annotators
* Challenging when relevant concepts are distributed across a long document

This project addresses the problem by transforming unstructured learning content into structured topic and competency information and evaluating the resulting mappings against human-reviewed annotations.

---

# 3. Project Objectives

The project aims to build a system that can:

1. Process different learning-content formats.
2. Extract and normalize content.
3. Identify relevant topics from the content.
4. Generate semantic representations for retrieval.
5. Retrieve relevant competencies from the Saudi Skills Taxonomy.
6. Combine evidence across content chunks where applicable.
7. Produce structured competency predictions.
8. Evaluate predictions against a human-reviewed benchmark.
9. Serve the pipeline through an API.
10. Benchmark and monitor the deployed AI system.

---

# 4. High-Level Architecture

The overall project workflow is:

```text
Learning Content
       │
       ▼
Content Extraction
       │
       ▼
Topic Tagging
       │
       ▼
Semantic Representation
       │
       ▼
Competency Retrieval
       │
       ├── Semantic Retrieval
       │
       └── Lexical Retrieval
       │
       ▼
Candidate Aggregation
       │
       ▼
Competency Selection / Ranking
       │
       ▼
Structured Output
       │
       ▼
Evaluation
       │
       ▼
Docker + Kubernetes Deployment
       │
       ▼
Benchmarking + Monitoring
```

The project contains several experiments and configurations. Therefore, not every component shown above belongs to the final selected configuration.

In particular, **LLM reranking was tested during development but was not retained as the final retrieval configuration because its evaluation performance was lower overall.**

---

# 5. Dataset and Evaluation Benchmark

## 5.1 Evaluation Dataset

The final benchmark contains:

```text
12 annotated learning-content files
```

The benchmark includes multiple educational-content formats, including:

* PowerPoint presentations (`.pptx`)
* Jupyter notebooks (`.ipynb`)
* Markdown (`.md`)
* Spreadsheet/structured learning-content files

The 12-file benchmark includes material covering topics such as:

* Machine learning
* Decision trees
* Ensemble learning
* Pandas
* SQL
* Prompt engineering
* Retrieval-Augmented Generation
* Window functions
* ML workflows

---

## 5.2 Topic Ground Truth

The 12 evaluation files were manually reviewed and assigned topic-level ground-truth tags.

These annotations are used to evaluate the **content-tagging stage** separately from competency retrieval.

The topic-tag evaluation measures whether the generated topic tags correctly represent the concepts present in each document.

This evaluation is separate from the competency-mapping evaluation.

---

## 5.3 Final Competency Ground Truth

The competency ground truth was subsequently reviewed and finalized.

The final competency evaluation set contains:

```text
12 files
34 validated competency mappings
```

The final mapping file is:

```text
competency_gold_labels_final.csv
```

Each mapping contains:

```text
file_name
ground_truth_concept
taxonomy_skill
match_decision
review_notes
```

Only mappings marked as **validated** are included in the final competency evaluation.

The earlier project evaluation used **39 proposed mappings**. Those should not be treated as the final ground truth.

---

# 6. Saudi Skills Taxonomy

The project uses the finalized taxonomy:

```text
saudi_skills_taxonomy_v1_final.csv
```

The taxonomy contains:

```text
134 competency entries
```

Main fields include:

```text
skill_name_en
description_en
subsector_en
related_job_families_en
needs_review
```

The taxonomy provides the standardized competency vocabulary used by the retrieval and mapping stages.

---

# 7. Evaluation Layers

The project evaluates the system at separate stages.

## Layer 1 — Topic Tagging

```text
Learning Content
      ↓
Generated Topic Tags
      ↓
Compared with
      ↓
Human-reviewed Topic Ground Truth
```

This measures the quality of the system's topic-level understanding.

---

## Layer 2 — E5 Competency Retrieval

```text
Predicted Topic Tags
      ↓
Domain-Aware Query Expansion
      ↓
Multilingual E5
      ↓
Top-5 Taxonomy Competencies
      ↓
Compared with
      ↓
34 Validated Competency Mappings
```

This measures whether semantic retrieval can find validated competencies from the taxonomy.

The final E5 evaluation is the source of the **83.33% Hit@1, 100% Hit@3, and 100% Hit@5** results described below.

---

## Layer 3 — End-to-End V8.4 Evaluation

The V8.4 pipeline was evaluated as a broader end-to-end system using an earlier competency benchmark.

Those results are retained as historical development results and should not be confused with the final 34-mapping E5 evaluation.

---

# 8. Final E5 Competency Retrieval Evaluation

The final E5 retrieval evaluation uses:

```text
Evaluation files:       12
Validated gold mappings: 34
Retrieval results:      60
Predictions per file:   Top-5
```

The retrieval output is:

```text
embedding_e5_domain_aware_results.csv
```

The evaluation output is:

```text
embedding_e5_final_evaluation_per_file.csv
embedding_e5_final_evaluation_summary.csv
```

## Final Results

| Metric      |      Result |
| ----------- | ----------: |
| Hit@1       |  **83.33%** |
| Hit@3       | **100.00%** |
| Hit@5       | **100.00%** |
| Precision@1 |  **83.33%** |
| Precision@3 |  **63.89%** |
| Precision@5 |  **45.00%** |
| Recall@1    |  **33.33%** |
| Recall@3    |  **70.14%** |
| Recall@5    |  **81.25%** |
| F1@1        |  **46.94%** |
| F1@3        |  **65.56%** |
| F1@5        |  **56.85%** |
| MRR         |  **90.28%** |

### Interpretation

The final E5 retrieval evaluation shows that:

* At least one validated competency was retrieved at **Top-1 for 83.33% of files**.
* At least one validated competency was retrieved within **Top-3 for all 12 files**.
* At least one validated competency was retrieved within **Top-5 for all 12 files**.
* **Recall@5 was 81.25%**, indicating that the Top-5 candidate sets covered a substantial portion of the validated competencies.
* **MRR was 90.28%**, indicating that the first relevant competency generally appeared near the top of the retrieved list.

Precision decreases as more candidates are included because the Top-5 set contains additional taxonomy competencies that are not part of the validated gold mapping for a particular document.

---

# 9. E5 Retrieval Methodology

The final domain-aware E5 evaluation uses:

```text
Model:
intfloat/multilingual-e5-base

Retrieval:
Top-5

Ranking:
60% skill-name similarity
40% full taxonomy-text similarity
```

The taxonomy representation combines the competency name and its description.

The query is constructed from the topic information generated for the learning content and expanded with broader domain terminology before being embedded.

The ground-truth competency mappings are **not used to construct the retrieval query**.

They are used only after retrieval for evaluation.

This separation prevents the gold competency labels from leaking into the retrieval stage.

---

# 10. E5 Experiment History

Several retrieval configurations were tested during development.

Earlier retrieval experiments produced:

```text
Initial E5:
Top-1 33.3%
Top-3 50.0%
Top-5 58.3%

Improved E5:
Top-1 33.3%
Top-3 83.3%
Top-5 91.7%

Previous domain-aware E5 evaluation:
Top-1 75.0%
Top-3 100%
Top-5 100%
```

These historical results were evaluated against an **earlier competency ground truth**.

They should therefore not be directly compared numerically with the final evaluation above, which uses the finalized **34 validated competency mappings**.

The final domain-aware E5 retrieval output is retained as:

```text
embedding_e5_domain_aware_results.csv
```

---

# 11. End-to-End V8.4 Pipeline

V8.4 represents the final evaluated checkpoint of the broader project pipeline.

The end-to-end pipeline includes:

```text
Content Extraction
       ↓
Topic / Concept Tagging
       ↓
Semantic Representation
       ↓
Hybrid Retrieval
       ↓
Cross-Chunk Aggregation
       ↓
Competency Selection
       ↓
Structured Output
```

The retrieval experiments included:

```text
Multilingual E5 semantic retrieval
+
Lexical retrieval
```

with the evaluated hybrid configuration using:

```text
E5 semantic retrieval: 60%
Lexical retrieval:     40%
```

---

# 12. Content Extraction

The API pipeline supports learning-content formats including:

```text
.pptx
.ipynb
.xlsx
.md
.txt
.csv
```

The extracted material is normalized into a representation that can be processed by the downstream AI pipeline.

---

# 13. Topic Tagging

The AI pipeline identifies relevant topic-level concepts from the extracted learning content.

These tags provide a compact semantic representation that can be used to construct retrieval queries.

The topic-tagging stage and competency-retrieval stage are evaluated separately so that errors can be analyzed by stage.

---

# 14. Semantic Retrieval

The project evaluated multilingual E5 embeddings for semantic competency retrieval.

The taxonomy entries are represented using their competency names and descriptions.

The final domain-aware E5 experiment uses a combined score based on:

```text
60% skill-name similarity
40% full taxonomy-text similarity
```

This configuration was evaluated using the final 12-file benchmark and 34 validated competency mappings.

---

# 15. Hybrid Retrieval

The broader V8.4 pipeline also evaluated a hybrid retrieval strategy combining:

```text
Semantic retrieval
+
Lexical retrieval
```

The purpose of the hybrid approach is to combine semantic similarity with lexical evidence when matching learning-content concepts to taxonomy terminology.

The hybrid retrieval results are documented separately from the final domain-aware E5 evaluation.

---

# 16. Cross-Chunk Aggregation

Longer learning materials may contain relevant competencies in different parts of the document.

The pipeline therefore supports processing content in chunks and aggregating candidate evidence across those chunks.

This allows competency candidates to be supported by multiple portions of the same learning material rather than relying on a single local section.

---

# 17. Reranking Experiment

An LLM-based reranking stage was evaluated during development.

The tested architecture was:

```text
Hybrid Retrieval
      ↓
Candidate Set
      ↓
LLM Reranking
      ↓
Final Competency Selection
```

However, the reranker was **not retained in the final selected configuration** because it performed worse overall during evaluation.

An important architectural constraint is:

> The reranker can only select from the retrieved candidate set.

Therefore, a competency that is not retrieved during the candidate-generation stage cannot be recovered by reranking.

---

# 18. Structured Output

The V8.4 pipeline produces structured results containing information such as:

```text
predicted tags
proposed competencies
difficulty level
confidence
notes
retrieval candidates
chunk count
```

This provides both the final prediction and supporting information that can be used for evaluation and analysis.

---

# 19. Earlier V8.4 Evaluation

Before the competency ground truth was finalized, the broader V8.4 pipeline was evaluated using an earlier set of **39 competency mappings**.

The recorded results were:

| Metric          | Earlier V8.4 Result |
| --------------- | ------------------: |
| Macro Precision |          **75.00%** |
| Macro Recall    |          **27.08%** |
| Macro F1        |          **39.17%** |
| Micro Precision |          **76.92%** |
| Micro Recall    |          **25.64%** |
| Micro F1        |          **38.46%** |
| Recall@10       |          **31.25%** |
| Hit Rate@10     |          **91.67%** |

These results are retained as **historical V8.4 benchmark results**.

They are not the final competency-retrieval evaluation because the underlying competency ground truth was subsequently reviewed and reduced from 39 proposed mappings to **34 validated mappings**.

---

# 20. Why the Two E5 Result Sets Differ

Two sets of results exist in the project because they answer different evaluation questions.

### Earlier V8.4 evaluation

```text
Earlier 39-mapping competency ground truth
        ↓
Broader V8.4 end-to-end pipeline
        ↓
Historical evaluation metrics
```

### Final E5 evaluation

```text
Final 34 validated competency mappings
        ↓
Domain-aware E5 retrieval
        ↓
Top-5 taxonomy candidates
        ↓
Final E5 retrieval metrics
```

Therefore, the two result sets should **not be presented as if they were measurements of exactly the same experiment**.

---

# 21. Deployment

The V8.4 API was containerized using Docker and deployed to Kubernetes.

The deployment architecture is:

```text
Client
  │
  ▼
FastAPI
  │
  ▼
AI Processing Pipeline
  │
  ▼
vLLM
  │
  ▼
Qwen 7B AWQ
  │
  ▼
NVIDIA RTX A6000
```

The deployed model-serving layer uses:

```text
Qwen/Qwen2.5-7B-Instruct-AWQ
```

served with:

```text
vLLM
```

The API provides:

```text
GET  /health
POST /predict
```

The container image used for the V8.4 API is:

```text
shahad09/beamdata-v8-api:v8.4
```

---

# 22. Kubernetes Infrastructure

The project was deployed in a Kubernetes environment with separate services for:

```text
V8.4 FastAPI
vLLM model serving
Monitoring
```

The deployment uses GPU-backed inference on an NVIDIA RTX A6000.

The infrastructure work includes:

* Docker containerization
* FastAPI serving
* Kubernetes deployment
* GPU-backed vLLM serving
* Service exposure
* Health checks
* End-to-end API testing

---

# 23. Benchmarking

The deployed V8.4 API was tested at multiple concurrency levels.

The authoritative benchmark contained:

```text
Total requests: 48
Successful requests: 43
Errors: 5
Overall error rate: 10.42%
```

The highest observed throughput in the recorded benchmark was:

```text
0.1783 requests/second
```

at concurrency 8.

A controlled monitoring run at concurrency 4 completed:

```text
12 / 12 successful requests
Average latency: 20.499 seconds
Throughput: 0.1700 requests/second
```

These are **serving-performance measurements**, not model-quality metrics.

---

# 24. GPU and Model Monitoring

The monitoring architecture is:

```text
NVIDIA GPU
    ↓
DCGM Exporter
    ↓
Prometheus
    ↓
Grafana
```

vLLM metrics are also collected through Prometheus.

The monitoring setup covers:

### GPU

* GPU utilization
* GPU memory usage
* GPU temperature
* GPU power

### vLLM

* Running requests
* Waiting requests
* KV-cache usage
* Generation throughput
* Token-level serving metrics where available

---

# 25. Monitoring Results

During the controlled monitoring benchmark, the following maximum/observed values were recorded:

| Metric                |       Observed |
| --------------------- | -------------: |
| GPU utilization       |       **100%** |
| GPU VRAM used         | **42,821 MiB** |
| GPU power             |  **297.104 W** |
| KV-cache usage        |     **43.30%** |
| vLLM running requests |          **2** |
| vLLM waiting requests |          **0** |

The monitoring stack successfully exposed GPU and vLLM metrics through Prometheus and Grafana.

---

# 26. Grafana

The monitoring dashboard provides visibility into:

```text
API / serving behavior
GPU utilization
GPU memory
GPU power
GPU temperature
vLLM request activity
KV-cache usage
Generation throughput
```

Grafana Cloud / Private Datasource Connect was also configured for remote monitoring of the Prometheus data.

---

# 27. Repository Structure

The repository contains the major components of the project:

```text
CONTENT-TAGGING-COMPETENCY-MAPPING-Team-6-
│
├── Model/
│   ├── v8/
│   │   ├── API / model pipeline
│   │   ├── retrieval
│   │   ├── evaluation
│   │   └── results
│   │
│   ├── evaluation_data/
│   └── saudi_skills_taxonomy_v1_final.csv
│
├── Infrastructure/
│   └── k8s/
│       ├── vLLM deployment
│       └── V8.4 API deployment
│
├── Infrastructure_Benchmark/
│   ├── benchmarking scripts
│   ├── benchmark reports
│   └── benchmark results
│
├── Dockerfile
└── README.md
```

The repository contains multiple experiment artifacts because the project was developed iteratively across several model and retrieval configurations.

---

# 28. Important Evaluation Files

The main finalized evaluation artifacts are:

```text
ground_truth_key_freezed.csv
```

Final content-level ground truth for the 12 evaluation files.

```text
competency_gold_labels_final.csv
```

Final **34 validated competency mappings** used for competency evaluation.

```text
saudi_skills_taxonomy_v1_final.csv
```

Final **134-entry Saudi Skills Taxonomy**.

```text
embedding_e5_domain_aware_results.csv
```

Final domain-aware E5 Top-5 retrieval output.

```text
embedding_e5_final_evaluation_per_file.csv
```

Per-file final E5 competency evaluation.

```text
embedding_e5_final_evaluation_summary.csv
```

Aggregate final E5 competency evaluation metrics.

---

# 29. Final Project Results

The project produced results at both the AI-quality and infrastructure levels.

## Final E5 competency retrieval

```text
12 evaluation files
34 validated competency mappings
60 Top-5 retrieval results

Hit@1:       83.33%
Hit@3:      100.00%
Hit@5:      100.00%
Recall@5:    81.25%
MRR:         90.28%
```

## Earlier V8.4 end-to-end evaluation

```text
Macro F1:       39.17%
Micro F1:       38.46%
Recall@10:      31.25%
Hit Rate@10:    91.67%
```

## Infrastructure benchmark

```text
48 total requests
43 successful
10.42% overall error rate
0.1783 req/s maximum recorded throughput
```

These metrics describe different stages of the project and should not be treated as interchangeable.

---

# 30. Project Status

| Workstream              | Status                     |
| ----------------------- | -------------------------- |
| Evaluation dataset      | Complete                   |
| Topic ground truth      | Complete                   |
| Competency ground truth | Complete                   |
| 34 validated mappings   | Complete                   |
| 134-entry taxonomy      | Complete                   |
| E5 retrieval evaluation | Complete                   |
| V8.4 model checkpoint   | Complete                   |
| Docker deployment       | Complete                   |
| FastAPI service         | Complete                   |
| Kubernetes deployment   | Complete                   |
| vLLM / Qwen serving     | Complete                   |
| API benchmarking        | Complete                   |
| GPU monitoring          | Complete                   |
| DCGM Exporter           | Complete                   |
| Prometheus              | Complete                   |
| Grafana dashboard       | Complete                   |
| Grafana Cloud / PDC     | Complete                   |
| AI Hub integration      | Pending final verification |
| Final documentation     | In progress                |
| Final presentation      | In progress                |

---

# 31. Final Technical Summary

This project demonstrates an end-to-end approach to **AI-assisted learning-content tagging and competency mapping**.

The system combines:

```text
Learning-content processing
        +
AI topic understanding
        +
Semantic competency retrieval
        +
Lexical retrieval experiments
        +
Taxonomy-based candidate generation
        +
Structured competency output
        +
Human-reviewed evaluation
        +
GPU-backed API deployment
        +
Kubernetes
        +
Monitoring
```

The final evaluation package is based on:

```text
12 learning-content files
34 validated competency mappings
134 taxonomy entries
```

The final domain-aware E5 retrieval evaluation achieved:

```text
Hit@1:    83.33%
Hit@3:   100.00%
Hit@5:   100.00%
Recall@5: 81.25%
MRR:      90.28%
```

The project also progressed from model experimentation to a containerized, GPU-backed V8.4 service with Kubernetes deployment, benchmarking, Prometheus monitoring, DCGM GPU metrics, and Grafana visualization.

**Final evaluated end-to-end checkpoint: V8.4**
