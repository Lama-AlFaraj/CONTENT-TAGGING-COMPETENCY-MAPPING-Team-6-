# Content Tagging & Competency Mapping

## Final End-to-End Benchmark & Evaluation Report

## 1. Executive Summary

This project developed and evaluated an end-to-end AI pipeline for **learning-content tagging and competency mapping** against the **Saudi Skills Taxonomy**.

The final system combines:

* Learning-content extraction
* AI-based topic and competency tagging
* Semantic summarization
* Multilingual E5 semantic retrieval
* Lexical retrieval
* Cross-chunk evidence aggregation
* Qwen-based candidate reranking and validation
* Final competency mapping
* FastAPI inference serving
* Qwen 7B AWQ inference through vLLM
* Docker containerization
* Kubernetes deployment
* Prometheus monitoring
* DCGM GPU monitoring
* Grafana visualization

The project was evaluated at three levels:

1. **Dataset/model quality**
2. **End-to-end API performance**
3. **Infrastructure and GPU behavior**

The final model checkpoint is **V8.4**.

The final competency evaluation produced:

| Metric      |       V8.4 |
| ----------- | ---------: |
| Macro F1    | **39.17%** |
| Micro F1    | **38.46%** |
| Recall@10   | **31.25%** |
| Hit Rate@10 | **91.67%** |

The final deployed API was tested under concurrency levels from **1 to 8**, with **48/48 requests successfully completed** and **0% errors**.

A controlled monitoring run at concurrency 4 produced:

| Metric          |       Result |
| --------------- | -----------: |
| Requests        |           12 |
| Successful      |        12/12 |
| Errors          |            0 |
| Average latency |     19.624 s |
| P50             |     17.368 s |
| P95             |     39.275 s |
| P99             |     44.100 s |
| Throughput      | 0.1795 req/s |

During this workload, Prometheus recorded:

* Maximum GPU utilization: **100%**
* Maximum GPU memory used: **42,821 MiB**
* Maximum GPU power: **297.104 W**
* Maximum vLLM running requests: **2**
* Maximum vLLM waiting requests: **0**
* Maximum observed KV-cache usage: **0.433%**

These results provide an end-to-end benchmark covering the project from the evaluation dataset through the final deployed AI inference system.

---

# 2. Project Objective

The objective was to build an automated system capable of transforming learning material into structured competency information.

The intended workflow is:

```text
Learning Content
      ↓
Content Extraction
      ↓
Topic / Competency Tagging
      ↓
Semantic Representation
      ↓
Competency Retrieval
      ↓
Saudi Skills Taxonomy
      ↓
Candidate Reranking
      ↓
Competency Mapping
      ↓
Difficulty / Confidence / Notes
      ↓
Structured Output
```

The system is designed to reduce the manual effort required to identify relevant learning topics and map them to standardized competencies.

---

# 3. Evaluation Dataset

The project uses a fixed benchmark consisting of **12 official evaluation files**.

The evaluation set contains multiple learning-content formats, including:

* PowerPoint presentations
* Jupyter notebooks
* Markdown content
* Excel content

The competency evaluation uses a human-reviewed gold-label reference.

The final V8.4 competency evaluation contained:

* **12 evaluation files**
* **39 gold competency mappings**

Using a fixed benchmark allows the different model versions and deployment configurations to be compared consistently.

The Saudi Skills Taxonomy was frozen for the evaluation and used as the standardized competency reference.

---

# 4. Dataset and Annotation Work

The dataset stage included:

* Preparing the evaluation files
* Preparing human-reviewed competency annotations
* Establishing valid taxonomy competency references
* Preparing ground-truth tags
* Preparing difficulty labels
* Preparing evaluation mappings
* Separating evaluation data from model inference logic

The taxonomy contains the standardized competency names and descriptions used during retrieval.

The final pipeline does not use the evaluation gold labels during inference.

Gold labels are used only during evaluation.

This separation prevents the evaluation reference from becoming part of the inference process.

---

# 5. Content Tagging Evaluation

The content-tagging stage was evaluated against the human-reviewed ground-truth tags.

The earlier semantic tag-matching evaluation showed:

* **Precision: 100%**
* **Recall: approximately 72.7%**

This indicates that the generated tags were generally specific enough to match the accepted ground-truth concepts, while some expected tags were still missed.

The tagging evaluation was performed separately from competency mapping because the two tasks measure different parts of the system.

---

# 6. Final V8.4 Model Pipeline

The final V8.4 architecture is:

```text
Learning Content
       ↓
Content Extraction
       ↓
Qwen Topic / Competency Tagging
       ↓
Semantic Summarization
       ↓
Hybrid Retrieval
 ┌───────────────┐
 │ E5 Retrieval  │
 │ Lexical Match │
 └───────┬───────┘
         ↓
Cross-Chunk Aggregation
         ↓
Top-K Competency Candidates
         ↓
Qwen Reranking / Validation
         ↓
Final Competency Mapping
         ↓
Difficulty / Confidence / Notes
         ↓
Structured Result
```

---

# 7. Model Components

## Qwen

Qwen is used for:

* Topic and competency tagging
* Semantic interpretation
* Candidate reranking
* Competency validation
* Structured output generation

## Multilingual E5

`intfloat/multilingual-e5-base` is used for semantic retrieval.

The model converts the content/query representation and taxonomy competency information into embeddings.

This allows semantically related competencies to be retrieved even when their wording is different.

## Lexical Retrieval

Lexical retrieval provides an additional matching signal based on textual overlap.

The final retrieval strategy combines semantic and lexical evidence.

## Saudi Skills Taxonomy

The frozen Saudi Skills Taxonomy provides the standardized competency reference used by the system.

---

# 8. Hybrid Retrieval

The retrieval system combines:

```text
E5 Semantic Similarity
        +
Lexical Similarity
        ↓
Hybrid Candidate Ranking
        ↓
Cross-Chunk Aggregation
        ↓
Top-10 Candidates
```

The hybrid retrieval configuration uses both semantic meaning and important terminology.

This is important because competency descriptions can contain specialized terminology that may not always be represented optimally by semantic similarity alone.

---

# 9. Cross-Chunk Aggregation

Learning documents can contain evidence for a competency across multiple sections.

The V8.4 pipeline therefore processes documents in chunks and aggregates evidence across chunks.

The final candidate set represents evidence collected from the document rather than relying on only one isolated chunk.

This allows competencies that are supported by multiple sections of a document to accumulate retrieval evidence.

---

# 10. Reranking

The retrieved candidates are passed to Qwen for reranking and validation.

The architecture therefore separates:

```text
Candidate Generation
        ↓
"Which competencies could be relevant?"
        ↓
Candidate Set
        ↓
Reranking
        ↓
"Which retrieved competencies best match?"
```

A competency that is absent from the candidate set cannot be selected by the reranker.

However, the evaluation shows that retrieval availability is not the only issue: the system can retrieve relevant candidates while still failing to produce the complete correct final mapping.

---

# 11. Final V8.4 Model Quality Results

The final V8.4 evaluation produced:

| Metric          |     Result |
| --------------- | ---------: |
| Macro Precision | **75.00%** |
| Macro Recall    | **27.08%** |
| Macro F1        | **39.17%** |
| Recall@1        |  **4.86%** |
| Recall@3        | **11.81%** |
| Recall@5        | **16.67%** |
| Recall@10       | **31.25%** |
| Hit Rate@10     | **91.67%** |
| Micro Precision | **76.92%** |
| Micro Recall    | **25.64%** |
| Micro F1        | **38.46%** |

The evaluation contained:

* 12 files
* 39 gold competency mappings
* 10 true positives
* 3 false positives
* 29 false negatives

---

# 12. Interpretation of Model Results

The results show an important distinction between **candidate retrieval** and **final competency mapping**.

The system achieved a **91.67% Hit Rate@10**, meaning that at least one relevant competency appeared within the top-10 candidate set for most evaluation files.

However, final mapping recall was substantially lower.

This means that finding a relevant candidate does not automatically result in selecting all of the correct competencies.

The remaining errors can therefore occur at several points:

```text
Content Understanding
        ↓
Query / Representation
        ↓
Candidate Retrieval
        ↓
Candidate Ranking
        ↓
LLM Reranking
        ↓
Final Mapping
```

Future improvements should therefore investigate both:

* retrieval coverage
* candidate selection and final mapping behavior

rather than treating retrieval as the sole source of error.

---

# 13. Infrastructure Architecture

The final deployed architecture is:

```text
                  User / AI Hub
                       │
                       ↓
              V8.4 FastAPI
                       │
          ┌────────────┴────────────┐
          ↓                         ↓
   E5 + Taxonomy              Qwen / vLLM
          │                         │
          └────────────┬────────────┘
                       ↓
              Competency Result


Monitoring:

 V8.4 API ────────┐
                  │
 vLLM ────────────┼──→ Prometheus ──→ Grafana
                  │
 DCGM Exporter ───┘
       ↓
  RTX A6000
```

---

# 14. Docker Deployment

The V8.4 API was containerized using Docker.

Final image:

```text
shahad09/beamdata-v8-api:v8.4
```

The container runs the FastAPI application and connects to the Kubernetes vLLM service.

The image was pushed to Docker Hub and deployed using Kubernetes.

---

# 15. Kubernetes Deployment

The deployment runs in the `shahad` namespace.

Main components:

```text
beamdata-v8-api
beamdata-vllm-7b
```

The API is exposed internally through:

```text
beamdata-v8-api:8000
```

The vLLM service is exposed internally through:

```text
beamdata-vllm:8000
```

The vLLM deployment uses:

* Qwen/Qwen2.5-7B-Instruct-AWQ
* AWQ quantization
* One NVIDIA GPU
* Maximum model length of 16,384 tokens
* GPU memory utilization target of 0.85
* Prefix caching

---

# 16. GPU Infrastructure

The deployed inference workload runs on:

```text
GPU: NVIDIA RTX A6000
VRAM: approximately 49 GB
Driver: 550.90.12
CUDA: 12.4
```

The Qwen 7B AWQ model is served through vLLM.

The GPU was monitored using NVIDIA DCGM Exporter.

---

# 17. Prometheus and GPU Monitoring

The monitoring architecture is:

```text
RTX A6000
     ↓
DCGM Exporter
     ↓
Prometheus
     ↓
Grafana
```

Prometheus successfully scraped:

### vLLM metrics

Including:

* Running requests
* Waiting requests
* KV-cache utilization
* Other vLLM engine metrics

### GPU metrics

Including:

* GPU utilization
* GPU framebuffer memory
* GPU power
* GPU temperature
* Additional DCGM metrics

The final monitoring configuration successfully reported the BeamData vLLM service and DCGM exporter as healthy Prometheus targets.

---

# 18. End-to-End API Benchmark

The deployed V8.4 API was benchmarked at four concurrency levels.

| Concurrency | Requests | Success | Errors |     Avg |     P50 |     P95 |     P99 |   Throughput |
| ----------: | -------: | ------: | -----: | ------: | ------: | ------: | ------: | -----------: |
|           1 |       12 |      12 |      0 | 10.014s |  7.259s | 22.930s | 25.349s | 0.0999 req/s |
|           2 |       12 |      12 |      0 | 13.789s | 10.726s | 34.401s | 35.799s | 0.1420 req/s |
|           4 |       12 |      12 |      0 | 21.048s | 18.046s | 42.470s | 46.983s | 0.1654 req/s |
|           8 |       12 |      12 |      0 | 33.361s | 31.002s | 43.654s | 66.892s | 0.1681 req/s |

Overall:

```text
48 requests
48 successful
0 errors
0% error rate
```

The results show that throughput increased with concurrency but began to plateau at higher concurrency while latency increased.

---

# 19. Final Controlled Monitoring Benchmark

A separate C=4 run was performed specifically to capture infrastructure behavior.

Results:

| Metric          |       Result |
| --------------- | -----------: |
| Concurrency     |            4 |
| Requests        |           12 |
| Successful      |           12 |
| Errors          |            0 |
| Wall time       |     66.8709s |
| Average latency |      19.624s |
| P50             |      17.368s |
| P95             |      39.275s |
| P99             |      44.100s |
| Throughput      | 0.1795 req/s |

This run is the dedicated workload used for the final monitoring observations.

---

# 20. GPU Monitoring Results

During the five-minute Prometheus observation window around the controlled benchmark, the maximum observed values were:

| GPU / vLLM Metric     | Maximum Observed |
| --------------------- | ---------------: |
| GPU utilization       |         **100%** |
| GPU memory used       |   **42,821 MiB** |
| GPU power             |    **297.104 W** |
| vLLM running requests |            **2** |
| vLLM waiting requests |            **0** |
| KV-cache usage        |       **0.433%** |

These values represent **maximum observed values from Prometheus over the monitoring window**, not averages.

The results demonstrate that the deployed inference workload can make substantial use of GPU compute while maintaining available GPU memory headroom.

---

# 21. API Concurrency Fix

During earlier testing, concurrent requests exposed a serving issue caused by synchronous model processing inside the FastAPI request handler.

The API was updated to execute the blocking processing function through FastAPI's threadpool:

```python
result = await run_in_threadpool(process_document, content)
```

After rebuilding and redeploying the API, the final concurrency benchmark achieved:

```text
C=1 → 12/12 successful
C=2 → 12/12 successful
C=4 → 12/12 successful
C=8 → 12/12 successful
```

The earlier pre-fix concurrency failures are **not included in the final benchmark results**.

---

# 22. Validation and Reproducibility

The final V8.4 implementation was validated through:

* Python syntax validation
* `git diff --check`
* Docker image build
* Docker image push
* Kubernetes deployment
* API health verification
* vLLM health verification
* Prometheus target verification
* DCGM exporter verification
* End-to-end API benchmarking
* GPU monitoring
* vLLM monitoring

The final model checkpoint was committed and pushed to the `hybrid-v8` branch.

---

# 23. Experiment History

Multiple V8 configurations were evaluated before selecting V8.4.

The experiments investigated:

* Retrieval improvements
* Semantic representations
* Hybrid retrieval
* Cross-chunk evidence
* Reranking
* Candidate selection
* Competency mapping

V8.3 was evaluated as an alternative configuration and subsequently reverted.

V8.4 became the final checkpoint after evaluation and validation.

The final project therefore reports V8.4 rather than an intermediate experimental configuration.

---

# 24. Limitations

The benchmark identifies several limitations.

### Competency Recall

Final competency recall remains substantially lower than candidate Hit Rate@10.

### Candidate Selection

The presence of a relevant candidate in the Top-10 set does not guarantee that it will be selected in the final mapping.

### Taxonomy Similarity

Some competencies have similar descriptions or overlapping terminology, making fine-grained distinction difficult.

### Specialized Competencies

Domain-specific competencies can require more precise representations and retrieval queries.

### Benchmark Size

The final competency benchmark contains 12 official evaluation files and 39 gold competency mappings.

A larger human-reviewed benchmark would provide stronger evidence for generalization.

### Latency

The full V8.4 pipeline is relatively computationally expensive because each request involves content processing, embedding retrieval, aggregation, and LLM inference.

### Resource Usage

The deployed Qwen 7B model occupies substantial GPU memory, leaving limited remaining VRAM capacity on the RTX A6000 during normal operation.

---

# 25. Final Results Summary

The complete project can therefore be summarized at three levels.

## Dataset / Model Quality

```text
12 evaluation files
39 gold competency mappings

Macro F1:       39.17%
Micro F1:       38.46%
Recall@10:      31.25%
Hit Rate@10:    91.67%
```

## API Performance

```text
48/48 successful requests
0% errors

Maximum tested concurrency: 8

C=4 controlled run:
Average:        19.624 s
P50:             17.368 s
P95:             39.275 s
P99:             44.100 s
Throughput:      0.1795 req/s
```

## Infrastructure

```text
GPU:             NVIDIA RTX A6000
Model:           Qwen2.5-7B-Instruct-AWQ
Serving:         vLLM
API:             FastAPI
Container:       Docker
Orchestration:   Kubernetes
GPU monitoring:  DCGM Exporter
Metrics:         Prometheus
Visualization:   Grafana
```

Observed maximum GPU metrics:

```text
GPU utilization:     100%
VRAM used:           42,821 MiB
GPU power:            297.104 W
vLLM running:        2
vLLM waiting:        0
KV cache:             0.433%
```

---

# 26. Final Project Status

The core project implementation and benchmark are complete.

### Completed

* Dataset preparation
* Human-reviewed evaluation reference
* Saudi Skills Taxonomy integration
* Topic/tagging pipeline
* Semantic retrieval
* Lexical retrieval
* Hybrid retrieval
* Cross-chunk aggregation
* Qwen reranking
* Competency mapping
* V8.4 model evaluation
* Docker packaging
* Docker Hub image
* Kubernetes deployment
* vLLM deployment
* FastAPI deployment
* Concurrency testing
* GPU monitoring
* vLLM monitoring
* Prometheus integration
* DCGM Exporter integration
* Grafana monitoring infrastructure
* End-to-end benchmark
* Final benchmark results
* V8.4 validation
* Git commit and `hybrid-v8` push

### Final implementation

```text
Version: V8.4
Model: Qwen/Qwen2.5-7B-Instruct-AWQ
Retrieval: E5 + Lexical
Serving: vLLM + FastAPI
Platform: Kubernetes
GPU: NVIDIA RTX A6000
Monitoring: Prometheus + DCGM + Grafana
Branch: hybrid-v8
```

---

# 27. Conclusion

The project successfully progressed from a human-reviewed learning-content dataset to a deployed AI competency-mapping system.

The final architecture integrates:

```text
Dataset
   ↓
Content Processing
   ↓
Qwen Tagging
   ↓
Semantic Representation
   ↓
E5 + Lexical Retrieval
   ↓
Cross-Chunk Aggregation
   ↓
Qwen Reranking
   ↓
Competency Mapping
   ↓
FastAPI
   ↓
Docker
   ↓
Kubernetes
   ↓
vLLM / Qwen
   ↓
RTX A6000
   ↓
Prometheus + DCGM + Grafana
```

The final V8.4 model achieved **39.17% Macro F1**, **38.46% Micro F1**, **31.25% Recall@10**, and **91.67% Hit Rate@10** on the fixed competency benchmark.

The deployed system completed **48/48 API benchmark requests successfully** across concurrency levels 1, 2, 4, and 8.

The infrastructure benchmark additionally verified that GPU, vLLM, Prometheus, and DCGM monitoring operate together during real inference workloads.

The principal model-level improvement opportunity is the gap between retrieving relevant candidates and producing the complete correct final competency mapping. The principal infrastructure observation is that the system reaches high GPU utilization under concurrent inference while maintaining stable API execution through the tested concurrency level.

**V8.4 is the final evaluated project checkpoint.**
