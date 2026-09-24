# Model Card — Content Tagging & Competency Mapping

## 1. Model Overview

**Project:** Content Tagging & Competency Mapping
**Final Version:** V8.4
**Use Case:** Automated tagging of educational content and mapping of identified concepts to competencies from the Saudi Skills Taxonomy.

The system processes learning-content files, identifies relevant topics and concepts, retrieves candidate competencies from a predefined Saudi Skills Taxonomy, and produces structured competency-mapping results.

## 2. Intended Use

The system is intended to support:

* Educational content platforms
* Learning and training providers
* Content and curriculum teams
* AI Hub users working with structured learning content
* Automated competency and skills-mapping workflows

The output can be used to assist with content organization, competency identification, and learning-content analysis.

## 3. Model Architecture

The final V8.4 pipeline consists of:

1. **Content Extraction** — extracts usable content from supported document formats.
2. **Topic Tagging** — Qwen identifies relevant topics and concepts.
3. **Semantic Query Expansion** — generates richer representations of the extracted concepts.
4. **Hybrid Retrieval** — combines multilingual E5 semantic retrieval with lexical matching.
5. **Cross-Chunk Aggregation** — combines evidence across document chunks.
6. **Taxonomy-Grounded Retrieval / RAG** — retrieves candidate competencies from the 134-entry Saudi Skills Taxonomy.
7. **Qwen Reranking** — evaluates and reranks retrieved candidates.
8. **Final Competency Mapping** — produces structured competency mappings together with relevant metadata.

## 4. Models and Components

### Generative Model

**Qwen/Qwen2.5-7B-Instruct-AWQ**

* Open-weight instruction-tuned LLM
* AWQ quantization
* Used for topic tagging, semantic processing, validation, and reranking
* Served using vLLM

### Embedding Model

**intfloat/multilingual-e5-base**

Used for semantic retrieval between extracted content and taxonomy competencies.

### Retrieval

The retrieval stage uses a hybrid approach combining:

* E5 semantic similarity
* Lexical matching
* Cross-chunk evidence aggregation
* LLM-based reranking

### Knowledge Source

The final competency mapping uses the **Saudi Skills Taxonomy**, containing 134 competency entries with skill names, descriptions, subsectors, and related job families.

The final pipeline therefore uses taxonomy-grounded retrieval rather than a separate conversational knowledge-base RAG system.

## 5. Evaluation Dataset

The evaluation uses **12 official learning-content files** with corresponding ground-truth topic labels and competency mappings.

The competency evaluation contains **39 gold competency mappings** derived from the evaluation dataset and Saudi Skills Taxonomy.

## 6. Evaluation Results

The final evaluation reports:

| Metric      | Result |
| ----------- | -----: |
| Macro F1    | 39.17% |
| Micro F1    | 38.46% |
| Recall@10   | 31.25% |
| Hit Rate@10 | 91.67% |

The system was also evaluated through API benchmarking and infrastructure testing.

## 7. Deployment

The model was deployed as a self-hosted inference service using:

* **FastAPI**
* **vLLM**
* **Docker**
* **Kubernetes**
* **RTX A6000 48 GB GPU**

The deployed API exposes health and prediction endpoints, including `/health` and `/predict`.

## 8. Infrastructure and Monitoring

The deployment is monitored using:

* Prometheus
* NVIDIA DCGM Exporter
* Grafana

Infrastructure measurements include GPU utilization, VRAM usage, power consumption, request behavior, latency, throughput, concurrency, and vLLM runtime metrics.

## 9. Model Comparison

V8.4 was compared with an OpenAI model as part of the project model evaluation.

The comparison considered:

* Output quality
* Evaluation performance
* Latency
* Token usage
* Cost
* Deployment approach

The OpenAI comparison is treated as a model-level evaluation, while detailed GPU and serving measurements are documented separately in the infrastructure evaluation.

## 10. Limitations

* The evaluation dataset contains 12 official learning-content files, so results may not represent all possible educational content.
* Competency mapping depends on the coverage and quality of the supplied Saudi Skills Taxonomy.
* The current taxonomy contains 134 competency entries.
* The system was evaluated primarily on the defined project dataset and taxonomy rather than a broad production-scale dataset.
* Multimodal content was not evaluated as a separate model capability; therefore, multimodal performance is outside the validated scope of the final V8.4 evaluation.
* Actual registration of the evaluation dataset in BeamData Data Hub and movement of the deployed model into the AI Hub Model Hub are not claimed unless separately verified.

## 11. Intended Deployment Context

The system is designed as a self-hosted AI inference service that can be integrated into an enterprise AI platform.

Its containerized deployment allows the model pipeline to run through Kubernetes and vLLM while exposing an API for downstream applications.

## 12. Contributors

**Team 7 — Content Tagging & Competency Mapping**

BeamData / AIDC Capstone Project
