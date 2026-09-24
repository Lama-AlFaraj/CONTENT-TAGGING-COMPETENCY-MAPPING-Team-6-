# Use Case & Requirements

## 1. Business Problem

Educational and training content often contains multiple topics, concepts, and skills that must be manually identified and mapped to a competency framework. Manual classification can be time-consuming and may result in inconsistent tagging and competency assignments.

The project addresses this problem by developing an automated AI pipeline that extracts meaningful topics from learning content and maps them to relevant competencies from the Saudi Skills Taxonomy.

## 2. Target Users

The solution is intended for:

* **Learning and training providers** — organize and analyze educational content.
* **Content and curriculum teams** — identify topics and relevant competencies across learning materials.
* **Enterprise AI / platform users** — integrate automated content analysis into AI-powered workflows.

## 3. Functional Requirements

The system should:

1. Accept supported learning-content files.
2. Extract usable textual content from the input.
3. Identify relevant topics and concepts.
4. Retrieve candidate competencies from the Saudi Skills Taxonomy.
5. Combine semantic and lexical evidence during retrieval.
6. Aggregate evidence across document chunks.
7. Rerank candidate competencies using an LLM.
8. Produce structured competency-mapping results.
9. Provide prediction results through an API.
10. Support evaluation against a predefined ground-truth dataset.

## 4. Non-Functional Requirements

### Accuracy

The system should produce competency mappings that are relevant to the supplied learning content and should be evaluated using established classification and retrieval metrics.

### Latency

The deployed service should provide measurable and reproducible inference latency suitable for API-based use, with latency and throughput evaluated under controlled benchmark conditions.

### Scalability

The inference service should support concurrent requests and be deployable using containerized infrastructure and Kubernetes.

### Observability

The deployment should expose infrastructure and model-serving metrics that allow GPU utilization, memory usage, power, latency, and request behavior to be monitored.

### Reproducibility

The model pipeline, evaluation dataset, benchmark process, and deployment configuration should be documented so that the system can be reproduced and evaluated consistently.

## 5. Data Requirements

The project uses:

* 12 official evaluation files
* Ground-truth topic tags
* 39 gold competency mappings
* A 134-entry Saudi Skills Taxonomy

The evaluation dataset has been prepared in structured formats suitable for the required evaluation and Data Hub registration process.

## 6. Infrastructure Requirements

The final system requires GPU-based inference infrastructure capable of serving the quantized Qwen model.

The validated deployment uses:

* NVIDIA RTX A6000 48 GB GPU
* Docker
* Kubernetes
* vLLM
* FastAPI
* Prometheus
* NVIDIA DCGM Exporter
* Grafana

## 7. Production Considerations

For production use, the system should be evaluated on a larger and more diverse content collection and monitored continuously for mapping quality, latency, resource utilization, and taxonomy coverage.

The current project evaluation provides a controlled validation of the V8.4 pipeline on the defined evaluation dataset rather than claiming universal production performance.

## 8. Scope

The implemented scope focuses on **content tagging and competency mapping using text-based learning content and a predefined Saudi Skills Taxonomy**.

Multimodal evaluation was not included as a validated component of the final V8.4 evaluation and is therefore considered outside the demonstrated scope.
