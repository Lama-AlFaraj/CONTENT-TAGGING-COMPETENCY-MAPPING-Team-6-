# Model / Pipeline

## 1. Overview

The Model / Pipeline stage implements the actual **Content Tagging and Saudi Skills Competency Mapping** system.

The pipeline takes learning content as input and produces structured AI-generated metadata:

* Topic tags
* Saudi Skills Taxonomy competencies
* Difficulty level
* Learning objectives
* Confidence
* Notes

The final output is available in both CSV and structured JSON formats.

---

## 2. Model

### Model Used

**Qwen/Qwen2.5-7B-Instruct**

The V6 implementation uses:

* Qwen 2.5 7B Instruct
* 4-bit NF4 quantization
* Automatic device mapping
* GPU acceleration

### Model Version

**V6**

V6 is the main model/pipeline implementation currently being prepared for deployment.

V7 was also implemented as a separate experiment and will be compared with V6 later.

---

## 3. Input Content

The pipeline supports multiple learning-content formats, including:

* `.pptx`
* `.ipynb`
* `.xlsx`
* `.md`
* `.txt`

The evaluation dataset contains 12 official learning-content files.

---

## 4. Pipeline Flow

```text
Learning Content
      ↓
Content Extraction
      ↓
Content Chunking
      ↓
Qwen 2.5 7B Inference
      ↓
Topic Tagging
      ↓
Competency Mapping
      ↓
Difficulty Classification
      ↓
Learning Objective Generation
      ↓
Final Aggregation
      ↓
Structured JSON / CSV
```

---

## 5. Content Extraction

The pipeline first extracts usable content from the input files.

Supported extraction includes:

* PowerPoint slide text and content
* Jupyter Notebook cells
* Markdown text
* Excel workbook content
* Plain text

The extracted content is then prepared for model inference.

---

## 6. Chunking

Large documents are divided into smaller chunks before inference.

Current V6 configuration:

* Chunk size: approximately 6000 characters
* Chunk overlap: approximately 500 characters

Chunking allows larger learning materials to be processed within the model's context limitations.

---

## 7. Model Inference

Each content chunk is passed to Qwen 2.5 7B with instructions to identify relevant learning-content information.

The model generates structured information for each chunk.

The chunk-level outputs are then aggregated into a final prediction for each source file.

---

## 8. Topic Tagging

The model generates specific topic tags describing the main concepts present in the learning material.

Examples include:

```text
Decision Trees
Gini Impurity
Feature Engineering
SQL
Join Types
Retrieval-Augmented Generation
Natural Language Processing
```

Tags are intended to represent the actual technical concepts covered by the content rather than broad generic categories.

---

## 9. Saudi Skills Taxonomy Mapping

The pipeline maps extracted content to the frozen Saudi Skills Taxonomy:

```text
saudi_skills_taxonomy_v1_final.csv
```

The taxonomy contains:

* Skill name
* Skill description
* Subsector
* Related job families
* Review status

The model uses the taxonomy to produce competency mappings using valid taxonomy skill names.

---

## 10. Difficulty Classification

Each learning-content file receives a difficulty level:

```text
Beginner
Intermediate
Advanced
```

The difficulty is generated based on the concepts and level of the learning material.

---

## 11. Learning Objectives

The pipeline generates learning objectives describing what a learner should be able to understand or accomplish after studying the content.

Example:

```text
Understand the fundamental concepts of decision trees
Analyze how decision trees are used for machine learning tasks
```

---

## 12. Final Output

The final V6 prediction contains:

```text
file_name
predicted_tags
proposed_competencies
difficulty_level
learning_objectives
confidence
notes
```

### CSV Output

```text
qwen7b_final_predictions_v6_FULL.csv
```

### JSON Output

```text
qwen7b_final_predictions_v6.json
```

The final JSON structure contains:

```json
{
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "pipeline_version": "V6",
  "task": "Content Tagging and Saudi Skills Competency Mapping",
  "num_predictions": 12,
  "predictions": []
}
```

---

## 13. V6 vs. V7

A separate V7 implementation was developed as an additional experiment.

V7 produced significantly different evaluation results from V6 and is being retained for comparison.

The final V6 vs. V7 comparison will be completed after deployment and benchmarking.

The comparison will consider:

* Tagging quality
* Competency mapping
* Difficulty classification
* Overall output quality
* Performance characteristics

---

## 14. Model / Pipeline Status

### Completed

* [x] Select model
* [x] Implement content extraction
* [x] Implement content chunking
* [x] Implement model inference
* [x] Implement topic tagging
* [x] Implement competency mapping
* [x] Implement difficulty classification
* [x] Implement learning-objective generation
* [x] Aggregate chunk-level results
* [x] Generate CSV output
* [x] Generate structured JSON output
* [x] Process the 12-file evaluation set

**Status: COMPLETE**

---

## 15. Next Stage

The Model / Pipeline stage is complete.

The next stage is Infrastructure.

The existing AIDC infrastructure will be used to prepare the environment for deploying and serving the model.

Completed Model / Pipeline
          ↓
Infrastructure
          ↓
Deployment
          ↓
Benchmarking

The Infrastructure stage will cover the existing AIDC environment, including:

Kubernetes
GPU resources
Containerized model serving
vLLM
Kubernetes services
Prometheus
Grafana
Monitoring infrastructure
