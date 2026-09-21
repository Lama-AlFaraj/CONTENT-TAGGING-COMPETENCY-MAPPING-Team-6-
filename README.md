# Content Tagging & Competency Mapping

## 1. Project Overview

This project develops an AI-based pipeline for **content tagging and competency mapping**. The system analyzes learning content, identifies relevant topics and competencies, and maps the extracted content to competencies from the **Saudi Skills Taxonomy**.

The project focuses on building and evaluating an end-to-end pipeline that combines:

* AI-based topic and competency extraction
* Semantic summarization
* Semantic and lexical retrieval
* Cross-chunk aggregation
* LLM-based reranking
* Final competency mapping
* Automated evaluation against a human-reviewed benchmark

The final implementation is **V8.4**, which represents the final evaluated checkpoint of the project.

---

## 2. Problem

Learning materials can contain multiple concepts, topics, and skills distributed across different sections or content chunks. Manually identifying the relevant competencies and mapping them to a standardized skills taxonomy is time-consuming and can lead to inconsistent results.

The project addresses this problem by creating an automated pipeline that transforms learning content into structured competency mappings.

---

## 3. Project Objective

The objective is to build a pipeline that can:

1. Process learning content.
2. Identify relevant topics and competencies.
3. Generate a semantic representation of the content.
4. Retrieve relevant competencies from the Saudi Skills Taxonomy.
5. Aggregate evidence across content chunks.
6. Rerank candidate competencies using an LLM.
7. Produce final competency mappings.
8. Evaluate the results against human-reviewed gold labels.

---

## 4. System Architecture

The overall system follows this pipeline:

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
(E5 + Lexical)
       ↓
Cross-Chunk Aggregation
       ↓
Qwen Reranking
       ↓
Final Competency Mapping
       ↓
Structured Results
       ↓
Evaluation
```

The system uses multiple stages because competency mapping requires both **semantic understanding** and **precise matching against the taxonomy**.

---

## 5. End-to-End Pipeline

### Step 1 — Content Processing

The selected learning materials are processed and converted into usable content representations.

The benchmark contains **12 official evaluation files**, which form the evaluation set for the project.

---

### Step 2 — Qwen Topic and Competency Tagging

Qwen is used to analyze the learning content and identify relevant topics and competency-related concepts.

This stage provides the semantic information used by later retrieval stages.

---

### Step 3 — Semantic Summarization

The extracted content is summarized into a representation that preserves the important concepts relevant to competency mapping.

The summarized representation is used to improve the quality of the retrieval query.

---

### Step 4 — Hybrid Retrieval

The system retrieves candidate competencies from the Saudi Skills Taxonomy using two complementary approaches:

* **Semantic retrieval using multilingual E5**
* **Lexical retrieval**

The two retrieval signals are combined to generate a candidate set.

This hybrid strategy allows the system to capture both:

* semantically related competencies
* competencies with strong lexical overlap

---

### Step 5 — Cross-Chunk Aggregation

Learning content can contain evidence for the same competency across multiple chunks.

Cross-chunk aggregation combines retrieval evidence across the content rather than treating every chunk independently.

This provides a broader representation of the competency evidence contained in the learning material.

---

### Step 6 — Qwen Reranking

The retrieved candidate competencies are passed to Qwen for reranking.

The purpose of this stage is to use the LLM's semantic understanding to determine which candidates are most relevant to the learning content.

The reranker operates on the candidates produced by retrieval.

Therefore, if a correct competency is not retrieved into the candidate set, the reranker cannot select it.

---

### Step 7 — Final Competency Mapping

The highest-ranked relevant competencies are selected as the final competency mapping.

The final mappings are then compared against the gold-label evaluation reference.

---

## 6. Models and Main Components

### Qwen

Qwen is used for:

* Topic and competency tagging
* Semantic understanding
* Reranking retrieved competency candidates

### Multilingual E5

The multilingual E5 model is used for semantic retrieval.

It converts the content/query representation and taxonomy competency information into embeddings so that semantically related competencies can be retrieved even when the wording differs.

### Lexical Retrieval

Lexical retrieval provides an additional matching signal based on textual overlap.

It complements semantic retrieval by helping identify competencies that contain important matching terminology.

### Saudi Skills Taxonomy

The Saudi Skills Taxonomy provides the standardized competency reference used for competency mapping.

---

## 7. Retrieval Strategy

The retrieval system uses a **hybrid E5 + lexical approach**.

The process can be summarized as:

```text
Content Representation
        ↓
Semantic Query
        │
        ├──────────────→ E5 Semantic Retrieval
        │
        └──────────────→ Lexical Retrieval
                              │
                              ↓
                     Candidate Combination
                              ↓
                    Cross-Chunk Aggregation
                              ↓
                       Top-K Candidates
```

The retrieval stage is particularly important because it determines which competencies are available to the reranker.

A correct competency that is absent from the retrieved candidate set cannot be recovered by later reranking.

---

## 8. Reranking Strategy

After candidate retrieval, Qwen is used to rerank the retrieved competencies.

This creates a two-stage ranking process:

```text
Candidate Generation
        ↓
Hybrid Retrieval
        ↓
Top-K Candidate Set
        ↓
Qwen Reranking
        ↓
Final Competency Ranking
```

This separates the task into:

1. **Retrieval:** Find potentially relevant competencies.
2. **Reranking:** Determine which retrieved competencies are most relevant.

This distinction is important when interpreting the evaluation results.

---

## 9. Evaluation Dataset

The project uses a fixed benchmark consisting of **12 official evaluation files**.

A gold-label reference was prepared for these files and is used as the basis for evaluating the competency mappings.

Keeping the evaluation set fixed allows the different pipeline versions to be compared consistently.

---

## 10. Evaluation Methodology

The system is evaluated using competency-level retrieval and mapping metrics.

### Macro F1

Macro F1 measures F1 performance across evaluation samples while giving each sample equal weight.

### Micro F1

Micro F1 aggregates the predictions across samples before calculating precision and recall.

### Recall@10

Recall@10 measures how often the correct competency is present within the top 10 retrieved candidates.

This metric is particularly useful for diagnosing the retrieval stage.

### Hit Rate@10

Hit Rate@10 measures the proportion of evaluation samples for which at least one relevant competency appears within the top 10 candidates.

---

## 11. Experiment History

Several versions of the pipeline were evaluated before reaching the final V8.4 checkpoint.

The experiments were used to investigate the effects of changes to retrieval, aggregation, ranking, and competency mapping.

### V8.1

V8.1 was evaluated as an intermediate pipeline configuration.

The experiment provided a comparison point for subsequent changes and helped establish the behavior of the improved retrieval and mapping pipeline.

### V8.2

V8.2 introduced further changes to the pipeline and was evaluated against the same benchmark.

The results were used to identify areas where retrieval and ranking could be improved.

### V8.3

V8.3 was tested as an additional experiment.

The changes were evaluated, but the configuration was ultimately **reverted** rather than retained in the final system.

This experiment was therefore useful primarily as an evaluation of an alternative approach.

### V8.4

V8.4 became the final checkpoint.

The version was evaluated on the full benchmark, cleaned up, syntax-validated, committed to Git, and pushed to the `hybrid-v8` branch.

V8.4 is therefore the final implementation used for reporting the project results.

---

## 12. Final V8.4 Results

The final V8.4 benchmark results are:

| Metric      |       V8.4 |
| ----------- | ---------: |
| Macro F1    | **39.17%** |
| Micro F1    | **38.46%** |
| Recall@10   | **31.25%** |
| Hit Rate@10 | **91.67%** |

These metrics should be interpreted together.

The high Hit Rate@10 indicates that the system frequently retrieves at least one relevant candidate within the Top-10 set. However, the lower Recall@10 and F1 results show that retrieving a relevant candidate does not necessarily result in the correct final competency mapping.

---

## 13. Key Finding

The experiments identified **retrieval as the main current bottleneck**.

The reranker can only select from the candidates supplied by the retrieval stage. When the correct competency does not reach the Top-10 candidate set, the reranker cannot recover it.

The current system can therefore be viewed as having two distinct areas of performance:

```text
Retrieval
    ↓
"Did we find the correct competency?"

        ↓

Reranking / Mapping
    ↓
"Did we select the correct competency from the candidates?"
```

Improving the first stage is therefore a major direction for future development.

---

## 14. Limitations

The main limitations identified during the experiments are:

### Retrieval Coverage

Some correct competencies are not retrieved into the Top-10 candidate set.

### Taxonomy Matching

Competencies can be difficult to distinguish when their descriptions are semantically similar or use overlapping terminology.

### Specialized Competencies

Some specialized competencies may require more precise domain-specific retrieval or query representations.

### Benchmark Size

The evaluation is based on a fixed set of 12 official evaluation files. A larger benchmark would provide a broader basis for evaluating generalization.

### Error Propagation

Errors in earlier stages can affect later stages. In particular, missing a relevant competency during retrieval prevents the reranker from selecting it.

---

## 15. Future Work

Future improvements can focus primarily on retrieval and candidate generation.

Potential directions include:

* Improving retrieval query generation
* Creating more precise competency representations
* Improving semantic retrieval
* Improving specialized competency retrieval
* Exploring more advanced reranking strategies
* Expanding the evaluation benchmark
* Adding more human-reviewed evaluation samples

These are future improvements rather than changes included in the final V8.4 checkpoint.

---

## 16. Project Validation

Before finalizing V8.4, the project was checked for implementation consistency.

The following checks were completed:

* Python syntax validation — **Passed**
* `git diff --check` — **Passed**
* Output/result cleanup — **Completed**
* V8.4 Git commit — **Completed**
* GitHub push — **Completed**
* `hybrid-v8` branch — **Created and pushed**

---

## 17. Repository / Version Status

The final project checkpoint is:

```text
Final Version: V8.4
Branch: hybrid-v8
Status: Completed and pushed to GitHub
```

Earlier versions remain useful as experiment checkpoints, while V8.4 represents the final selected configuration.

---

## 18. Conclusion

The project developed and evaluated an end-to-end AI pipeline for learning-content tagging and competency mapping against the Saudi Skills Taxonomy.

The final V8.4 system combines:

```text
Qwen
  +
Semantic Summarization
  +
E5 Semantic Retrieval
  +
Lexical Retrieval
  +
Cross-Chunk Aggregation
  +
Qwen Reranking
  =
Final Competency Mapping
```

The technical pipeline, benchmark, evaluation process, and experiments have been completed.

The final V8.4 results are:

* **39.17% Macro F1**
* **38.46% Micro F1**
* **31.25% Recall@10**
* **91.67% Hit Rate@10**

The experiments indicate that **retrieval coverage remains the primary area for future improvement**. The project is therefore at the final documentation and presentation stage rather than requiring another experimental pipeline version.
