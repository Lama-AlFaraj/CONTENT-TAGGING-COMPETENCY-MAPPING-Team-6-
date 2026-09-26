## Dataset

* **Dataset:** 12 annotated learning-content files
* **Final competency ground truth:** 34 validated competency mappings
* **Taxonomy:** 134 competency entries
* **Embedding model:** `intfloat/multilingual-e5-base`
* **Retrieval:** Domain-aware query expansion with Top-5 retrieval
* **Ranking:** 60% skill-name similarity + 40% full taxonomy-text similarity

---

## Results

Final evaluation against the **34 validated competency mappings**:

* **Hit@1:** 83.33%
* **Hit@3:** 100.00%
* **Hit@5:** 100.00%
* **Precision@5:** 45.00%
* **Recall@5:** 81.25%
* **F1@5:** 56.85%
* **MRR:** 90.28%

The evaluation covers all **12 files** and **60 Top-5 retrieval results**.

---

## Evaluation Methodology

The final evaluation uses the domain-aware E5 retrieval configuration with:

* `intfloat/multilingual-e5-base`
* Domain-aware query expansion
* Top-5 competency retrieval
* 60% skill-name similarity
* 40% full taxonomy-text similarity

The **34 validated competency mappings are used only as evaluation ground truth** and are not provided to the retrieval model.

---

## Document Baseline vs. Improved Model

Historical retrieval experiments showed progressive improvement in Top-K retrieval coverage:

* **Initial E5:** Top-1 33.3%, Top-3 50.0%, Top-5 58.3%
* **Improved E5:** Top-1 33.3%, Top-3 83.3%, Top-5 91.7%
* **Previous domain-aware E5 evaluation:** Top-1 75.0%, Top-3 100%, Top-5 100%

> **Note:** The historical results above were evaluated using the earlier competency ground truth. The final results reported in this README use the finalized **34 validated competency mappings** and therefore should not be directly compared numerically with the earlier evaluation.

A reranker was also tested during development but was not retained in the final configuration because its evaluation performance was lower overall.

---

## Finalized Evaluation Dataset

The evaluation package was finalized with:

* **12 annotated learning-content files**
* **34 validated competency mappings**
* **134-entry Saudi Skills Taxonomy**

The competency ground truth was reviewed and validated before the final E5 evaluation.

---

## Important Files

* `embedding_e5_domain_aware_results.csv` — final domain-aware E5 Top-5 retrieval output
* `competency_gold_labels_final.csv` — 34 validated competency mappings used as evaluation ground truth
* `saudi_skills_taxonomy_v1_final.csv` — finalized 134-entry Saudi Skills Taxonomy
* `embedding_e5_final_evaluation_per_file.csv` — per-file competency retrieval evaluation
* `embedding_e5_final_evaluation_summary.csv` — final aggregate evaluation metrics
