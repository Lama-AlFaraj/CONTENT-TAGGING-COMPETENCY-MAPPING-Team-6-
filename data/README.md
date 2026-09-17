## Dataset

Dataset: 12 annotated learning-content files

Taxonomy: 134 competency entries

Model: multilingual-e5-base

Retrieval: domain-aware query expansion

Ranking: 60% skill-name similarity + 40% taxonomy-text similarity



--------------


## Results 

Top-1: 75.0%

Top-3: 100.0%

Top-5: 100.0%

MRR: 0.875

Precision: 46.7%

Recall: 88.9%

F1: 0.607


----------------------

## Evaluation Methodology 

Final methodology is the domain-aware E5 retrieval using intfloat/multilingual-e5-base, Top-5 retrieval, 60% skill-name + 40% full-taxonomy similarity, with ground truth used only for evaluation.

----------------------

## Document baseline vs. improved model

- Initial E5: Top-1 33.3%, Top-3 50.0%, Top-5 58.3%
- Improved E5: Top-1 33.3%, Top-3 83.3%, Top-5 91.7%
- Final domain-aware E5: Top-1 75.0%, Top-3 100%, Top-5 100%
- Reranker was tested and rejected because it performed worse overall.

-------------------------

## Finalize the evaluation dataset

We finalized the 12-file evaluation set, the ground-truth competency mappings, and the 134-skill Saudi Skills Taxonomy.
----------------------

## Important files

retrieval output file `embedding_e5_domain_aware_results.csv`
