# OpenAI vs V8.4 Competency Comparison

> This is a model-to-model agreement comparison on the same 12 evaluation files. It is not an accuracy benchmark because no gold competency labels were provided.

## Summary

- Files compared: **12**
- Exact competency-set agreement: **3/12 (25.0%)**
- Difficulty agreement: **10/12 (83.3%)**
- Average competency Jaccard similarity: **0.465**
- OpenAI average latency: **19.24s**
- V8.4 average latency: **16.39s**
- Average latency difference (OpenAI − V8.4): **2.86s**

## File-by-file comparison

| File | OpenAI competencies | V8.4 competencies | Competency match | Jaccard | Difficulty match | OpenAI latency | V8.4 latency |
|---|---|---|---|---:|---|---:|---:|
| Decision Trees Revised.pptx | Machine Learning (ML) | Machine Learning (ML) | Yes | 1.000 | Yes | 12.74s | 11.72s |
| HandsOn1_Window_Functions.md | Data processing | Data processing; Data collection and analysis | No | 0.500 | Yes | 14.22s | 11.14s |
| Introduction to Pandas_.pptx | Data processing | Data processing; Data collection and analysis | No | 0.500 | Yes | 18.88s | 11.37s |
| Lecture - Decision Trees.ipynb | Machine Learning (ML) | Machine Learning (ML) | Yes | 1.000 | Yes | 20.40s | 17.03s |
| Lecture - Pandas Basics.ipynb | Data collection and analysis; Data preparation | Data cleaning | No | 0.000 | Yes | 13.75s | 13.81s |
| Lecture_ML_Workflow.ipynb | Machine Learning (ML); Data collection and analysis | Machine Learning (ML) | No | 0.500 | No | 19.10s | 19.16s |
| ML Workflow Introduction.pptx | Machine Learning (ML); Data preparation; Data collection and analysis | Machine Learning (ML); Data processing | No | 0.250 | Yes | 17.36s | 11.37s |
| Prompt_Engineering.ipynb | Natural language Processing (NLP); Machine Learning (ML) | Natural language Processing (NLP); Machine Learning (ML) | Yes | 1.000 | Yes | 38.92s | 39.27s |
| Retrieval_Augmented_Generation.ipynb | Natural language Processing (NLP); Machine Learning (ML) | Natural language Processing (NLP); Data collection and analysis | No | 0.333 | Yes | 40.56s | 37.04s |
| SQL_Foundations_for_Data_Science.pptx | Data preparation; Database modelling | Database management and configuration; Data collection and analysis | No | 0.000 | Yes | 17.50s | 12.08s |
| WK2_D1_Quiz1_IntroToML_LectureNotebook.xlsx | Machine Learning (ML); Data processing | Machine Learning (ML) | No | 0.500 | No | 9.54s | 6.42s |
| WK4_D2_Quiz2_EnsembleLearningScenarios_Afternoon.xlsx | Machine Learning (ML) | Data processing | No | 0.000 | Yes | 7.92s | 6.21s |

## Differences

### HandsOn1_Window_Functions.md
- V8.4 only: data collection and analysis
- Difficulty: OpenAI = Intermediate; V8.4 = Intermediate

### Introduction to Pandas_.pptx
- V8.4 only: data collection and analysis
- Difficulty: OpenAI = Beginner; V8.4 = Beginner

### Lecture - Pandas Basics.ipynb
- OpenAI only: data collection and analysis, data preparation
- V8.4 only: data cleaning
- Difficulty: OpenAI = Beginner; V8.4 = Beginner

### Lecture_ML_Workflow.ipynb
- OpenAI only: data collection and analysis
- Difficulty: OpenAI = Beginner; V8.4 = Intermediate

### ML Workflow Introduction.pptx
- OpenAI only: data collection and analysis, data preparation
- V8.4 only: data processing
- Difficulty: OpenAI = Beginner; V8.4 = Beginner

### Retrieval_Augmented_Generation.ipynb
- OpenAI only: machine learning (ml)
- V8.4 only: data collection and analysis
- Difficulty: OpenAI = Intermediate; V8.4 = Intermediate

### SQL_Foundations_for_Data_Science.pptx
- OpenAI only: data preparation, database modelling
- V8.4 only: data collection and analysis, database management and configuration
- Difficulty: OpenAI = Beginner; V8.4 = Beginner

### WK2_D1_Quiz1_IntroToML_LectureNotebook.xlsx
- OpenAI only: data processing
- Difficulty: OpenAI = Beginner; V8.4 = Intermediate

### WK4_D2_Quiz2_EnsembleLearningScenarios_Afternoon.xlsx
- OpenAI only: machine learning (ml)
- V8.4 only: data processing
- Difficulty: OpenAI = Intermediate; V8.4 = Intermediate

## Methodology

- Both systems were evaluated on the same 12 course-content files.
- Competency predictions were compared as sets, ignoring ordering.
- Jaccard similarity = intersection / union of competency sets.
- Difficulty agreement checks exact equality of the difficulty label.
- Latency is measured independently by each evaluation run.
- No gold labels were available, so no accuracy, precision, recall, or F1 conclusions are made.