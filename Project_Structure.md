# BeamData Capstone Project — Team 7
## Content Tagging & Competency Mapping

---

# 1. Project Overview

### Goal

Build an AI pipeline that takes learning content and produces structured metadata:

Learning Content
→ Content Extraction
→ Topic Tagging
→ Competency Mapping
→ Difficulty Classification
→ Learning Objectives
→ Structured JSON Output

### Main Output

For each learning-content file:

```json
{
  "file_name": "...",
  "topic_tags": [],
  "competencies": [],
  "difficulty_level": "...",
  "learning_objectives": []
}
