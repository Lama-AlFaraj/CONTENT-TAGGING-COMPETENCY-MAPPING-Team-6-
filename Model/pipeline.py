import pandas as pd
import ast
import json

INPUT = "qwen7b_final_predictions_v6_FULL.csv"
OUTPUT = "qwen7b_final_predictions_v6.json"

OFFICIAL_FILES = {
    "Decision Trees Revised.pptx",
    "HandsOn1_Window_Functions.md",
    "Introduction to Pandas_.pptx",
    "Lecture - Decision Trees.ipynb",
    "Lecture - Pandas Basics.ipynb",
    "Lecture_ML_Workflow.ipynb",
    "ML Workflow Introduction.pptx",
    "Prompt_Engineering.ipynb",
    "Retrieval_Augmented_Generation.ipynb",
    "SQL_Foundations_for_Data_Science.pptx",
    "WK2_D1_Quiz1_IntroToML_LectureNotebook.xlsx",
    "WK4_D2_Quiz2_EnsembleLearningScenarios_Afternoon.xlsx",
}

def parse_list(value):
    if pd.isna(value):
        return []

    value = str(value).strip()

    try:
        result = ast.literal_eval(value)
        return result if isinstance(result, list) else [str(result)]
    except Exception:
        return [x.strip() for x in value.split(";") if x.strip()]


df = pd.read_csv(INPUT)

print("Input columns:")
print(df.columns.tolist())

df = df[df["file_name"].isin(OFFICIAL_FILES)].copy()

predictions = []

for _, row in df.iterrows():

    prediction = {
        "file_name": row["file_name"],
        "tags": parse_list(row["predicted_tags"]),
        "competencies": parse_list(row["proposed_competencies"]),
        "difficulty": row["difficulty_level"],
        "learning_objectives": parse_list(row["learning_objectives"]),
        "confidence": float(row["confidence"]),
        "notes": "" if pd.isna(row["notes"]) else str(row["notes"])
    }

    predictions.append(prediction)


output = {
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "pipeline_version": "V6",
    "task": "Content Tagging and Saudi Skills Competency Mapping",
    "num_predictions": len(predictions),
    "predictions": predictions
}

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print()
print(f"Created: {OUTPUT}")
print(f"Predictions: {len(predictions)}")
print("Output fields:")
print(list(predictions[0].keys()))
