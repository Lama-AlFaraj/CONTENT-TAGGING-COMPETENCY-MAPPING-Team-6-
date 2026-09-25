import ast
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PRED_PATH = ROOT / "Model" / "v8" / "results" / "v8_predictions.csv"
GOLD_PATH = ROOT / "data" / "competency_gold_labels_review.csv"
OUT_PATH = ROOT / "Model" / "v8" / "results" / "v8_evaluation.csv"


def parse_list(value):
    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value:
        return []

    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass

    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass

    return []


def f1(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


pred = pd.read_csv(PRED_PATH)
gold = pd.read_csv(GOLD_PATH)

# Only the 12 official evaluation files that have Gold mappings.
gold = gold[gold["match_decision"].astype(str).str.lower() == "proposed"].copy()

gold_by_file = (
    gold.groupby("file_name")["taxonomy_skill"]
    .apply(lambda x: set(str(v).strip() for v in x if str(v).strip()))
    .to_dict()
)

pred_by_file = {
    row["file_name"]: {
        "competencies": set(parse_list(row["proposed_competencies"])),
        "retrieval": parse_list(row["retrieval_candidates"]),
    }
    for _, row in pred.iterrows()
}


rows = []

for file_name, gold_set in gold_by_file.items():
    if file_name not in pred_by_file:
        continue

    result = pred_by_file[file_name]
    predicted = result["competencies"]
    retrieval = result["retrieval"]

    tp = len(predicted & gold_set)
    fp = len(predicted - gold_set)
    fn = len(gold_set - predicted)

    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(gold_set) if gold_set else 0.0
    score_f1 = f1(precision, recall)

    retrieval_names = [
        str(x.get("skill_name_en", "")).strip()
        for x in retrieval
        if isinstance(x, dict)
    ]

    recall_at = {}

    for k in [1, 3, 5, 10]:
        top_k = set(retrieval_names[:k])
        hits = len(top_k & gold_set)
        recall_at[k] = hits / len(gold_set) if gold_set else 0.0

    hit_at_10 = 1.0 if (set(retrieval_names[:10]) & gold_set) else 0.0

    rows.append({
        "file_name": file_name,
        "gold_count": len(gold_set),
        "predicted_count": len(predicted),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": score_f1,
        "recall_at_1": recall_at[1],
        "recall_at_3": recall_at[3],
        "recall_at_5": recall_at[5],
        "recall_at_10": recall_at[10],
        "hit_rate_at_10": hit_at_10,
        "gold": " | ".join(sorted(gold_set)),
        "predicted": " | ".join(sorted(predicted)),
    })


df = pd.DataFrame(rows)

if df.empty:
    raise RuntimeError("No overlapping evaluation files found.")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_PATH, index=False)


# Macro average across files
metrics = [
    "precision",
    "recall",
    "f1",
    "recall_at_1",
    "recall_at_3",
    "recall_at_5",
    "recall_at_10",
    "hit_rate_at_10",
]

print("=" * 72)
print("V8 COMPETENCY EVALUATION")
print("=" * 72)

print(f"Evaluation files: {len(df)}")
print(f"Gold mappings:    {int(df['gold_count'].sum())}")
print()

print("MACRO AVERAGE")
print("-" * 72)

for metric in metrics:
    print(f"{metric:18s}: {df[metric].mean() * 100:6.2f}%")

print()
print("MICRO / GLOBAL")
print("-" * 72)

total_tp = int(df["true_positive"].sum())
total_fp = int(df["false_positive"].sum())
total_fn = int(df["false_negative"].sum())

micro_precision = (
    total_tp / (total_tp + total_fp)
    if total_tp + total_fp
    else 0.0
)

micro_recall = (
    total_tp / (total_tp + total_fn)
    if total_tp + total_fn
    else 0.0
)

micro_f1 = f1(micro_precision, micro_recall)

print(f"True positives:    {total_tp}")
print(f"False positives:   {total_fp}")
print(f"False negatives:   {total_fn}")
print(f"Precision:          {micro_precision * 100:.2f}%")
print(f"Recall:             {micro_recall * 100:.2f}%")
print(f"F1:                 {micro_f1 * 100:.2f}%")

print()
print("RETRIEVAL")
print("-" * 72)

for k in [1, 3, 5, 10]:
    print(
        f"Recall@{k:<2}:          "
        f"{df[f'recall_at_{k}'].mean() * 100:.2f}%"
    )

print(
    f"Hit Rate@10:       "
    f"{df['hit_rate_at_10'].mean() * 100:.2f}%"
)

print()
print("PER-FILE RESULTS")
print("-" * 72)

display_cols = [
    "file_name",
    "gold_count",
    "predicted_count",
    "precision",
    "recall",
    "f1",
    "recall_at_10",
]

display_df = df[display_cols].copy()

for col in [
    "precision",
    "recall",
    "f1",
    "recall_at_10",
]:
    display_df[col] = (display_df[col] * 100).round(1)

print(display_df.to_string(index=False))

print()
print(f"Saved: {OUT_PATH}")
print("=" * 72)
