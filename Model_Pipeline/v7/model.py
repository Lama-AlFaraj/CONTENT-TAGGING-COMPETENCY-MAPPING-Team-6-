# ============================================================
# FINAL MODEL PROMPTS
# ============================================================

import json
import re
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import gc
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

# ============================================================
# QWEN 2.5 7B MODEL
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quant_config,
    device_map="auto"
)

model.eval()

print("Qwen2.5-7B-Instruct loaded.")

# ============================================================
# RUN QWEN ON ONE CHUNK
# ============================================================

def run_qwen(content):

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": build_prompt(content)
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=False
        )

    generated_tokens = outputs[0][
        inputs["input_ids"].shape[1]:
    ]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    del inputs
    del outputs
    del generated_tokens

    gc.collect()
    torch.cuda.empty_cache()

    try:

        result = clean_qwen_json(response)

        return validate_result(result)

    except Exception as e:

        return {
            "predicted_tags": [],
            "proposed_competencies": [],
            "difficulty_level": "Beginner",
            "confidence": 0.0,
            "notes": f"Qwen parsing error: {str(e)}"
        }



# Load the frozen taxonomy
taxonomy = pd.read_csv("saudi_skills_taxonomy_v1_final.csv")

taxonomy["skill_name_en"] = taxonomy["skill_name_en"].fillna("").astype(str).str.strip()
taxonomy["description_en"] = taxonomy["description_en"].fillna("").astype(str).str.strip()

VALID_COMPETENCIES = set(
    taxonomy.loc[
        taxonomy["skill_name_en"] != "",
        "skill_name_en"
    ].tolist()
)

print("Taxonomy skills:", len(VALID_COMPETENCIES))


# ------------------------------------------------------------
# System prompt
# ------------------------------------------------------------

SYSTEM_PROMPT = f"""
You are an AI learning-content tagging and competency-mapping system.

Your task is to analyze educational learning content and return structured JSON.

You MUST follow these rules:

1. Return 3 to 8 specific topic tags.
2. Tags must come from concepts actually present in the content.
3. Do not invent information.
4. Return 1 to 3 competencies.
5. Every competency MUST exactly match one of the allowed taxonomy skill names.
6. Difficulty must be exactly one of:
   Beginner
   Intermediate
   Advanced
7. Confidence must be a number between 0 and 1.
8. Return ONLY valid JSON.
9. Do not use Markdown or code fences.

VALID TAXONOMY COMPETENCIES:

{chr(10).join("- " + x for x in sorted(VALID_COMPETENCIES))}
"""


# ------------------------------------------------------------
# Content prompt
# ------------------------------------------------------------

def build_prompt(content):

    return f"""
Analyze the following learning content.

CONTENT:
========================
{content}
========================

TASK

Identify the main concepts, techniques, algorithms, tools, and procedures
actually taught or demonstrated in the content.

TAGS
Return 3 to 8 specific tags.

Good tags identify concrete concepts such as:
- decision trees
- entropy
- Gini impurity
- information gain
- Pandas DataFrame
- SQL joins
- prompt engineering

Do NOT use the filename as evidence.
Do NOT invent concepts.

COMPETENCIES

Select 1 to 3 competencies from the allowed taxonomy list.

A competency should be selected only when the content meaningfully
teaches, explains, demonstrates, or practices that skill.

Do not select a competency merely because it is broadly related.

The competency name MUST be copied exactly from the taxonomy.

DIFFICULTY

Choose exactly one:
Beginner
Intermediate
Advanced

Base this on the actual technical complexity of the content.

CONFIDENCE

Return a number from 0 to 1.

NOTES

Briefly explain the evidence supporting the selected competencies.

OUTPUT

Return ONLY this JSON structure:

{{
  "predicted_tags": [
    "tag 1",
    "tag 2",
    "tag 3"
  ],
  "proposed_competencies": [
    "EXACT taxonomy skill name"
  ],
  "difficulty_level": "Beginner",
  "confidence": 0.0,
  "notes": "Brief evidence-based explanation."
}}
"""

def clean_qwen_json(response):

    response = response.strip()

    response = re.sub(
        r"^```json\s*",
        "",
        response,
        flags=re.IGNORECASE
    )

    response = re.sub(
        r"^```\s*",
        "",
        response
    )

    response = re.sub(
        r"\s*```$",
        "",
        response
    )

    start = response.find("{")
    end = response.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(
            "No JSON object found in Qwen response."
        )

    json_text = response[start:end + 1]

    return json.loads(json_text)

# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------

def validate_result(result):

    tags = result.get("predicted_tags", [])

    if not isinstance(tags, list):
        tags = []

    tags = [
        str(x).strip()
        for x in tags
        if str(x).strip()
    ]

    # Remove duplicate tags
    unique_tags = []
    seen = set()

    for tag in tags:
        key = tag.lower()

        if key not in seen:
            unique_tags.append(tag)
            seen.add(key)

    tags = unique_tags[:8]


    competencies = result.get("proposed_competencies", [])

    if not isinstance(competencies, list):
        competencies = []

    valid_competencies = []

    for competency in competencies:

        competency = str(competency).strip()

        if competency in VALID_COMPETENCIES:
            if competency not in valid_competencies:
                valid_competencies.append(competency)

    valid_competencies = valid_competencies[:3]


    difficulty = result.get(
        "difficulty_level",
        "Beginner"
    )

    if difficulty not in [
        "Beginner",
        "Intermediate",
        "Advanced"
    ]:
        difficulty = "Beginner"


    try:
        confidence = float(
            result.get("confidence", 0)
        )
    except:
        confidence = 0.0

    confidence = max(
        0.0,
        min(1.0, confidence)
    )


    notes = str(
        result.get("notes", "")
    ).strip()


    return {
        "predicted_tags": tags,
        "proposed_competencies": valid_competencies,
        "difficulty_level": difficulty,
        "confidence": confidence,
        "notes": notes
    }


print("Qwen prompt + validation ready.")


########################################################################## AGGREGATION

# ============================================================
# FINAL AGGREGATION
# ============================================================

def build_aggregation_prompt(chunk_results):

    # Collect evidence from all chunks
    chunk_evidence = []

    for i, result in enumerate(chunk_results, start=1):

        chunk_evidence.append(
            f"""
CHUNK {i}

Tags:
{", ".join(result.get("predicted_tags", []))}

Suggested competencies:
{", ".join(result.get("proposed_competencies", []))}

Difficulty:
{result.get("difficulty_level", "")}

Confidence:
{result.get("confidence", 0)}

Evidence:
{result.get("notes", "")}
"""
        )

    evidence_text = "\n".join(chunk_evidence)


    # Give the final model the taxonomy candidates
    candidate_rows = taxonomy[
        ["skill_name_en", "description_en"]
    ].to_dict("records")

    skills_text = "\n".join(
        f"- {row['skill_name_en']}: {row['description_en']}"
        for row in candidate_rows
    )


    return f"""
You are the FINAL decision stage of a learning-content
tagging and competency-mapping system.

The following chunk outputs are evidence extracted from
the same learning document.

They are NOT ground truth.

========================
AVAILABLE TAXONOMY
========================

{skills_text}

========================
CHUNK EVIDENCE
========================

{evidence_text}

========================
FINAL DECISION
========================

Use evidence across ALL chunks.

Select 1 to 3 competencies that are genuinely supported
by the learning content.

A competency is valid only when the document meaningfully
teaches, explains, demonstrates, or practices that skill.

Do NOT select a competency merely because it is broadly
related to the subject.

Every competency MUST be copied EXACTLY from the taxonomy.

TAGS

Return 3 to 8 specific tags representing actual concepts,
algorithms, techniques, tools, or procedures in the content.

Remove duplicates.

Do not invent tags.

DIFFICULTY

Choose exactly one:

Beginner
Intermediate
Advanced

Choose based on the actual technical complexity.

CONFIDENCE

Return a number between 0 and 1.

NOTES

Briefly explain the concrete evidence supporting the
final competency selection.

========================
OUTPUT
========================

Return ONLY valid JSON.

No Markdown.
No code fences.

{{
  "predicted_tags": [
    "tag 1",
    "tag 2",
    "tag 3"
  ],
  "proposed_competencies": [
    "EXACT taxonomy skill name"
  ],
  "difficulty_level": "Beginner",
  "confidence": 0.0,
  "notes": "Evidence-based explanation."
}}
"""


def run_final_aggregator(chunk_results):

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": build_aggregation_prompt(chunk_results)
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=False
        )

    generated_tokens = outputs[0][
        inputs["input_ids"].shape[1]:
    ]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    del inputs
    del outputs
    del generated_tokens

    gc.collect()
    torch.cuda.empty_cache()

    try:

        result = clean_qwen_json(response)

        return validate_result(result)

    except Exception as e:

        return {
            "predicted_tags": [],
            "proposed_competencies": [],
            "difficulty_level": "Beginner",
            "confidence": 0.0,
            "notes": f"Final aggregation error: {str(e)}"
        }


print("Final aggregator ready.")

#################################################################
import os
import gc
import torch

from pptx import Presentation
from docx import Document

# ============================================================
# CONTENT EXTRACTION
# ============================================================

def extract_pptx(path):
    prs = Presentation(path)

    text_parts = []

    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text = shape.text.strip()

                if text:
                    text_parts.append(text)

    return "\n".join(text_parts)


def extract_ipynb(path):
    with open(path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    text_parts = []

    for cell in notebook.get("cells", []):

        source = "".join(
            cell.get("source", [])
        ).strip()

        if source:
            text_parts.append(source)

    return "\n\n".join(text_parts)


def extract_xlsx(path):
    workbook = pd.ExcelFile(path)

    text_parts = []

    for sheet_name in workbook.sheet_names:

        df = pd.read_excel(
            path,
            sheet_name=sheet_name,
            header=None
        )

        text_parts.append(
            f"Sheet: {sheet_name}"
        )

        text_parts.append(
            df.to_string(
                index=False,
                header=False
            )
        )

    return "\n".join(text_parts)


def extract_text_file(path):
    with open(
        path,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:
        return f.read()


def extract_content(path):

    extension = Path(path).suffix.lower()

    if extension == ".pptx":
        return extract_pptx(path)

    elif extension == ".ipynb":
        return extract_ipynb(path)

    elif extension == ".xlsx":
        return extract_xlsx(path)

    elif extension in [
        ".md",
        ".txt",
        ".csv"
    ]:
        return extract_text_file(path)

    else:
        raise ValueError(
            f"Unsupported file type: {extension}"
        )


def split_content(
    content,
    chunk_size=6000,
    overlap=500
):

    if not content:
        return []

    chunks = []

    start = 0
    content_length = len(content)

    while start < content_length:

        end = min(
            start + chunk_size,
            content_length
        )

        chunks.append(
            content[start:end]
        )

        if end >= content_length:
            break

        start = end - overlap

    return chunks


print("Content extraction ready.")

# ============================================================
# LOAD FINAL 12-FILE EVALUATION SET
# ============================================================

ANNOTATION_FILE = "annotation_sheet_full_Lama.csv"
EVAL_DIR = Path("evaluation_data")

annotations = pd.read_csv(ANNOTATION_FILE)

FILE_COLUMN = "file_name"

evaluation_files = (
    annotations[FILE_COLUMN]
    .dropna()
    .astype(str)
    .str.strip()
    .tolist()
)

print("Evaluation files:", len(evaluation_files))

for i, file_name in enumerate(evaluation_files, start=1):

    file_path = EVAL_DIR / file_name
    exists = file_path.exists()

    print(
        f"{i:02d}. {file_name} "
        f"{'FOUND' if exists else 'MISSING'}"
    )

missing_files = [
    f for f in evaluation_files
    if not (EVAL_DIR / f).exists()
]

if missing_files:
    raise FileNotFoundError(
        f"Missing evaluation files: {missing_files}"
    )

print("\nAll evaluation files are available.")


# ============================================================
# RUN QWEN ON ALL EVALUATION FILES
# ============================================================

chunk_results_by_file = {}

for file_index, file_name in enumerate(
    evaluation_files,
    start=1
):

    print("\n" + "=" * 80)
    print(
        f"FILE {file_index}/{len(evaluation_files)}: "
        f"{file_name}"
    )
    print("=" * 80)

    file_path = EVAL_DIR / file_name

    content = extract_content(str(file_path))

    if not content:
        print("No content extracted.")
        chunk_results_by_file[file_name] = []
        continue

    print("Characters:", len(content))

    chunks = split_content(
        content,
        chunk_size=6000,
        overlap=500
    )

    print("Chunks:", len(chunks))

    chunk_results = []

    for chunk_index, chunk in enumerate(
        chunks,
        start=1
    ):

        print(
            f"\nProcessing chunk "
            f"{chunk_index}/{len(chunks)}"
        )

        try:

            result = run_qwen(chunk)

            chunk_results.append(result)

            print(
                "Tags:",
                result["predicted_tags"]
            )

            print(
                "Competencies:",
                result["proposed_competencies"]
            )

            print(
                "Difficulty:",
                result["difficulty_level"]
            )

        except Exception as e:

            print(
                "ERROR:",
                str(e)
            )

            chunk_results.append({
                "predicted_tags": [],
                "proposed_competencies": [],
                "difficulty_level": "Beginner",
                "confidence": 0.0,
                "notes": f"Chunk error: {str(e)}"
            })

        gc.collect()
        torch.cuda.empty_cache()

    chunk_results_by_file[file_name] = chunk_results

    print(
        f"\nCompleted {file_name}: "
        f"{len(chunk_results)} chunks"
    )

print("\nALL FILES PROCESSED.")

# ============================================================
# FINAL AGGREGATION FOR ALL FILES
# ============================================================

final_results_v7 = []

for file_index, file_name in enumerate(
    evaluation_files,
    start=1
):

    print("\n" + "=" * 80)
    print(
        f"FINAL AGGREGATION "
        f"{file_index}/{len(evaluation_files)}: "
        f"{file_name}"
    )
    print("=" * 80)

    chunk_results = chunk_results_by_file.get(
        file_name,
        []
    )

    if not chunk_results:

        result = {
            "predicted_tags": [],
            "proposed_competencies": [],
            "difficulty_level": "Beginner",
            "confidence": 0.0,
            "notes": "No chunk results."
        }

    else:

        result = run_final_aggregator(
            chunk_results
        )

    result["file_name"] = file_name

    final_results_v7.append(result)

    print("Tags:", result["predicted_tags"])
    print(
        "Competencies:",
        result["proposed_competencies"]
    )
    print(
        "Difficulty:",
        result["difficulty_level"]
    )
    print(
        "Confidence:",
        result["confidence"]
    )

    gc.collect()
    torch.cuda.empty_cache()


results_df_v7 = pd.DataFrame(
    final_results_v7
)

print(
    results_df_v7[[
        "file_name",
        "predicted_tags",
        "proposed_competencies",
        "difficulty_level",
        "confidence"
    ]].to_string(index=False)
)


# ============================================================
# SAVE FINAL MODEL PREDICTIONS
# ============================================================

OUTPUT_FILE = "qwen7b_final_predictions_v7.csv"

save_df = results_df_v7.copy()

save_df["predicted_tags"] = save_df[
    "predicted_tags"
].apply(
    lambda x: "; ".join(x)
    if isinstance(x, list)
    else str(x)
)

save_df["proposed_competencies"] = save_df[
    "proposed_competencies"
].apply(
    lambda x: "; ".join(x)
    if isinstance(x, list)
    else str(x)
)

save_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("Saved:", OUTPUT_FILE)
print("Rows:", len(save_df))

# ============================================================
# COMPETENCY + DIFFICULTY EVALUATION
# ============================================================

def split_values(value):

    if pd.isna(value):
        return []

    return [
        x.strip()
        for x in str(value).split(";")
        if x.strip()
    ]


gt = pd.read_csv(
    "annotation_sheet_full_Lama.csv"
)

pred = pd.read_csv(
    "qwen7b_final_predictions_v7.csv"
)

COMPETENCY_COLUMN = (
    "proposed_competencies "
    "(1-3 skill names, COPIED EXACTLY from valid_taxonomy_skill_reference.csv)"
)

TAG_COLUMN = (
    "predicted_tags (3-8, semicolon-separated, specific not broad)"
)

merged = gt.merge(
    pred,
    on="file_name",
    how="inner"
)

print("Matched evaluation files:", len(merged))


# ------------------------------------------------------------
# Competency metrics
# ------------------------------------------------------------

competency_rows = []

total_tp = 0
total_fp = 0
total_fn = 0

for _, row in merged.iterrows():

    gt_comp = set(
        split_values(
            row[COMPETENCY_COLUMN]
        )
    )

    pred_comp = set(
        split_values(
            row["proposed_competencies"]
        )
    )

    tp = len(
        gt_comp & pred_comp
    )

    fp = len(
        pred_comp - gt_comp
    )

    fn = len(
        gt_comp - pred_comp
    )

    total_tp += tp
    total_fp += fp
    total_fn += fn

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    exact_match = (
        gt_comp == pred_comp
    )

    competency_rows.append({
        "file_name": row["file_name"],
        "gt_competencies": "; ".join(gt_comp),
        "pred_competencies": "; ".join(pred_comp),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": exact_match
    })


competency_eval = pd.DataFrame(
    competency_rows
)

micro_precision = (
    total_tp / (total_tp + total_fp)
    if total_tp + total_fp > 0
    else 0
)

micro_recall = (
    total_tp / (total_tp + total_fn)
    if total_tp + total_fn > 0
    else 0
)

micro_f1 = (
    2 * micro_precision * micro_recall /
    (micro_precision + micro_recall)
    if micro_precision + micro_recall > 0
    else 0
)

macro_f1 = competency_eval["f1"].mean()

exact_match_accuracy = competency_eval[
    "exact_match"
].mean()


print("\n" + "=" * 70)
print("COMPETENCY RESULTS")
print("=" * 70)

print(
    f"Micro Precision: {micro_precision:.3f}"
)

print(
    f"Micro Recall:    {micro_recall:.3f}"
)

print(
    f"Micro F1:        {micro_f1:.3f}"
)

print(
    f"Macro F1:        {macro_f1:.3f}"
)

print(
    f"Exact Match:     {exact_match_accuracy:.3f}"
)


# ------------------------------------------------------------
# Difficulty
# ------------------------------------------------------------

difficulty_gt = (
    merged["difficulty_level_y"]
    if "difficulty_level_y" in merged.columns
    else merged["difficulty_level"]
)

difficulty_pred = merged[
    "difficulty_level_x"
] if "difficulty_level_x" in merged.columns else None

# Because the prediction CSV and GT have the same column name,
# identify them explicitly from the merged dataframe.
difficulty_columns = [
    c for c in merged.columns
    if c.startswith("difficulty_level")
]

print("\nDifficulty columns:", difficulty_columns)

# Ground truth is first occurrence, prediction second occurrence
gt_difficulty = merged[
    difficulty_columns[0]
]

pred_difficulty = merged[
    difficulty_columns[1]
]

difficulty_accuracy = accuracy_score(
    gt_difficulty,
    pred_difficulty
)

print("\n" + "=" * 70)
print("DIFFICULTY RESULTS")
print("=" * 70)

print(
    f"Accuracy: {difficulty_accuracy:.3f}"
)


# ============================================================
# TAG EVALUATION
# ============================================================

tag_rows = []

total_tp = 0
total_fp = 0
total_fn = 0

for _, row in merged.iterrows():

    gt_tags = set(
        x.lower().strip()
        for x in split_values(
            row[TAG_COLUMN]
        )
    )

    pred_tags = set(
        x.lower().strip()
        for x in split_values(
            row["predicted_tags"]
        )
    )

    matched = gt_tags & pred_tags

    tp = len(matched)

    fp = len(
        pred_tags - gt_tags
    )

    fn = len(
        gt_tags - pred_tags
    )

    total_tp += tp
    total_fp += fp
    total_fn += fn

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if precision + recall > 0
        else 0
    )

    tag_rows.append({
        "file_name": row["file_name"],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched": len(matched),
        "ground_truth_count": len(gt_tags),
        "predicted_count": len(pred_tags)
    })


tag_eval = pd.DataFrame(tag_rows)

tag_micro_precision = (
    total_tp / (total_tp + total_fp)
    if total_tp + total_fp > 0
    else 0
)

tag_micro_recall = (
    total_tp / (total_tp + total_fn)
    if total_tp + total_fn > 0
    else 0
)

tag_micro_f1 = (
    2 * tag_micro_precision * tag_micro_recall /
    (tag_micro_precision + tag_micro_recall)
    if tag_micro_precision + tag_micro_recall > 0
    else 0
)

tag_macro_f1 = tag_eval["f1"].mean()


print("\n" + "=" * 70)
print("TAG RESULTS")
print("=" * 70)

print(
    f"Micro Precision: {tag_micro_precision:.3f}"
)

print(
    f"Micro Recall:    {tag_micro_recall:.3f}"
)

print(
    f"Micro F1:        {tag_micro_f1:.3f}"
)

print(
    f"Macro F1:        {tag_macro_f1:.3f}"
)


# ============================================================
# SAVE EVALUATION RESULTS
# ============================================================

competency_eval.to_csv(
    "qwen7b_competency_evaluation_v7.csv",
    index=False
)

tag_eval.to_csv(
    "qwen7b_tag_evaluation_v7.csv",
    index=False
)

summary = pd.DataFrame([{
    "model": "Qwen2.5-7B-Instruct-4bit",

    "tag_micro_precision":
        tag_micro_precision,

    "tag_micro_recall":
        tag_micro_recall,

    "tag_micro_f1":
        tag_micro_f1,

    "tag_macro_f1":
        tag_macro_f1,

    "competency_micro_precision":
        micro_precision,

    "competency_micro_recall":
        micro_recall,

    "competency_micro_f1":
        micro_f1,

    "competency_macro_f1":
        macro_f1,

    "competency_exact_match":
        exact_match_accuracy,

    "difficulty_accuracy":
        difficulty_accuracy
}])

summary.to_csv(
    "qwen7b_final_evaluation_summary_v7.csv",
    index=False
)

print(summary.to_string(index=False))
