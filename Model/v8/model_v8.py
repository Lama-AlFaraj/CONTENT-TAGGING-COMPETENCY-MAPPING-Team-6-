"""
V8 Hybrid Qwen + E5 competency mapping.

Architecture:

Content
  -> chunking
  -> Qwen topic tagging
  -> E5 semantic retrieval
  -> Top-K taxonomy candidates
  -> Qwen validation/reranking
  -> final competencies

Ground-truth data is NEVER used during inference.
"""

import gc
import json
import re
from pathlib import Path

import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from retrieval_v8 import TaxonomyRetriever


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

TAXONOMY_PATH = "Model/saudi_skills_taxonomy_v1_final.csv"

E5_MODEL = "intfloat/multilingual-e5-base"

CHUNK_SIZE = 6000
CHUNK_OVERLAP = 500

RETRIEVAL_TOP_K_PER_CHUNK = 5
RETRIEVAL_FINAL_K = 10

MAX_NEW_TOKENS_TAGGER = 350
MAX_NEW_TOKENS_RERANKER = 350


# ============================================================
# MODEL LOADING
# ============================================================

print("=" * 70)
print("V8: Loading Qwen2.5-7B-Instruct")
print("=" * 70)

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quant_config,
    device_map="auto",
)

model.eval()

print("Qwen loaded.")

print("=" * 70)
print("V8: Loading E5 retriever")
print("=" * 70)

retriever = TaxonomyRetriever(
    taxonomy_path=TAXONOMY_PATH,
    model_name=E5_MODEL,
    top_k=RETRIEVAL_TOP_K_PER_CHUNK,
)


taxonomy = retriever.taxonomy

VALID_COMPETENCIES = set(
    taxonomy["skill_name_en"].tolist()
)

print(
    "Taxonomy skills:",
    len(VALID_COMPETENCIES)
)


# ============================================================
# JSON HELPERS
# ============================================================

def clean_qwen_json(response):

    response = response.strip()

    response = re.sub(
        r"^```json\s*",
        "",
        response,
        flags=re.IGNORECASE,
    )

    response = re.sub(
        r"^```\s*",
        "",
        response,
    )

    response = re.sub(
        r"\s*```$",
        "",
        response,
    )

    start = response.find("{")
    end = response.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(
            "No JSON object found."
        )

    return json.loads(
        response[start:end + 1]
    )


def normalize_result(result):

    tags = result.get(
        "predicted_tags",
        [],
    )

    if not isinstance(tags, list):
        tags = []

    clean_tags = []
    seen = set()

    for tag in tags:

        tag = str(tag).strip()

        if not tag:
            continue

        key = tag.lower()

        if key not in seen:

            seen.add(key)
            clean_tags.append(tag)

    tags = clean_tags[:8]

    difficulty = result.get(
        "difficulty_level",
        "Beginner",
    )

    if difficulty not in {
        "Beginner",
        "Intermediate",
        "Advanced",
    }:
        difficulty = "Beginner"

    try:
        confidence = float(
            result.get(
                "confidence",
                0.0,
            )
        )
    except Exception:
        confidence = 0.0

    confidence = max(
        0.0,
        min(1.0, confidence),
    )

    return {
        "predicted_tags": tags,
        "difficulty_level": difficulty,
        "confidence": confidence,
        "notes": str(
            result.get("notes", "")
        ).strip(),
    }


# ============================================================
# QWEN GENERATION
# ============================================================

def generate_json(
    system_prompt,
    user_prompt,
    max_new_tokens,
):

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_tokens = outputs[0][
        inputs["input_ids"].shape[1]:
    ]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    del inputs
    del outputs
    del generated_tokens

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return clean_qwen_json(response)


# ============================================================
# STAGE 1 — TOPIC TAGGING
# ============================================================

TAGGER_SYSTEM_PROMPT = """
You are the topic-tagging stage of an educational
learning-content analysis system.

Your job is to identify concrete concepts actually taught
or demonstrated in the supplied content.

Rules:

1. Return 3 to 8 specific topic tags.
2. Tags must describe actual concepts, algorithms,
   techniques, tools, procedures, or methods present.
3. Do not use the filename as evidence.
4. Do not invent concepts.
5. Do not select competencies.
6. Difficulty must be Beginner, Intermediate, or Advanced.
7. Confidence must be between 0 and 1.
8. Return only valid JSON.
"""

def build_tagger_prompt(content):

    return f"""
Analyze this learning content.

CONTENT
========================
{content}
========================

Return:

{{
  "predicted_tags": [
    "specific concept 1",
    "specific concept 2",
    "specific concept 3"
  ],
  "difficulty_level": "Beginner",
  "confidence": 0.0,
  "notes": "Brief evidence-based explanation."
}}

Focus on concrete concepts rather than broad subjects.
"""


def run_tagger(content):

    try:

        result = generate_json(
            TAGGER_SYSTEM_PROMPT,
            build_tagger_prompt(content),
            MAX_NEW_TOKENS_TAGGER,
        )

        return normalize_result(result)

    except Exception as exc:

        return {
            "predicted_tags": [],
            "difficulty_level": "Beginner",
            "confidence": 0.0,
            "notes": f"Tagger error: {exc}",
        }


# ============================================================
# STAGE 2 — RETRIEVAL
# ============================================================

def build_retrieval_query(
    content,
    predicted_tags,
):

    tags_text = ", ".join(
        predicted_tags
    )

    # E5 works best when the query is concise.
    # Keep the semantic query focused on tags + an excerpt.
    excerpt = str(content).strip()[:2500]

    return (
        f"Topics: {tags_text}\n"
        f"Content evidence: {excerpt}"
    )


def retrieve_candidates(
    chunk_results,
):

    queries = []

    for item in chunk_results:

        query = build_retrieval_query(
            item["content"],
            item["result"]["predicted_tags"],
        )

        queries.append(query)

    return retriever.retrieve_from_chunks(
        queries,
        top_k_per_chunk=RETRIEVAL_TOP_K_PER_CHUNK,
        final_k=RETRIEVAL_FINAL_K,
    )


# ============================================================
# STAGE 3 — QWEN RERANK / VALIDATION
# ============================================================

RERANK_SYSTEM_PROMPT = """
You are the final competency validation stage of an
educational content tagging system.

You receive:
1. evidence extracted from document chunks
2. semantic retrieval candidates from a fixed taxonomy

Your job is NOT to search the entire taxonomy.

Choose only competencies from the supplied candidate list.

Rules:

1. Select 1 to 3 competencies.
2. A competency must be genuinely supported by the
   document content.
3. The content must meaningfully teach, explain,
   demonstrate, or practice the skill.
4. Do not select a candidate merely because it is broadly
   related.
5. Never invent a competency.
6. Copy competency names exactly.
7. If a retrieved candidate is not supported, reject it.
8. Return only valid JSON.
"""


def build_rerank_prompt(
    chunk_results,
    candidates,
):

    evidence_parts = []

    for i, item in enumerate(
        chunk_results,
        start=1,
    ):

        result = item["result"]

        evidence_parts.append(
            f"""
CHUNK {i}

Tags:
{", ".join(result["predicted_tags"])}

Difficulty:
{result["difficulty_level"]}

Evidence:
{result["notes"]}

Content excerpt:
{item["content"][:1800]}
"""
        )

    evidence_text = "\n".join(
        evidence_parts
    )

    candidate_parts = []

    for i, candidate in enumerate(
        candidates,
        start=1,
    ):

        candidate_parts.append(
            f"""
CANDIDATE {i}
Skill: {candidate["skill_name_en"]}
Description: {candidate["description_en"]}
Retrieval score: {candidate["retrieval_score"]:.4f}
Retrieved from chunks: {candidate["hits"]}
"""
        )

    candidate_text = "\n".join(
        candidate_parts
    )

    return f"""
DOCUMENT EVIDENCE
========================
{evidence_text}

RETRIEVED TAXONOMY CANDIDATES
========================
{candidate_text}

FINAL TASK

Select 1 to 3 genuinely supported competencies.

Return:

{{
  "proposed_competencies": [
    "EXACT candidate skill name"
  ],
  "confidence": 0.0,
  "notes": "Explain the concrete evidence supporting the selection."
}}

The competency names MUST be copied exactly from the
candidate list.
"""


def run_reranker(
    chunk_results,
    candidates,
):

    if not candidates:

        return {
            "proposed_competencies": [],
            "confidence": 0.0,
            "notes": "No retrieval candidates.",
        }

    try:

        result = generate_json(
            RERANK_SYSTEM_PROMPT,
            build_rerank_prompt(
                chunk_results,
                candidates,
            ),
            MAX_NEW_TOKENS_RERANKER,
        )

        competencies = result.get(
            "proposed_competencies",
            [],
        )

        if not isinstance(
            competencies,
            list,
        ):
            competencies = []

        valid = []

        candidate_names = {
            x["skill_name_en"]
            for x in candidates
        }

        for competency in competencies:

            competency = str(
                competency
            ).strip()

            if (
                competency in candidate_names
                and competency not in valid
            ):
                valid.append(
                    competency
                )

        try:
            confidence = float(
                result.get(
                    "confidence",
                    0.0,
                )
            )
        except Exception:
            confidence = 0.0

        confidence = max(
            0.0,
            min(1.0, confidence),
        )

        return {
            "proposed_competencies": valid[:3],
            "confidence": confidence,
            "notes": str(
                result.get(
                    "notes",
                    "",
                )
            ).strip(),
        }

    except Exception as exc:

        return {
            "proposed_competencies": [],
            "confidence": 0.0,
            "notes": f"Reranker error: {exc}",
        }


# ============================================================
# DOCUMENT PIPELINE
# ============================================================

def split_content(
    content,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP,
):

    if not content:
        return []

    chunks = []

    start = 0

    while start < len(content):

        end = min(
            start + chunk_size,
            len(content),
        )

        chunks.append(
            content[start:end]
        )

        if end >= len(content):
            break

        start = end - overlap

    return chunks


def process_document(
    content,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP,
):

    chunks = split_content(
        content,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    chunk_results = []

    for chunk_index, chunk in enumerate(
        chunks,
        start=1,
    ):

        print(
            f"  Chunk {chunk_index}/{len(chunks)}"
        )

        result = run_tagger(chunk)

        chunk_results.append({
            "chunk_index": chunk_index,
            "content": chunk,
            "result": result,
        })

    candidates = retrieve_candidates(
        chunk_results
    )

    reranked = run_reranker(
        chunk_results,
        candidates,
    )

    all_tags = []

    for item in chunk_results:

        all_tags.extend(
            item["result"]["predicted_tags"]
        )

    unique_tags = []
    seen = set()

    for tag in all_tags:

        key = tag.lower()

        if key not in seen:

            seen.add(key)
            unique_tags.append(tag)

    # Keep the first 8 tags for compatibility
    # with the V7 output schema.
    final_tags = unique_tags[:8]

    difficulty_votes = {}

    for item in chunk_results:

        difficulty = item["result"][
            "difficulty_level"
        ]

        difficulty_votes[difficulty] = (
            difficulty_votes.get(
                difficulty,
                0,
            ) + 1
        )

    final_difficulty = (
        max(
            difficulty_votes,
            key=difficulty_votes.get,
        )
        if difficulty_votes
        else "Beginner"
    )

    return {
        "predicted_tags": final_tags,
        "proposed_competencies": reranked[
            "proposed_competencies"
        ],
        "difficulty_level": final_difficulty,
        "confidence": reranked[
            "confidence"
        ],
        "notes": reranked["notes"],
        "retrieval_candidates": candidates,
        "chunk_count": len(chunks),
    }
