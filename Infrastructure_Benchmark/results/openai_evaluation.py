import json
import os
import re
import sys
import time
from pathlib import Path

# Make the project root importable when this script is run directly.
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from openai import OpenAI

# Reuse the exact content extraction and retrieval logic from V8.
from Model.v8.api import extract_content
from Model.v8.retrieval_v8 import TaxonomyRetriever


DATA_DIR = ROOT_DIR / "Model" / "evaluation_data"
TAXONOMY_PATH = ROOT_DIR / "Model" / "saudi_skills_taxonomy_v1_final.csv"
OUTPUT_DIR = ROOT_DIR / "Infrastructure_Benchmark" / "results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
# One output file per model so a second run never overwrites the first.
_SAFE = re.sub(r"[^A-Za-z0-9._-]+", "_", MODEL_NAME)
OUTPUT_NAME = "openai_evaluation.json" if MODEL_NAME == "gpt-5.6-luna" else f"openai_evaluation_{_SAFE}.json"

CHUNK_SIZE = 6000
CHUNK_OVERLAP = 500

RETRIEVAL_TOP_K_PER_CHUNK = 5
RETRIEVAL_FINAL_K = 10

MAX_OUTPUT_TOKENS = 350

client = OpenAI()

print("=" * 70)
print("OPENAI COMPETENCY EVALUATION")
print("=" * 70)
print(f"Model: {MODEL_NAME}")
print(f"Data:  {DATA_DIR}")
print()


# ============================================================
# PROMPTS
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


SEMANTIC_SYSTEM_PROMPT = """
You create concise semantic search queries for competency retrieval.

Return ONLY valid JSON with this exact structure:
{"summary": "one concise sentence"}

The summary must describe technical concepts actually taught or
demonstrated in the supplied content.

Do not:
- name Saudi taxonomy competencies
- invent unsupported topics
- discuss difficulty
- explain your reasoning
- use bullet points
"""


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


# ============================================================
# HELPERS
# ============================================================

def clean_json(text):
    text = str(text).strip()

    text = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"^```\s*",
        "",
        text,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model response.")

    return json.loads(text[start:end + 1])


def normalize_tags(tags):
    if not isinstance(tags, list):
        return []

    result = []
    seen = set()

    for tag in tags:
        tag = str(tag).strip()

        if not tag:
            continue

        key = tag.lower()

        if key not in seen:
            seen.add(key)
            result.append(tag)

    return result[:8]


def normalize_difficulty(value):
    if value not in {
        "Beginner",
        "Intermediate",
        "Advanced",
    }:
        return "Beginner"

    return value


def normalize_confidence(value):
    try:
        value = float(value)
    except Exception:
        value = 0.0

    return max(0.0, min(1.0, value))


def split_content(content):
    if not content:
        return []

    chunks = []
    start = 0

    while start < len(content):
        end = min(
            start + CHUNK_SIZE,
            len(content),
        )

        chunks.append(content[start:end])

        if end >= len(content):
            break

        start = end - CHUNK_OVERLAP

    return chunks


def openai_json(system_prompt, user_prompt):
    response = client.responses.create(
        model=MODEL_NAME,
        input=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )

    return clean_json(response.output_text)


# ============================================================
# STAGE 1 — TOPIC TAGGING
# ============================================================

def run_tagger(content):
    prompt = f"""
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

    result = openai_json(
        TAGGER_SYSTEM_PROMPT,
        prompt,
    )

    return {
        "predicted_tags": normalize_tags(
            result.get("predicted_tags", [])
        ),
        "difficulty_level": normalize_difficulty(
            result.get("difficulty_level")
        ),
        "confidence": normalize_confidence(
            result.get("confidence")
        ),
        "notes": str(
            result.get("notes", "")
        ).strip(),
    }


# ============================================================
# STAGE 2 — SEMANTIC SUMMARY
# ============================================================

def generate_semantic_summary(content, predicted_tags):
    tags_text = ", ".join(predicted_tags)

    prompt = f"""
Predicted topic tags:
{tags_text}

Content:
{str(content).strip()[:3500]}

Write one concise semantic summary containing the most important
technical concepts, methods, tools, and learning topics.
"""

    result = openai_json(
        SEMANTIC_SYSTEM_PROMPT,
        prompt,
    )

    summary = (
        result.get("summary")
        or result.get("semantic_summary")
        or result.get("text")
        or ""
    )

    return str(summary).strip()


# ============================================================
# STAGE 3 — RETRIEVAL
# ============================================================

retriever = TaxonomyRetriever(
    taxonomy_path=TAXONOMY_PATH,
    model_name="intfloat/multilingual-e5-base",
)


def build_retrieval_query(
    content,
    predicted_tags,
    semantic_summary,
):
    parts = []

    if predicted_tags:
        parts.append(
            "Topic tags: "
            + ", ".join(predicted_tags)
        )

    if semantic_summary:
        parts.append(
            "Semantic summary: "
            + semantic_summary
        )

    content_preview = " ".join(
        str(content).split()
    )[:2500]

    if content_preview:
        parts.append(
            "Content context: "
            + content_preview
        )

    return "\n".join(parts)


def retrieve_candidates(chunk_results):
    queries = []

    for item in chunk_results:
        result = item["result"]
        content = item["content"]

        semantic_summary = generate_semantic_summary(
            content,
            result["predicted_tags"],
        )

        item["semantic_summary"] = semantic_summary

        queries.append(
            build_retrieval_query(
                content,
                result["predicted_tags"],
                semantic_summary,
            )
        )

    return retriever.retrieve_from_chunks(
        queries,
        top_k_per_chunk=RETRIEVAL_TOP_K_PER_CHUNK,
        final_k=RETRIEVAL_FINAL_K,
    )


# ============================================================
# STAGE 4 — COMPETENCY VALIDATION
# ============================================================

def run_reranker(chunk_results, candidates):
    if not candidates:
        return {
            "proposed_competencies": [],
            "confidence": 0.0,
            "notes": "No retrieval candidates.",
        }

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

    prompt = f"""
DOCUMENT EVIDENCE
========================
{"".join(evidence_parts)}

RETRIEVED TAXONOMY CANDIDATES
========================
{"".join(candidate_parts)}

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

    result = openai_json(
        RERANK_SYSTEM_PROMPT,
        prompt,
    )

    candidate_names = {
        x["skill_name_en"]
        for x in candidates
    }

    competencies = result.get(
        "proposed_competencies",
        [],
    )

    if not isinstance(competencies, list):
        competencies = []

    valid = []

    for competency in competencies:
        competency = str(competency).strip()

        if (
            competency in candidate_names
            and competency not in valid
        ):
            valid.append(competency)

    return {
        "proposed_competencies": valid[:3],
        "confidence": normalize_confidence(
            result.get("confidence")
        ),
        "notes": str(
            result.get("notes", "")
        ).strip(),
    }


# ============================================================
# DOCUMENT PROCESSING
# ============================================================

def process_document(content):
    chunks = split_content(content)

    chunk_results = []

    for chunk_index, chunk in enumerate(
        chunks,
        start=1,
    ):
        print(
            f"    Chunk {chunk_index}/{len(chunks)}"
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

    difficulty_votes = {}

    for item in chunk_results:
        difficulty = item["result"]["difficulty_level"]

        difficulty_votes[difficulty] = (
            difficulty_votes.get(difficulty, 0) + 1
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
        "predicted_tags": unique_tags[:8],
        "proposed_competencies": reranked[
            "proposed_competencies"
        ],
        "difficulty_level": final_difficulty,
        "confidence": reranked["confidence"],
        "notes": reranked["notes"],
        "retrieval_candidates": candidates,
        "chunk_count": len(chunks),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    files = sorted(
        p for p in DATA_DIR.iterdir()
        if p.suffix.lower() in {
            ".pptx",
            ".ipynb",
            ".xlsx",
            ".md",
        }
        and p.name != "README_STUDENTS.md"
    )

    print(f"Files found: {len(files)}")

    if len(files) != 12:
        raise RuntimeError(
            f"Expected 12 evaluation files, found {len(files)}"
        )

    results = []

    for index, path in enumerate(files, start=1):
        print()
        print(
            f"[{index}/{len(files)}] {path.name}"
        )

        started = time.perf_counter()

        record = {
            "file_name": path.name,
            "status": "error",
        }

        try:
            content = extract_content(path)

            if not content or not content.strip():
                raise RuntimeError(
                    "No text could be extracted."
                )

            output = process_document(content)

            record.update(output)
            record["status"] = "success"

        except Exception as exc:
            record["error"] = (
                f"{type(exc).__name__}: {exc}"
            )

        record["latency_s"] = (
            time.perf_counter() - started
        )

        results.append(record)

        print(
            f"    Status: {record['status']}"
        )
        print(
            f"    Latency: {record['latency_s']:.2f}s"
        )

        if record["status"] == "success":
            print(
                f"    Competencies: "
                f"{record['proposed_competencies']}"
            )
            print(
                f"    Difficulty: "
                f"{record['difficulty_level']}"
            )

    output = {
        "timestamp": time.strftime(
            "%Y-%m-%dT%H:%M:%S"
        ),
        "model": MODEL_NAME,
        "data_dir": str(DATA_DIR),
        "taxonomy": str(TAXONOMY_PATH),
        "file_count": len(files),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "retrieval_top_k_per_chunk": (
            RETRIEVAL_TOP_K_PER_CHUNK
        ),
        "retrieval_final_k": RETRIEVAL_FINAL_K,
        "results": results,
    }

    output_path = OUTPUT_DIR / OUTPUT_NAME

    output_path.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    success_count = sum(
        r["status"] == "success"
        for r in results
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print(
        f"Success: {success_count}/{len(results)}"
    )
    print(
        f"Output: {output_path}"
    )


if __name__ == "__main__":
    main()
