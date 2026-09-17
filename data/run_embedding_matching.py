import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

TAXONOMY_PATH = "saudi_skills_taxonomy.csv"
GROUND_TRUTH_PATH = "evaluation_dataset_v1_with_competencies.csv"
TOP_K = 5

# Load data
taxonomy = pd.read_csv(TAXONOMY_PATH)
ground_truth = pd.read_csv(GROUND_TRUTH_PATH)

required_taxonomy = [
    "skill_name_en",
    "description_en",
    "subsector_en",
    "related_job_families_en",
    "needs_review",
]
missing = [c for c in required_taxonomy if c not in taxonomy.columns]
if missing:
    raise ValueError(f"Missing taxonomy columns: {missing}")

# E5 multilingual retrieval model
model = SentenceTransformer("intfloat/multilingual-e5-base")

# Candidate/reference representation.
# E5 is trained with passage: for retrieval candidates.
taxonomy["taxonomy_text"] = (
    "passage: "
    + taxonomy["skill_name_en"].fillna("").astype(str)
    + ". "
    + taxonomy["description_en"].fillna("").astype(str)
)

taxonomy_embeddings = model.encode(
    taxonomy["taxonomy_text"].tolist(),
    normalize_embeddings=True,
    show_progress_bar=True,
)

def retrieve_skills(query_text, top_k=TOP_K):
    # Query representation.
    query_embedding = model.encode(
        ["query: " + str(query_text)],
        normalize_embeddings=True,
    )[0]

    # Normalized dot product = cosine similarity.
    scores = taxonomy_embeddings @ query_embedding
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = taxonomy.iloc[top_indices][
        ["skill_name_en", "description_en"]
    ].copy()
    results["similarity"] = scores[top_indices]
    return results

rows = []

for _, row in ground_truth.iterrows():
    # Initial evaluation query: use the independently created content tags.
    query_text = str(row["ground_truth_tags"])

    results = retrieve_skills(query_text)

    record = {
        "file_name": row["file_name"],
        "query_type": "ground_truth_tags",
        "query_text": query_text,
    }

    for rank in range(TOP_K):
        if rank < len(results):
            record[f"predicted_skill_{rank+1}"] = results.iloc[rank]["skill_name_en"]
            record[f"score_{rank+1}"] = float(results.iloc[rank]["similarity"])
        else:
            record[f"predicted_skill_{rank+1}"] = ""
            record[f"score_{rank+1}"] = np.nan

    record["latency_ms"] = np.nan  # Fill only if inference time is measured.
    rows.append(record)

embedding_results = pd.DataFrame(rows)
embedding_results.to_csv("embedding_results.csv", index=False)

print("Saved embedding_results.csv")
