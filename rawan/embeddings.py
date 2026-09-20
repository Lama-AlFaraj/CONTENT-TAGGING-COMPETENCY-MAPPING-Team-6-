import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------
# 1. Load Saudi Skills Taxonomy
# ---------------------------------------

taxonomy_path = "../data/validated-dataset.csv"

taxonomy = pd.read_csv(taxonomy_path)

taxonomy["embedding_text"] = (
    taxonomy["skill_name_en"].fillna("")
    + ": "
    + taxonomy["description_en"].fillna("")
)

print("Taxonomy loaded:", len(taxonomy), "skills")


# ---------------------------------------
# 2. Load Embedding Model
# ---------------------------------------

model = SentenceTransformer(
    "intfloat/multilingual-e5-base"
)


# ---------------------------------------
# 3. Create Taxonomy Embeddings
# ---------------------------------------

skill_texts = [
    "passage: " + text
    for text in taxonomy["embedding_text"].tolist()
]

skill_embeddings = model.encode(
    skill_texts,
    normalize_embeddings=True
)

print("Taxonomy embeddings created:", skill_embeddings.shape)


# ---------------------------------------
# 4. Load Ground Truth Dataset
# ---------------------------------------

ground_truth_path = "../data/ground_truth_key.csv"

ground_truth = pd.read_csv(ground_truth_path)

print("Ground truth loaded:", len(ground_truth), "files")


# ---------------------------------------
# 5. Concept-to-Competency Mapping
# ---------------------------------------

concept_results = []

for _, row in ground_truth.iterrows():

    file_name = row["file_name"]
    tags = str(row["ground_truth_tags"]).split(";")

    for tag in tags:

        tag = tag.strip()

        if not tag:
            continue

        tag_embedding = model.encode(
            ["query: " + tag],
            normalize_embeddings=True
        )

        scores = cosine_similarity(
            tag_embedding,
            skill_embeddings
        )[0]

        top_indices = np.argsort(scores)[::-1][:3]

        for rank, idx in enumerate(top_indices, start=1):

            concept_results.append({
                "file_name": file_name,
                "source_tag": tag,
                "rank": rank,
                "mapped_skill": taxonomy.iloc[idx]["skill_name_en"],
                "similarity_score": round(float(scores[idx]), 4)
            })


concept_df = pd.DataFrame(concept_results)

concept_df.to_csv(
    "concept_to_competency_mapping.csv",
    index=False
)

print("Saved: concept_to_competency_mapping.csv")


# ---------------------------------------
# 6. File-Level Competency Mapping
# ---------------------------------------

file_results = []

for _, row in ground_truth.iterrows():

    file_name = row["file_name"]

    combined_concepts = (
        "This learning content teaches the following concepts and skills: "
        + str(row["ground_truth_tags"]).replace(";", ". ")
    )

    file_embedding = model.encode(
        ["query: " + combined_concepts],
        normalize_embeddings=True
    )

    scores = cosine_similarity(
        file_embedding,
        skill_embeddings
    )[0]

    top_indices = np.argsort(scores)[::-1][:5]

    for rank, idx in enumerate(top_indices, start=1):

        file_results.append({
            "file_name": file_name,
            "rank": rank,
            "mapped_skill": taxonomy.iloc[idx]["skill_name_en"],
            "similarity_score": round(float(scores[idx]), 4)
        })


file_df = pd.DataFrame(file_results)

file_df.to_csv(
    "file_level_competency_mapping.csv",
    index=False
)

print("Saved: file_level_competency_mapping.csv")


# ---------------------------------------
# 7. Display Sample Results
# ---------------------------------------

print("\nFile-level results:\n")

print(
    file_df.head(20).to_string(index=False)
)
    