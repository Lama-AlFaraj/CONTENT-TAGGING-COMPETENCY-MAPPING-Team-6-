"""
V8 semantic retrieval layer.

IMPORTANT:
- This module NEVER reads ground-truth labels.
- It embeds only the frozen taxonomy and inference queries.
- E5 is used for candidate retrieval, not as the final decision maker.
"""

from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


DEFAULT_E5_MODEL = "intfloat/multilingual-e5-base"


class TaxonomyRetriever:
    def __init__(
        self,
        taxonomy_path: str,
        model_name: str = DEFAULT_E5_MODEL,
        top_k: int = 5,
    ):
        self.taxonomy_path = Path(taxonomy_path)
        self.model_name = model_name
        self.top_k = top_k

        self.taxonomy = pd.read_csv(self.taxonomy_path)

        required = {
            "skill_name_en",
            "description_en",
        }

        missing = required - set(self.taxonomy.columns)

        if missing:
            raise ValueError(
                f"Taxonomy missing required columns: {sorted(missing)}"
            )

        self.taxonomy["skill_name_en"] = (
            self.taxonomy["skill_name_en"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        self.taxonomy["description_en"] = (
            self.taxonomy["description_en"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        self.taxonomy = self.taxonomy[
            self.taxonomy["skill_name_en"] != ""
        ].reset_index(drop=True)

        print(
            f"Loading E5 retriever: {self.model_name}"
        )

        self.encoder = SentenceTransformer(
            self.model_name
        )

        self.taxonomy_texts = [
            (
                "passage: "
                f"{row.skill_name_en}. "
                f"{row.description_en}"
            )
            for row in self.taxonomy.itertuples()
        ]

        print(
            f"Encoding {len(self.taxonomy_texts)} taxonomy skills..."
        )

        self.taxonomy_embeddings = self.encoder.encode(
            self.taxonomy_texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

        print("E5 taxonomy index ready.")

    def retrieve(
        self,
        query_text: str,
        top_k: int = None,
    ) -> List[Dict]:

        if top_k is None:
            top_k = self.top_k

        query_text = str(query_text).strip()

        if not query_text:
            return []

        query = "query: " + query_text

        query_embedding = self.encoder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]

        scores = np.dot(
            self.taxonomy_embeddings,
            query_embedding,
        )

        indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for rank, idx in enumerate(indices, start=1):

            row = self.taxonomy.iloc[int(idx)]

            results.append({
                "rank": rank,
                "skill_name_en": row["skill_name_en"],
                "description_en": row["description_en"],
                "score": float(scores[idx]),
            })

        return results

    def retrieve_from_chunks(
        self,
        chunk_queries: List[str],
        top_k_per_chunk: int = 5,
        final_k: int = 10,
    ) -> List[Dict]:
        """
        Retrieve candidates from multiple chunks.

        Candidate score:
        - max semantic score across chunks
        - plus a small frequency bonus for repeated retrieval

        This prevents one chunk from dominating the complete document.
        """

        candidate_map = {}

        for chunk_index, query_text in enumerate(
            chunk_queries,
            start=1
        ):

            results = self.retrieve(
                query_text,
                top_k=top_k_per_chunk,
            )

            for item in results:

                skill = item["skill_name_en"]
                score = item["score"]

                if skill not in candidate_map:

                    candidate_map[skill] = {
                        "skill_name_en": skill,
                        "description_en": item["description_en"],
                        "max_score": score,
                        "sum_score": score,
                        "hits": 1,
                        "chunk_indices": [chunk_index],
                    }

                else:

                    current = candidate_map[skill]

                    current["max_score"] = max(
                        current["max_score"],
                        score,
                    )

                    current["sum_score"] += score
                    current["hits"] += 1
                    current["chunk_indices"].append(
                        chunk_index
                    )

        ranked = []

        for item in candidate_map.values():

            # Primary signal = strongest semantic match.
            # Repeated retrieval is a small supporting signal.
            final_score = (
                item["max_score"]
                + 0.02 * min(item["hits"], 3)
            )

            item["retrieval_score"] = final_score
            ranked.append(item)

        ranked.sort(
            key=lambda x: x["retrieval_score"],
            reverse=True,
        )

        return ranked[:final_k]
