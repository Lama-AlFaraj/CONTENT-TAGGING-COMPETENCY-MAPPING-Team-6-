"""
V8 Hybrid Retrieval Layer.

Architecture:
    E5 semantic retrieval
        +
    lexical token-overlap retrieval
        ->
    hybrid candidate ranking

IMPORTANT:
- This module NEVER reads ground-truth labels.
- It embeds only the frozen taxonomy and inference queries.
- E5 is used for candidate retrieval, not as the final decision maker.
- Hybrid weights are fixed from the retrieval evaluation:
      E5      = 0.6
      Lexical = 0.4
"""

from pathlib import Path
from typing import List, Dict

import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


DEFAULT_E5_MODEL = "intfloat/multilingual-e5-base"

DEFAULT_E5_WEIGHT = 0.6
DEFAULT_LEXICAL_WEIGHT = 0.4


def normalize_text(text: str) -> str:
    """
    Normalize text for lexical token matching.
    """
    text = str(text).lower()

    text = re.sub(
        r"[^a-z0-9\u0600-\u06ff\s]",
        " ",
        text,
    )

    text = re.sub(r"\s+", " ", text).strip()

    return text

def lexical_overlap_score(
    query_text: str,
    taxonomy_text: str,
) -> float:
    """
    Token-overlap score for hybrid retrieval.

    Uses query-token coverage, but removes very common
    generic English stopwords so that technical terms
    contribute more strongly to lexical matching.
    """

    STOPWORDS = {
        "the", "a", "an", "and", "or", "of", "to", "in",
        "on", "for", "with", "from", "by", "is", "are",
        "be", "as", "at", "this", "that", "these", "those",
        "using", "used", "use", "based", "including",
        "content", "topics", "semantic", "summary",
    }

    query_tokens = {
        token
        for token in normalize_text(query_text).split()
        if token not in STOPWORDS
    }

    taxonomy_tokens = set(
        normalize_text(taxonomy_text).split()
    )

    if not query_tokens or not taxonomy_tokens:
        return 0.0

    return (
        len(query_tokens & taxonomy_tokens)
        / len(query_tokens)
    )


class TaxonomyRetriever:

    def __init__(
        self,
        taxonomy_path: str,
        model_name: str = DEFAULT_E5_MODEL,
        top_k: int = 5,
        e5_weight: float = DEFAULT_E5_WEIGHT,
        lexical_weight: float = DEFAULT_LEXICAL_WEIGHT,
    ):

        self.taxonomy_path = Path(taxonomy_path)
        self.model_name = model_name
        self.top_k = top_k

        self.e5_weight = float(e5_weight)
        self.lexical_weight = float(lexical_weight)

        if not np.isclose(
            self.e5_weight + self.lexical_weight,
            1.0,
        ):
            raise ValueError(
                "e5_weight + lexical_weight must equal 1.0"
            )

        self.taxonomy = pd.read_csv(
            self.taxonomy_path
        )

        required = {
            "skill_name_en",
            "description_en",
        }

        missing = required - set(
            self.taxonomy.columns
        )

        if missing:
            raise ValueError(
                "Taxonomy missing required columns: "
                f"{sorted(missing)}"
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

        # Combined text used by E5 and lexical matching.
        self.taxonomy["search_text"] = (
            self.taxonomy["skill_name_en"]
            + ". "
            + self.taxonomy["description_en"]
        )

        self.taxonomy["lexical_text"] = (
            self.taxonomy["search_text"]
            .map(normalize_text)
        )

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
            f"Encoding {len(self.taxonomy_texts)} "
            "taxonomy skills..."
        )

        self.taxonomy_embeddings = (
            self.encoder.encode(
                self.taxonomy_texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=True,
            )
        )

        print(
            "E5 taxonomy index ready."
        )

        print(
            "Hybrid retrieval weights: "
            f"E5={self.e5_weight:.1f}, "
            f"Lexical={self.lexical_weight:.1f}"
        )

    def _normalize_e5_scores(
        self,
        scores: np.ndarray,
    ) -> np.ndarray:
        """
        Normalize semantic scores to [0, 1].

        This matches the normalization used in the
        hybrid retrieval evaluation.
        """

        score_min = scores.min()
        score_max = scores.max()

        if score_max > score_min:
            return (
                (scores - score_min)
                / (score_max - score_min)
            )

        return np.zeros_like(scores)

    def retrieve(
        self,
        query_text: str,
        top_k: int = None,
    ) -> List[Dict]:
        """
        Retrieve taxonomy candidates using:

            hybrid =
                0.6 * normalized_E5
                +
                0.4 * lexical_overlap
        """

        if top_k is None:
            top_k = self.top_k

        query_text = str(
            query_text
        ).strip()

        if not query_text:
            return []

        query = "query: " + query_text

        query_embedding = (
            self.encoder.encode(
                [query],
                normalize_embeddings=True,
                convert_to_numpy=True,
            )[0]
        )

        # -------------------------------------------------
        # 1. Semantic E5 score
        # -------------------------------------------------

        e5_scores = np.dot(
            self.taxonomy_embeddings,
            query_embedding,
        )

        e5_normalized = (
            self._normalize_e5_scores(
                e5_scores
            )
        )

        # -------------------------------------------------
        # 2. Lexical score
        # -------------------------------------------------

        lexical_scores = np.array([
            lexical_overlap_score(
                query_text,
                taxonomy_text,
            )
            for taxonomy_text
            in self.taxonomy["lexical_text"]
        ])

        # -------------------------------------------------
        # 3. Hybrid score
        # -------------------------------------------------

        hybrid_scores = (
            self.e5_weight * e5_normalized
            +
            self.lexical_weight * lexical_scores
        )

        indices = np.argsort(
            hybrid_scores
        )[::-1][:top_k]

        results = []

        for rank, idx in enumerate(
            indices,
            start=1,
        ):

            idx = int(idx)

            row = self.taxonomy.iloc[idx]

            results.append({
                "rank": rank,
                "skill_name_en": row[
                    "skill_name_en"
                ],
                "description_en": row[
                    "description_en"
                ],

                # Keep semantic score for diagnostics.
                "e5_score": float(
                    e5_scores[idx]
                ),

                # Normalized semantic score used
                # in the hybrid formula.
                "e5_normalized_score": float(
                    e5_normalized[idx]
                ),

                # Lexical component.
                "lexical_score": float(
                    lexical_scores[idx]
                ),

                # Final candidate-generation score.
                "score": float(
                    hybrid_scores[idx]
                ),

                "hybrid_score": float(
                    hybrid_scores[idx]
                ),
            })

        return results

    def retrieve_from_chunks(
        self,
        chunk_queries: List[str],
        top_k_per_chunk: int = 10,
        final_k: int = 10,
    ) -> List[Dict]:
        """
        Retrieve candidates from multiple chunks.

        For each chunk:
            1. Hybrid E5 + lexical ranking.
            2. Keep top_k_per_chunk candidates.

        Across chunks:
            - max hybrid score is the primary signal.
            - repeated retrieval receives a small frequency bonus.

        No ground-truth information is used.
        """

        candidate_map = {}

        for chunk_index, query_text in enumerate(
            chunk_queries,
            start=1,
        ):

            results = self.retrieve(
                query_text,
                top_k=top_k_per_chunk,
            )

            for item in results:

                skill = item[
                    "skill_name_en"
                ]

                score = item["score"]

                if skill not in candidate_map:

                    candidate_map[skill] = {
                        "skill_name_en": skill,
                        "description_en": item[
                            "description_en"
                        ],
                        "max_score": score,
                        "sum_score": score,
                        "hits": 1,
                        "chunk_indices": [
                            chunk_index
                        ],
                        "best_e5_score": item[
                            "e5_score"
                        ],
                        "best_lexical_score": item[
                            "lexical_score"
                        ],
                    }

                else:

                    current = candidate_map[
                        skill
                    ]

                    if score > current[
                        "max_score"
                    ]:
                        current[
                            "best_e5_score"
                        ] = item[
                            "e5_score"
                        ]

                        current[
                            "best_lexical_score"
                        ] = item[
                            "lexical_score"
                        ]

                    current[
                        "max_score"
                    ] = max(
                        current["max_score"],
                        score,
                    )

                    current[
                        "sum_score"
                    ] += score

                    current[
                        "hits"
                    ] += 1

                    current[
                        "chunk_indices"
                    ].append(
                        chunk_index
                    )

        ranked = []

        for item in candidate_map.values():

            # Primary signal:
            # strongest hybrid match across chunks.
            #
            # Repeated retrieval receives a small bonus.
            final_score = (
                item["max_score"]
                + 0.02 * min(
                    item["hits"],
                    3,
                )
            )

            item[
                "retrieval_score"
            ] = final_score

            ranked.append(item)

        ranked.sort(
            key=lambda x: x["retrieval_score"],
            reverse=True,
        )

        print("\n" + "=" * 70)
        print("V8 RETRIEVAL CANDIDATES")
        print("=" * 70)

        for rank, item in enumerate(
            ranked[:final_k],
            start=1,
        ):
            print(
                f"{rank:02d}. "
                f"{item['skill_name_en']} | "
                f"retrieval={item['retrieval_score']:.4f} | "
                f"E5={item['best_e5_score']:.4f} | "
                f"lexical={item['best_lexical_score']:.4f} | "
                f"hits={item['hits']}"
            )

        print("=" * 70)

        # Reassign final candidate rank.
        ranked = ranked[:final_k]

        for rank, item in enumerate(
            ranked,
            start=1,
        ):
            item["rank"] = rank

        return ranked
