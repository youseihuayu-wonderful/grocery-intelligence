"""XGBoost Learn-to-Rank model for grocery search.

Trains a lightweight ranking model that learns to combine multiple signals:
- BM25 keyword relevance score
- Semantic embedding similarity
- Product popularity (order count, reorder rate)
- Nutrition quality (nutri-score grade)
- Query-product text overlap features

The model learns optimal weights for these features from relevance labels,
outperforming both fixed-weight RRF and individual retrieval methods.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger

MODELS_DIR = Path(__file__).parent.parent.parent / "models"

# Nutrition grade to numeric mapping
GRADE_MAP = {"a": 5, "b": 4, "c": 3, "d": 2, "e": 1}


class LTRModel:
    """XGBoost learn-to-rank model for search result reranking."""

    def __init__(self, model_path: Path | None = None):
        self.model: xgb.XGBRanker | None = None
        self.feature_names = [
            "bm25_score",
            "semantic_score",
            "rrf_score",
            "bm25_rank",
            "semantic_rank",
            "order_count_log",
            "reorder_rate",
            "nutrition_grade_num",
            "has_nutrition_data",
            "query_name_overlap",
            "query_category_overlap",
            "name_length",
        ]

        if model_path and model_path.exists():
            self.load(model_path)

    def extract_features(
        self,
        query: str,
        candidate_indices: list[int],
        bm25_scores: np.ndarray,
        semantic_scores: np.ndarray,
        catalog: pd.DataFrame,
    ) -> np.ndarray:
        """Extract ranking features for candidate products.

        Args:
            query: Search query string
            candidate_indices: Indices into the catalog for candidate products
            bm25_scores: Full BM25 score array (all products)
            semantic_scores: Full semantic similarity array (all products)
            catalog: Product catalog DataFrame

        Returns:
            Feature matrix of shape (n_candidates, n_features)
        """
        query_tokens = set(query.lower().split())
        features = []

        # Pre-compute ranks for BM25 and semantic
        bm25_ranks = np.argsort(np.argsort(-bm25_scores))
        semantic_ranks = np.argsort(np.argsort(-semantic_scores))

        for idx in candidate_indices:
            row = catalog.iloc[idx]
            bm25 = float(bm25_scores[idx])
            sem = float(semantic_scores[idx])

            # RRF score
            rrf = 1 / (60 + bm25_ranks[idx] + 1) + 1 / (60 + semantic_ranks[idx] + 1)

            # Popularity features
            order_count = row.get("order_count", 0) or 0
            reorder_rate = row.get("reorder_rate", 0) or 0

            # Nutrition
            grade_str = row.get("nutrition_grade", "")
            grade_num = GRADE_MAP.get(str(grade_str).lower(), 0)
            has_nutrition = 1 if pd.notna(row.get("calories_100g")) else 0

            # Query-product text overlap
            name_tokens = set(str(row.get("product_name", "")).lower().split())
            cat_tokens = set(str(row.get("category", "")).lower().replace("_", " ").split())

            name_overlap = len(query_tokens & name_tokens) / max(len(query_tokens), 1)
            cat_overlap = len(query_tokens & cat_tokens) / max(len(query_tokens), 1)

            features.append([
                bm25,
                sem,
                rrf,
                float(bm25_ranks[idx]),
                float(semantic_ranks[idx]),
                np.log1p(order_count),
                reorder_rate,
                grade_num,
                has_nutrition,
                name_overlap,
                cat_overlap,
                len(name_tokens),
            ])

        return np.array(features)

    def train(
        self,
        queries: list[str],
        candidate_indices_per_query: list[list[int]],
        labels_per_query: list[list[int]],
        bm25_scores_per_query: list[np.ndarray],
        semantic_scores_per_query: list[np.ndarray],
        catalog: pd.DataFrame,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.1,
    ):
        """Train the LTR model.

        Args:
            queries: List of query strings
            candidate_indices_per_query: For each query, list of candidate product indices
            labels_per_query: For each query, relevance labels (0-4) for each candidate
            bm25_scores_per_query: For each query, full BM25 score array
            semantic_scores_per_query: For each query, full semantic similarity array
            catalog: Product catalog DataFrame
            n_estimators: Number of boosting rounds
            max_depth: Max tree depth
            learning_rate: Learning rate
        """
        all_features = []
        all_labels = []
        group_sizes = []

        for i, query in enumerate(queries):
            candidates = candidate_indices_per_query[i]
            labels = labels_per_query[i]

            X = self.extract_features(
                query, candidates,
                bm25_scores_per_query[i],
                semantic_scores_per_query[i],
                catalog,
            )
            all_features.append(X)
            all_labels.extend(labels)
            group_sizes.append(len(candidates))

        X_train = np.vstack(all_features)
        y_train = np.array(all_labels)

        self.model = xgb.XGBRanker(
            objective="rank:ndcg",
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            tree_method="hist",
            lambdarank_num_pair_per_sample=8,
            lambdarank_pair_method="topk",
        )

        self.model.fit(X_train, y_train, group=group_sizes)
        logger.info(
            f"LTR model trained on {len(queries)} queries, "
            f"{X_train.shape[0]} candidates, {X_train.shape[1]} features"
        )

    def predict(
        self,
        query: str,
        candidate_indices: list[int],
        bm25_scores: np.ndarray,
        semantic_scores: np.ndarray,
        catalog: pd.DataFrame,
    ) -> np.ndarray:
        """Predict relevance scores for candidates.

        Returns:
            Score array of shape (n_candidates,), higher = more relevant.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() or load() first.")

        X = self.extract_features(
            query, candidate_indices, bm25_scores, semantic_scores, catalog
        )
        return self.model.predict(X)

    def rerank(
        self,
        query: str,
        candidate_indices: list[int],
        bm25_scores: np.ndarray,
        semantic_scores: np.ndarray,
        catalog: pd.DataFrame,
        top_k: int = 10,
    ) -> list[dict]:
        """Rerank candidates using the LTR model and return enriched results."""
        scores = self.predict(
            query, candidate_indices, bm25_scores, semantic_scores, catalog
        )
        ranked_order = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank_pos in ranked_order:
            idx = candidate_indices[rank_pos]
            row = catalog.iloc[idx].to_dict()
            row["relevance_score"] = float(scores[rank_pos])
            results.append(row)
        return results

    def save(self, path: Path | None = None):
        """Save trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save")
        path = path or MODELS_DIR / "ltr_model.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))
        logger.info(f"LTR model saved to {path}")

    def load(self, path: Path | None = None):
        """Load trained model from disk."""
        path = path or MODELS_DIR / "ltr_model.json"
        self.model = xgb.XGBRanker()
        self.model.load_model(str(path))
        logger.info(f"LTR model loaded from {path}")

    def feature_importance(self) -> dict[str, float]:
        """Get feature importance scores."""
        if self.model is None:
            return {}
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))


def generate_training_data(
    engine,
    eval_queries: list[dict],
) -> tuple[list, list, list, list, list]:
    """Generate training data from evaluation queries with category-based labels.

    Uses category match as a proxy for relevance:
    - 3 = product category matches expected category
    - 1 = same department but different category
    - 0 = unrelated

    Args:
        engine: Initialized GrocerySearchEngine
        eval_queries: List of dicts with 'query' and 'cats' keys

    Returns:
        Tuple of (queries, candidate_indices, labels, bm25_scores, semantic_scores)
    """
    queries = []
    candidates_list = []
    labels_list = []
    bm25_list = []
    semantic_list = []

    for item in eval_queries:
        query = item["query"]
        target_cats = set(item["cats"])

        # Get BM25 and semantic scores
        bm25_scores = engine.bm25.get_scores(query.lower().split())
        query_emb = engine.embedder.embed_query(query)
        semantic_scores = np.dot(engine.embeddings, query_emb)

        # Get top candidates from both methods
        bm25_top = set(np.argsort(bm25_scores)[-100:][::-1].tolist())
        sem_top = set(np.argsort(semantic_scores)[-100:][::-1].tolist())
        all_candidates = sorted(bm25_top | sem_top)

        # Get department for target categories
        target_depts = set()
        for cat in target_cats:
            dept_rows = engine.catalog[engine.catalog["category"] == cat]
            if not dept_rows.empty:
                target_depts.update(dept_rows["department"].dropna().unique())

        # Assign relevance labels
        labels = []
        for idx in all_candidates:
            row = engine.catalog.iloc[idx]
            cat = row.get("category", "")
            dept = row.get("department", "")
            if cat in target_cats:
                labels.append(3)
            elif dept in target_depts:
                labels.append(1)
            else:
                labels.append(0)

        queries.append(query)
        candidates_list.append(all_candidates)
        labels_list.append(labels)
        bm25_list.append(bm25_scores)
        semantic_list.append(semantic_scores)

    return queries, candidates_list, labels_list, bm25_list, semantic_list
