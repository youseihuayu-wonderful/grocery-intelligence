"""Smart substitute recommendation engine.

When a product is out of stock, find the best alternatives based on:
- Semantic similarity (embedding distance)
- Category match (same aisle/department)
- Nutritional similarity
- Price proximity
- Dietary compatibility
"""

import numpy as np
import pandas as pd
from loguru import logger

from src.models.embeddings import ProductEmbedder, build_product_text


class SubstituteRecommender:
    """Find substitute products when items are out of stock."""

    def __init__(
        self,
        catalog: pd.DataFrame,
        embedder: ProductEmbedder | None = None,
        embeddings: np.ndarray | None = None,
    ):
        self.catalog = catalog
        self.embedder = embedder or ProductEmbedder()

        # Build product texts and embeddings
        self.product_texts = [
            build_product_text(row) for _, row in catalog.iterrows()
        ]

        if embeddings is not None:
            self.embeddings = embeddings
        else:
            self.embeddings = self.embedder.embed_texts(self.product_texts)

        logger.info(f"SubstituteRecommender initialized with {len(catalog)} products")

    def find_substitutes(
        self,
        product_id: int,
        top_k: int = 5,
        same_category: bool = True,
        max_price_ratio: float = 1.5,
    ) -> list[dict]:
        """Find substitute products for a given product.

        Args:
            product_id: ID of the out-of-stock product
            top_k: Number of substitutes to return
            same_category: Whether to restrict to same category
            max_price_ratio: Max price as ratio of original (1.5 = 50% more expensive)

        Returns:
            List of substitute product dicts with similarity scores.
        """
        # Find the original product
        product_mask = self.catalog["product_id"] == product_id
        if not product_mask.any():
            logger.warning(f"Product {product_id} not found in catalog")
            return []

        product_idx = self.catalog[product_mask].index[0]
        original = self.catalog.iloc[product_idx]
        original_embedding = self.embeddings[product_idx]

        # Compute similarity to all products
        similarities = np.dot(self.embeddings, original_embedding)

        # Build candidate mask
        candidate_mask = np.ones(len(self.catalog), dtype=bool)
        candidate_mask[product_idx] = False  # Exclude self

        # Filter: same category if requested
        if same_category and pd.notna(original.get("category")):
            category_match = self.catalog["category"] == original["category"]
            candidate_mask &= category_match.values

        # Filter: price within range
        if "price" in self.catalog.columns and pd.notna(original.get("price")):
            max_price = original["price"] * max_price_ratio
            price_ok = (self.catalog["price"] <= max_price) | self.catalog["price"].isna()
            candidate_mask &= price_ok.values

        # Filter: only in-stock items
        if "stock_status" in self.catalog.columns:
            in_stock = self.catalog["stock_status"] == "in_stock"
            candidate_mask &= in_stock.values

        # Apply mask and get top-K
        masked_similarities = similarities.copy()
        masked_similarities[~candidate_mask] = -1

        top_indices = np.argsort(masked_similarities)[-top_k:][::-1]

        # Build results
        results = []
        for idx in top_indices:
            if masked_similarities[idx] <= 0:
                break
            sub = self.catalog.iloc[idx].to_dict()
            sub["similarity_score"] = float(masked_similarities[idx])
            sub["substitution_reasons"] = self._explain_substitution(
                original.to_dict(), sub
            )
            results.append(sub)

        logger.info(
            f"Found {len(results)} substitutes for product {product_id} "
            f"({original['product_name']})"
        )
        return results

    def find_healthier_alternatives(
        self, product_id: int, top_k: int = 5
    ) -> list[dict]:
        """Find healthier substitutes (lower sugar, higher protein, fewer calories)."""
        product_mask = self.catalog["product_id"] == product_id
        if not product_mask.any():
            return []

        original = self.catalog[product_mask].iloc[0]

        # Get same-category products
        if pd.notna(original.get("category")):
            candidates = self.catalog[
                (self.catalog["category"] == original["category"])
                & (self.catalog["product_id"] != product_id)
            ].copy()
        else:
            candidates = self.catalog[
                self.catalog["product_id"] != product_id
            ].copy()

        # Score by health improvement
        candidates["health_score"] = 0.0

        if "sugar_100g" in candidates.columns:
            orig_sugar = original.get("sugar_100g", 0) or 0
            candidates["health_score"] += (
                orig_sugar - candidates["sugar_100g"].fillna(orig_sugar)
            ).clip(lower=0) * 2

        if "protein_100g" in candidates.columns:
            orig_protein = original.get("protein_100g", 0) or 0
            candidates["health_score"] += (
                candidates["protein_100g"].fillna(orig_protein) - orig_protein
            ).clip(lower=0)

        if "calories_100g" in candidates.columns:
            orig_cal = original.get("calories_100g", 0) or 0
            candidates["health_score"] += (
                orig_cal - candidates["calories_100g"].fillna(orig_cal)
            ).clip(lower=0) * 0.5

        top = candidates.nlargest(top_k, "health_score")
        return top.to_dict("records")

    def find_cheaper_alternatives(
        self, product_id: int, top_k: int = 5
    ) -> list[dict]:
        """Find cheaper substitutes in the same category.

        Falls back to most popular same-category products when price data
        is unavailable (e.g. Instacart dataset has no prices).
        """
        product_mask = self.catalog["product_id"] == product_id
        if not product_mask.any():
            return []

        original = self.catalog[product_mask].iloc[0]

        has_price = "price" in self.catalog.columns and pd.notna(original.get("price"))

        # Same-category candidates (excluding original)
        candidates = self.catalog[
            (self.catalog["category"] == original.get("category"))
            & (self.catalog["product_id"] != product_id)
        ].copy()

        if has_price:
            candidates = candidates[candidates["price"] < original["price"]]
            candidates = candidates.sort_values("price", ascending=True)
        elif "order_count" in candidates.columns:
            # No price data — return most popular same-category products instead
            logger.info("No price data; returning popular same-category alternatives")
            candidates = candidates.sort_values("order_count", ascending=False)
        else:
            return []

        results = candidates.head(top_k).to_dict("records")
        for r in results:
            r["substitution_reasons"] = self._explain_substitution(
                original.to_dict(), r
            )
            if not has_price:
                r["substitution_reasons"].append("Popular in category (no price data)")
        return results

    @staticmethod
    def _explain_substitution(original: dict, substitute: dict) -> list[str]:
        """Generate human-readable reasons why this is a good substitute."""
        reasons = []

        if original.get("category") == substitute.get("category"):
            reasons.append(f"Same category: {original['category']}")

        if original.get("brand") == substitute.get("brand"):
            reasons.append(f"Same brand: {original['brand']}")

        if substitute.get("price") and original.get("price"):
            if substitute["price"] < original["price"]:
                savings = original["price"] - substitute["price"]
                reasons.append(f"${savings:.2f} cheaper")

        if substitute.get("sugar_100g") and original.get("sugar_100g"):
            if substitute["sugar_100g"] < original["sugar_100g"]:
                reasons.append("Lower sugar")

        if substitute.get("protein_100g") and original.get("protein_100g"):
            if substitute["protein_100g"] > original["protein_100g"]:
                reasons.append("Higher protein")

        return reasons
