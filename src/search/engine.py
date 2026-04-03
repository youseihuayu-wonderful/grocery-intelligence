"""Hybrid search engine combining BM25 keyword search and semantic search.

Search pipeline:
1. User query → LLM query rewriting (optional)
2. BM25 keyword search → top-100 candidates
3. Semantic embedding search → top-100 candidates
4. Merge candidates (reciprocal rank fusion)
5. Cross-encoder reranking → top-K results
6. Business rule filtering (price, dietary, stock)
7. LLM explanation generation
"""

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from loguru import logger

from src.models.embeddings import ProductEmbedder, build_product_text
from src.models.reranker import ProductReranker


class GrocerySearchEngine:
    """Hybrid search engine for grocery products."""

    def __init__(
        self,
        catalog: pd.DataFrame,
        embedder: ProductEmbedder | None = None,
        reranker: ProductReranker | None = None,
    ):
        self.catalog = catalog
        self.embedder = embedder or ProductEmbedder()
        self.reranker = reranker or ProductReranker()

        # Build product text representations
        self.product_texts = [
            build_product_text(row) for _, row in catalog.iterrows()
        ]
        self.product_ids = catalog["product_id"].astype(str).tolist()

        # Initialize BM25 index
        tokenized = [text.lower().split() for text in self.product_texts]
        self.bm25 = BM25Okapi(tokenized)

        # Build or load embeddings
        self.embeddings = self.embedder.build_product_embeddings(
            self.product_texts, self.product_ids, save=True
        )

        logger.info(
            f"Search engine initialized with {len(self.catalog)} products"
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        bm25_candidates: int = 100,
        semantic_candidates: int = 100,
        use_reranker: bool = True,
    ) -> list[dict]:
        """Execute hybrid search pipeline.

        Args:
            query: Natural language search query
            top_k: Number of final results
            bm25_candidates: BM25 pre-filter count
            semantic_candidates: Semantic pre-filter count
            use_reranker: Whether to apply cross-encoder reranking

        Returns:
            List of product dicts with scores and metadata.
        """
        # Step 1: BM25 keyword search
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_top_idx = np.argsort(bm25_scores)[-bm25_candidates:][::-1]

        # Step 2: Semantic search
        query_embedding = self.embedder.embed_query(query)
        similarities = np.dot(self.embeddings, query_embedding)
        semantic_top_idx = np.argsort(similarities)[-semantic_candidates:][::-1]

        # Step 3: Reciprocal Rank Fusion
        candidate_scores = {}
        k = 60  # RRF constant

        for rank, idx in enumerate(bm25_top_idx):
            candidate_scores[idx] = candidate_scores.get(idx, 0) + 1 / (k + rank + 1)

        for rank, idx in enumerate(semantic_top_idx):
            candidate_scores[idx] = candidate_scores.get(idx, 0) + 1 / (k + rank + 1)

        # Sort by fused score
        fused_candidates = sorted(
            candidate_scores.keys(),
            key=lambda x: candidate_scores[x],
            reverse=True,
        )

        # Take top candidates for reranking
        rerank_count = min(50, len(fused_candidates))
        top_candidates = fused_candidates[:rerank_count]

        # Step 4: Cross-encoder reranking
        if use_reranker and top_candidates:
            candidate_texts = [self.product_texts[i] for i in top_candidates]
            candidate_ids = [self.product_ids[i] for i in top_candidates]

            reranked = self.reranker.rerank(
                query, candidate_texts, candidate_ids, top_k=top_k
            )

            # Enrich with full product data
            results = []
            for item in reranked:
                pid = int(item["product_id"])
                product_row = self.catalog[
                    self.catalog["product_id"] == pid
                ].iloc[0]
                results.append({
                    **product_row.to_dict(),
                    "relevance_score": item["relevance_score"],
                })
            return results

        # Fallback: return by fused score without reranking
        results = []
        for idx in top_candidates[:top_k]:
            product_row = self.catalog.iloc[idx]
            results.append({
                **product_row.to_dict(),
                "relevance_score": float(candidate_scores[idx]),
            })
        return results

    def search_by_category(
        self, category: str, top_k: int = 20
    ) -> list[dict]:
        """Get top products in a specific category."""
        filtered = self.catalog[
            self.catalog["category"].str.contains(category, case=False, na=False)
        ]
        return filtered.head(top_k).to_dict("records")
