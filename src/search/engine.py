"""Hybrid search engine combining BM25 keyword search and semantic search.

Search pipeline:
1. User query -> BM25 keyword search -> top-N candidates
2. User query -> Semantic embedding search -> top-N candidates
3. Merge candidates (reciprocal rank fusion)
4. Cross-encoder reranking -> top-K results
5. Return enriched product results
"""

import time

import numpy as np
import pandas as pd
from pathlib import Path
from rank_bm25 import BM25Okapi
from loguru import logger

from src.models.embeddings import ProductEmbedder, build_product_text

DATA_DIR = Path(__file__).parent.parent.parent / "data"


class GrocerySearchEngine:
    """Hybrid search engine for grocery products."""

    def __init__(
        self,
        catalog: pd.DataFrame,
        embeddings: np.ndarray | None = None,
        embedder: ProductEmbedder | None = None,
        reranker=None,
    ):
        self.catalog = catalog.reset_index(drop=True)
        self.embedder = embedder or ProductEmbedder()
        self.reranker = reranker  # Lazy-loaded on first use if None

        # Per-stage latency (ms) of the most recent search() call, for profiling.
        self.last_timings: dict[str, float] = {}

        # Build product text representations
        self.product_texts = [
            build_product_text(row) for _, row in self.catalog.iterrows()
        ]
        self.product_ids = self.catalog["product_id"].astype(str).tolist()

        # Initialize BM25 index
        tokenized = [text.lower().split() for text in self.product_texts]
        self.bm25 = BM25Okapi(tokenized)

        # Load pre-computed embeddings or build them
        if embeddings is not None:
            self.embeddings = embeddings
        else:
            emb_path = DATA_DIR / "embeddings" / "product_embeddings.npy"
            if emb_path.exists():
                self.embeddings = np.load(emb_path)
                logger.info(f"Loaded pre-computed embeddings: {self.embeddings.shape}")
            else:
                logger.info("No pre-computed embeddings found, building...")
                self.embeddings = self.embedder.embed_texts(self.product_texts)

        logger.info(
            f"Search engine initialized with {len(self.catalog)} products"
        )

    def _get_reranker(self):
        """Lazy-load the cross-encoder reranker on first use."""
        if self.reranker is None:
            from src.models.reranker import ProductReranker
            self.reranker = ProductReranker()
        return self.reranker

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
        timings: dict[str, float] = {}

        # Step 1: BM25 keyword search
        _t = time.perf_counter()
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_top_idx = np.argsort(bm25_scores)[-bm25_candidates:][::-1]
        timings["bm25"] = (time.perf_counter() - _t) * 1000

        # Step 2: Semantic search
        _t = time.perf_counter()
        query_embedding = self.embedder.embed_query(query)
        similarities = np.dot(self.embeddings, query_embedding)
        semantic_top_idx = np.argsort(similarities)[-semantic_candidates:][::-1]
        timings["semantic"] = (time.perf_counter() - _t) * 1000

        # Step 3: Reciprocal Rank Fusion
        _t = time.perf_counter()
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
        timings["fusion"] = (time.perf_counter() - _t) * 1000

        # Step 4: Cross-encoder reranking (top 50 candidates)
        rerank_count = min(50, len(fused_candidates))
        top_candidates = fused_candidates[:rerank_count]

        if use_reranker and top_candidates:
            try:
                _t = time.perf_counter()
                reranker = self._get_reranker()
                candidate_texts = [self.product_texts[i] for i in top_candidates]
                candidate_ids = [self.product_ids[i] for i in top_candidates]

                reranked = reranker.rerank(
                    query, candidate_texts, candidate_ids, top_k=top_k
                )
                timings["rerank"] = (time.perf_counter() - _t) * 1000

                # Enrich with full product data
                _t = time.perf_counter()
                results = []
                for item in reranked:
                    pid = int(item["product_id"])
                    match = self.catalog[self.catalog["product_id"] == pid]
                    if not match.empty:
                        row = match.iloc[0].to_dict()
                        row["relevance_score"] = item["relevance_score"]
                        results.append(row)
                timings["enrich"] = (time.perf_counter() - _t) * 1000
                self.last_timings = timings
                return results
            except Exception as e:
                logger.warning(f"Reranker failed, falling back to RRF: {e}")

        # Fallback: return by fused score without reranking
        _t = time.perf_counter()
        results = []
        for idx in top_candidates[:top_k]:
            row = self.catalog.iloc[idx].to_dict()
            row["relevance_score"] = float(candidate_scores[idx])
            results.append(row)
        timings["enrich"] = (time.perf_counter() - _t) * 1000
        self.last_timings = timings
        return results

    def search_by_category(
        self, category: str, top_k: int = 20
    ) -> list[dict]:
        """Get top products in a specific category."""
        filtered = self.catalog[
            self.catalog["category"].str.contains(category, case=False, na=False)
        ]
        if "order_count" in filtered.columns:
            filtered = filtered.sort_values("order_count", ascending=False)
        return filtered.head(top_k).to_dict("records")

    def get_categories(self) -> list[str]:
        """Return all unique product categories."""
        return sorted(self.catalog["category"].dropna().unique().tolist())
