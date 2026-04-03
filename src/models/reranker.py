"""Cross-encoder reranker for improving search result precision.

Uses the pre-trained ms-marco-MiniLM-L-6-v2 cross-encoder model.
Unlike bi-encoders, cross-encoders process query-document pairs together
for more accurate relevance scoring (at the cost of speed).
"""

from sentence_transformers import CrossEncoder
from loguru import logger

DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class ProductReranker:
    """Rerank search results using a cross-encoder model."""

    def __init__(self, model_name: str = DEFAULT_RERANKER):
        logger.info(f"Loading reranker model: {model_name}")
        self.model = CrossEncoder(model_name)
        logger.info("Reranker model loaded")

    def rerank(
        self,
        query: str,
        product_texts: list[str],
        product_ids: list[str],
        top_k: int = 10,
    ) -> list[dict]:
        """Rerank products by relevance to query.

        Args:
            query: User search query
            product_texts: Text representations of candidate products
            product_ids: Corresponding product IDs
            top_k: Number of results to return

        Returns:
            List of dicts with product_id, text, and relevance_score,
            sorted by relevance (highest first).
        """
        pairs = [[query, text] for text in product_texts]
        scores = self.model.predict(pairs)

        results = [
            {
                "product_id": pid,
                "text": text,
                "relevance_score": float(score),
            }
            for pid, text, score in zip(product_ids, product_texts, scores)
        ]

        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:top_k]
