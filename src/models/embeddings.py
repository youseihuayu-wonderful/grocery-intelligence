"""Product and query embedding using sentence-transformers.

Uses the pre-trained all-MiniLM-L6-v2 model (no training required).
This encoder transformer converts text into 384-dimensional vectors
for semantic similarity search.
"""

from pathlib import Path

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

# Pre-trained model — no fine-tuning needed for MVP
DEFAULT_MODEL = "all-MiniLM-L6-v2"
EMBEDDINGS_DIR = Path(__file__).parent.parent.parent / "data" / "embeddings"


class ProductEmbedder:
    """Generate and manage embeddings for grocery products."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")

    def embed_texts(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Encode a list of texts into embedding vectors."""
        logger.info(f"Embedding {len(texts)} texts (batch_size={batch_size})")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Encode a single search query."""
        return self.model.encode(
            query,
            normalize_embeddings=True,
        )

    def build_product_embeddings(
        self,
        product_texts: list[str],
        product_ids: list[str],
        save: bool = True,
    ) -> np.ndarray:
        """Build embeddings for the entire product catalog.

        Args:
            product_texts: Combined text representation of each product
                (e.g., "Chobani Greek Yogurt | dairy | yogurt | gluten-free low-fat")
            product_ids: Corresponding product IDs
            save: Whether to save embeddings to disk
        """
        embeddings = self.embed_texts(product_texts)

        if save:
            EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
            np.save(EMBEDDINGS_DIR / "product_embeddings.npy", embeddings)
            np.save(EMBEDDINGS_DIR / "product_ids.npy", np.array(product_ids))
            logger.info(f"Saved {len(embeddings)} embeddings to {EMBEDDINGS_DIR}")

        return embeddings

    def load_product_embeddings(self) -> tuple[np.ndarray, np.ndarray]:
        """Load pre-computed product embeddings from disk."""
        embeddings = np.load(EMBEDDINGS_DIR / "product_embeddings.npy")
        product_ids = np.load(EMBEDDINGS_DIR / "product_ids.npy")
        logger.info(f"Loaded {len(embeddings)} product embeddings")
        return embeddings, product_ids


def build_product_text(row: dict) -> str:
    """Create a rich text representation of a product for embedding.

    Combines name, category, brand, and dietary info into a single string
    that captures the product's semantic meaning.
    """
    parts = [str(row.get("product_name", ""))]

    if row.get("category"):
        parts.append(str(row["category"]))
    if row.get("department"):
        parts.append(str(row["department"]))
    if row.get("brand"):
        parts.append(str(row["brand"]))
    if row.get("ingredients"):
        # Truncate long ingredient lists
        ingredients = str(row["ingredients"])[:200]
        parts.append(ingredients)

    return " | ".join(parts)
