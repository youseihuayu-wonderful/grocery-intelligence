"""Serving layer for the Two-Tower recommender (candidate generation).

The two-tower model is trained offline (``scripts/train_two_tower.py``) and
ships three artifacts in ``models/``:

    two_tower.pt      full model state_dict (we only need the user tower here)
    item_vectors.npy  (n_items, d) L2-normalized item-tower vectors
    item_ids.npy      (n_items,) product_id for each item_vectors row

At serving time we never recompute item vectors — they are precomputed and
queried with FAISS (``ANNIndex``), so retrieval is sublinear. The user vector
is computed ONLINE from the user's purchase history:

    user history (product_ids) -> mean of their 384-d MiniLM embeddings
                               -> user_mlp -> L2-normalize -> d-vector

This mirrors the trained ``user_mode='history'`` tower exactly, and because it
is content-based it generalizes to users not seen at train time (no user-ID
cold-start) and to ad-hoc baskets. No mock data — every purchase is real
(behavior.db: 1M purchases).

Validated offline (leave-last-out, reorder task): Recall@10 0.125 vs popularity
0.080 (+56%), NDCG@10 +62%.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from src.search.ann_index import ANNIndex

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"


class ColdStartError(Exception):
    """Raised when a user has no purchase history we can pool into a vector."""


def _build_user_mlp(state_dict: dict, item_in: int, dim: int) -> nn.Sequential:
    """Reconstruct the trained history-mode user tower from a state_dict.

    Matches ``TwoTower`` (user_mode='history'): Linear(item_in, 256) -> ReLU
    -> Linear(256, dim). Only the ``user_mlp.*`` tensors are needed for serving.
    """
    mlp = nn.Sequential(
        nn.Linear(item_in, 256), nn.ReLU(), nn.Linear(256, dim),
    )
    user_mlp_state = {
        k[len("user_mlp."):]: v
        for k, v in state_dict.items()
        if k.startswith("user_mlp.")
    }
    if not user_mlp_state:
        raise ValueError(
            "state_dict has no 'user_mlp.*' tensors — was the model trained "
            "with user_mode='history'?"
        )
    mlp.load_state_dict(user_mlp_state)
    mlp.eval()
    return mlp


class TwoTowerRecommender:
    """Personalized candidate generation via the trained two-tower model."""

    def __init__(
        self,
        item_vectors: np.ndarray,
        item_ids: np.ndarray,
        user_mlp: nn.Sequential,
        product_emb: np.ndarray,
        product_id_to_row: dict[int, int],
        user_history: dict[int, list[int]],
    ):
        self.dim = int(item_vectors.shape[1])
        self.user_mlp = user_mlp
        self.product_emb = product_emb              # (n_products, 384) float32
        self.product_id_to_row = product_id_to_row  # product_id -> row in product_emb
        self.user_history = user_history            # user_id -> [product_id, ...]
        # Exact inner-product index over the 64-d item-tower vectors. Vectors are
        # already L2-normalized at train time, so inner product == cosine.
        self.index = ANNIndex.build(
            item_vectors, [int(i) for i in item_ids], index_type="flat"
        )
        self._known_items = set(self.index.product_ids)
        logger.info(
            f"TwoTowerRecommender ready: {len(self._known_items):,} items, "
            f"{len(self.user_history):,} users with history, dim={self.dim}"
        )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    @classmethod
    def load(
        cls,
        models_dir: Path | None = None,
        data_dir: Path | None = None,
        behavior_db: Path | None = None,
    ) -> "TwoTowerRecommender":
        """Load artifacts from disk and preload user purchase history.

        Raises FileNotFoundError if the two-tower artifacts are missing, so the
        API can degrade gracefully (fall back to popularity feeds).
        """
        models_dir = models_dir or MODELS_DIR
        data_dir = data_dir or DATA_DIR
        behavior_db = behavior_db or data_dir / "processed" / "behavior.db"

        item_vectors = np.load(models_dir / "item_vectors.npy").astype(np.float32)
        item_ids = np.load(models_dir / "item_ids.npy")
        product_ids = np.load(data_dir / "embeddings" / "product_ids.npy")
        product_emb = np.load(
            data_dir / "embeddings" / "product_embeddings.npy"
        ).astype(np.float32)

        state_dict = torch.load(
            models_dir / "two_tower.pt", map_location="cpu", weights_only=True
        )
        user_mlp = _build_user_mlp(
            state_dict, item_in=product_emb.shape[1], dim=item_vectors.shape[1]
        )

        # product_ids.npy is a string array; behavior.db ids are ints.
        product_id_to_row = {int(p): i for i, p in enumerate(product_ids)}

        user_history = _load_user_history(behavior_db)

        return cls(
            item_vectors=item_vectors,
            item_ids=item_ids,
            user_mlp=user_mlp,
            product_emb=product_emb,
            product_id_to_row=product_id_to_row,
            user_history=user_history,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def _pool_user_vector(self, product_ids: list[int]) -> np.ndarray:
        """Pool a list of product_ids into a single d-dim user-tower vector.

        Mean of the products' frozen 384-d embeddings -> user_mlp -> normalize.
        Returns None-equivalent by raising ColdStartError if none of the
        product_ids have an embedding.
        """
        rows = [
            self.product_id_to_row[pid]
            for pid in product_ids
            if pid in self.product_id_to_row
        ]
        if not rows:
            raise ColdStartError("no purchased products have embeddings")
        hist = self.product_emb[rows].mean(axis=0)  # (384,)
        with torch.no_grad():
            vec = self.user_mlp(torch.from_numpy(hist).float())
            vec = F.normalize(vec, dim=-1)
        return vec.numpy().astype(np.float32)

    def recommend(
        self, user_id: int, top_k: int = 20, exclude_purchased: bool = False,
    ) -> list[tuple[int, float]]:
        """Top-k (product_id, score) recommendations for a known user.

        exclude_purchased=False (default) is the validated grocery reorder task:
        69% of held-out next-purchases are repurchases, so keeping previously
        bought items in the pool is correct. Set True for novel discovery.
        """
        history = self.user_history.get(int(user_id))
        if not history:
            raise ColdStartError(f"user {user_id} has no purchase history")
        return self._recommend_from_history(
            history, top_k=top_k,
            exclude=set(history) if exclude_purchased else None,
        )

    def recommend_from_items(
        self, product_ids: list[int], top_k: int = 20,
        exclude_seed: bool = True,
    ) -> list[tuple[int, float]]:
        """Top-k recommendations for an ad-hoc basket of product_ids.

        Powers basket / "complete your cart" style recs and works for users
        with no stored history (true content-based cold-start handling).
        """
        return self._recommend_from_history(
            product_ids, top_k=top_k,
            exclude=set(product_ids) if exclude_seed else None,
        )

    def _recommend_from_history(
        self, product_ids: list[int], top_k: int, exclude: set[int] | None,
    ) -> list[tuple[int, float]]:
        user_vec = self._pool_user_vector(product_ids)
        # Over-fetch so post-filtering still leaves top_k results.
        fetch_k = top_k + (len(exclude) if exclude else 0)
        hits = self.index.search(user_vec, top_k=min(fetch_k, len(self._known_items)))
        if exclude:
            hits = [(pid, s) for pid, s in hits if pid not in exclude]
        return hits[:top_k]

    def has_user(self, user_id: int) -> bool:
        return int(user_id) in self.user_history


def _load_user_history(behavior_db: Path) -> dict[int, list[int]]:
    """Load each user's purchased product_ids from behavior.db (once, at startup)."""
    if not Path(behavior_db).exists():
        logger.warning(f"behavior.db not found at {behavior_db}; no user history")
        return {}
    con = sqlite3.connect(str(behavior_db))
    try:
        rows = con.execute(
            "SELECT user_id, product_id FROM events "
            "WHERE event_type='purchase' ORDER BY timestamp"
        ).fetchall()
    finally:
        con.close()
    history: dict[int, list[int]] = {}
    for user_id, product_id in rows:
        history.setdefault(int(user_id), []).append(int(product_id))
    return history
