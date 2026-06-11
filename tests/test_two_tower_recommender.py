"""Tests for the two-tower serving layer (:mod:`src.recommend.two_tower_recommender`).

Uses tiny synthetic vectors/weights to exercise the retrieval, pooling,
exclusion, and cold-start logic without loading the multi-hundred-MB production
artifacts. (The real artifacts are validated separately via the API.)
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.recommend.two_tower_recommender import (
    ColdStartError,
    TwoTowerRecommender,
    _build_user_mlp,
)


def _make_recommender():
    """Build a recommender over 5 items with a deterministic user tower."""
    rng = np.random.default_rng(0)
    item_in, dim = 8, 4
    n_items = 5
    item_ids = np.array([10, 20, 30, 40, 50], dtype=np.int64)

    # L2-normalized item-tower vectors (inner product == cosine).
    item_vectors = rng.standard_normal((n_items, dim)).astype(np.float32)
    item_vectors /= np.linalg.norm(item_vectors, axis=1, keepdims=True)

    # Frozen 384-d-analogue product embeddings (here item_in=8), one per item.
    product_emb = rng.standard_normal((n_items, item_in)).astype(np.float32)
    product_id_to_row = {int(pid): i for i, pid in enumerate(item_ids)}

    torch.manual_seed(0)
    user_mlp = nn.Sequential(nn.Linear(item_in, 16), nn.ReLU(), nn.Linear(16, dim))
    user_mlp.eval()

    user_history = {
        1: [10, 20],   # active user
        2: [],         # cold-start (no usable purchases)
    }

    return TwoTowerRecommender(
        item_vectors=item_vectors,
        item_ids=item_ids,
        user_mlp=user_mlp,
        product_emb=product_emb,
        product_id_to_row=product_id_to_row,
        user_history=user_history,
    )


def test_recommend_returns_known_items_with_scores():
    rec = _make_recommender()
    hits = rec.recommend(1, top_k=3)
    assert 1 <= len(hits) <= 3
    for pid, score in hits:
        assert pid in {10, 20, 30, 40, 50}
        assert isinstance(score, float)
    # Scores must be sorted descending.
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)


def test_exclude_purchased_removes_history_items():
    rec = _make_recommender()
    hits = rec.recommend(1, top_k=5, exclude_purchased=True)
    returned = {pid for pid, _ in hits}
    assert returned.isdisjoint({10, 20})  # user 1's history excluded


def test_recommend_unknown_user_is_cold_start():
    rec = _make_recommender()
    with pytest.raises(ColdStartError):
        rec.recommend(99999, top_k=3)


def test_recommend_empty_history_is_cold_start():
    rec = _make_recommender()
    with pytest.raises(ColdStartError):
        rec.recommend(2, top_k=3)


def test_recommend_from_items_excludes_seed_by_default():
    rec = _make_recommender()
    hits = rec.recommend_from_items([10], top_k=5)
    assert all(pid != 10 for pid, _ in hits)


def test_recommend_from_items_unknown_products_raises():
    rec = _make_recommender()
    with pytest.raises(ColdStartError):
        rec.recommend_from_items([777, 888], top_k=3)


def test_has_user():
    rec = _make_recommender()
    assert rec.has_user(1)
    assert not rec.has_user(12345)


def test_build_user_mlp_requires_history_weights():
    # A state_dict with no user_mlp.* tensors must be rejected loudly.
    with pytest.raises(ValueError):
        _build_user_mlp({"item_mlp.0.weight": torch.zeros(4, 8)}, item_in=8, dim=4)


def test_build_user_mlp_loads_matching_arch():
    src = nn.Sequential(nn.Linear(8, 256), nn.ReLU(), nn.Linear(256, 4))
    state = {f"user_mlp.{k}": v for k, v in src.state_dict().items()}
    mlp = _build_user_mlp(state, item_in=8, dim=4)
    x = torch.randn(8)
    assert torch.allclose(mlp(x), src(x), atol=1e-6)
