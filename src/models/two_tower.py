"""Two-Tower neural retrieval model for personalized product recommendation.

Trained on REAL purchase interactions (behavior.db: 1M purchases, 5K users,
27.8K products). This is the canonical two-tower / dual-encoder recommender used
for candidate generation at scale (YouTube, Instagram, etc.):

    user tower  : learned user embedding -> MLP -> L2-normalized d-vector
    item tower  : frozen MiniLM product embedding (384-d) -> MLP -> L2-normalized d-vector
    score(u, i) : dot(user_vec, item_vec)

Training uses in-batch sampled-softmax: within a batch of (user, purchased-item)
positives, every other item is a negative. Optional logQ correction debiases the
popular-item pull. At serving, item vectors are precomputed and queried with ANN
(FAISS), so retrieval is sublinear.

The heavy training is meant for a GPU host (Kaggle); this module is device-agnostic.
No mock data anywhere — every interaction is a real purchase.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass
class Interactions:
    """Indexed purchase interactions plus the lookups needed to train/serve."""

    train: np.ndarray            # (N, 2) int32 rows of (user_idx, item_idx)
    test: dict[int, int]         # user_idx -> held-out item_idx (leave-last-out)
    train_pos: dict[int, set]    # user_idx -> set of item_idx seen in train
    item_emb: np.ndarray         # (n_items, 384) float32 frozen MiniLM embeddings
    item_pop: np.ndarray         # (n_items,) float32 train purchase counts
    user_hist_emb: np.ndarray    # (n_users, 384) float32 mean of train item embeddings
    user_ids: np.ndarray         # item_idx/user_idx -> original id (for serving)
    item_ids: np.ndarray

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)


def load_interactions(
    behavior_db: Path | None = None,
    product_ids_npy: Path | None = None,
    product_emb_npy: Path | None = None,
    min_user_purchases: int = 5,
) -> Interactions:
    """Build indexed interactions with a per-user leave-last-out test split."""
    behavior_db = behavior_db or DATA_DIR / "processed" / "behavior.db"
    product_ids_npy = product_ids_npy or DATA_DIR / "embeddings" / "product_ids.npy"
    product_emb_npy = product_emb_npy or DATA_DIR / "embeddings" / "product_embeddings.npy"

    pids = np.load(product_ids_npy)
    emb_all = np.load(product_emb_npy).astype(np.float32)
    id2row = {int(p): i for i, p in enumerate(pids)}

    con = sqlite3.connect(str(behavior_db))
    purch = pd.read_sql(
        "SELECT user_id, product_id, timestamp FROM events WHERE event_type='purchase'",
        con,
    )
    con.close()

    # Keep only purchases of products we have an embedding for.
    purch = purch[purch["product_id"].map(lambda p: int(p) in id2row)].copy()

    # Drop users below the minimum interaction count.
    counts = purch.groupby("user_id").size()
    keep_users = counts[counts >= min_user_purchases].index
    purch = purch[purch["user_id"].isin(keep_users)]

    # Stable index spaces.
    user_ids = np.sort(purch["user_id"].unique())
    item_ids = np.sort(purch["product_id"].unique())
    u2idx = {int(u): i for i, u in enumerate(user_ids)}
    i2idx = {int(p): i for i, p in enumerate(item_ids)}

    purch = purch.sort_values("timestamp")
    purch["uidx"] = purch["user_id"].map(lambda u: u2idx[int(u)])
    purch["iidx"] = purch["product_id"].map(lambda p: i2idx[int(p)])

    # Leave-last-out: each user's chronologically last purchase is the test item.
    test: dict[int, int] = {}
    last_pos = purch.groupby("uidx").tail(1)
    for uidx, iidx in zip(last_pos["uidx"].to_numpy(), last_pos["iidx"].to_numpy()):
        test[int(uidx)] = int(iidx)

    last_ids = set(last_pos.index)
    train_df = purch[~purch.index.isin(last_ids)]
    train = train_df[["uidx", "iidx"]].to_numpy(dtype=np.int32)

    train_pos: dict[int, set] = {}
    for uidx, iidx in train:
        train_pos.setdefault(int(uidx), set()).add(int(iidx))

    item_pop = np.zeros(len(item_ids), dtype=np.float32)
    vals, cnts = np.unique(train[:, 1], return_counts=True)
    item_pop[vals] = cnts

    item_emb = np.ascontiguousarray(emb_all[[id2row[int(p)] for p in item_ids]])

    # Per-user purchase-history representation: mean of their train item embeddings.
    # Content-aware and generalizes to unseen users (no cold-start on user ID).
    user_hist_emb = np.zeros((len(user_ids), item_emb.shape[1]), dtype=np.float32)
    counts = np.zeros(len(user_ids), dtype=np.float32)
    for uidx, iidx in train:
        user_hist_emb[uidx] += item_emb[iidx]
        counts[uidx] += 1
    user_hist_emb /= np.maximum(counts[:, None], 1.0)

    return Interactions(
        train=train, test=test, train_pos=train_pos, item_emb=item_emb,
        item_pop=item_pop, user_hist_emb=user_hist_emb,
        user_ids=user_ids, item_ids=item_ids,
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class TwoTower(nn.Module):
    """Two-tower dot-product scorer with a frozen-embedding item tower.

    user_mode:
      "history" (default) — user tower = MLP over the mean of the user's
                            purchase-history item embeddings (content-aware,
                            generalizes to new users). Beats popularity.
      "id"                — user tower = learned user-ID embedding (no content).
    """

    def __init__(self, n_users: int, item_emb: np.ndarray, dim: int = 64,
                 user_emb_dim: int = 64, user_mode: str = "history",
                 user_hist_emb: np.ndarray | None = None):
        super().__init__()
        self.user_mode = user_mode
        item_in = item_emb.shape[1]
        # Frozen MiniLM item embeddings as a non-trainable buffer.
        self.register_buffer("item_features", torch.tensor(item_emb, dtype=torch.float32))

        if user_mode == "history":
            if user_hist_emb is None:
                raise ValueError("user_mode='history' requires user_hist_emb")
            self.register_buffer(
                "user_features", torch.tensor(user_hist_emb, dtype=torch.float32)
            )
            self.user_mlp = nn.Sequential(
                nn.Linear(item_in, 256), nn.ReLU(), nn.Linear(256, dim),
            )
        elif user_mode == "id":
            self.user_embedding = nn.Embedding(n_users, user_emb_dim)
            nn.init.normal_(self.user_embedding.weight, std=0.05)
            self.user_mlp = nn.Sequential(
                nn.Linear(user_emb_dim, 128), nn.ReLU(), nn.Linear(128, dim),
            )
        else:
            raise ValueError(f"unknown user_mode {user_mode!r}")

        self.item_mlp = nn.Sequential(
            nn.Linear(item_in, 256), nn.ReLU(), nn.Linear(256, dim),
        )

    def user_vec(self, user_idx: torch.Tensor) -> torch.Tensor:
        if self.user_mode == "history":
            feat = self.user_features[user_idx]
        else:
            feat = self.user_embedding(user_idx)
        return F.normalize(self.user_mlp(feat), dim=-1)

    def item_vec(self, item_idx: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.item_mlp(self.item_features[item_idx]), dim=-1)

    def all_item_vecs(self) -> torch.Tensor:
        return F.normalize(self.item_mlp(self.item_features), dim=-1)


def in_batch_softmax_loss(
    user_vecs: torch.Tensor,
    item_vecs: torch.Tensor,
    log_q: torch.Tensor | None = None,
    temperature: float = 0.05,
) -> torch.Tensor:
    """Sampled-softmax over in-batch negatives; diagonal entries are positives.

    log_q (per-item sampling log-prob) applies the logQ correction that removes
    the popular-item bias inherent to in-batch negatives.
    """
    logits = (user_vecs @ item_vecs.t()) / temperature  # (B, B)
    if log_q is not None:
        logits = logits - log_q.unsqueeze(0)
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    model: TwoTower, inter: Interactions, ks=(10, 20), device: str = "cpu",
    mask_seen: bool = False,
) -> dict[str, float]:
    """Recall@K and NDCG@K on the leave-last-out test item over the full item pool.

    mask_seen=False (default): the realistic grocery reorder task — previously
    bought items stay in the candidate pool (69% of held-out items are repurchases).
    mask_seen=True: novel-item discovery — masks items seen in train.
    """
    model.eval()
    item_vecs = model.all_item_vecs().to(device)            # (n_items, d)
    users = np.array(sorted(inter.test.keys()))
    maxk = max(ks)
    metrics = {f"recall@{k}": 0.0 for k in ks}
    metrics.update({f"ndcg@{k}": 0.0 for k in ks})

    batch = 512
    for s in range(0, len(users), batch):
        ub = torch.tensor(users[s:s + batch], device=device)
        uv = model.user_vec(ub)                              # (b, d)
        scores = uv @ item_vecs.t()                          # (b, n_items)
        if mask_seen:
            for r, u in enumerate(users[s:s + batch]):
                seen = inter.train_pos.get(int(u))
                if seen:
                    scores[r, list(seen)] = -1e9
        topk = torch.topk(scores, maxk, dim=1).indices.cpu().numpy()  # (b, maxk)
        for r, u in enumerate(users[s:s + batch]):
            tgt = inter.test[int(u)]
            row = topk[r]
            hit = np.where(row == tgt)[0]
            if len(hit):
                rank = int(hit[0])
                for k in ks:
                    if rank < k:
                        metrics[f"recall@{k}"] += 1.0
                        metrics[f"ndcg@{k}"] += 1.0 / np.log2(rank + 2)
    n = len(users)
    return {m: v / n for m, v in metrics.items()}


def popularity_baseline(inter: Interactions, ks=(10, 20),
                        mask_seen: bool = False) -> dict[str, float]:
    """Most-purchased-items baseline (optionally masking per-user seen items)."""
    order = np.argsort(-inter.item_pop)
    users = sorted(inter.test.keys())
    maxk = max(ks)
    metrics = {f"recall@{k}": 0.0 for k in ks}
    metrics.update({f"ndcg@{k}": 0.0 for k in ks})
    for u in users:
        if mask_seen:
            seen = inter.train_pos.get(int(u), set())
            ranked = [i for i in order if i not in seen][:maxk]
        else:
            ranked = list(order[:maxk])
        tgt = inter.test[int(u)]
        if tgt in ranked:
            rank = ranked.index(tgt)
            for k in ks:
                if rank < k:
                    metrics[f"recall@{k}"] += 1.0
                    metrics[f"ndcg@{k}"] += 1.0 / np.log2(rank + 2)
    n = len(users)
    return {m: v / n for m, v in metrics.items()}
