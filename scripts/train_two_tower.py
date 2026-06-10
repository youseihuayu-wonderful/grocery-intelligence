"""Train the Two-Tower recommender on real purchase interactions.

Device-agnostic: uses CUDA (Kaggle) > MPS > CPU. The full run is meant for a GPU
host; --smoke runs a tiny config on CPU to validate the pipeline end-to-end.

Usage:
  python scripts/train_two_tower.py --smoke                 # quick pipeline check
  python scripts/train_two_tower.py --epochs 20 --batch 4096   # full (Kaggle GPU)

Outputs (to models/): two_tower.pt + item_vectors.npy + item_ids.npy
Reports Recall@K / NDCG@K vs the popularity baseline.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.two_tower import (  # noqa: E402
    TwoTower, evaluate, in_batch_softmax_loss, load_interactions, popularity_baseline,
)

MODELS_DIR = ROOT / "models"


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--user-mode", default="history", choices=["history", "id"],
                    help="user tower: 'history' (pool purchase embeddings) or 'id'")
    ap.add_argument("--logq", action="store_true", help="apply logQ popularity correction")
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny CPU run to test the pipeline")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.smoke:
        args.epochs, args.batch, args.device = 3, 2048, "cpu"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device or pick_device()

    print("Loading real purchase interactions...", file=sys.stderr)
    inter = load_interactions()
    print(f"  users={inter.n_users:,}  items={inter.n_items:,}  "
          f"train={len(inter.train):,}  test_users={len(inter.test):,}", file=sys.stderr)

    base = popularity_baseline(inter)
    print(f"  popularity baseline: "
          f"R@10={base['recall@10']:.4f} R@20={base['recall@20']:.4f} "
          f"NDCG@10={base['ndcg@10']:.4f}", file=sys.stderr)

    model = TwoTower(
        inter.n_users, inter.item_emb, dim=args.dim,
        user_mode=args.user_mode, user_hist_emb=inter.user_hist_emb,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # logQ correction term from train popularity (sampling prob ~ frequency).
    log_q = None
    if args.logq:
        q = inter.item_pop / inter.item_pop.sum()
        log_q_full = torch.tensor(np.log(q + 1e-12), dtype=torch.float32, device=device)

    users_t = torch.tensor(inter.train[:, 0], dtype=torch.long)
    items_t = torch.tensor(inter.train[:, 1], dtype=torch.long)
    loader = DataLoader(TensorDataset(users_t, items_t), batch_size=args.batch,
                        shuffle=True, drop_last=True)

    print(f"\nTraining on {device} ({args.epochs} epochs, batch {args.batch})...", file=sys.stderr)
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        total = 0.0
        for ub, ib in loader:
            ub, ib = ub.to(device), ib.to(device)
            uv = model.user_vec(ub)
            iv = model.item_vec(ib)
            lq = log_q_full[ib] if args.logq else None
            loss = in_batch_softmax_loss(uv, iv, log_q=lq, temperature=args.temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        avg = total / len(loader)
        m = evaluate(model, inter, device=device)
        print(f"epoch {epoch:2d}  loss={avg:.4f}  "
              f"R@10={m['recall@10']:.4f} R@20={m['recall@20']:.4f} "
              f"NDCG@10={m['ndcg@10']:.4f}  ({time.perf_counter()-t0:.1f}s)", file=sys.stderr)

    print("\n" + "=" * 60)
    print(f"Two-Tower ({args.user_mode}) vs popularity  (leave-last-out)")
    print("=" * 60)
    for label, mask in [("reorder task (no mask)", False), ("novel-item (mask seen)", True)]:
        tt = evaluate(model, inter, device=device, mask_seen=mask)
        bl = popularity_baseline(inter, mask_seen=mask)
        print(f"\n  [{label}]")
        for k in (10, 20):
            lift = (tt[f'recall@{k}'] / bl[f'recall@{k}'] - 1) * 100 if bl[f'recall@{k}'] else float('nan')
            print(f"    Recall@{k:<2d}: two-tower {tt[f'recall@{k}']:.4f}  vs  "
                  f"popularity {bl[f'recall@{k}']:.4f}   ({lift:+.0f}%)")
        print(f"    NDCG@10  : two-tower {tt['ndcg@10']:.4f}  vs  popularity {bl['ndcg@10']:.4f}")

    if not args.smoke:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), MODELS_DIR / "two_tower.pt")
        item_vecs = model.all_item_vecs().detach().cpu().numpy()
        np.save(MODELS_DIR / "item_vectors.npy", item_vecs)
        np.save(MODELS_DIR / "item_ids.npy", inter.item_ids)
        print(f"\nSaved model + {item_vecs.shape} item vectors to {MODELS_DIR}", file=sys.stderr)


if __name__ == "__main__":
    main()
