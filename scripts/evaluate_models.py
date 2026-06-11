"""Compute and persist REAL offline evaluation metrics for the demo UI.

Everything here is measured from real artifacts — no hard-coded or illustrative
numbers:

* Two-tower recommender: Recall@K / NDCG@K vs a popularity baseline, on the
  per-user leave-last-out test split (the held-out last purchase is the ground
  truth, so no manual relevance labels are needed). Reported for both the
  grocery reorder task (repurchases allowed) and novel-item discovery
  (previously bought items masked).
* Search latency: read from the benchmark CSV produced by benchmark/bench_search.py.
* Data scale & nutrition coverage: measured straight from the catalog/behavior db.

Output: benchmark/results/model_eval.json (small, committed so the UI has real
numbers without a GPU retrain).

Usage:
  python scripts/evaluate_models.py
"""

from __future__ import annotations

import csv
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.two_tower import (  # noqa: E402
    TwoTower, evaluate, load_interactions, popularity_baseline,
)

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
LATENCY_CSV = ROOT / "benchmark" / "results" / "search_latency_rerank.csv"
# Small, reproducible results snapshot — committed (unlike bulk benchmark/results
# outputs) so the Model Performance UI has real numbers on a fresh clone.
OUT_DIR = DATA_DIR / "eval"
OUT_JSON = OUT_DIR / "model_eval.json"


def _lift_pct(model_val: float, base_val: float) -> float | None:
    return round((model_val / base_val - 1) * 100, 1) if base_val else None


def evaluate_two_tower() -> dict:
    """Reconstruct the saved two-tower model and re-measure it vs popularity."""
    state_dict = torch.load(MODELS_DIR / "two_tower.pt", map_location="cpu", weights_only=True)
    # Infer the embedding dim from the item tower's final layer.
    dim = int(state_dict["item_mlp.2.weight"].shape[0])

    inter = load_interactions()
    model = TwoTower(
        inter.n_users, inter.item_emb, dim=dim,
        user_mode="history", user_hist_emb=inter.user_hist_emb,
    )
    model.load_state_dict(state_dict)
    model.eval()

    tasks = {}
    for label, mask in [("reorder", False), ("novel", True)]:
        tt = evaluate(model, inter, ks=(10, 20), device="cpu", mask_seen=mask)
        bl = popularity_baseline(inter, ks=(10, 20), mask_seen=mask)
        tasks[label] = {
            "two_tower": {k: round(v, 4) for k, v in tt.items()},
            "popularity": {k: round(v, 4) for k, v in bl.items()},
            "lift_pct": {
                k: _lift_pct(tt[k], bl[k]) for k in tt
            },
        }

    return {
        "dim": dim,
        "n_users": int(inter.n_users),
        "n_items": int(inter.n_items),
        "n_train_interactions": int(len(inter.train)),
        "n_test_users": int(len(inter.test)),
        "tasks": tasks,
    }


def read_latency() -> dict:
    if not LATENCY_CSV.exists():
        return {}
    out = {}
    with open(LATENCY_CSV) as f:
        for row in csv.DictReader(f):
            out[row["metric"]] = float(row["value_ms"])
    return out


def data_stats() -> dict:
    catalog = pd.read_parquet(DATA_DIR / "processed" / "product_catalog.parquet")
    n_products = len(catalog)
    nutri_fields = ["calories_100g", "protein_100g", "sugar_100g", "fat_100g", "fiber_100g"]
    coverage = {
        f: round(100 * catalog[f].notna().sum() / n_products, 1)
        for f in nutri_fields if f in catalog.columns
    }
    any_nutri = catalog[nutri_fields].notna().any(axis=1).sum() if all(
        f in catalog.columns for f in nutri_fields
    ) else 0

    purchases = users = None
    behavior_db = DATA_DIR / "processed" / "behavior.db"
    if behavior_db.exists():
        con = sqlite3.connect(str(behavior_db))
        try:
            purchases = con.execute(
                "SELECT COUNT(*) FROM events WHERE event_type='purchase'"
            ).fetchone()[0]
            users = con.execute(
                "SELECT COUNT(DISTINCT user_id) FROM events WHERE event_type='purchase'"
            ).fetchone()[0]
        finally:
            con.close()

    return {
        "n_products": int(n_products),
        "n_purchases": int(purchases) if purchases is not None else None,
        "n_users": int(users) if users is not None else None,
        "nutrition_coverage_pct": coverage,
        "nutrition_any_field_pct": round(100 * int(any_nutri) / n_products, 1),
    }


def main() -> None:
    print("Evaluating two-tower recommender (this loads behavior.db)...", file=sys.stderr)
    two_tower = evaluate_two_tower()
    print("Reading search latency + data stats...", file=sys.stderr)
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "two_tower": two_tower,
        "search_latency_ms": read_latency(),
        "data": data_stats(),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)

    r = two_tower["tasks"]["reorder"]
    print(f"\nWrote {OUT_JSON.relative_to(ROOT)}")
    print(f"  Two-tower reorder Recall@10: {r['two_tower']['recall@10']} "
          f"vs popularity {r['popularity']['recall@10']} "
          f"({r['lift_pct']['recall@10']:+}%)")


if __name__ == "__main__":
    main()
