"""Seed the behavioral event log with simulated purchase events.

Why
---
Gap #7 in the project plan calls for behavioral signal tracking so the
LTR model can train on real interaction data. The MVP demo has only
one live user, so we bootstrap the event log from Instacart's prior
order history: each (user_id, product_id) purchase in the dataset
becomes one ``purchase`` event in our log.

Algorithm
---------
1. Read ``data/raw/instacart/orders.csv`` and keep ``eval_set == 'prior'``,
   yielding an ``order_id -> user_id`` mapping.
2. Read ``data/raw/instacart/order_products__prior.csv``, the 32M-row
   (order_id, product_id, reordered) table.
3. Merge to produce user-product purchase pairs.
4. Sample down to ~1M rows for the demo, stratified by user so we
   still cover thousands of users with reasonable history each.
5. Synthesize a timestamp for each row inside a 90-day recent window
   so the events look "recent" to downstream code.
6. Bulk-insert into ``data/processed/behavior.db`` in 50K-row batches.

Performance
-----------
Target wall-clock < 3 min for 1M events. The main cost is loading
the 32M-row CSV; the SQLite inserts themselves take a few seconds in
WAL mode with one transaction per batch.

Usage
-----
    python3 scripts/seed_behavioral_events.py
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Make ``src`` importable when running this script directly.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.recommend.behavior import BehaviorLogger, DEFAULT_DB_PATH  # noqa: E402


ORDERS_PATH = ROOT / "data" / "raw" / "instacart" / "orders.csv"
ORDER_PRODUCTS_PATH = (
    ROOT / "data" / "raw" / "instacart" / "order_products__prior.csv"
)
DB_PATH = ROOT / DEFAULT_DB_PATH

# Total purchase events we want in the log. 1M is enough for the demo
# without making the DB file huge.
TARGET_EVENTS = 1_000_000

# How many distinct users to cover after sampling. Stratifying by
# user keeps the per-user distribution realistic (a few heavy users,
# a long tail of light users) instead of one-event-each.
TARGET_USERS = 5_000

# 90-day "recent" window for synthetic timestamps so behavior looks
# fresh to downstream consumers.
SECONDS_IN_DAY = 24 * 3600
WINDOW_DAYS = 90

BATCH_SIZE = 50_000

# Reproducible sampling.
RNG_SEED = 42


def _synthesize_timestamps(n: int, *, rng: np.random.Generator) -> np.ndarray:
    """Return ``n`` unix-epoch seconds uniformly inside a recent window."""
    now = time.time()
    window_seconds = WINDOW_DAYS * SECONDS_IN_DAY
    # Uniform within [now - window, now]
    return now - rng.random(n) * window_seconds


def main() -> None:
    wall_t0 = time.time()

    for required in (ORDERS_PATH, ORDER_PRODUCTS_PATH):
        if not required.exists():
            raise FileNotFoundError(required)

    rng = np.random.default_rng(RNG_SEED)
    random.seed(RNG_SEED)

    # ---- 1. Orders.csv -> order_id -> user_id (prior only) ----
    print(f"Reading orders from {ORDERS_PATH}")
    t0 = time.time()
    orders = pd.read_csv(
        ORDERS_PATH,
        usecols=["order_id", "user_id", "eval_set"],
        dtype={
            "order_id": "int32",
            "user_id": "int32",
            "eval_set": "category",
        },
    )
    print(f"  loaded {len(orders):,} order rows in {time.time() - t0:.1f}s")
    prior = orders.loc[orders["eval_set"] == "prior", ["order_id", "user_id"]]
    print(f"  kept {len(prior):,} prior orders")

    # ---- 2. Order products ----
    print(f"Reading order_products from {ORDER_PRODUCTS_PATH}")
    t0 = time.time()
    order_products = pd.read_csv(
        ORDER_PRODUCTS_PATH,
        usecols=["order_id", "product_id"],
        dtype={"order_id": "int32", "product_id": "int32"},
    )
    print(
        f"  loaded {len(order_products):,} order-product rows in "
        f"{time.time() - t0:.1f}s"
    )

    # ---- 3. Merge ----
    print("Merging orders <-> order_products...")
    t0 = time.time()
    merged = order_products.merge(prior, on="order_id", how="inner")[
        ["user_id", "product_id"]
    ]
    print(
        f"  produced {len(merged):,} (user, product) rows in "
        f"{time.time() - t0:.1f}s"
    )

    # ---- 4. Sample down (stratified by user) ----
    # Strategy: pick the top TARGET_USERS users by purchase count
    # (these are the people with realistic order histories), then
    # randomly sample within them until we hit TARGET_EVENTS. This
    # guarantees broad user coverage *and* good per-user depth.
    print(
        f"Stratified-sampling down to ~{TARGET_EVENTS:,} events "
        f"across ~{TARGET_USERS:,} users..."
    )
    t0 = time.time()
    user_counts = merged["user_id"].value_counts()
    if len(user_counts) <= TARGET_USERS:
        chosen_users = user_counts.index
    else:
        # Take the TARGET_USERS most-active users so each has enough
        # history to be useful for LTR training.
        chosen_users = user_counts.head(TARGET_USERS).index
    in_chosen = merged["user_id"].isin(chosen_users)
    pool = merged.loc[in_chosen].reset_index(drop=True)
    print(
        f"  pool restricted to {len(pool):,} rows "
        f"from {len(chosen_users):,} users"
    )

    if len(pool) > TARGET_EVENTS:
        sample_idx = rng.choice(len(pool), size=TARGET_EVENTS, replace=False)
        sample = pool.iloc[sample_idx].reset_index(drop=True)
    else:
        sample = pool
    print(
        f"  sampled {len(sample):,} (user, product) rows in "
        f"{time.time() - t0:.1f}s"
    )

    # ---- 5. Synthesize timestamps ----
    print("Synthesizing timestamps in a 90-day recent window...")
    t0 = time.time()
    timestamps = _synthesize_timestamps(len(sample), rng=rng)
    sample = sample.assign(timestamp=timestamps)
    print(f"  done in {time.time() - t0:.1f}s")

    # ---- 6. Bulk insert ----
    # Start clean: remove any prior behavior.db so re-running the
    # seed script is idempotent.
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        print(f"Removing existing {DB_PATH}")
        DB_PATH.unlink()
    for sidecar_suffix in ("-journal", "-wal", "-shm"):
        sidecar = DB_PATH.with_name(DB_PATH.name + sidecar_suffix)
        if sidecar.exists():
            sidecar.unlink()

    print(f"Writing events to {DB_PATH}")
    t_insert = time.time()

    total_inserted = 0
    user_arr = sample["user_id"].to_numpy()
    product_arr = sample["product_id"].to_numpy()
    ts_arr = sample["timestamp"].to_numpy()

    with BehaviorLogger(DB_PATH) as log:
        for start in range(0, len(sample), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(sample))
            batch = [
                {
                    "product_id": int(product_arr[i]),
                    "event_type": "purchase",
                    "user_id": int(user_arr[i]),
                    "query": None,
                    "position": None,
                    "timestamp": float(ts_arr[i]),
                }
                for i in range(start, end)
            ]
            n = log.log_events_bulk(batch)
            total_inserted += n
            if (start // BATCH_SIZE) % 5 == 0 or end == len(sample):
                pct = end / len(sample) * 100
                elapsed = time.time() - t_insert
                print(
                    f"  {end:,}/{len(sample):,} ({pct:5.1f}%) "
                    f"inserted; elapsed={elapsed:.1f}s"
                )

    insert_secs = time.time() - t_insert
    print(
        f"  inserted {total_inserted:,} events in {insert_secs:.1f}s "
        f"({total_inserted / insert_secs:.0f} rows/sec)"
    )

    # ---- 7. Final stats ----
    wall_secs = time.time() - wall_t0
    size_mb = DB_PATH.stat().st_size / (1024 * 1024)
    distinct_users = int(sample["user_id"].nunique())
    distinct_products = int(sample["product_id"].nunique())

    print()
    print("=" * 60)
    print("Seed complete.")
    print("=" * 60)
    print(f"  events seeded:         {total_inserted:,}")
    print(f"  distinct users:        {distinct_users:,}")
    print(f"  distinct products:     {distinct_products:,}")
    print(f"  behavior.db size:      {size_mb:.2f} MB")
    print(
        f"  wall-clock total:      {wall_secs:.1f}s "
        f"({wall_secs/60:.2f} min)"
    )


if __name__ == "__main__":
    main()
