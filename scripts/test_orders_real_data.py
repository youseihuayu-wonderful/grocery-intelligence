"""Real-data sanity check for ``src.shopping.orders.OrderHistoryStore``.

Loads the full Instacart CSVs and exercises the public API on a couple
of real users:

1. ``user_id=1`` — the very first user in the dataset.
2. ``user_id=206105`` — a known high-volume user from
   ``data/processed/user_profiles.parquet``.

Run with::

    cd /Users/shihuayu/grocery-intelligence
    source venv/bin/activate
    python -m scripts.test_orders_real_data

or::

    python scripts/test_orders_real_data.py
"""

from __future__ import annotations

import os
import resource
import sys
import time
from pathlib import Path

# Make ``src.*`` importable when run as a plain script.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.shopping.orders import OrderHistoryStore  # noqa: E402


ORDERS_CSV = ROOT / "data" / "raw" / "instacart" / "orders.csv"
ORDER_PRODUCTS_CSV = ROOT / "data" / "raw" / "instacart" / "order_products__prior.csv"
CATALOG_PARQUET = ROOT / "data" / "processed" / "product_catalog.parquet"


def _rss_mb() -> float:
    """Resident set size of this process in MB (macOS reports bytes)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS returns bytes; Linux returns kilobytes.
    if sys.platform == "darwin":
        return rss / (1024 * 1024)
    return rss / 1024


def main() -> int:
    print(f"[start] RSS = {_rss_mb():.0f} MB")
    print(f"[start] loading {ORDERS_CSV}")
    print(f"[start] loading {ORDER_PRODUCTS_CSV}")
    t0 = time.time()
    store = OrderHistoryStore(
        orders_csv=ORDERS_CSV,
        order_products_csv=ORDER_PRODUCTS_CSV,
        user_cap=None,  # load everyone
        verbose=True,
    )
    load_dt = time.time() - t0
    print(
        f"[loaded] in {load_dt:.1f}s — "
        f"{store.n_users:,} users / {store.n_orders:,} orders / "
        f"{store.n_items_total:,} items"
    )
    print(f"[loaded] RSS = {_rss_mb():.0f} MB")

    print(f"[catalog] loading {CATALOG_PARQUET}")
    catalog = pd.read_parquet(
        CATALOG_PARQUET, columns=["product_id", "product_name"]
    )
    pid_to_name = dict(
        zip(
            catalog["product_id"].astype(int).tolist(),
            catalog["product_name"].astype(str).tolist(),
        )
    )

    # ------------------------------------------------------------------
    # 1) user_id=1
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("user_id=1")
    print("=" * 70)
    orders_1 = store.get_user_orders(1)
    print(f"order count: {len(orders_1)}")
    if orders_1:
        most_recent = orders_1[0]
        print(
            f"most recent order_id={most_recent['order_id']} "
            f"order_number={most_recent['order_number']} "
            f"dow={most_recent['order_dow']} "
            f"hour={most_recent['order_hour_of_day']} "
            f"days_since_prior={most_recent['days_since_prior_order']} "
            f"n_items={most_recent['n_items']}"
        )
        print("items:")
        for item in most_recent["items"]:
            pid = item["product_id"]
            name = pid_to_name.get(pid, "(unknown)")
            reordered = "R" if item["reordered"] else " "
            print(f"  [{reordered}] {pid:>6}  {name}")

    # ------------------------------------------------------------------
    # 2) user_id=206105 — buy_again_top(10)
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("user_id=206105 — buy_again_top(10)")
    print("=" * 70)
    top = store.buy_again_top(user_id=206105, top_k=10)
    if not top:
        print("(no orders for this user)")
    else:
        # Total order count for context.
        n_orders = len(store.get_user_orders(206105))
        print(f"order count: {n_orders}")
        print("buy_again_top(10):")
        for rank, pid in enumerate(top, start=1):
            name = pid_to_name.get(pid, "(unknown)")
            print(f"  {rank:>2}. {pid:>6}  {name}")

    print()
    print(f"[end] RSS = {_rss_mb():.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
