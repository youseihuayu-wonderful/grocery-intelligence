"""Build per-user personalization profiles from Instacart prior orders.

Reads:
    data/raw/instacart/orders.csv
    data/raw/instacart/order_products__prior.csv
    data/processed/product_catalog.parquet

Writes:
    data/processed/user_profiles.parquet

Usage:
    python3 scripts/build_user_profiles.py
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import pandas as pd

# Make ``src`` importable when running this script directly.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.recommend.personalization import (  # noqa: E402
    UserPersonalizationStore,
    compute_user_profiles,
)

ORDERS_PATH = ROOT / "data" / "raw" / "instacart" / "orders.csv"
ORDER_PRODUCTS_PATH = ROOT / "data" / "raw" / "instacart" / "order_products__prior.csv"
CATALOG_PATH = ROOT / "data" / "processed" / "product_catalog.parquet"
OUTPUT_PATH = ROOT / "data" / "processed" / "user_profiles.parquet"


def main() -> None:
    for required in (ORDERS_PATH, ORDER_PRODUCTS_PATH, CATALOG_PATH):
        if not required.exists():
            raise FileNotFoundError(required)

    wall_t0 = time.time()

    # ---- Load catalog ------------------------------------------------
    print(f"Reading catalog from {CATALOG_PATH}")
    catalog = pd.read_parquet(CATALOG_PATH)
    print(f"  catalog has {len(catalog):,} products")

    # ---- Load orders -------------------------------------------------
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
    print(
        f"  loaded {len(orders):,} order rows in {time.time() - t0:.1f}s"
    )
    # Restrict to the prior split upfront -- train/test rows would
    # otherwise add noise to the join. compute_user_profiles also does
    # this, but filtering early shrinks the merge.
    prior_orders = orders[orders["eval_set"] == "prior"][
        ["order_id", "user_id"]
    ].copy()
    print(f"  {len(prior_orders):,} prior orders kept")

    # ---- Load order products -----------------------------------------
    print(f"Reading order_products from {ORDER_PRODUCTS_PATH}")
    t0 = time.time()
    order_products = pd.read_csv(
        ORDER_PRODUCTS_PATH,
        usecols=["order_id", "product_id", "reordered"],
        dtype={
            "order_id": "int32",
            "product_id": "int32",
            "reordered": "int8",
        },
    )
    print(
        f"  loaded {len(order_products):,} order-product rows in "
        f"{time.time() - t0:.1f}s"
    )

    # ---- Build profiles ----------------------------------------------
    print("Computing user profiles (min_orders=5)...")
    t_fit = time.time()
    profiles = compute_user_profiles(
        prior_orders, order_products, catalog, min_orders=5
    )
    fit_secs = time.time() - t_fit
    print(
        f"  built {len(profiles):,} profiles in {fit_secs:.1f}s "
        f"({fit_secs/60:.2f} min)"
    )

    # ---- Save --------------------------------------------------------
    print(f"Saving profiles to {OUTPUT_PATH}")
    store = UserPersonalizationStore(profiles)
    t_save = time.time()
    store.save(OUTPUT_PATH)
    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"  saved in {time.time() - t_save:.1f}s ({size_mb:.2f} MB)")

    wall_secs = time.time() - wall_t0
    print(
        f"Total wall-clock: {wall_secs:.1f}s ({wall_secs/60:.2f} min)"
    )

    # ---- Stats -------------------------------------------------------
    order_counts = [p.total_orders for p in profiles.values()]
    basket_sizes = [p.avg_basket_size for p in profiles.values()]

    print()
    print("Profile stats:")
    print(f"  total users:               {len(profiles):,}")
    print(
        f"  median total_orders/user:  {statistics.median(order_counts):.1f}"
    )
    print(
        f"  mean total_orders/user:    {statistics.mean(order_counts):.2f}"
    )
    print(
        f"  median avg_basket_size:    {statistics.median(basket_sizes):.2f}"
    )
    print(
        f"  mean avg_basket_size:      {statistics.mean(basket_sizes):.2f}"
    )

    # ---- Top-20 most-active users -----------------------------------
    by_orders = sorted(
        profiles.values(), key=lambda p: p.total_orders, reverse=True
    )
    print()
    print("Top 20 most-active users:")
    for rank, p in enumerate(by_orders[:20], start=1):
        top_dept = max(p.favorite_departments, key=p.favorite_departments.get) \
            if p.favorite_departments else "?"
        print(
            f"  #{rank:>2}  user_id={p.user_id:>6}  "
            f"orders={p.total_orders:>3}  basket={p.avg_basket_size:5.2f}  "
            f"top_dept={top_dept}"
        )

    # ---- Example profile --------------------------------------------
    sample_uid = 1
    if sample_uid in profiles:
        sample = profiles[sample_uid]
        name_lookup = catalog.set_index("product_id")["product_name"].to_dict()
        print()
        print(f"Sample profile for user_id={sample_uid}:")
        print(f"  total_orders:        {sample.total_orders}")
        print(f"  avg_basket_size:     {sample.avg_basket_size:.2f}")
        print(f"  favorite_categories: {sample.favorite_categories}")
        print(f"  favorite_departments: {sample.favorite_departments}")
        print(f"  favorite_brands:     {sample.favorite_brands}")
        print("  top 5 favorite products:")
        for pid in sample.favorite_products[:5]:
            name = name_lookup.get(pid, f"<unknown {pid}>")
            print(f"    - {pid:>6}  {name}")


if __name__ == "__main__":
    main()
