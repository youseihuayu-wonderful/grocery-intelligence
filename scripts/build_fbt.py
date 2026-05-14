"""Build the Frequently Bought Together model from Instacart prior orders.

Reads:
    data/raw/instacart/order_products__prior.csv
    data/processed/product_catalog.parquet  (for the popularity filter)

Writes:
    data/processed/fbt_model.parquet

Usage:
    python3 scripts/build_fbt.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

# Make ``src`` importable when running directly.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.recommend.fbt import FrequentlyBoughtTogether  # noqa: E402


ORDER_PRODUCTS_PATH = ROOT / "data" / "raw" / "instacart" / "order_products__prior.csv"
CATALOG_PATH = ROOT / "data" / "processed" / "product_catalog.parquet"
OUTPUT_PATH = ROOT / "data" / "processed" / "fbt_model.parquet"

# Tunable knobs. Filtering by popularity is essential to keep RAM use
# bounded — Instacart has 49K products and a full 49K x 49K dense matrix
# would be 2.5B cells (way too big for memory). With popularity >= 100,
# we keep ~10K products which gives a much more tractable pair space.
MIN_PRODUCT_ORDER_COUNT = 100  # only keep items ordered at least this many times
MIN_SUPPORT = 50  # min co-occurrences for a pair to be considered
TOP_K = 20  # partners kept per product


def main() -> None:
    if not ORDER_PRODUCTS_PATH.exists():
        raise FileNotFoundError(ORDER_PRODUCTS_PATH)
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(CATALOG_PATH)

    print(f"Reading catalog from {CATALOG_PATH}")
    catalog = pd.read_parquet(CATALOG_PATH)
    print(f"  catalog has {len(catalog):,} products")

    popular = catalog[catalog["order_count"] >= MIN_PRODUCT_ORDER_COUNT]
    candidate_ids = set(popular["product_id"].astype(int).tolist())
    print(
        f"  {len(candidate_ids):,} products meet popularity filter "
        f"(order_count >= {MIN_PRODUCT_ORDER_COUNT})"
    )

    print(f"Reading order_products from {ORDER_PRODUCTS_PATH}")
    t0 = time.time()
    order_products = pd.read_csv(
        ORDER_PRODUCTS_PATH,
        usecols=["order_id", "product_id"],
        dtype={"order_id": "int32", "product_id": "int32"},
    )
    print(
        f"  loaded {len(order_products):,} order-product rows "
        f"in {time.time() - t0:.1f}s"
    )

    # Restrict to popular items up front to slash the work.
    before = len(order_products)
    order_products = order_products[
        order_products["product_id"].isin(candidate_ids)
    ]
    print(
        f"  filtered to {len(order_products):,} rows "
        f"(dropped {before - len(order_products):,} unpopular item rows)"
    )

    print(
        f"Fitting FBT model (min_support={MIN_SUPPORT}, top_k={TOP_K})..."
    )
    t_fit = time.time()
    fbt = FrequentlyBoughtTogether().fit(
        order_products,
        min_support=MIN_SUPPORT,
        top_k=TOP_K,
        # candidate_ids is already pre-applied above, but pass it anyway
        # so .fit() can short-circuit on filter checks.
        candidate_ids=candidate_ids,
        progress_every=200_000,
    )
    fit_secs = time.time() - t_fit
    print(f"  fit complete in {fit_secs:.1f}s ({fit_secs/60:.2f} min)")

    # ---- Save ---------------------------------------------------------
    print(f"Saving model to {OUTPUT_PATH}")
    fbt.save(OUTPUT_PATH)
    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"  saved ({size_mb:.2f} MB)")

    # ---- Stats --------------------------------------------------------
    total_items = len(fbt)
    partner_counts = [len(p) for p in fbt.related_map.values()]
    avg_partners = sum(partner_counts) / max(1, len(partner_counts))
    print()
    print("Model stats:")
    print(f"  products covered:        {total_items:,}")
    print(f"  average partners/item:   {avg_partners:.2f}")
    print(f"  max partners/item:       {max(partner_counts) if partner_counts else 0}")

    # ---- Example lookups ---------------------------------------------
    name_lookup = catalog.set_index("product_id")["product_name"].to_dict()

    def show_lookup(product_name_substr: str) -> None:
        match = catalog[
            catalog["product_name"].str.contains(
                product_name_substr, case=False, na=False
            )
        ].sort_values("order_count", ascending=False)
        if match.empty:
            print(f"  [no product matched '{product_name_substr}']")
            return
        target = match.iloc[0]
        pid = int(target["product_id"])
        print()
        print(
            f"Top 5 related for {pid} '{target['product_name']}' "
            f"(order_count={int(target['order_count']):,}):"
        )
        related = fbt.get_related(pid, top_k=5)
        if not related:
            print("  (no related items)")
            return
        for partner_id, lift in related:
            partner_name = name_lookup.get(partner_id, f"<unknown {partner_id}>")
            print(f"  - {partner_id:>6}  lift={lift:7.2f}  {partner_name}")

    for name in ("Banana", "Strawberries", "Milk"):
        show_lookup(name)


if __name__ == "__main__":
    main()
