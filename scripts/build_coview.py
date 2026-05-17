"""Build the Customers Also Viewed model from the behavior log.

Reads:
    data/processed/behavior.db        (1M+ seeded events via BehaviorLogger)
    data/processed/product_catalog.parquet  (for nice example printouts)

Writes:
    data/processed/coview_model.parquet

Usage:
    python3 scripts/build_coview.py
"""

from __future__ import annotations

# macOS libomp guard: harmless when no OpenMP-backed lib is imported,
# but keeps the script working if a future change pulls in xgboost/torch.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import time
from pathlib import Path

import pandas as pd

# Make ``src`` importable when running directly.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.recommend.behavior import BehaviorLogger  # noqa: E402
from src.recommend.coview import CustomersAlsoViewed  # noqa: E402


BEHAVIOR_DB = ROOT / "data" / "processed" / "behavior.db"
CATALOG_PATH = ROOT / "data" / "processed" / "product_catalog.parquet"
OUTPUT_PATH = ROOT / "data" / "processed" / "coview_model.parquet"

# Tunables -- match the FBT side so the two models feel "calibrated".
EVENT_TYPES: tuple[str, ...] = ("view", "click", "purchase")
MIN_SUPPORT = 30
TOP_K = 20


def main() -> None:
    if not BEHAVIOR_DB.exists():
        raise FileNotFoundError(BEHAVIOR_DB)
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(CATALOG_PATH)

    print(f"Reading catalog from {CATALOG_PATH}")
    catalog = pd.read_parquet(CATALOG_PATH)
    print(f"  catalog has {len(catalog):,} products")
    name_lookup = catalog.set_index("product_id")["product_name"].to_dict()

    print(f"Opening behavior log at {BEHAVIOR_DB}")
    t_start = time.time()
    with BehaviorLogger(BEHAVIOR_DB) as behavior:
        total = behavior.count_events()
        print(f"  log has {total:,} events total")
        for et in EVENT_TYPES:
            print(f"    {et}: {behavior.count_events(et):,}")

        print(
            f"Fitting CustomersAlsoViewed "
            f"(event_types={EVENT_TYPES}, min_support={MIN_SUPPORT}, top_k={TOP_K})..."
        )
        t_fit = time.time()
        coview = CustomersAlsoViewed().fit(
            behavior,
            event_types=EVENT_TYPES,
            min_support=MIN_SUPPORT,
            top_k=TOP_K,
        )
        fit_secs = time.time() - t_fit
        print(f"  fit complete in {fit_secs:.1f}s ({fit_secs/60:.2f} min)")

    # ---- Save -------------------------------------------------------------
    print(f"Saving model to {OUTPUT_PATH}")
    coview.save(OUTPUT_PATH)
    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"  saved ({size_mb:.2f} MB)")

    # ---- Stats ------------------------------------------------------------
    total_items = len(coview)
    partner_counts = [len(p) for p in coview.related_map.values()]
    avg_partners = sum(partner_counts) / max(1, len(partner_counts))
    print()
    print("Model stats:")
    print(f"  products covered:        {total_items:,}")
    print(f"  average partners/item:   {avg_partners:.2f}")
    print(f"  max partners/item:       {max(partner_counts) if partner_counts else 0}")

    # ---- Example lookups -------------------------------------------------
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
            f"Top 5 'customers also viewed' for {pid} "
            f"'{target['product_name']}' "
            f"(order_count={int(target['order_count']):,}):"
        )
        related = coview.get_related(pid, top_k=5)
        if not related:
            print("  (no related items)")
            return
        for partner_id, lift in related:
            partner_name = name_lookup.get(partner_id, f"<unknown {partner_id}>")
            print(f"  - {partner_id:>6}  lift={lift:7.2f}  {partner_name}")

    for name in ("Banana", "Strawberries", "Milk"):
        show_lookup(name)

    print()
    print(f"Build wall-clock: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
