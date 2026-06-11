"""Smoke test for the pricing modules.

Run from the repo root::

    source venv/bin/activate
    python scripts/test_pricing.py

It loads the real catalog, builds the full price_map, prints a few
sampled prices across departments, generates a price-history sample,
and evaluates a sample cart against the promotion engine.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Make ``src`` importable when invoked as ``python scripts/test_pricing.py``.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from src.pricing.history import detect_price_drop, generate_price_history  # noqa: E402
from src.pricing.synthetic_prices import attach_prices, build_price_map  # noqa: E402
from src.pricing.promotions import cart_pricing_summary, evaluate_promotions  # noqa: E402

CATALOG_PATH = REPO_ROOT / "data" / "processed" / "product_catalog.parquet"


def _hr(title: str) -> None:
    """Print a tidy section divider."""
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def step_1_build_prices() -> tuple[pd.DataFrame, dict[int, float], float]:
    """Load catalog and build the full price map; return elapsed seconds."""
    _hr("Step 1 — Load catalog + build price_map for all products")
    print(f"Loading {CATALOG_PATH}...")
    catalog = pd.read_parquet(CATALOG_PATH)
    print(f"Loaded {len(catalog):,} products")

    t0 = time.perf_counter()
    price_map = build_price_map(catalog)
    elapsed = time.perf_counter() - t0

    print(f"build_price_map: {len(price_map):,} prices in {elapsed:.3f}s "
          f"({elapsed * 1000 / max(1, len(price_map)):.4f} ms/row)")
    return catalog, price_map, elapsed


def step_2_sample_prices(catalog: pd.DataFrame, price_map: dict[int, float]) -> None:
    """Print 10 representative products from a mix of departments."""
    _hr("Step 2 — Sample prices across departments")

    # Pick one product per department, for breadth, then trim to ~10.
    target_departments = [
        "produce", "dairy eggs", "meat seafood", "bakery", "snacks",
        "beverages", "frozen", "pantry", "household", "alcohol",
    ]
    rows: list[dict] = []
    for dept in target_departments:
        match = catalog[catalog["department"].str.lower() == dept].head(1)
        for _, r in match.iterrows():
            rows.append(r.to_dict())

    products = [
        {
            "product_id": int(r["product_id"]),
            "product_name": r.get("product_name"),
            "department": r.get("department"),
            "nutrition_grade": r.get("nutrition_grade"),
            "order_count": r.get("order_count"),
        }
        for r in rows
    ]
    attach_prices(products, price_map)

    header = f"{'Department':<18} {'Grade':<6} {'Orders':>10} {'Price':>8}  Name"
    print(header)
    print("-" * 72)
    for p in products:
        print(
            f"{str(p.get('department') or ''):<18} "
            f"{str(p.get('nutrition_grade') or ''):<6} "
            f"{int(p.get('order_count') or 0):>10,} "
            f"${p.get('price', 0.0):>7.2f}  "
            f"{(p.get('product_name') or '')[:48]}"
        )


def step_3_history(catalog: pd.DataFrame, price_map: dict[int, float]) -> None:
    """Generate a 90-day history for one product and print summary stats."""
    _hr("Step 3 — 90-day price history for a sample product")

    # Pick a product we can clearly see in output: a popular dairy-eggs item.
    dairy = catalog[catalog["department"].str.lower() == "dairy eggs"]
    if dairy.empty:
        sample = catalog.iloc[0]
    else:
        # Pick the most-ordered dairy item.
        sample = dairy.sort_values("order_count", ascending=False).iloc[0]

    pid = int(sample["product_id"])
    current = float(price_map[pid])

    history = generate_price_history(product_id=pid, current_price=current)
    prices = [row["price"] for row in history]

    print(f"Product: {sample['product_name']} (id={pid})")
    print(f"Department: {sample['department']}, current price: ${current:.2f}")
    print(f"History length: {len(history)} days "
          f"({history[0]['date']} → {history[-1]['date']})")
    print(f"  min price : ${min(prices):.2f}")
    print(f"  max price : ${max(prices):.2f}")
    print(f"  mean      : ${sum(prices) / len(prices):.2f}")
    print(f"  last day  : ${history[-1]['price']:.2f}  (matches current = "
          f"{abs(history[-1]['price'] - current) < 1e-6})")

    drop = detect_price_drop(history, lookback_days=30)
    if drop is None:
        print("  detect_price_drop: insufficient history")
    else:
        flag = " (SIGNIFICANT)" if drop["is_significant"] else ""
        print(
            f"  drop vs 30-day median: {drop['drop_pct']:+.2f}% "
            f"(current ${drop['current']:.2f} vs median "
            f"${drop['previous_median']:.2f}){flag}"
        )


def step_4_promotions(catalog: pd.DataFrame, price_map: dict[int, float]) -> None:
    """Build a sample cart, run the promotion engine, print results."""
    _hr("Step 4 — Promotions on a sample cart")

    # Pick 5 items across produce / dairy / snacks for realistic mix.
    picks: list[dict] = []
    for dept in ("produce", "produce", "produce", "dairy eggs", "snacks"):
        choices = catalog[catalog["department"].str.lower() == dept]
        if choices.empty:
            continue
        row = choices.iloc[len(picks) % max(1, len(choices))]
        picks.append(
            {
                "product_id": int(row["product_id"]),
                "product_name": row.get("product_name"),
                "department": row.get("department"),
                "category": row.get("department"),
                "qty": 1,
            }
        )
    attach_prices(picks, price_map)

    print(f"Cart ({len(picks)} items):")
    for p in picks:
        print(
            f"  [{p.get('department'):<14}] "
            f"${p.get('price', 0.0):>6.2f} x{p['qty']}  "
            f"{(p.get('product_name') or '')[:46]}"
        )

    summary = cart_pricing_summary(picks)
    print(f"\nSubtotal       : ${summary['subtotal']:.2f}")
    print(f"Total discount : ${summary['total_discount']:.2f}")
    print(f"Final total    : ${summary['total']:.2f}")
    print(f"Item count     : {summary['n_items']}")

    print("\nApplied promotions:")
    if not summary["promotions_applied"]:
        print("  (none)")
    for p in summary["promotions_applied"]:
        print(
            f"  [APPLIED ] {p.code:<20} -${p.discount_amount:>5.2f}  "
            f"{p.title} — {p.description}"
        )

    print("\nIncentive promotions (not yet qualified):")
    for p in summary["promotions_available"]:
        bar_n = int(round(p.progress * 20))
        bar = "#" * bar_n + "." * (20 - bar_n)
        print(
            f"  [{bar}] {p.progress * 100:>5.1f}%  "
            f"{p.code:<20} {p.title} — {p.description}"
        )


def main() -> int:
    catalog, price_map, elapsed = step_1_build_prices()
    step_2_sample_prices(catalog, price_map)
    step_3_history(catalog, price_map)
    step_4_promotions(catalog, price_map)

    _hr("Done")
    print(f"build_price_map elapsed: {elapsed:.3f}s for "
          f"{len(price_map):,} products")
    return 0


if __name__ == "__main__":
    sys.exit(main())
