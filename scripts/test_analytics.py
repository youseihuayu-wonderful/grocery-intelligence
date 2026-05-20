"""Sanity-check the analytics dashboard against the real behavior log.

Opens ``data/processed/behavior.db`` if it exists (the seeded 1M-event
log), otherwise falls back to a fresh empty in-memory logger so the
script doesn't crash. Runs every public function in
``src.analytics.dashboard`` once and prints the result in a human-
readable shape so we can eyeball numbers before wiring a UI.

Usage:
    cd /Users/shihuayu/grocery-intelligence && source venv/bin/activate
    python scripts/test_analytics.py
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

# Make sure ``src.*`` imports resolve when run from the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analytics.dashboard import (  # noqa: E402
    category_breakdown,
    daily_event_counts,
    funnel_metrics,
    hot_products,
    search_quality_signals,
    top_queries,
    user_activity_summary,
)
from src.recommend.behavior import BehaviorLogger  # noqa: E402


DEFAULT_LOG_PATH = REPO_ROOT / "data" / "processed" / "behavior.db"
CATALOG_PATH = REPO_ROOT / "data" / "processed" / "product_catalog.parquet"


def _hr(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def _timed(label: str, fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    dt = time.perf_counter() - t0
    print(f"[{label}: {dt*1000:.0f} ms]")
    return out


def main() -> int:
    if DEFAULT_LOG_PATH.exists():
        print(f"Opening behavior log: {DEFAULT_LOG_PATH}")
        logger = BehaviorLogger(DEFAULT_LOG_PATH)
        n_events = logger.count_events()
        print(f"Total events in log: {n_events:,}")
    else:
        print(f"!! No log at {DEFAULT_LOG_PATH}, falling back to in-memory empty log.")
        logger = BehaviorLogger(":memory:")
        n_events = 0

    catalog: pd.DataFrame | None = None
    if CATALOG_PATH.exists():
        catalog = pd.read_parquet(CATALOG_PATH)
        print(f"Catalog loaded: {len(catalog):,} products")

    _hr("top_queries(limit=10)")
    rows = _timed("top_queries", top_queries, logger, limit=10)
    if not rows:
        print("(no query events in log)")
    for r in rows:
        ap = (
            f"{r.avg_position_clicked:.2f}"
            if r.avg_position_clicked is not None else "n/a"
        )
        print(
            f"  {r.query!r:30s}  count={r.count:>8,}  "
            f"users={r.distinct_users:>6,}  avg_pos_clicked={ap}"
        )

    _hr("funnel_metrics()")
    funnel = _timed("funnel_metrics", funnel_metrics, logger)
    print(json.dumps(asdict(funnel), indent=2))

    _hr("hot_products(limit=10)")
    hp = _timed("hot_products", hot_products, logger, event_type="purchase", limit=10)
    if not hp:
        print("(no purchase events in log)")
    for pid, n in hp:
        print(f"  product_id={pid:<10} purchases={n:,}")

    _hr("daily_event_counts(days_back=7)")
    daily = _timed("daily_event_counts", daily_event_counts, logger, days_back=7)
    for r in daily:
        print(
            f"  {r['date']}  views={r['n_views']:>6,}  "
            f"clicks={r['n_clicks']:>6,}  cart={r['n_add_to_cart']:>5,}  "
            f"purchases={r['n_purchases']:>5,}"
        )

    if catalog is not None:
        _hr("category_breakdown(event_type='purchase')")
        cats = _timed(
            "category_breakdown",
            category_breakdown,
            logger, catalog, event_type="purchase",
        )
        for r in cats[:15]:
            print(
                f"  {r['department']:<20s} count={r['count']:>8,}  "
                f"share={r['share']*100:5.1f}%"
            )

    _hr("search_quality_signals()")
    sigs = _timed("search_quality_signals", search_quality_signals, logger)
    print(json.dumps(sigs, indent=2))

    _hr("user_activity_summary(user_id=27)")
    summary = _timed("user_activity_summary", user_activity_summary, logger, user_id=27)
    # ``by_type`` -> JSON, timestamps stay as floats for readability.
    print(json.dumps(summary, indent=2))

    logger.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
