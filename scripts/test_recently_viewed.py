"""Sanity-check the Recently Viewed feature end-to-end.

Spins up a temp BehaviorLogger, logs a handful of events for a fake
user, and prints the three Recently-Viewed views (rail, view-count,
session-pairs) so a human can eyeball them.

Usage:
    python3 scripts/test_recently_viewed.py
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

# Make ``src`` importable when running directly.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.recommend.behavior import BehaviorLogger  # noqa: E402
from src.recommend.recently_viewed import (  # noqa: E402
    get_co_viewed_in_session,
    get_recently_viewed,
    get_view_count,
    log_view,
)


USER_ID = 42


def main() -> None:
    # Use a real on-disk file inside a tmp dir so the BehaviorLogger
    # can WAL-journal exactly as in production. Cleans up automatically.
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "behavior.db"
        logger = BehaviorLogger(db_path)

        try:
            _seed_events(logger)
            _print_rail(logger)
            _print_view_counts(logger)
            _print_session_pairs(logger)
        finally:
            logger.close()


def _seed_events(logger: BehaviorLogger) -> None:
    """Log 8 view events for user 42, with intentional duplicates."""
    now = time.time()

    # 8 events, mix of duplicates and unique products, in chronological
    # order with a clear "most recent" winner.
    events = [
        # 3 minutes ago: viewed product 101.
        (101, now - 180),
        # 2 minutes ago: viewed product 102.
        (102, now - 120),
        # 90 seconds ago: viewed product 101 AGAIN (duplicate).
        (101, now - 90),
        # 75 seconds ago: viewed product 103.
        (103, now - 75),
        # 60 seconds ago: viewed product 104.
        (104, now - 60),
        # 45 seconds ago: viewed product 102 again (duplicate).
        (102, now - 45),
        # 20 seconds ago: viewed product 105.
        (105, now - 20),
        # 5 seconds ago: most recent — product 103 again (duplicate).
        (103, now - 5),
    ]

    for product_id, ts in events:
        logger.log_event(
            product_id=product_id,
            event_type="view",
            user_id=USER_ID,
            timestamp=ts,
        )

    print("Seeded 8 view events for user", USER_ID)
    print("  Products touched: 101, 102, 103, 104, 105 (with duplicates)")
    print()


def _print_rail(logger: BehaviorLogger) -> None:
    print("=== get_recently_viewed(user_id=42, limit=5) ===")
    rail = get_recently_viewed(logger, user_id=USER_ID, limit=5)
    print("  Result:", rail)
    print()
    print("  Expected ordering (most-recent first, deduplicated):")
    print("    [103, 105, 102, 104, 101]")
    print("    -> 103 was last (5s ago), 105 next (20s), 102 (45s),")
    print("       104 (60s), 101 (90s, latest of its 2 visits).")
    print()
    assert rail == [103, 105, 102, 104, 101], (
        f"Unexpected rail: {rail}"
    )
    print("  OK: matches expectation.")
    print()


def _print_view_counts(logger: BehaviorLogger) -> None:
    print("=== get_view_count for two specific products ===")
    for pid in (101, 103):
        n = get_view_count(logger, user_id=USER_ID, product_id=pid)
        print(f"  product {pid}: {n} views")
    # Also show that a never-viewed product correctly reports 0.
    n_missing = get_view_count(logger, user_id=USER_ID, product_id=999)
    print(f"  product 999 (never viewed): {n_missing} views")
    print()


def _print_session_pairs(logger: BehaviorLogger) -> None:
    print("=== get_co_viewed_in_session(user_id=42) ===")
    pairs = get_co_viewed_in_session(logger, user_id=USER_ID)
    if not pairs:
        print("  (no pairs at or above min_co_occurrences=2)")
    else:
        for a, b, count in pairs:
            print(f"  ({a}, {b}) co-viewed x {count}")
    print()
    print("  Note: products 101, 102, and 103 were each viewed >= 2 times in")
    print("  the last 60 minutes, so their pairwise co-view counts are >= 2.")
    print()


if __name__ == "__main__":
    main()
