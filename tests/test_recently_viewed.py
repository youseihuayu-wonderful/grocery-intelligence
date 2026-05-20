"""Tests for the Recently Viewed feature.

Each test uses a temp database via the pytest ``tmp_path`` fixture so
no production state is touched. The fixture mirrors the style used in
``tests/test_behavior.py``.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from src.recommend.behavior import BehaviorLogger
from src.recommend.recently_viewed import (
    get_co_viewed_in_session,
    get_recently_viewed,
    get_view_count,
    log_view,
)


# ---------------------------------------------------------------------------
# Fixture: a fresh logger backed by a tmp_path file
# ---------------------------------------------------------------------------
@pytest.fixture
def logger(tmp_path: Path) -> BehaviorLogger:
    db = tmp_path / "behavior.db"
    log = BehaviorLogger(db)
    yield log
    log.close()


# ---------------------------------------------------------------------------
# log_view + get_recently_viewed: 1 view → returns 1 product_id
# ---------------------------------------------------------------------------
def test_log_view_and_get_recently_viewed_single(logger: BehaviorLogger) -> None:
    event_id = log_view(logger, user_id=1, product_id=42)
    assert event_id > 0

    result = get_recently_viewed(logger, user_id=1)
    assert result == [42]


# ---------------------------------------------------------------------------
# get_recently_viewed dedupes repeat views
# ---------------------------------------------------------------------------
def test_get_recently_viewed_dedupes_repeats(logger: BehaviorLogger) -> None:
    base = 1_700_000_000.0
    # 3 view events for the same product at different times.
    for i in range(3):
        logger.log_event(
            product_id=42,
            event_type="view",
            user_id=1,
            timestamp=base + i,
        )

    result = get_recently_viewed(logger, user_id=1)
    assert result == [42]
    # And it is at position 0 (most-recent).
    assert result[0] == 42


# ---------------------------------------------------------------------------
# get_recently_viewed returns MOST RECENT FIRST
# ---------------------------------------------------------------------------
def test_get_recently_viewed_most_recent_first(logger: BehaviorLogger) -> None:
    base = 1_700_000_000.0
    # View 1 first, then 2, then 3.
    logger.log_event(product_id=1, event_type="view", user_id=7, timestamp=base + 10)
    logger.log_event(product_id=2, event_type="view", user_id=7, timestamp=base + 20)
    logger.log_event(product_id=3, event_type="view", user_id=7, timestamp=base + 30)

    result = get_recently_viewed(logger, user_id=7)
    # Latest (3) at index 0, oldest (1) at the end.
    assert result == [3, 2, 1]


# ---------------------------------------------------------------------------
# get_recently_viewed honors the limit parameter
# ---------------------------------------------------------------------------
def test_get_recently_viewed_respects_limit(logger: BehaviorLogger) -> None:
    base = 1_700_000_000.0
    # 30 distinct products, one view each.
    for i in range(30):
        logger.log_event(
            product_id=100 + i,
            event_type="view",
            user_id=5,
            timestamp=base + i,
        )

    result = get_recently_viewed(logger, user_id=5, limit=5)
    assert len(result) == 5
    # And they should be the 5 most recent (product 129 down to 125).
    assert result == [129, 128, 127, 126, 125]


# ---------------------------------------------------------------------------
# get_recently_viewed for a user with no events
# ---------------------------------------------------------------------------
def test_get_recently_viewed_empty_user(logger: BehaviorLogger) -> None:
    # Logger has no events at all.
    assert get_recently_viewed(logger, user_id=999) == []

    # Now add events for a *different* user; the target user is still empty.
    logger.log_event(product_id=1, event_type="view", user_id=1)
    assert get_recently_viewed(logger, user_id=999) == []


# ---------------------------------------------------------------------------
# 'click' events are included when event_types contains 'click'
# ---------------------------------------------------------------------------
def test_get_recently_viewed_includes_click(logger: BehaviorLogger) -> None:
    base = 1_700_000_000.0
    logger.log_event(product_id=1, event_type="view", user_id=3, timestamp=base + 1)
    logger.log_event(product_id=2, event_type="click", user_id=3, timestamp=base + 2)

    # Default event_types = ("view", "click") — both should be picked up.
    result = get_recently_viewed(logger, user_id=3)
    assert result == [2, 1]

    # And explicitly asking for only views excludes the click.
    result_view_only = get_recently_viewed(logger, user_id=3, event_types=("view",))
    assert result_view_only == [1]


# ---------------------------------------------------------------------------
# 'purchase' / 'add_to_cart' are EXCLUDED from default Recently Viewed
# ---------------------------------------------------------------------------
def test_get_recently_viewed_excludes_purchase_and_cart(
    logger: BehaviorLogger,
) -> None:
    base = 1_700_000_000.0
    logger.log_event(product_id=1, event_type="view", user_id=4, timestamp=base + 1)
    logger.log_event(
        product_id=2, event_type="add_to_cart", user_id=4, timestamp=base + 2
    )
    logger.log_event(
        product_id=3, event_type="purchase", user_id=4, timestamp=base + 3
    )

    result = get_recently_viewed(logger, user_id=4)
    # Only the view of product 1 counts.
    assert result == [1]
    assert 2 not in result
    assert 3 not in result


# ---------------------------------------------------------------------------
# get_view_count
# ---------------------------------------------------------------------------
def test_get_view_count_never_viewed(logger: BehaviorLogger) -> None:
    # No events at all.
    assert get_view_count(logger, user_id=1, product_id=42) == 0


def test_get_view_count_counts_views_and_clicks(logger: BehaviorLogger) -> None:
    base = 1_700_000_000.0
    # 3 views and 2 clicks → 5.
    for i in range(3):
        logger.log_event(
            product_id=42, event_type="view", user_id=1, timestamp=base + i
        )
    for i in range(2):
        logger.log_event(
            product_id=42, event_type="click", user_id=1, timestamp=base + 10 + i
        )
    # An add_to_cart for the same product must NOT be counted.
    logger.log_event(
        product_id=42, event_type="add_to_cart", user_id=1, timestamp=base + 20
    )
    # And an unrelated product must not pollute the count.
    logger.log_event(product_id=99, event_type="view", user_id=1, timestamp=base + 30)

    assert get_view_count(logger, user_id=1, product_id=42) == 5
    assert get_view_count(logger, user_id=1, product_id=99) == 1
    assert get_view_count(logger, user_id=1, product_id=12345) == 0


# ---------------------------------------------------------------------------
# get_co_viewed_in_session
# ---------------------------------------------------------------------------
def test_get_co_viewed_in_session_filters_below_min(logger: BehaviorLogger) -> None:
    now = time.time()
    # 2 products viewed within the last minute. With default
    # min_co_occurrences=2 the single co-view is filtered out.
    logger.log_event(
        product_id=1, event_type="view", user_id=1, timestamp=now - 60
    )
    logger.log_event(
        product_id=2, event_type="view", user_id=1, timestamp=now - 50
    )

    pairs = get_co_viewed_in_session(logger, user_id=1)
    # Pair (1, 2) has count=1, which is below the default
    # min_co_occurrences=2 → result is empty.
    assert pairs == []


def test_get_co_viewed_in_session_returns_pair_when_count_meets_min(
    logger: BehaviorLogger,
) -> None:
    now = time.time()
    # View the pair (1, 2) twice each → pair count = min(2, 2) = 2.
    logger.log_event(
        product_id=1, event_type="view", user_id=1, timestamp=now - 60
    )
    logger.log_event(
        product_id=2, event_type="view", user_id=1, timestamp=now - 50
    )
    logger.log_event(
        product_id=1, event_type="view", user_id=1, timestamp=now - 40
    )
    logger.log_event(
        product_id=2, event_type="view", user_id=1, timestamp=now - 30
    )

    pairs = get_co_viewed_in_session(logger, user_id=1)
    assert pairs == [(1, 2, 2)]


def test_get_co_viewed_in_session_excludes_outside_window(
    logger: BehaviorLogger,
) -> None:
    now = time.time()
    # Two old events (way outside the 60-min default window)
    # and one fresh event. With < 2 events in the window the
    # function must return [].
    logger.log_event(
        product_id=1, event_type="view", user_id=1, timestamp=now - 7200
    )  # 2h ago
    logger.log_event(
        product_id=2, event_type="view", user_id=1, timestamp=now - 7100
    )  # ~2h ago
    logger.log_event(
        product_id=3, event_type="view", user_id=1, timestamp=now - 30
    )  # fresh

    assert get_co_viewed_in_session(logger, user_id=1) == []

    # Add 4 fresh events of (3, 4) → only those count for the pair.
    logger.log_event(
        product_id=4, event_type="view", user_id=1, timestamp=now - 25
    )
    logger.log_event(
        product_id=3, event_type="view", user_id=1, timestamp=now - 20
    )
    logger.log_event(
        product_id=4, event_type="view", user_id=1, timestamp=now - 15
    )

    pairs = get_co_viewed_in_session(logger, user_id=1)
    # Inside window we have: 3×2 (counts 2) and 4×2 (counts 2),
    # so pair (3, 4) → min(2,2) = 2 ≥ 2 → returned.
    # (Pairs involving the in-window single touch of product 3 with
    # product 4 dominate; no pair with product 1 or 2 because they
    # are outside the window.)
    assert (3, 4, 2) in pairs
    # Out-of-window products 1 and 2 must not appear at all.
    for a, b, _ in pairs:
        assert a not in (1, 2)
        assert b not in (1, 2)
