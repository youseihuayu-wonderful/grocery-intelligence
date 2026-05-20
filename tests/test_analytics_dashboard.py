"""Tests for the search analytics dashboard backend.

Each test seeds a tiny ``BehaviorLogger`` (backed by ``tmp_path``) with
hand-crafted events covering every code path -- queries with and without
clicks, multiple event types, an empty log, a fake catalog for the
department breakdown, etc. We assert on counts AND on the shape of the
returned data structures so the API layer can rely on a stable contract.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest

from src.analytics.dashboard import (
    FunnelMetrics,
    TopQuery,
    category_breakdown,
    daily_event_counts,
    funnel_metrics,
    hot_products,
    search_quality_signals,
    top_queries,
    user_activity_summary,
)
from src.recommend.behavior import BehaviorLogger


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def seeded_logger(tmp_path: Path) -> BehaviorLogger:
    """A tmp logger seeded with the canonical fixture event set.

    Layout:
      * 5 view events with query="yogurt"
      * 3 view events with query="milk"
      * 2 click events with query="yogurt", positions 1 and 3
      * 1 add_to_cart event (no query, no position)
      * 1 purchase event (no query, no position)

    All events are anchored at a fixed UTC timestamp so the
    ``daily_event_counts`` window doesn't accidentally include them
    (the window covers the last N days; the fixture is in the past).
    """
    db_path = tmp_path / "behavior.db"
    log = BehaviorLogger(db_path)

    base = 1_700_000_000.0  # ~2023-11-14 UTC; well outside any 30-day window

    # 5 yogurt views, distinct users so distinct_users for yogurt > 1.
    for i in range(5):
        log.log_event(
            product_id=100 + i,
            event_type="view",
            user_id=10 + i,
            query="yogurt",
            timestamp=base + i,
        )
    # 3 milk views, all same user.
    for i in range(3):
        log.log_event(
            product_id=200 + i,
            event_type="view",
            user_id=20,
            query="milk",
            timestamp=base + 10 + i,
        )
    # 2 yogurt clicks, positions 1 and 3.
    log.log_event(
        product_id=100,
        event_type="click",
        user_id=10,
        query="yogurt",
        position=1,
        timestamp=base + 20,
    )
    log.log_event(
        product_id=101,
        event_type="click",
        user_id=11,
        query="yogurt",
        position=3,
        timestamp=base + 21,
    )
    # 1 add_to_cart, 1 purchase (no query, no position).
    log.log_event(
        product_id=100,
        event_type="add_to_cart",
        user_id=10,
        timestamp=base + 30,
    )
    log.log_event(
        product_id=100,
        event_type="purchase",
        user_id=10,
        timestamp=base + 31,
    )

    yield log
    log.close()


@pytest.fixture
def empty_logger(tmp_path: Path) -> BehaviorLogger:
    """A logger with no events at all -- for the safe-defaults tests."""
    db_path = tmp_path / "empty.db"
    log = BehaviorLogger(db_path)
    yield log
    log.close()


# ---------------------------------------------------------------------------
# top_queries
# ---------------------------------------------------------------------------
def test_top_queries_counts_and_order(seeded_logger: BehaviorLogger) -> None:
    rows = top_queries(seeded_logger, limit=10)

    by_query = {r.query: r for r in rows}
    assert set(by_query) == {"yogurt", "milk"}

    # yogurt = 5 views + 2 clicks = 7 events
    assert by_query["yogurt"].count == 7
    # milk = 3 views
    assert by_query["milk"].count == 3

    # Ordered descending by count.
    assert [r.query for r in rows] == ["yogurt", "milk"]

    # avg_position_clicked for yogurt = mean(1, 3) = 2.0
    assert by_query["yogurt"].avg_position_clicked == pytest.approx(2.0)
    # milk has no clicks at all -> None
    assert by_query["milk"].avg_position_clicked is None


def test_top_queries_distinct_users(seeded_logger: BehaviorLogger) -> None:
    rows = top_queries(seeded_logger, limit=10)
    by_query = {r.query: r for r in rows}

    # yogurt: views from users 10..14 + clicks from 10, 11 -> distinct = 5
    assert by_query["yogurt"].distinct_users == 5
    # milk: all 3 views from user 20 -> distinct = 1
    assert by_query["milk"].distinct_users == 1


def test_top_queries_empty_log(empty_logger: BehaviorLogger) -> None:
    assert top_queries(empty_logger, limit=10) == []


# ---------------------------------------------------------------------------
# funnel_metrics
# ---------------------------------------------------------------------------
def test_funnel_metrics_counts(seeded_logger: BehaviorLogger) -> None:
    m = funnel_metrics(seeded_logger)
    assert m.n_views == 8       # 5 yogurt + 3 milk views
    assert m.n_clicks == 2
    assert m.n_add_to_cart == 1
    assert m.n_purchases == 1


def test_funnel_metrics_rates(seeded_logger: BehaviorLogger) -> None:
    m = funnel_metrics(seeded_logger)
    assert m.view_to_click_rate == pytest.approx(2 / 8)
    assert m.click_to_cart_rate == pytest.approx(1 / 2)
    assert m.cart_to_purchase_rate == pytest.approx(1 / 1)
    assert m.overall_conversion == pytest.approx(1 / 8)


def test_funnel_metrics_empty_log_zero_safe(empty_logger: BehaviorLogger) -> None:
    m = funnel_metrics(empty_logger)
    # Counts are all zero, but every rate must be 0.0 (NOT NaN, NOT a crash).
    assert m == FunnelMetrics(
        n_views=0,
        n_clicks=0,
        n_add_to_cart=0,
        n_purchases=0,
        view_to_click_rate=0.0,
        click_to_cart_rate=0.0,
        cart_to_purchase_rate=0.0,
        overall_conversion=0.0,
    )


# ---------------------------------------------------------------------------
# hot_products
# ---------------------------------------------------------------------------
def test_hot_products_purchase(seeded_logger: BehaviorLogger) -> None:
    # The seed has exactly one purchase for product 100.
    rows = hot_products(seeded_logger, event_type="purchase", limit=10)
    assert rows == [(100, 1)]


def test_hot_products_view_sorted_desc(seeded_logger: BehaviorLogger) -> None:
    rows = hot_products(seeded_logger, event_type="view", limit=10)
    counts = [c for _, c in rows]
    # Sorted desc -- each step must be >= the next.
    assert counts == sorted(counts, reverse=True)
    # 5 yogurt + 3 milk = 8 view events spread over 8 distinct product_ids.
    assert sum(counts) == 8


def test_hot_products_empty_log(empty_logger: BehaviorLogger) -> None:
    assert hot_products(empty_logger, event_type="purchase", limit=10) == []


# ---------------------------------------------------------------------------
# daily_event_counts
# ---------------------------------------------------------------------------
def test_daily_event_counts_full_window(seeded_logger: BehaviorLogger) -> None:
    rows = daily_event_counts(seeded_logger, days_back=30)
    assert len(rows) == 30

    # Oldest-first ordering.
    dates = [r["date"] for r in rows]
    assert dates == sorted(dates)

    # Each entry has the expected shape and zero counts (fixture is from
    # 2023; the last 30 days definitely don't include it).
    for r in rows:
        assert set(r) == {
            "date", "n_views", "n_clicks", "n_add_to_cart", "n_purchases",
        }
        assert r["n_views"] == 0
        assert r["n_clicks"] == 0
        assert r["n_add_to_cart"] == 0
        assert r["n_purchases"] == 0


def test_daily_event_counts_today_present(seeded_logger: BehaviorLogger) -> None:
    """Adding a 'now' event should land in today's bucket."""
    seeded_logger.log_event(
        product_id=999, event_type="view", user_id=1,
        timestamp=_dt.datetime.now(_dt.timezone.utc).timestamp(),
    )
    rows = daily_event_counts(seeded_logger, days_back=7)
    today_iso = _dt.datetime.now(_dt.timezone.utc).date().isoformat()
    today_row = next(r for r in rows if r["date"] == today_iso)
    assert today_row["n_views"] >= 1


def test_daily_event_counts_empty_log(empty_logger: BehaviorLogger) -> None:
    rows = daily_event_counts(empty_logger, days_back=7)
    assert len(rows) == 7
    for r in rows:
        assert r["n_views"] == 0
        assert r["n_clicks"] == 0
        assert r["n_add_to_cart"] == 0
        assert r["n_purchases"] == 0


# ---------------------------------------------------------------------------
# category_breakdown
# ---------------------------------------------------------------------------
def test_category_breakdown_share_sums_to_one(seeded_logger: BehaviorLogger) -> None:
    # The seed only has one purchase (product 100). Build a fake catalog
    # so the function has a department to attribute to.
    catalog = pd.DataFrame({
        "product_id": [100, 101, 102, 200, 201, 202],
        "category": ["yogurt"] * 3 + ["milk"] * 3,
        "department": ["dairy"] * 3 + ["beverages"] * 3,
    })
    rows = category_breakdown(seeded_logger, catalog, event_type="purchase")
    # Only one purchase -> dairy=1, beverages absent.
    assert len(rows) == 1
    assert rows[0]["department"] == "dairy"
    assert rows[0]["count"] == 1
    assert rows[0]["share"] == pytest.approx(1.0)


def test_category_breakdown_view_groups_correctly(
    seeded_logger: BehaviorLogger,
) -> None:
    catalog = pd.DataFrame({
        "product_id": [100, 101, 102, 103, 104, 200, 201, 202],
        "category": ["yogurt"] * 5 + ["milk"] * 3,
        "department": ["dairy"] * 5 + ["beverages"] * 3,
    })
    rows = category_breakdown(seeded_logger, catalog, event_type="view")
    by_dept = {r["department"]: r for r in rows}
    assert by_dept["dairy"]["count"] == 5
    assert by_dept["beverages"]["count"] == 3
    # Shares must sum to 1.0 (subject to float rounding).
    assert sum(r["share"] for r in rows) == pytest.approx(1.0)
    # Sorted descending by count.
    counts = [r["count"] for r in rows]
    assert counts == sorted(counts, reverse=True)


def test_category_breakdown_empty_log(
    empty_logger: BehaviorLogger,
) -> None:
    catalog = pd.DataFrame({
        "product_id": [1, 2],
        "category": ["a", "b"],
        "department": ["x", "y"],
    })
    assert category_breakdown(empty_logger, catalog, event_type="purchase") == []


# ---------------------------------------------------------------------------
# user_activity_summary
# ---------------------------------------------------------------------------
def test_user_activity_summary_shape(seeded_logger: BehaviorLogger) -> None:
    # User 10: 1 view + 1 click + 1 add_to_cart + 1 purchase = 4 events.
    summary = user_activity_summary(seeded_logger, user_id=10)
    assert summary["user_id"] == 10
    assert summary["total_events"] == 4
    assert summary["by_type"] == {
        "view": 1, "click": 1, "add_to_cart": 1, "purchase": 1,
    }
    # 4 events touch product 100 three times + product 100 (view) -> 1 distinct.
    # Actually: view product 100, click product 100, add_to_cart 100, purchase 100.
    assert summary["distinct_products"] == 1
    assert summary["first_event_at"] is not None
    assert summary["last_event_at"] is not None
    assert summary["last_event_at"] >= summary["first_event_at"]
    # All seeded events for user 10 fall on the same UTC date.
    assert summary["active_days"] == 1


def test_user_activity_summary_unknown_user(seeded_logger: BehaviorLogger) -> None:
    summary = user_activity_summary(seeded_logger, user_id=99999)
    assert summary["total_events"] == 0
    assert summary["by_type"] == {
        "view": 0, "click": 0, "add_to_cart": 0, "purchase": 0,
    }
    assert summary["distinct_products"] == 0
    assert summary["first_event_at"] is None
    assert summary["last_event_at"] is None
    assert summary["active_days"] == 0


def test_user_activity_summary_active_days_multi(tmp_path: Path) -> None:
    """A user with events on N distinct days reports active_days=N."""
    log = BehaviorLogger(tmp_path / "multiday.db")
    try:
        # 2023-11-14, 2023-11-15, 2023-11-15
        day0 = _dt.datetime(2023, 11, 14, 12, 0, tzinfo=_dt.timezone.utc).timestamp()
        day1 = _dt.datetime(2023, 11, 15, 9, 0, tzinfo=_dt.timezone.utc).timestamp()
        day1b = _dt.datetime(2023, 11, 15, 18, 0, tzinfo=_dt.timezone.utc).timestamp()
        log.log_event(product_id=1, event_type="view", user_id=7, timestamp=day0)
        log.log_event(product_id=2, event_type="view", user_id=7, timestamp=day1)
        log.log_event(product_id=3, event_type="view", user_id=7, timestamp=day1b)
        summary = user_activity_summary(log, user_id=7)
        assert summary["active_days"] == 2
        assert summary["total_events"] == 3
    finally:
        log.close()


# ---------------------------------------------------------------------------
# search_quality_signals
# ---------------------------------------------------------------------------
def test_search_quality_signals_ctr(seeded_logger: BehaviorLogger) -> None:
    signals = search_quality_signals(seeded_logger)
    # query-bearing events: 5 yogurt views + 3 milk views + 2 yogurt clicks = 10.
    assert signals["total_search_events"] == 10
    # distinct queries: 'yogurt', 'milk'
    assert signals["distinct_queries"] == 2
    # CTR = 2 clicks / 8 views (over query-bearing events).
    assert signals["click_through_rate"] == pytest.approx(2 / 8)
    # avg click position = mean(1, 3) = 2.0
    assert signals["avg_click_position"] == pytest.approx(2.0)
    # zero_result_rate is documented as None.
    assert signals["zero_result_rate"] is None


def test_search_quality_signals_empty_log(empty_logger: BehaviorLogger) -> None:
    signals = search_quality_signals(empty_logger)
    assert signals["total_search_events"] == 0
    assert signals["distinct_queries"] == 0
    assert signals["click_through_rate"] == 0.0
    assert signals["avg_click_position"] == 0.0
    assert signals["zero_result_rate"] is None


# ---------------------------------------------------------------------------
# Dataclass serialization
# ---------------------------------------------------------------------------
def test_top_query_serializable(seeded_logger: BehaviorLogger) -> None:
    rows = top_queries(seeded_logger, limit=5)
    assert rows
    d = asdict(rows[0])
    assert set(d) == {"query", "count", "distinct_users", "avg_position_clicked"}


def test_funnel_metrics_serializable(seeded_logger: BehaviorLogger) -> None:
    m = funnel_metrics(seeded_logger)
    d = asdict(m)
    assert set(d) == {
        "n_views", "n_clicks", "n_add_to_cart", "n_purchases",
        "view_to_click_rate", "click_to_cart_rate",
        "cart_to_purchase_rate", "overall_conversion",
    }
