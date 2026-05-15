"""Tests for the behavioral signal tracking module.

Each test uses a temp database via the pytest ``tmp_path`` fixture so
no production state is touched.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pytest

from src.recommend.behavior import (
    DEFAULT_DB_PATH,
    EVENT_TYPES,
    BehaviorLogger,
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
# 1. validation
# ---------------------------------------------------------------------------
def test_log_event_rejects_unknown_event_type(logger: BehaviorLogger) -> None:
    with pytest.raises(ValueError):
        logger.log_event(product_id=1, event_type="hover")

    # And the known types should all be accepted.
    for et in EVENT_TYPES:
        rowid = logger.log_event(product_id=1, event_type=et, user_id=1)
        assert rowid > 0


# ---------------------------------------------------------------------------
# 2. round-trip
# ---------------------------------------------------------------------------
def test_log_event_get_events_round_trip(logger: BehaviorLogger) -> None:
    ts = 1_700_000_000.0
    rowid = logger.log_event(
        product_id=42,
        event_type="click",
        user_id=7,
        query="milk",
        position=3,
        timestamp=ts,
    )
    assert rowid > 0

    rows = logger.get_events(user_id=7)
    assert len(rows) == 1
    row = rows[0]
    assert row["user_id"] == 7
    assert row["product_id"] == 42
    assert row["event_type"] == "click"
    assert row["query"] == "milk"
    assert row["position"] == 3
    assert row["timestamp"] == pytest.approx(ts)


# ---------------------------------------------------------------------------
# 3. filters
# ---------------------------------------------------------------------------
def test_get_events_filters(logger: BehaviorLogger) -> None:
    base = 1_700_000_000.0
    logger.log_event(product_id=1, event_type="view", user_id=1, timestamp=base + 1)
    logger.log_event(product_id=2, event_type="click", user_id=1, timestamp=base + 2)
    logger.log_event(product_id=1, event_type="view", user_id=2, timestamp=base + 3)
    logger.log_event(product_id=3, event_type="purchase", user_id=1, timestamp=base + 4)

    # Filter by user_id.
    rows = logger.get_events(user_id=1)
    assert len(rows) == 3
    assert all(r["user_id"] == 1 for r in rows)

    # Filter by product_id.
    rows = logger.get_events(product_id=1)
    assert len(rows) == 2
    assert all(r["product_id"] == 1 for r in rows)

    # Filter by event_type.
    rows = logger.get_events(event_type="view")
    assert len(rows) == 2
    assert all(r["event_type"] == "view" for r in rows)

    # Filter by since (>= timestamp).
    rows = logger.get_events(since=base + 3)
    assert len(rows) == 2
    assert all(r["timestamp"] >= base + 3 for r in rows)

    # Combined filters.
    rows = logger.get_events(user_id=1, event_type="view")
    assert len(rows) == 1
    assert rows[0]["product_id"] == 1


def test_get_events_orders_newest_first(logger: BehaviorLogger) -> None:
    base = 1_700_000_000.0
    logger.log_event(product_id=1, event_type="view", timestamp=base + 10)
    logger.log_event(product_id=2, event_type="view", timestamp=base + 30)
    logger.log_event(product_id=3, event_type="view", timestamp=base + 20)

    rows = logger.get_events()
    timestamps = [r["timestamp"] for r in rows]
    assert timestamps == sorted(timestamps, reverse=True)


def test_get_events_limit(logger: BehaviorLogger) -> None:
    for i in range(10):
        logger.log_event(product_id=i, event_type="view")
    rows = logger.get_events(limit=3)
    assert len(rows) == 3


# ---------------------------------------------------------------------------
# 4. count_events
# ---------------------------------------------------------------------------
def test_count_events_overall_and_by_type(logger: BehaviorLogger) -> None:
    logger.log_event(product_id=1, event_type="view")
    logger.log_event(product_id=1, event_type="view")
    logger.log_event(product_id=1, event_type="click")
    logger.log_event(product_id=2, event_type="purchase")

    assert logger.count_events() == 4
    assert logger.count_events("view") == 2
    assert logger.count_events("click") == 1
    assert logger.count_events("purchase") == 1
    assert logger.count_events("add_to_cart") == 0


# ---------------------------------------------------------------------------
# 5. summaries
# ---------------------------------------------------------------------------
def test_user_event_summary_shape(logger: BehaviorLogger) -> None:
    logger.log_event(product_id=1, event_type="view", user_id=42)
    logger.log_event(product_id=1, event_type="click", user_id=42)
    logger.log_event(product_id=2, event_type="purchase", user_id=42)
    # noise: different user
    logger.log_event(product_id=1, event_type="view", user_id=99)

    summary = logger.user_event_summary(42)
    assert summary["total"] == 3
    assert summary["distinct_products"] == 2
    assert summary["by_type"] == {
        "view": 1,
        "click": 1,
        "add_to_cart": 0,
        "purchase": 1,
    }


def test_product_event_summary_shape(logger: BehaviorLogger) -> None:
    logger.log_event(product_id=24852, event_type="view", user_id=1)
    logger.log_event(product_id=24852, event_type="click", user_id=2)
    logger.log_event(product_id=24852, event_type="purchase", user_id=1)
    # anonymous event should still count to total but not to distinct_users.
    logger.log_event(product_id=24852, event_type="view", user_id=None)
    # noise: different product
    logger.log_event(product_id=999, event_type="view", user_id=1)

    summary = logger.product_event_summary(24852)
    assert summary["total"] == 4
    assert summary["distinct_users"] == 2  # users 1 and 2, anonymous excluded
    assert summary["by_type"] == {
        "view": 2,
        "click": 1,
        "add_to_cart": 0,
        "purchase": 1,
    }


# ---------------------------------------------------------------------------
# 6. bulk insert: correctness + speed
# ---------------------------------------------------------------------------
def test_log_events_bulk_inserts_everything(logger: BehaviorLogger) -> None:
    events = [
        {
            "product_id": i,
            "event_type": "view",
            "user_id": i % 3,
            "timestamp": 1_700_000_000.0 + i,
        }
        for i in range(250)
    ]
    n = logger.log_events_bulk(events)
    assert n == 250
    assert logger.count_events() == 250
    assert logger.count_events("view") == 250


def test_log_events_bulk_validates_event_type(logger: BehaviorLogger) -> None:
    events = [
        {"product_id": 1, "event_type": "view"},
        {"product_id": 2, "event_type": "BOGUS"},
    ]
    with pytest.raises(ValueError):
        logger.log_events_bulk(events)
    # the failed call should have inserted nothing (transactional).
    assert logger.count_events() == 0


def test_log_events_bulk_is_much_faster_than_individual(
    tmp_path: Path,
) -> None:
    """The headline reason for ``log_events_bulk`` is throughput.

    With WAL + a single ``COMMIT`` we should be able to insert
    1,000 rows in well under the time it takes to do them
    individually. The threshold is conservative (>=3x faster);
    on a typical laptop the speedup is more like 50-200x.
    """
    n_rows = 1000
    events = [
        {"product_id": i, "event_type": "view", "user_id": i}
        for i in range(n_rows)
    ]

    # individual path
    log_a = BehaviorLogger(tmp_path / "a.db")
    t0 = time.perf_counter()
    for ev in events:
        log_a.log_event(**ev)
    t_individual = time.perf_counter() - t0
    log_a.close()

    # bulk path
    log_b = BehaviorLogger(tmp_path / "b.db")
    t0 = time.perf_counter()
    log_b.log_events_bulk(events)
    t_bulk = time.perf_counter() - t0
    log_b.close()

    assert t_bulk * 3 < t_individual, (
        f"Expected bulk insert to be at least 3x faster, "
        f"got individual={t_individual:.4f}s, bulk={t_bulk:.4f}s"
    )


# ---------------------------------------------------------------------------
# 7. feature matrix
# ---------------------------------------------------------------------------
def test_get_feature_matrix_aggregates_correctly(logger: BehaviorLogger) -> None:
    """Five events on two (user, product) pairs aggregate as expected."""
    base = 1_700_000_000.0
    # user 1, product 1: 2 views + 1 click + 1 purchase
    logger.log_event(product_id=1, event_type="view", user_id=1, timestamp=base + 1)
    logger.log_event(product_id=1, event_type="view", user_id=1, timestamp=base + 2)
    logger.log_event(product_id=1, event_type="click", user_id=1, timestamp=base + 3)
    logger.log_event(product_id=1, event_type="purchase", user_id=1, timestamp=base + 4)
    # user 2, product 2: 1 add_to_cart (no purchase -> conversion=0)
    logger.log_event(
        product_id=2, event_type="add_to_cart", user_id=2, timestamp=base + 5
    )

    df = logger.get_feature_matrix()
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) >= {
        "user_id",
        "product_id",
        "n_views",
        "n_clicks",
        "n_add_to_cart",
        "n_purchases",
        "last_event_ts",
        "conversion",
    }
    # two (user, product) pairs
    assert len(df) == 2

    row1 = df[(df["user_id"] == 1) & (df["product_id"] == 1)].iloc[0]
    assert row1["n_views"] == 2
    assert row1["n_clicks"] == 1
    assert row1["n_add_to_cart"] == 0
    assert row1["n_purchases"] == 1
    assert row1["last_event_ts"] == pytest.approx(base + 4)
    assert row1["conversion"] == 1

    row2 = df[(df["user_id"] == 2) & (df["product_id"] == 2)].iloc[0]
    assert row2["n_views"] == 0
    assert row2["n_clicks"] == 0
    assert row2["n_add_to_cart"] == 1
    assert row2["n_purchases"] == 0
    assert row2["last_event_ts"] == pytest.approx(base + 5)
    assert row2["conversion"] == 0


def test_get_feature_matrix_empty(logger: BehaviorLogger) -> None:
    df = logger.get_feature_matrix()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    # columns are still defined so downstream code never KeyErrors.
    assert {
        "user_id",
        "product_id",
        "n_views",
        "n_clicks",
        "n_add_to_cart",
        "n_purchases",
        "last_event_ts",
        "conversion",
    } <= set(df.columns)


# ---------------------------------------------------------------------------
# 8. persistence: close + reopen
# ---------------------------------------------------------------------------
def test_close_and_reopen_persists(tmp_path: Path) -> None:
    db = tmp_path / "persist.db"

    log = BehaviorLogger(db)
    log.log_event(product_id=11, event_type="view", user_id=99)
    log.log_event(product_id=11, event_type="purchase", user_id=99)
    log.close()

    # File should exist on disk.
    assert db.exists()

    log2 = BehaviorLogger(db)
    try:
        assert log2.count_events() == 2
        rows = log2.get_events(user_id=99)
        assert len(rows) == 2
        types = sorted(r["event_type"] for r in rows)
        assert types == ["purchase", "view"]
    finally:
        log2.close()


# ---------------------------------------------------------------------------
# 9. context-manager use
# ---------------------------------------------------------------------------
def test_context_manager(tmp_path: Path) -> None:
    db = tmp_path / "ctx.db"
    with BehaviorLogger(db) as log:
        log.log_event(product_id=1, event_type="view")
        assert log.count_events() == 1
    # After exit, the file is still there with 1 row.
    with BehaviorLogger(db) as log:
        assert log.count_events() == 1


# ---------------------------------------------------------------------------
# 10. DEFAULT_DB_PATH constant
# ---------------------------------------------------------------------------
def test_default_db_path_is_path() -> None:
    assert isinstance(DEFAULT_DB_PATH, Path)
    assert str(DEFAULT_DB_PATH).endswith("behavior.db")
