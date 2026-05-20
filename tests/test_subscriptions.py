"""Tests for the Subscribe & Save module."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from src.shopping.cart import CartStore
from src.shopping.subscriptions import (
    FREQUENCIES,
    FREQUENCY_DAYS,
    SubscriptionStore,
    fulfill_due_subscriptions,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def store(tmp_path: Path) -> SubscriptionStore:
    """A fresh SubscriptionStore backed by a per-test temp SQLite file."""
    db_path = tmp_path / "test_subscriptions.db"
    s = SubscriptionStore(db_path)
    yield s
    s.close()


@pytest.fixture
def cart(tmp_path: Path) -> CartStore:
    """A fresh CartStore backed by a per-test temp SQLite file."""
    db_path = tmp_path / "test_cart_for_subs.db"
    c = CartStore(db_path)
    yield c
    c.close()


# ----------------------------------------------------------------------
# subscribe()
# ----------------------------------------------------------------------
def test_subscribe_returns_id_and_lists(store: SubscriptionStore) -> None:
    sub_id = store.subscribe(
        user_id=1,
        product_id=100,
        frequency="weekly",
        qty=2,
        first_delivery_offset_days=7,
    )
    assert isinstance(sub_id, int)
    assert sub_id > 0

    subs = store.get_user_subscriptions(user_id=1)
    assert len(subs) == 1
    s = subs[0]
    assert s["id"] == sub_id
    assert s["product_id"] == 100
    assert s["qty"] == 2
    assert s["frequency"] == "weekly"
    assert s["active"] == 1
    assert s["created_at"] > 0
    # ~7 days from now, give a little tolerance for test runtime.
    assert 6.5 < s["days_until_next"] < 7.5


def test_subscribe_invalid_frequency_raises(store: SubscriptionStore) -> None:
    with pytest.raises(ValueError):
        store.subscribe(user_id=1, product_id=100, frequency="daily")
    with pytest.raises(ValueError):
        store.subscribe(user_id=1, product_id=100, frequency="")


def test_subscribe_all_valid_frequencies(store: SubscriptionStore) -> None:
    for f in FREQUENCIES:
        sub_id = store.subscribe(user_id=1, product_id=1, frequency=f)
        assert sub_id > 0


def test_subscribe_invalid_qty(store: SubscriptionStore) -> None:
    with pytest.raises(ValueError):
        store.subscribe(user_id=1, product_id=100, frequency="weekly", qty=0)
    with pytest.raises(ValueError):
        store.subscribe(user_id=1, product_id=100, frequency="weekly", qty=-2)


# ----------------------------------------------------------------------
# cancel()
# ----------------------------------------------------------------------
def test_cancel_soft_deletes(store: SubscriptionStore) -> None:
    sub_id = store.subscribe(
        user_id=1, product_id=100, frequency="weekly"
    )
    store.cancel(sub_id)
    # active_only=True (default) excludes cancelled rows.
    assert store.get_user_subscriptions(user_id=1) == []
    # active_only=False still returns them with active=0.
    all_subs = store.get_user_subscriptions(user_id=1, active_only=False)
    assert len(all_subs) == 1
    assert all_subs[0]["id"] == sub_id
    assert all_subs[0]["active"] == 0


# ----------------------------------------------------------------------
# update_frequency / update_qty
# ----------------------------------------------------------------------
def test_update_frequency_round_trip(store: SubscriptionStore) -> None:
    sub_id = store.subscribe(
        user_id=1, product_id=100, frequency="weekly"
    )
    store.update_frequency(sub_id, "monthly")
    subs = store.get_user_subscriptions(user_id=1)
    assert subs[0]["frequency"] == "monthly"


def test_update_frequency_validates(store: SubscriptionStore) -> None:
    sub_id = store.subscribe(
        user_id=1, product_id=100, frequency="weekly"
    )
    with pytest.raises(ValueError):
        store.update_frequency(sub_id, "fortnightly")


def test_update_qty_round_trip(store: SubscriptionStore) -> None:
    sub_id = store.subscribe(
        user_id=1, product_id=100, frequency="weekly", qty=1
    )
    store.update_qty(sub_id, 5)
    subs = store.get_user_subscriptions(user_id=1)
    assert subs[0]["qty"] == 5


def test_update_qty_validates(store: SubscriptionStore) -> None:
    sub_id = store.subscribe(
        user_id=1, product_id=100, frequency="weekly"
    )
    with pytest.raises(ValueError):
        store.update_qty(sub_id, 0)
    with pytest.raises(ValueError):
        store.update_qty(sub_id, -1)


# ----------------------------------------------------------------------
# get_user_subscriptions sort order
# ----------------------------------------------------------------------
def test_get_user_subscriptions_sorted_by_next_delivery(
    store: SubscriptionStore,
) -> None:
    # Schedule 3 subs with increasing offsets.
    a = store.subscribe(
        user_id=1, product_id=10, frequency="weekly",
        first_delivery_offset_days=30,
    )
    b = store.subscribe(
        user_id=1, product_id=20, frequency="weekly",
        first_delivery_offset_days=1,
    )
    c = store.subscribe(
        user_id=1, product_id=30, frequency="weekly",
        first_delivery_offset_days=10,
    )
    subs = store.get_user_subscriptions(user_id=1)
    ids = [s["id"] for s in subs]
    # Ascending by next_delivery_at = b (1d), c (10d), a (30d).
    assert ids == [b, c, a]


# ----------------------------------------------------------------------
# get_due_subscriptions
# ----------------------------------------------------------------------
def test_get_due_subscriptions_returns_only_past_due_active(
    store: SubscriptionStore,
) -> None:
    # One due (offset 0), one not yet (offset +10d), one cancelled-but-due (-1d).
    due_id = store.subscribe(
        user_id=1, product_id=10, frequency="weekly",
        first_delivery_offset_days=0,
    )
    store.subscribe(
        user_id=1, product_id=20, frequency="weekly",
        first_delivery_offset_days=10,
    )
    cancelled_id = store.subscribe(
        user_id=1, product_id=30, frequency="weekly",
        first_delivery_offset_days=-1,
    )
    store.cancel(cancelled_id)

    due = store.get_due_subscriptions()
    due_ids = [d["id"] for d in due]
    assert due_ids == [due_id]


def test_get_due_subscriptions_as_of_in_future(
    store: SubscriptionStore,
) -> None:
    """as_of allows the caller to ask 'what would be due at time T?'"""
    sub_id = store.subscribe(
        user_id=1, product_id=10, frequency="weekly",
        first_delivery_offset_days=5,
    )
    # Now: nothing due.
    assert store.get_due_subscriptions() == []
    # Future: it's due.
    future = time.time() + 10 * 86400
    due = store.get_due_subscriptions(as_of=future)
    assert [d["id"] for d in due] == [sub_id]


# ----------------------------------------------------------------------
# advance_next_delivery
# ----------------------------------------------------------------------
def test_advance_next_delivery_uses_frequency_days(
    store: SubscriptionStore,
) -> None:
    sub_id = store.subscribe(
        user_id=1, product_id=10, frequency="biweekly",
        first_delivery_offset_days=0,
    )
    before = store.get_user_subscriptions(user_id=1)[0]
    before_at = before["next_delivery_at"]

    store.advance_next_delivery(sub_id)

    after = store.get_user_subscriptions(user_id=1)[0]
    delta_days = (after["next_delivery_at"] - before_at) / 86400
    assert delta_days == pytest.approx(FREQUENCY_DAYS["biweekly"], abs=1e-6)


def test_advance_next_delivery_unknown_id_noop(
    store: SubscriptionStore,
) -> None:
    # Should not raise; simply does nothing.
    store.advance_next_delivery(999999)


# ----------------------------------------------------------------------
# count_active
# ----------------------------------------------------------------------
def test_count_active(store: SubscriptionStore) -> None:
    assert store.count_active(user_id=1) == 0

    a = store.subscribe(user_id=1, product_id=10, frequency="weekly")
    store.subscribe(user_id=1, product_id=20, frequency="weekly")
    store.subscribe(user_id=2, product_id=30, frequency="weekly")
    assert store.count_active(user_id=1) == 2
    assert store.count_active(user_id=2) == 1

    store.cancel(a)
    assert store.count_active(user_id=1) == 1


# ----------------------------------------------------------------------
# estimated_monthly_value
# ----------------------------------------------------------------------
def test_estimated_monthly_value_weekly_5dollars(
    store: SubscriptionStore,
) -> None:
    store.subscribe(user_id=1, product_id=100, frequency="weekly", qty=1)
    price_map = {100: 5.0}
    # weekly => 30/7 deliveries per month * $5 * 1 ≈ $21.43
    val = store.estimated_monthly_value(user_id=1, price_map=price_map)
    assert val == pytest.approx(21.43, abs=0.01)


def test_estimated_monthly_value_mixed_frequencies(
    store: SubscriptionStore,
) -> None:
    # User 1: 2x weekly $5 ($42.86) + 1x monthly $10 ($10.00).
    store.subscribe(user_id=1, product_id=100, frequency="weekly", qty=2)
    store.subscribe(user_id=1, product_id=200, frequency="monthly", qty=1)
    price_map = {100: 5.0, 200: 10.0}

    expected = (30 / 7) * 2 * 5.0 + (30 / 30) * 1 * 10.0
    val = store.estimated_monthly_value(user_id=1, price_map=price_map)
    assert val == pytest.approx(round(expected, 2), abs=0.01)


def test_estimated_monthly_value_missing_price_is_zero(
    store: SubscriptionStore,
) -> None:
    store.subscribe(user_id=1, product_id=100, frequency="weekly")
    # Empty price map → revenue is zero (no crash).
    assert store.estimated_monthly_value(user_id=1, price_map={}) == 0.0


def test_estimated_monthly_value_ignores_cancelled(
    store: SubscriptionStore,
) -> None:
    a = store.subscribe(user_id=1, product_id=100, frequency="weekly")
    store.cancel(a)
    assert store.estimated_monthly_value(
        user_id=1, price_map={100: 5.0}
    ) == 0.0


# ----------------------------------------------------------------------
# fulfill_due_subscriptions
# ----------------------------------------------------------------------
def test_fulfill_due_subscriptions_adds_to_cart_and_advances(
    store: SubscriptionStore, cart: CartStore
) -> None:
    # Two due now, one not.
    sub_a = store.subscribe(
        user_id=42, product_id=100, frequency="weekly", qty=2,
        first_delivery_offset_days=0,
    )
    sub_b = store.subscribe(
        user_id=42, product_id=200, frequency="monthly", qty=1,
        first_delivery_offset_days=0,
    )
    sub_c = store.subscribe(
        user_id=42, product_id=300, frequency="weekly", qty=1,
        first_delivery_offset_days=5,  # not due yet
    )

    # Capture the "before" next_delivery_at for the two due rows.
    before_map = {
        s["id"]: s["next_delivery_at"]
        for s in store.get_user_subscriptions(user_id=42)
    }

    fired = fulfill_due_subscriptions(store, cart)

    # Exactly the two due rows fired.
    assert len(fired) == 2
    fired_ids = {f["subscription_id"] for f in fired}
    assert fired_ids == {sub_a, sub_b}

    # Cart now has products 100 (qty 2) and 200 (qty 1).
    cart_items = cart.get_cart(42)
    by_pid = {it["product_id"]: it["qty"] for it in cart_items}
    assert by_pid == {100: 2, 200: 1}

    # next_delivery_at advanced for the two fired, untouched for sub_c.
    after = {
        s["id"]: s["next_delivery_at"]
        for s in store.get_user_subscriptions(user_id=42)
    }
    assert (after[sub_a] - before_map[sub_a]) / 86400 == pytest.approx(7)
    assert (after[sub_b] - before_map[sub_b]) / 86400 == pytest.approx(30)
    assert after[sub_c] == before_map[sub_c]

    # Each fired entry exposes the contract the API will hand to the UI.
    for f in fired:
        assert set(f.keys()) == {
            "subscription_id", "user_id", "product_id", "qty"
        }
        assert f["user_id"] == 42


def test_fulfill_due_no_due_returns_empty(
    store: SubscriptionStore, cart: CartStore
) -> None:
    store.subscribe(
        user_id=1, product_id=100, frequency="weekly",
        first_delivery_offset_days=7,
    )
    assert fulfill_due_subscriptions(store, cart) == []
    assert cart.get_cart(1) == []


# ----------------------------------------------------------------------
# Persistence / context manager
# ----------------------------------------------------------------------
def test_persistence_across_open(tmp_path: Path) -> None:
    db_path = tmp_path / "persist.db"
    s = SubscriptionStore(db_path)
    sub_id = s.subscribe(
        user_id=7, product_id=42, frequency="monthly", qty=3,
        first_delivery_offset_days=2,
    )
    s.close()

    # Reopen — data should still be there.
    s2 = SubscriptionStore(db_path)
    subs = s2.get_user_subscriptions(user_id=7)
    assert len(subs) == 1
    assert subs[0]["id"] == sub_id
    assert subs[0]["product_id"] == 42
    assert subs[0]["qty"] == 3
    assert subs[0]["frequency"] == "monthly"
    s2.close()


def test_context_manager(tmp_path: Path) -> None:
    db_path = tmp_path / "ctx.db"
    with SubscriptionStore(db_path) as s:
        s.subscribe(user_id=1, product_id=100, frequency="weekly")
        assert s.count_active(user_id=1) == 1
    # After exit, reopening reads back what we wrote.
    with SubscriptionStore(db_path) as s2:
        assert s2.count_active(user_id=1) == 1
