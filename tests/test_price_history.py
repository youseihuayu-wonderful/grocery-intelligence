"""Tests for :mod:`src.pricing.history`."""

from __future__ import annotations

import datetime as _dt

import pytest

from src.pricing.history import detect_price_drop, generate_price_history


# ----------------------------------------------------------------------
# generate_price_history
# ----------------------------------------------------------------------


def test_generate_price_history_default_length() -> None:
    """Default call returns exactly 90 rows."""
    history = generate_price_history(product_id=1, current_price=5.00)
    assert len(history) == 90


def test_generate_price_history_custom_length() -> None:
    """``days`` parameter is honored."""
    for days in (1, 7, 30, 120):
        history = generate_price_history(product_id=1, current_price=5.00, days=days)
        assert len(history) == days


def test_generate_price_history_last_day_equals_current() -> None:
    """The final entry must pin to the requested current price."""
    for current in (0.99, 1.23, 5.00, 19.99):
        history = generate_price_history(product_id=7, current_price=current)
        assert history[-1]["price"] == round(current, 2)


def test_generate_price_history_last_date_is_today() -> None:
    """Final row is dated today."""
    history = generate_price_history(product_id=1, current_price=5.0)
    today = _dt.date.today().isoformat()
    assert history[-1]["date"] == today


def test_generate_price_history_dates_are_consecutive_and_oldest_first() -> None:
    """No gaps, no duplicates, sorted oldest-first."""
    history = generate_price_history(product_id=42, current_price=4.0, days=30)
    dates = [_dt.date.fromisoformat(row["date"]) for row in history]
    for prev, curr in zip(dates, dates[1:]):
        assert (curr - prev).days == 1


def test_generate_price_history_deterministic_same_id() -> None:
    """Same product_id + price → identical series."""
    h1 = generate_price_history(product_id=123, current_price=10.0)
    h2 = generate_price_history(product_id=123, current_price=10.0)
    assert h1 == h2


def test_generate_price_history_different_ids_differ() -> None:
    """Different ids should diverge (vanishingly tiny chance of collision)."""
    h1 = generate_price_history(product_id=1, current_price=10.0)
    h2 = generate_price_history(product_id=2, current_price=10.0)
    assert h1 != h2


def test_generate_price_history_all_prices_positive() -> None:
    """No negative or zero prices in the simulation."""
    history = generate_price_history(product_id=99, current_price=1.50)
    assert all(row["price"] > 0 for row in history)


def test_generate_price_history_handles_nonpositive_current() -> None:
    """A bogus zero/negative current price must not crash."""
    history = generate_price_history(product_id=1, current_price=0.0)
    assert len(history) == 90
    assert all(row["price"] > 0 for row in history)


# ----------------------------------------------------------------------
# detect_price_drop
# ----------------------------------------------------------------------


def _flat_history(price: float, days: int = 90) -> list[dict]:
    """Helper: build a flat-line history at ``price``."""
    today = _dt.date.today()
    return [
        {
            "date": (today - _dt.timedelta(days=days - 1 - i)).isoformat(),
            "price": price,
        }
        for i in range(days)
    ]


def test_detect_price_drop_flat_history_not_significant() -> None:
    history = _flat_history(price=5.0)
    result = detect_price_drop(history, lookback_days=30)
    assert result is not None
    assert result["is_significant"] is False
    assert result["drop_pct"] == 0.0
    assert result["current"] == 5.0
    assert result["previous_median"] == 5.0


def test_detect_price_drop_large_drop_significant() -> None:
    """20% drop on the last day → flagged significant."""
    history = _flat_history(price=10.0, days=60)
    history[-1] = {"date": history[-1]["date"], "price": 8.0}  # 20% drop
    result = detect_price_drop(history, lookback_days=30)
    assert result is not None
    assert result["is_significant"] is True
    assert result["drop_pct"] == pytest.approx(20.0, abs=0.1)


def test_detect_price_drop_small_drop_not_significant() -> None:
    """5% drop is below the 10% threshold."""
    history = _flat_history(price=10.0, days=60)
    history[-1] = {"date": history[-1]["date"], "price": 9.5}  # 5% drop
    result = detect_price_drop(history, lookback_days=30)
    assert result is not None
    assert result["is_significant"] is False
    assert result["drop_pct"] == pytest.approx(5.0, abs=0.1)


def test_detect_price_drop_returns_none_when_short() -> None:
    """Not enough history → None."""
    history = _flat_history(price=5.0, days=10)
    assert detect_price_drop(history, lookback_days=30) is None


def test_detect_price_drop_empty_history() -> None:
    assert detect_price_drop([], lookback_days=30) is None


def test_detect_price_drop_price_increase() -> None:
    """Price increase produces a negative drop_pct; not significant."""
    history = _flat_history(price=10.0, days=60)
    history[-1] = {"date": history[-1]["date"], "price": 12.0}
    result = detect_price_drop(history, lookback_days=30)
    assert result is not None
    assert result["drop_pct"] < 0  # negative means a rise
    assert result["is_significant"] is False
