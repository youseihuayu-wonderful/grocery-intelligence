"""Tests for online (A/B) evaluation metrics.

Every test uses hand-crafted event dicts that match the schema
returned by ``BehaviorLogger.get_events`` -- a list of dicts with
``event_type``, ``user_id``, ``product_id``, ``position``, etc.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.evaluation.online_metrics import (
    VariantMetrics,
    compare_variants,
    compute_avg_click_position,
    compute_conversion_rate,
    compute_ctr,
    compute_mrr_at_k,
    compute_variant_metrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ev(
    event_type: str,
    user_id: int | None = 1,
    product_id: int = 1,
    position: int | None = None,
    timestamp: float = 1_700_000_000.0,
) -> dict[str, Any]:
    """Build one event dict with the same shape ``get_events`` returns."""
    return {
        "id": 0,
        "timestamp": timestamp,
        "user_id": user_id,
        "product_id": product_id,
        "event_type": event_type,
        "query": None,
        "position": position,
    }


class _FakeLogger:
    """Tiny stand-in for ``BehaviorLogger`` -- only ``get_events`` is used
    by :func:`compute_variant_metrics`, so that's all we mock."""

    def __init__(self, events: list[dict]) -> None:
        self._events = list(events)

    def get_events(
        self, since: float | None = None, limit: int = 100, **_: Any
    ) -> list[dict]:
        rows = self._events
        if since is not None:
            rows = [r for r in rows if r["timestamp"] >= since]
        return rows[:limit]


# ---------------------------------------------------------------------------
# compute_ctr
# ---------------------------------------------------------------------------
class TestComputeCtr:
    def test_ten_views_three_clicks_is_thirty_percent(self) -> None:
        events = [_ev("view") for _ in range(10)] + [_ev("click") for _ in range(3)]
        assert compute_ctr(events) == pytest.approx(0.30)

    def test_zero_views_returns_zero_no_division_error(self) -> None:
        # Pure clicks (or no events at all) must not blow up.
        events = [_ev("click") for _ in range(5)]
        assert compute_ctr(events) == 0.0
        assert compute_ctr([]) == 0.0

    def test_ignores_unrelated_event_types(self) -> None:
        # Purchases and add_to_cart don't count toward CTR numerator
        # or denominator.
        events = (
            [_ev("view") for _ in range(4)]
            + [_ev("click") for _ in range(1)]
            + [_ev("purchase") for _ in range(10)]
        )
        assert compute_ctr(events) == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# compute_conversion_rate
# ---------------------------------------------------------------------------
class TestComputeConversionRate:
    def test_five_impressions_two_converted_is_forty_percent(self) -> None:
        # 5 distinct (user, product) impressions, 2 end in purchase.
        events: list[dict[str, Any]] = []
        for pid in range(1, 6):
            events.append(_ev("view", user_id=1, product_id=pid))
        # Users 1, products 1 and 2 convert.
        events.append(_ev("purchase", user_id=1, product_id=1))
        events.append(_ev("purchase", user_id=1, product_id=2))
        assert compute_conversion_rate(events) == pytest.approx(0.40)

    def test_no_impressions_returns_zero(self) -> None:
        events = [_ev("purchase", user_id=1, product_id=1)]
        # No views/clicks => denominator is 0 => 0.0 by contract.
        assert compute_conversion_rate(events) == 0.0
        assert compute_conversion_rate([]) == 0.0

    def test_purchase_without_impression_is_excluded(self) -> None:
        # User 1 viewed product 1, but the purchase is from product 2
        # which never had an impression. So 0 of 1 impressions converted.
        events = [
            _ev("view", user_id=1, product_id=1),
            _ev("purchase", user_id=1, product_id=2),
        ]
        assert compute_conversion_rate(events) == 0.0


# ---------------------------------------------------------------------------
# compute_mrr_at_k
# ---------------------------------------------------------------------------
class TestComputeMrrAtK:
    def test_clicks_at_positions_1_2_5(self) -> None:
        events = [
            _ev("click", position=1),
            _ev("click", position=2),
            _ev("click", position=5),
        ]
        expected = (1.0 + 0.5 + 0.2) / 3
        assert compute_mrr_at_k(events, k=10) == pytest.approx(expected)
        assert compute_mrr_at_k(events, k=10) == pytest.approx(0.5666666666)

    def test_ignores_clicks_above_k(self) -> None:
        events = [
            _ev("click", position=1),
            _ev("click", position=2),
            _ev("click", position=15),  # outside top-10
            _ev("click", position=99),  # outside top-10
        ]
        # Only positions 1 and 2 are counted at k=10.
        expected = (1.0 + 0.5) / 2
        assert compute_mrr_at_k(events, k=10) == pytest.approx(expected)

    def test_ignores_clicks_with_null_position(self) -> None:
        events = [
            _ev("click", position=None),
            _ev("click", position=2),
        ]
        # Only the position-2 click counts.
        assert compute_mrr_at_k(events, k=10) == pytest.approx(0.5)

    def test_no_qualifying_clicks_returns_zero(self) -> None:
        # Pure views shouldn't contribute, and there are no clicks at all.
        events = [_ev("view") for _ in range(5)]
        assert compute_mrr_at_k(events, k=10) == 0.0
        assert compute_mrr_at_k([], k=10) == 0.0


# ---------------------------------------------------------------------------
# compute_avg_click_position
# ---------------------------------------------------------------------------
class TestComputeAvgClickPosition:
    def test_mean_of_positions(self) -> None:
        events = [
            _ev("click", position=1),
            _ev("click", position=3),
            _ev("click", position=8),
        ]
        # mean(1, 3, 8) = 4.0
        assert compute_avg_click_position(events) == pytest.approx(4.0)

    def test_ignores_null_positions_and_non_clicks(self) -> None:
        events = [
            _ev("view", position=99),  # view, ignored
            _ev("click", position=None),  # click but no position, ignored
            _ev("click", position=2),
            _ev("click", position=4),
        ]
        # mean(2, 4) = 3.0
        assert compute_avg_click_position(events) == pytest.approx(3.0)

    def test_no_clicks_returns_zero(self) -> None:
        assert compute_avg_click_position([]) == 0.0
        assert compute_avg_click_position([_ev("view")]) == 0.0


# ---------------------------------------------------------------------------
# compute_variant_metrics
# ---------------------------------------------------------------------------
class TestComputeVariantMetrics:
    def test_three_users_two_variants_aggregate_correctly(self) -> None:
        events = [
            # user 1 (control): 2 views + 1 click + 1 purchase
            _ev("view", user_id=1, product_id=10),
            _ev("view", user_id=1, product_id=10),
            _ev("click", user_id=1, product_id=10, position=1),
            _ev("purchase", user_id=1, product_id=10),
            # user 2 (control): 1 view, no click
            _ev("view", user_id=2, product_id=20),
            # user 3 (treatment): 2 views + 2 clicks
            _ev("view", user_id=3, product_id=30),
            _ev("view", user_id=3, product_id=31),
            _ev("click", user_id=3, product_id=30, position=2),
            _ev("click", user_id=3, product_id=31, position=4),
        ]
        logger = _FakeLogger(events)
        user_variant_map = {1: "control", 2: "control", 3: "treatment"}

        result = compute_variant_metrics(logger, user_variant_map)

        assert set(result.keys()) == {"control", "treatment"}

        control = result["control"]
        assert control.variant == "control"
        assert control.n_users == 2
        assert control.n_views == 3  # 2 from user1 + 1 from user2
        assert control.n_clicks == 1
        assert control.n_purchases == 1
        assert control.ctr == pytest.approx(1 / 3)
        # 2 distinct (user, product) impressions: (1, 10) and (2, 20).
        # (1, 10) has a purchase. So conversion = 1/2.
        assert control.conversion_rate == pytest.approx(0.5)

        treatment = result["treatment"]
        assert treatment.variant == "treatment"
        assert treatment.n_users == 1
        assert treatment.n_views == 2
        assert treatment.n_clicks == 2
        assert treatment.n_purchases == 0
        assert treatment.ctr == pytest.approx(1.0)
        # MRR over clicks at positions 2 and 4 => (0.5 + 0.25) / 2 = 0.375
        assert treatment.mrr_at_10 == pytest.approx(0.375)
        # Avg click position = (2 + 4) / 2 = 3.0
        assert treatment.avg_click_position == pytest.approx(3.0)

    def test_events_from_users_not_in_map_are_excluded(self) -> None:
        events = [
            _ev("view", user_id=1, product_id=10),
            _ev("click", user_id=1, product_id=10, position=1),
            # user 999 is NOT in the experiment - drop these events
            _ev("view", user_id=999, product_id=10),
            _ev("view", user_id=999, product_id=10),
            _ev("click", user_id=999, product_id=10, position=1),
            # anonymous (user_id = None) - drop as well
            _ev("view", user_id=None, product_id=10),
        ]
        logger = _FakeLogger(events)
        user_variant_map = {1: "control"}

        result = compute_variant_metrics(logger, user_variant_map)
        assert "control" in result
        assert result["control"].n_users == 1
        assert result["control"].n_views == 1
        assert result["control"].n_clicks == 1
        # The user-999 events must NOT have inflated any counter.

    def test_variant_with_no_events_still_appears(self) -> None:
        # User 1 has events; user 2 is in the experiment but logged nothing.
        events = [_ev("view", user_id=1, product_id=10)]
        logger = _FakeLogger(events)
        user_variant_map = {1: "control", 2: "treatment"}

        result = compute_variant_metrics(logger, user_variant_map)
        # Both variants must be reported -- a zero row is information.
        assert set(result.keys()) == {"control", "treatment"}
        assert result["treatment"].n_views == 0
        assert result["treatment"].n_clicks == 0
        assert result["treatment"].n_users == 0


# ---------------------------------------------------------------------------
# compare_variants
# ---------------------------------------------------------------------------
class TestCompareVariants:
    def _vm(
        self,
        name: str,
        ctr: float,
        conv: float = 0.0,
        mrr: float = 0.0,
        n_users: int = 100,
    ) -> VariantMetrics:
        return VariantMetrics(
            variant=name,
            n_users=n_users,
            n_views=0,
            n_clicks=0,
            n_purchases=0,
            ctr=ctr,
            conversion_rate=conv,
            mrr_at_10=mrr,
            avg_click_position=0.0,
        )

    def test_treatment_wins_with_higher_ctr(self) -> None:
        a = self._vm("control", ctr=0.10, conv=0.02, mrr=0.30, n_users=500)
        b = self._vm("treatment", ctr=0.12, conv=0.025, mrr=0.36, n_users=510)

        out = compare_variants(a, b)
        assert out["control"] == "control"
        assert out["treatment"] == "treatment"
        # (0.12 - 0.10) / 0.10 * 100 = 20.0
        assert out["ctr_lift_pct"] == pytest.approx(20.0)
        # (0.025 - 0.02) / 0.02 * 100 = 25.0
        assert out["conversion_lift_pct"] == pytest.approx(25.0)
        # (0.36 - 0.30) / 0.30 * 100 = 20.0
        assert out["mrr_lift_pct"] == pytest.approx(20.0)
        assert out["winner"] == "treatment"
        assert out["sample_size_a"] == 500
        assert out["sample_size_b"] == 510

    def test_control_wins_when_treatment_ctr_lower(self) -> None:
        a = self._vm("control", ctr=0.20)
        b = self._vm("treatment", ctr=0.10)
        out = compare_variants(a, b)
        # negative lift
        assert out["ctr_lift_pct"] == pytest.approx(-50.0)
        assert out["winner"] == "control"

    def test_tie_when_ctrs_equal(self) -> None:
        a = self._vm("control", ctr=0.10)
        b = self._vm("treatment", ctr=0.10)
        out = compare_variants(a, b)
        assert out["ctr_lift_pct"] == pytest.approx(0.0)
        assert out["winner"] == "tie"

    def test_zero_control_ctr_yields_none_lift_not_crash(self) -> None:
        # Control had no views at all; treatment had a few clicks.
        # Lift is undefined; we report None, not inf or NaN.
        a = self._vm("control", ctr=0.0, conv=0.0, mrr=0.0)
        b = self._vm("treatment", ctr=0.05, conv=0.10, mrr=0.40)
        out = compare_variants(a, b)
        assert out["ctr_lift_pct"] is None
        assert out["conversion_lift_pct"] is None
        assert out["mrr_lift_pct"] is None
        # Treatment still wins on CTR despite undefined lift.
        assert out["winner"] == "treatment"
