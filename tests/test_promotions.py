"""Tests for :mod:`src.pricing.promotions`."""

from __future__ import annotations

import pytest

from src.pricing.promotions import (
    Promotion,
    cart_pricing_summary,
    evaluate_promotions,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _item(price: float, qty: int = 1, department: str = "pantry",
          product_name: str = "Generic", **extra) -> dict:
    """Tiny factory for a cart-line dict."""
    base = {
        "product_id": extra.get("product_id", 0),
        "product_name": product_name,
        "department": department,
        "category": extra.get("category", department),
        "price": price,
        "qty": qty,
    }
    base.update(extra)
    return base


def _find(promos: list[Promotion], code: str) -> Promotion:
    """Find a promo by its code, or fail with a clear message."""
    for p in promos:
        if p.code == code:
            return p
    raise AssertionError(f"Promotion {code!r} not in {[p.code for p in promos]}")


# ----------------------------------------------------------------------
# evaluate_promotions
# ----------------------------------------------------------------------


def test_empty_cart_no_applied_all_at_zero_progress() -> None:
    promos = evaluate_promotions([])
    assert all(not p.is_applied for p in promos)
    # Every promo starts at 0 progress on an empty cart.
    assert all(p.progress == 0.0 for p in promos)
    # And no $ discounts.
    assert all(p.discount_amount == 0.0 for p in promos)
    # All five built-in rules should be present.
    expected_codes = {
        "SPEND_50_SAVE_5",
        "SPEND_100_SAVE_15",
        "BUY_2_DAIRY",
        "BUY_3_PRODUCE",
        "FIRST_ORGANIC",
    }
    assert {p.code for p in promos} == expected_codes


def test_spend_50_threshold_applied() -> None:
    """A $60 cart triggers SPEND_50_SAVE_5 but not the $100 tier."""
    cart = [_item(price=20.0, qty=3)]  # $60 subtotal
    promos = evaluate_promotions(cart)
    s50 = _find(promos, "SPEND_50_SAVE_5")
    s100 = _find(promos, "SPEND_100_SAVE_15")
    assert s50.is_applied is True
    assert s50.discount_amount == 5.0
    assert s50.progress == 1.0
    assert s100.is_applied is False
    assert 0.0 < s100.progress < 1.0


def test_spend_100_both_thresholds_qualify() -> None:
    """A $110 cart triggers both spend-thresholds."""
    cart = [_item(price=22.0, qty=5)]  # $110 subtotal
    promos = evaluate_promotions(cart)
    s50 = _find(promos, "SPEND_50_SAVE_5")
    s100 = _find(promos, "SPEND_100_SAVE_15")
    assert s50.is_applied is True
    assert s100.is_applied is True
    assert s100.discount_amount == 15.0


def test_buy_3_produce_applied() -> None:
    """3 produce items → BUY_3_PRODUCE applied with 15% off produce."""
    cart = [
        _item(price=2.00, department="produce", product_name="Apple"),
        _item(price=3.00, department="produce", product_name="Banana"),
        _item(price=5.00, department="produce", product_name="Spinach"),
    ]
    promos = evaluate_promotions(cart)
    p3 = _find(promos, "BUY_3_PRODUCE")
    assert p3.is_applied is True
    # Produce subtotal $10 * 15% = $1.50
    assert p3.discount_amount == pytest.approx(1.50, abs=0.01)


def test_buy_3_produce_progress_when_not_qualified() -> None:
    """Only 2 produce items → progress 2/3, not applied."""
    cart = [
        _item(price=2.00, department="produce"),
        _item(price=3.00, department="produce"),
    ]
    promos = evaluate_promotions(cart)
    p3 = _find(promos, "BUY_3_PRODUCE")
    assert p3.is_applied is False
    assert p3.progress == pytest.approx(2.0 / 3.0, abs=0.01)


def test_buy_2_dairy_applied() -> None:
    """2 dairy items → 10% off dairy."""
    cart = [
        _item(price=4.00, department="dairy eggs", product_name="Milk"),
        _item(price=6.00, department="dairy eggs", product_name="Yogurt"),
    ]
    promos = evaluate_promotions(cart)
    d2 = _find(promos, "BUY_2_DAIRY")
    assert d2.is_applied is True
    # Dairy subtotal $10 * 10% = $1.00
    assert d2.discount_amount == pytest.approx(1.00, abs=0.01)


def test_first_organic_applied_by_attribute() -> None:
    """Organic flag on attributes is enough."""
    cart = [_item(price=3.0, attributes=["organic"])]
    promos = evaluate_promotions(cart)
    org = _find(promos, "FIRST_ORGANIC")
    assert org.is_applied is True
    assert org.discount_amount == 2.0


def test_first_organic_applied_by_name() -> None:
    """'Organic' in the product name also qualifies."""
    cart = [_item(price=3.0, product_name="Organic Apples")]
    promos = evaluate_promotions(cart)
    org = _find(promos, "FIRST_ORGANIC")
    assert org.is_applied is True


def test_first_organic_not_applied_without_signal() -> None:
    cart = [_item(price=3.0, product_name="Regular Apples")]
    promos = evaluate_promotions(cart)
    org = _find(promos, "FIRST_ORGANIC")
    assert org.is_applied is False
    assert org.discount_amount == 0.0


# ----------------------------------------------------------------------
# cart_pricing_summary
# ----------------------------------------------------------------------


def test_cart_pricing_summary_math_is_consistent() -> None:
    """subtotal - total_discount must equal total exactly."""
    cart = [
        _item(price=20.0, qty=3),  # $60 → triggers SPEND_50_SAVE_5
        _item(price=2.0, department="produce", product_name="Organic Apple"),
    ]
    summary = cart_pricing_summary(cart)
    assert summary["subtotal"] == pytest.approx(62.0, abs=0.01)
    expected_total = summary["subtotal"] - summary["total_discount"]
    assert summary["total"] == pytest.approx(expected_total, abs=0.01)


def test_cart_pricing_summary_empty() -> None:
    summary = cart_pricing_summary([])
    assert summary["subtotal"] == 0.0
    assert summary["total"] == 0.0
    assert summary["total_discount"] == 0.0
    assert summary["promotions_applied"] == []
    assert len(summary["promotions_available"]) >= 1
    assert summary["n_items"] == 0


def test_cart_pricing_summary_discount_never_exceeds_subtotal() -> None:
    """Tiny cart but every promo applied → total still >= 0."""
    # Construct a cart that qualifies for several promotions but has a
    # very small subtotal — but spend thresholds require >=$50 so the
    # only realistic "discount > subtotal" risk is FIRST_ORGANIC ($2).
    # We make a $1 cart of an organic item: subtotal $1, discount $2.
    cart = [_item(price=1.0, product_name="Organic Lemon",
                  department="produce")]
    summary = cart_pricing_summary(cart)
    assert summary["total"] >= 0
    assert summary["total_discount"] <= summary["subtotal"] + 1e-6


def test_cart_pricing_summary_promotions_split() -> None:
    """Applied vs. available promos are split correctly."""
    cart = [_item(price=20.0, qty=3)]  # $60
    summary = cart_pricing_summary(cart)
    applied_codes = {p.code for p in summary["promotions_applied"]}
    available_codes = {p.code for p in summary["promotions_available"]}
    # No overlap.
    assert applied_codes.isdisjoint(available_codes)
    # SPEND_50 qualified; SPEND_100 did not.
    assert "SPEND_50_SAVE_5" in applied_codes
    assert "SPEND_100_SAVE_15" in available_codes


def test_cart_pricing_summary_n_items_counts_qty() -> None:
    cart = [
        _item(price=5.0, qty=2),
        _item(price=3.0, qty=4),
    ]
    summary = cart_pricing_summary(cart)
    assert summary["n_items"] == 6


def test_promotion_to_dict_roundtrip() -> None:
    """to_dict should expose every field."""
    p = Promotion(
        code="X", title="T", description="D",
        discount_amount=1.0, is_applied=True, progress=1.0,
    )
    d = p.to_dict()
    assert d == {
        "code": "X", "title": "T", "description": "D",
        "discount_amount": 1.0, "is_applied": True, "progress": 1.0,
    }
