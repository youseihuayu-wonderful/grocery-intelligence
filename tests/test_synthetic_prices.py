"""Tests for :mod:`src.pricing.synthetic_prices`."""

from __future__ import annotations

import math
import time
from pathlib import Path

import pandas as pd
import pytest

from src.pricing.synthetic_prices import (
    BASE_PRICES,
    DEFAULT_BASE_PRICE,
    attach_prices,
    build_price_map,
    generate_synthetic_price,
)


# ----------------------------------------------------------------------
# Helpers / fixtures
# ----------------------------------------------------------------------


REAL_CATALOG = Path("data/processed/product_catalog.parquet")


def _make_product(**overrides) -> dict:
    """Tiny factory for a plausible catalog row."""
    base = {
        "product_id": 1,
        "product_name": "Plain Yogurt",
        "department": "dairy eggs",
        "nutrition_grade": "b",
        "order_count": 100,
    }
    base.update(overrides)
    return base


# ----------------------------------------------------------------------
# generate_synthetic_price — single product
# ----------------------------------------------------------------------


def test_generate_synthetic_price_is_deterministic() -> None:
    """Same input must yield exactly the same output."""
    product = _make_product()
    p1 = generate_synthetic_price(product)
    p2 = generate_synthetic_price(product)
    p3 = generate_synthetic_price(dict(product))
    assert p1 == p2 == p3


def test_generate_synthetic_price_different_departments() -> None:
    """Different department → different base price → different output."""
    produce = generate_synthetic_price(_make_product(department="produce", order_count=0))
    meat = generate_synthetic_price(
        _make_product(department="meat seafood", order_count=0)
    )
    alcohol = generate_synthetic_price(
        _make_product(department="alcohol", order_count=0)
    )
    assert produce < meat < alcohol


def test_generate_synthetic_price_organic_bump() -> None:
    """The word 'organic' in the name should raise the price ~30%."""
    base = generate_synthetic_price(
        _make_product(
            product_name="Banana", department="produce",
            order_count=0, nutrition_grade="",
        )
    )
    organic = generate_synthetic_price(
        _make_product(
            product_name="Organic Banana", department="produce",
            order_count=0, nutrition_grade="",
        )
    )
    assert organic > base
    # 1.30x means the organic version should be at least ~25% higher.
    assert organic / base >= 1.25


def test_generate_synthetic_price_nutrition_grade_a_higher() -> None:
    """Grade A items get a 10% bump."""
    a_grade = generate_synthetic_price(
        _make_product(nutrition_grade="a", order_count=0, product_name="X")
    )
    no_grade = generate_synthetic_price(
        _make_product(nutrition_grade="", order_count=0, product_name="X")
    )
    assert a_grade > no_grade


def test_generate_synthetic_price_nutrition_grade_low_lower() -> None:
    """Grade D / E get a 10% reduction."""
    for low in ("d", "e"):
        low_grade = generate_synthetic_price(
            _make_product(nutrition_grade=low, order_count=0, product_name="X")
        )
        no_grade = generate_synthetic_price(
            _make_product(nutrition_grade="", order_count=0, product_name="X")
        )
        assert low_grade < no_grade


def test_generate_synthetic_price_minimum_floor() -> None:
    """Even on a (theoretical) zero base, we shouldn't drop below $0.99."""
    # 'unknown' department + grade E + no popularity = the lowest realistic combo.
    price = generate_synthetic_price(
        {
            "product_name": "x",
            "department": "no_such_department_at_all",
            "nutrition_grade": "e",
            "order_count": 0,
        }
    )
    assert price >= 0.99


def test_generate_synthetic_price_handles_missing_fields() -> None:
    """Robust to missing / None / NaN values."""
    p = generate_synthetic_price({})
    assert p >= 0.99
    p2 = generate_synthetic_price(
        {"product_name": None, "department": None, "nutrition_grade": None,
         "order_count": None}
    )
    assert p2 >= 0.99
    p3 = generate_synthetic_price(
        {"product_name": float("nan"), "department": "produce",
         "nutrition_grade": float("nan"), "order_count": float("nan")}
    )
    assert p3 >= 0.99


def test_generate_synthetic_price_popularity_boost_is_capped() -> None:
    """Popular items get a boost, but it caps out at +$5."""
    quiet = generate_synthetic_price(
        _make_product(department="pantry", order_count=1, nutrition_grade="")
    )
    busy = generate_synthetic_price(
        _make_product(department="pantry", order_count=10**9, nutrition_grade="")
    )
    # Cap is $5; pantry base $4. Multiplier 1.0, so max is ~$9 strictly.
    assert busy <= quiet + 5.001
    assert busy > quiet


# ----------------------------------------------------------------------
# build_price_map — bulk catalog
# ----------------------------------------------------------------------


def test_build_price_map_synthetic_catalog() -> None:
    """End-to-end on a tiny in-memory DataFrame."""
    df = pd.DataFrame(
        [
            {"product_id": 1, "product_name": "Organic Apple",
             "department": "produce", "nutrition_grade": "a", "order_count": 500},
            {"product_id": 2, "product_name": "Regular Apple",
             "department": "produce", "nutrition_grade": "b", "order_count": 500},
            {"product_id": 3, "product_name": "Wild Salmon",
             "department": "meat seafood", "nutrition_grade": "a",
             "order_count": 100},
            {"product_id": 4, "product_name": "Generic Item",
             "department": "totally unknown", "nutrition_grade": "",
             "order_count": 0},
        ]
    )
    pmap = build_price_map(df)
    assert set(pmap.keys()) == {1, 2, 3, 4}
    # Every entry is a positive float.
    assert all(isinstance(v, float) and v >= 0.99 for v in pmap.values())
    # Organic is more expensive than its non-organic twin.
    assert pmap[1] > pmap[2]


def test_build_price_map_matches_per_row() -> None:
    """The bulk path must agree with the per-row function."""
    df = pd.DataFrame(
        [
            {"product_id": 10, "product_name": "Organic Spinach",
             "department": "produce", "nutrition_grade": "a", "order_count": 50},
            {"product_id": 11, "product_name": "Wild Tuna",
             "department": "meat seafood", "nutrition_grade": "d",
             "order_count": 999_999},
            {"product_id": 12, "product_name": "Cola",
             "department": "beverages", "nutrition_grade": "e",
             "order_count": 0},
        ]
    )
    pmap = build_price_map(df)
    for _, row in df.iterrows():
        expected = generate_synthetic_price(row.to_dict())
        # Round both sides; FP differences can show up in the last
        # decimal between the per-row and vectorized paths.
        assert pmap[int(row["product_id"])] == pytest.approx(expected, abs=0.01)


def test_build_price_map_empty_input() -> None:
    """Empty catalog yields empty map (no crash)."""
    assert build_price_map(pd.DataFrame()) == {}


@pytest.mark.skipif(
    not REAL_CATALOG.exists(),
    reason="Real catalog parquet not present; skipping perf sanity test.",
)
def test_build_price_map_real_catalog_perf() -> None:
    """All 49,688 real products must price in under 3 seconds."""
    df = pd.read_parquet(REAL_CATALOG)
    t0 = time.perf_counter()
    pmap = build_price_map(df)
    elapsed = time.perf_counter() - t0
    assert len(pmap) == len(df)
    # Every product got a sensible price.
    assert all(v >= 0.99 for v in pmap.values())
    assert elapsed < 3.0, f"build_price_map took {elapsed:.2f}s (>3.0s)"


# ----------------------------------------------------------------------
# attach_prices — mutates dicts in place
# ----------------------------------------------------------------------


def test_attach_prices_mutates_list() -> None:
    products = [
        {"product_id": 1, "product_name": "Apple"},
        {"product_id": 2, "product_name": "Banana"},
    ]
    price_map = {1: 1.50, 2: 0.75}
    result = attach_prices(products, price_map)
    # Same list returned (not a copy).
    assert result is products
    assert products[0]["price"] == 1.50
    assert products[1]["price"] == 0.75


def test_attach_prices_skips_unknown_ids() -> None:
    products = [
        {"product_id": 1},
        {"product_id": 99},  # not in the map
    ]
    attach_prices(products, {1: 2.0})
    assert products[0]["price"] == 2.0
    assert "price" not in products[1]


def test_attach_prices_empty_inputs() -> None:
    assert attach_prices([], {1: 2.0}) == []
    items = [{"product_id": 1}]
    assert attach_prices(items, {}) is items
    assert "price" not in items[0]


# ----------------------------------------------------------------------
# BASE_PRICES sanity
# ----------------------------------------------------------------------


def test_base_prices_table_has_expected_departments() -> None:
    """Spot-check the table against the spec."""
    expected = {
        "produce": 2.50,
        "dairy eggs": 3.50,
        "meat seafood": 9.00,
        "bakery": 4.00,
        "snacks": 3.00,
        "beverages": 3.50,
        "frozen": 5.00,
        "pantry": 4.00,
        "household": 6.00,
        "babies": 12.00,
        "personal care": 7.00,
        "alcohol": 14.00,
    }
    for dept, expected_price in expected.items():
        assert BASE_PRICES[dept] == expected_price
    assert DEFAULT_BASE_PRICE == 4.00
