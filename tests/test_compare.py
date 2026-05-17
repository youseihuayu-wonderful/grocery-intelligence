"""Tests for the side-by-side product comparison module."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.recommend.compare import COMPARE_ATTRS, compare_products


# ---------------------------------------------------------------------------
# Synthetic catalog fixture
# ---------------------------------------------------------------------------
@pytest.fixture
def catalog() -> pd.DataFrame:
    """Three products carefully chosen so each attribute has a clear winner.

    Product 1: high protein/fiber, low sugar/fat/calories, grade 'a',
               popular, high reorder.   -> winner on most nutrition axes.
    Product 2: middling everywhere, grade 'c'.
    Product 3: opposite of P1 -- low protein/fiber, high sugar/fat/cal,
               grade 'e', unpopular.    -> loser on most axes.
    """
    return pd.DataFrame([
        {
            "product_id": 1,
            "product_name": "Healthy Granola",
            "brand": "BrandX",
            "category": "cereal",
            "department": "breakfast",
            "calories_100g": 100.0,
            "protein_100g": 20.0,
            "sugar_100g": 2.0,
            "fat_100g": 3.0,
            "fiber_100g": 10.0,
            "nutrition_grade": "a",
            "order_count": 1000,
            "reorder_rate": 0.8,
        },
        {
            "product_id": 2,
            "product_name": "Plain Granola",
            "brand": "BrandY",
            "category": "cereal",
            "department": "breakfast",
            "calories_100g": 200.0,
            "protein_100g": 10.0,
            "sugar_100g": 10.0,
            "fat_100g": 8.0,
            "fiber_100g": 5.0,
            "nutrition_grade": "c",
            "order_count": 500,
            "reorder_rate": 0.5,
        },
        {
            "product_id": 3,
            "product_name": "Sugar Bombs",
            "brand": "BrandZ",
            "category": "cereal",
            "department": "breakfast",
            "calories_100g": 400.0,
            "protein_100g": 3.0,
            "sugar_100g": 40.0,
            "fat_100g": 20.0,
            "fiber_100g": 1.0,
            "nutrition_grade": "e",
            "order_count": 100,
            "reorder_rate": 0.2,
        },
    ])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _attr(result: dict, key: str) -> dict:
    """Pluck a single attribute row out of compare_products' output."""
    for a in result["attributes"]:
        if a["key"] == key:
            return a
    raise KeyError(key)


# ---------------------------------------------------------------------------
# 1. Basic shape & winner detection
# ---------------------------------------------------------------------------
def test_compare_two_products_winners(catalog: pd.DataFrame) -> None:
    out = compare_products(catalog, [1, 3])
    assert out["product_ids"] == [1, 3]
    assert len(out["products"]) == 2

    # Lower-is-better wins for product 1.
    for k in ("calories_100g", "sugar_100g", "fat_100g"):
        attr = _attr(out, k)
        assert attr["best_index"] == 0, f"{k} should pick index 0"
        assert attr["lower_is_better"] is True

    # Higher-is-better wins for product 1.
    for k in ("protein_100g", "fiber_100g", "order_count", "reorder_rate"):
        attr = _attr(out, k)
        assert attr["best_index"] == 0
        assert attr["lower_is_better"] is False

    # Nutrition grade: 'a' beats 'e' -> index 0.
    grade = _attr(out, "nutrition_grade")
    assert grade["best_index"] == 0
    assert grade["lower_is_better"] is False


def test_compare_three_products_picks_middle_or_best(catalog: pd.DataFrame) -> None:
    out = compare_products(catalog, [3, 2, 1])  # reversed order
    assert out["product_ids"] == [3, 2, 1]
    # Best calories (lowest) -> product 1, which is at index 2.
    assert _attr(out, "calories_100g")["best_index"] == 2
    # Best protein -> product 1 at index 2.
    assert _attr(out, "protein_100g")["best_index"] == 2
    # Best nutrition grade -> product 1 ('a') at index 2.
    assert _attr(out, "nutrition_grade")["best_index"] == 2


# ---------------------------------------------------------------------------
# 2. Text fields: no winner ever
# ---------------------------------------------------------------------------
def test_text_fields_have_no_winner(catalog: pd.DataFrame) -> None:
    out = compare_products(catalog, [1, 2, 3])
    for key in ("product_name", "brand", "category", "department"):
        a = _attr(out, key)
        assert a["best_index"] is None, f"{key} should have no winner"
        assert a["lower_is_better"] is False


# ---------------------------------------------------------------------------
# 3. Nutrition grade ordering a > b > c > d > e
# ---------------------------------------------------------------------------
def test_nutrition_grade_ranking() -> None:
    rows = []
    grades = ["e", "d", "c", "b", "a"]
    for i, g in enumerate(grades, start=1):
        rows.append({
            "product_id": i,
            "product_name": f"P{i}",
            "brand": "X",
            "category": "c",
            "department": "d",
            "calories_100g": float("nan"),
            "protein_100g": float("nan"),
            "sugar_100g": float("nan"),
            "fat_100g": float("nan"),
            "fiber_100g": float("nan"),
            "nutrition_grade": g,
            "order_count": 0,
            "reorder_rate": 0.0,
        })
    cat = pd.DataFrame(rows)

    # Compare 'e' (idx 0), 'a' (idx 4): winner -> idx 4.
    out = compare_products(cat, [1, 5])
    assert _attr(out, "nutrition_grade")["best_index"] == 1

    # Compare 'b' (idx 3), 'c' (idx 2): winner -> idx 0 (the 'b').
    out = compare_products(cat, [4, 3])
    assert _attr(out, "nutrition_grade")["best_index"] == 0

    # Compare 'd' (idx 1), 'e' (idx 0): winner -> idx 1 (the 'd').
    out = compare_products(cat, [2, 1])
    assert _attr(out, "nutrition_grade")["best_index"] == 0


# ---------------------------------------------------------------------------
# 4. NaN handling
# ---------------------------------------------------------------------------
def test_nan_does_not_break(catalog: pd.DataFrame) -> None:
    """A NaN value should be emitted as None and excluded from the winner search."""
    cat = catalog.copy()
    # Drop product 1's calories — now only products 2 and 3 have it.
    cat.loc[cat["product_id"] == 1, "calories_100g"] = float("nan")
    out = compare_products(cat, [1, 2, 3])

    cal = _attr(out, "calories_100g")
    # The first value is None (was NaN); the winner is among 2 and 3.
    assert cal["values"][0] is None
    # Calories: lower is better. Product 2 (200) < Product 3 (400).
    assert cal["best_index"] == 1


def test_single_non_missing_value_means_no_winner(catalog: pd.DataFrame) -> None:
    """If only ONE product has a value, no winner can be declared."""
    cat = catalog.copy()
    cat.loc[cat["product_id"] != 1, "fiber_100g"] = float("nan")
    out = compare_products(cat, [1, 2])
    fiber = _attr(out, "fiber_100g")
    # Product 2's fiber is NaN -> None in output.
    assert fiber["values"][1] is None
    # Only one non-NaN value -> no winner.
    assert fiber["best_index"] is None


def test_all_nan_means_no_winner(catalog: pd.DataFrame) -> None:
    cat = catalog.copy()
    cat["fiber_100g"] = float("nan")
    out = compare_products(cat, [1, 2, 3])
    fiber = _attr(out, "fiber_100g")
    assert all(v is None for v in fiber["values"])
    assert fiber["best_index"] is None


def test_all_equal_means_no_winner(catalog: pd.DataFrame) -> None:
    cat = catalog.copy()
    cat["sugar_100g"] = 5.0
    out = compare_products(cat, [1, 2, 3])
    sugar = _attr(out, "sugar_100g")
    assert sugar["best_index"] is None


# ---------------------------------------------------------------------------
# 5. Input validation
# ---------------------------------------------------------------------------
def test_raises_for_fewer_than_two_products(catalog: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="at least 2"):
        compare_products(catalog, [1])
    with pytest.raises(ValueError, match="at least 2"):
        compare_products(catalog, [])


def test_raises_for_more_than_six_products(catalog: pd.DataFrame) -> None:
    # Make a catalog with 7 products so we can construct a too-long list.
    extra = catalog.copy()
    for pid in range(4, 8):
        extra = pd.concat([
            extra,
            pd.DataFrame([{
                "product_id": pid,
                "product_name": f"Extra {pid}",
                "brand": "X",
                "category": "c",
                "department": "d",
                "calories_100g": 100.0,
                "protein_100g": 1.0,
                "sugar_100g": 1.0,
                "fat_100g": 1.0,
                "fiber_100g": 1.0,
                "nutrition_grade": "a",
                "order_count": 1,
                "reorder_rate": 0.1,
            }]),
        ], ignore_index=True)
    with pytest.raises(ValueError, match="at most 6"):
        compare_products(extra, [1, 2, 3, 4, 5, 6, 7])


def test_raises_for_unknown_product_id(catalog: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="not found"):
        compare_products(catalog, [1, 99_999])


# ---------------------------------------------------------------------------
# 6. Output shape
# ---------------------------------------------------------------------------
def test_all_compare_attrs_in_output(catalog: pd.DataFrame) -> None:
    out = compare_products(catalog, [1, 2])
    keys = {a["key"] for a in out["attributes"]}
    assert set(COMPARE_ATTRS).issubset(keys)
    # Per-product records also expose every attribute.
    for product in out["products"]:
        for attr in COMPARE_ATTRS:
            assert attr in product


def test_output_preserves_input_order(catalog: pd.DataFrame) -> None:
    out = compare_products(catalog, [3, 1, 2])
    assert out["product_ids"] == [3, 1, 2]
    assert [p["product_id"] for p in out["products"]] == [3, 1, 2]
    # The 'values' lists must align with that order.
    grade = _attr(out, "nutrition_grade")
    assert grade["values"] == ["e", "a", "c"]


def test_label_is_human_readable(catalog: pd.DataFrame) -> None:
    out = compare_products(catalog, [1, 2])
    cal = _attr(out, "calories_100g")
    assert isinstance(cal["label"], str)
    assert "Calories" in cal["label"]
