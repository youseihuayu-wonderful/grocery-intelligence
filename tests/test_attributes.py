"""Tests for the automatic attribute extraction module.

Mixes synthetic edge-case products (to pin down each rule precisely) with a
real-catalog sanity check that guards against silent regressions in the
rule set.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.search.attributes import (
    ALL_ATTRIBUTES,
    ATTRIBUTE_LABELS,
    extract_attributes,
    extract_attributes_bulk,
)

DATA_DIR = Path(__file__).parent.parent / "data"


# ---------------------------------------------------------------------------
# Module-level invariants
# ---------------------------------------------------------------------------


def test_all_attributes_is_sorted_and_matches_labels():
    """ALL_ATTRIBUTES should be the sorted key-set of ATTRIBUTE_LABELS."""
    assert ALL_ATTRIBUTES == sorted(ATTRIBUTE_LABELS.keys())
    assert len(ALL_ATTRIBUTES) == len(set(ALL_ATTRIBUTES))


def test_every_label_is_a_non_empty_string():
    for attr_id, label in ATTRIBUTE_LABELS.items():
        assert isinstance(label, str) and label.strip(), (
            f"Label for {attr_id} is empty or non-string: {label!r}"
        )


# ---------------------------------------------------------------------------
# Direct keyword extraction
# ---------------------------------------------------------------------------


def test_organic_in_name():
    product = {"product_name": "Organic Almond Milk", "ingredients": None}
    attrs = extract_attributes(product)
    assert "organic" in attrs


def test_almond_milk_is_not_automatically_dairy_free():
    """Spec is strict: dairy-free needs an explicit keyword. A product whose
    name only says "Almond Milk" must NOT receive the dairy-free tag."""
    product = {"product_name": "Organic Almond Milk", "ingredients": None}
    attrs = extract_attributes(product)
    assert "dairy-free" not in attrs


def test_explicit_non_dairy_keyword_emits_dairy_free():
    product = {
        "product_name": "Non-Dairy Coconut Yogurt",
        "ingredients": "coconut cream, sugar",
    }
    attrs = extract_attributes(product)
    assert "dairy-free" in attrs


def test_keyword_match_in_ingredients_field():
    """Keywords in either name OR ingredients should fire — here only the
    ingredients string carries the signal."""
    product = {
        "product_name": "Trail Mix",
        "ingredients": "Vegan dark chocolate, organic raisins, organic oats",
    }
    attrs = extract_attributes(product)
    assert "vegan" in attrs
    assert "organic" in attrs


def test_gluten_free_with_hyphen_dash_and_gf_token():
    # "gf " with trailing space pattern requires the haystack padding
    # implemented in extract_attributes; this confirms it works.
    cases = [
        {"product_name": "Gluten Free Bread", "ingredients": None},
        {"product_name": "Gluten-Free Bread", "ingredients": None},
        {"product_name": "GF Crackers", "ingredients": None},
    ]
    for product in cases:
        assert "gluten-free" in extract_attributes(product), (
            f"Failed for product: {product!r}"
        )


def test_unsweetened_emits_sugar_free():
    product = {
        "product_name": "Robust Golden Unsweetened Oolong Tea",
        "ingredients": None,
    }
    assert "sugar-free" in extract_attributes(product)


# ---------------------------------------------------------------------------
# Nutrition threshold rules
# ---------------------------------------------------------------------------


def test_high_protein_threshold():
    product = {"product_name": "Whey Powder", "protein_100g": 20.0}
    assert "high-protein" in extract_attributes(product)


def test_high_protein_boundary_inclusive():
    """`>= 15` — exactly 15 must qualify."""
    product = {"product_name": "Whey Powder", "protein_100g": 15.0}
    assert "high-protein" in extract_attributes(product)


def test_low_sugar_boundary_inclusive():
    """`<= 5` — exactly 5 must qualify."""
    product = {"product_name": "Cracker", "sugar_100g": 5.0}
    assert "low-sugar" in extract_attributes(product)


def test_threshold_does_not_fire_when_field_missing():
    """The spec mandates that threshold rules only emit when the relevant
    field is present. A None protein should yield no high-protein tag even
    if the product name happens to suggest one."""
    product = {"product_name": "Protein Bar", "protein_100g": None}
    assert "high-protein" not in extract_attributes(product)


def test_low_calorie_and_low_fat_emit_together():
    product = {
        "product_name": "Lite Yogurt",
        "calories_100g": 80.0,
        "fat_100g": 2.0,
    }
    attrs = extract_attributes(product)
    assert "low-calorie" in attrs
    assert "low-fat" in attrs


# ---------------------------------------------------------------------------
# Natural-language nutrition intent parsing
# ---------------------------------------------------------------------------


def test_parse_intent_detects_multiple_constraints():
    from src.search.attributes import parse_nutrition_intent
    assert parse_nutrition_intent("high protein low sugar breakfast") == [
        "high-protein", "low-sugar"
    ]


def test_parse_intent_detects_dietary_terms():
    from src.search.attributes import parse_nutrition_intent
    assert parse_nutrition_intent("vegan gluten free snacks") == [
        "gluten-free", "vegan"
    ]


def test_parse_intent_is_conservative_on_bare_nouns():
    from src.search.attributes import parse_nutrition_intent
    # A bare "sugar" or "cheap milk" must NOT trigger a nutrition filter.
    assert parse_nutrition_intent("sugar") == []
    assert parse_nutrition_intent("cheap milk") == []


def test_parse_intent_empty_query():
    from src.search.attributes import parse_nutrition_intent
    assert parse_nutrition_intent("") == []


def test_parse_intent_only_returns_known_attributes():
    from src.search.attributes import parse_nutrition_intent, ALL_ATTRIBUTES
    detected = parse_nutrition_intent(
        "organic low calorie high fiber keto vegan dairy free"
    )
    assert set(detected).issubset(set(ALL_ATTRIBUTES))


# ---------------------------------------------------------------------------
# Allergen-derived attributes intentionally removed
# ---------------------------------------------------------------------------


def test_no_allergen_derived_attributes():
    """The Instacart catalog has no populated allergens_en, so no allergen-based
    attribute (e.g. the former 'nut-free') may be emitted — asserting one from
    sparse data would be an unsafe false claim. Even with allergen text present,
    extract_attributes must not invent a nut-free claim."""
    product = {
        "product_name": "Cheese Crackers",
        "allergens_en": "en:milk,en:gluten",
    }
    assert "nut-free" not in extract_attributes(product)


# ---------------------------------------------------------------------------
# NaN safety
# ---------------------------------------------------------------------------


def test_all_none_returns_empty_list():
    product = {
        "product_name": None,
        "ingredients": None,
        "calories_100g": None,
        "protein_100g": None,
        "sugar_100g": None,
        "fat_100g": None,
        "fiber_100g": None,
        "nutrition_grade": None,
        "allergens_en": None,
    }
    assert extract_attributes(product) == []


def test_all_nan_returns_empty_list():
    """Same as the None case but with floating-point NaN — the dominant
    representation when reading from parquet/pandas."""
    nan = float("nan")
    product = {
        "product_name": nan,
        "ingredients": nan,
        "calories_100g": nan,
        "protein_100g": nan,
        "sugar_100g": nan,
        "fat_100g": nan,
        "fiber_100g": nan,
        "nutrition_grade": nan,
        "allergens_en": nan,
    }
    assert extract_attributes(product) == []


def test_returns_sorted_list_with_no_duplicates():
    product = {
        "product_name": "Organic Vegan Gluten-Free Bar",
        "ingredients": "Organic oats, organic almonds",
        "protein_100g": 20.0,
    }
    attrs = extract_attributes(product)
    assert attrs == sorted(attrs)
    assert len(attrs) == len(set(attrs))
    # spot-check the obvious ones
    for expected in ("organic", "vegan", "gluten-free", "high-protein"):
        assert expected in attrs


# ---------------------------------------------------------------------------
# Bulk extraction
# ---------------------------------------------------------------------------


def test_bulk_returns_one_entry_per_row():
    df = pd.DataFrame(
        {
            "product_id": [1, 2, 3],
            "product_name": ["Organic Apples", "Plain Cheese", "Vegan Burger"],
            "ingredients": [None, None, "soy protein, organic oats"],
            "protein_100g": [np.nan, np.nan, 22.0],
            "sugar_100g": [np.nan, np.nan, np.nan],
            "calories_100g": [np.nan, np.nan, np.nan],
            "fat_100g": [np.nan, np.nan, np.nan],
            "fiber_100g": [np.nan, np.nan, np.nan],
            "nutrition_grade": [None, None, None],
            "allergens_en": [None, None, None],
        }
    )
    out = extract_attributes_bulk(df)

    assert isinstance(out, dict)
    assert set(out.keys()) == {1, 2, 3}
    assert "organic" in out[1]
    assert out[2] == []  # "Plain Cheese" has no keywords and no nutrition
    assert "vegan" in out[3]
    assert "organic" in out[3]
    assert "high-protein" in out[3]


def test_bulk_empty_dataframe():
    df = pd.DataFrame(
        columns=["product_id", "product_name", "ingredients", "protein_100g"]
    )
    assert extract_attributes_bulk(df) == {}


# ---------------------------------------------------------------------------
# Real-catalog sanity check
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def catalog() -> pd.DataFrame:
    """Load the real product catalog (skipping if it isn't built yet)."""
    path = DATA_DIR / "processed" / "product_catalog.parquet"
    if not path.exists():
        pytest.skip("Product catalog not found. Run data pipeline first.")
    return pd.read_parquet(path)


def test_real_catalog_yields_attributes_for_many_products(catalog):
    """At least ~5,000 products in the real catalog should have ≥1 attribute.
    If this regresses sharply, either the rule set lost a major keyword or
    the underlying data shape changed."""
    out = extract_attributes_bulk(catalog)
    with_any = sum(1 for attrs in out.values() if attrs)
    assert with_any >= 5_000, (
        f"Only {with_any} products got any attribute (expected >= 5,000). "
        "Did the rule set or the catalog schema change?"
    )
