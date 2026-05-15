"""Tests for the product emoji-icon module.

These cover both the priority logic (keyword > category > department >
fallback) and a sanity check against the real 49,688-row catalog so we know
the curated emoji tables actually achieve good coverage in production.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pytest

from src.recommend.images import (
    CATEGORY_ICONS,
    DEPARTMENT_ICONS,
    FALLBACK_EMOJI,
    KEYWORD_ICONS,
    build_emoji_map,
    get_emoji_for_product,
)

DATA_DIR = Path(__file__).parent.parent / "data"


# ---------------------------------------------------------------------------
# Unit tests on the priority ladder
# ---------------------------------------------------------------------------


def test_keyword_match_returns_banana():
    """Product-name keyword should be the highest-priority match."""
    emoji = get_emoji_for_product(
        {
            "product_name": "Bag of Organic Bananas",
            "category": "fresh fruits",
            "department": "produce",
        }
    )
    assert emoji == "🍌"


def test_keyword_beats_category_for_milk():
    """Even when category also has a match, the keyword path wins."""
    emoji = get_emoji_for_product(
        {
            "product_name": "Whole Milk",
            "category": "milk",
            "department": "dairy eggs",
        }
    )
    assert emoji == "🥛"


def test_category_falls_through_when_no_keyword():
    """If the name doesn't yield a keyword hit, category should fire."""
    emoji = get_emoji_for_product(
        {
            "product_name": "Some Branded Item",
            "category": "yogurt",
            "department": "dairy eggs",
        }
    )
    assert emoji == "🍦"


def test_department_fallback_for_generic_produce():
    """No keyword + no category hit -> department default."""
    emoji = get_emoji_for_product(
        {
            "product_name": "Generic",
            "category": "",
            "department": "produce",
        }
    )
    assert emoji == DEPARTMENT_ICONS["produce"]


def test_full_fallback_for_empty_input():
    """Empty input collapses to the box emoji, never None."""
    assert get_emoji_for_product({}) == FALLBACK_EMOJI


def test_full_fallback_for_all_missing_fields():
    """Explicit None / missing strings still resolve to the fallback."""
    emoji = get_emoji_for_product(
        {
            "product_name": None,
            "category": None,
            "department": "missing",
        }
    )
    # "missing" is a valid department key that maps to the box emoji.
    assert emoji == DEPARTMENT_ICONS["missing"]


def test_word_boundary_not_pineapple_for_apple():
    """The keyword matcher must use word boundaries so 'apple' doesn't fire
    on 'pineapple'."""
    emoji = get_emoji_for_product(
        {"product_name": "Pineapple Slices", "category": "", "department": ""}
    )
    assert emoji == "🍍"


def test_returns_string_never_none():
    """No input shape should ever produce None or an empty string."""
    for product in [
        {},
        {"product_name": ""},
        {"category": "totally-unknown"},
        {"department": "not-a-real-department"},
        {"product_name": "Frobnicator Widget"},
    ]:
        emoji = get_emoji_for_product(product)
        assert isinstance(emoji, str) and emoji != ""


# ---------------------------------------------------------------------------
# Required emoji tables
# ---------------------------------------------------------------------------


def test_department_icons_cover_expected_departments():
    """The icon table must include every department the catalog uses, so
    nothing relies on the box fallback because we forgot a department."""
    required = {
        "produce",
        "dairy eggs",
        "frozen",
        "beverages",
        "snacks",
        "pantry",
        "deli",
        "bakery",
        "meat seafood",
        "alcohol",
        "household",
        "personal care",
        "babies",
        "pets",
        "breakfast",
        "canned goods",
        "dry goods pasta",
        "international",
        "missing",
    }
    missing = required - set(DEPARTMENT_ICONS.keys())
    assert not missing, f"Departments missing from DEPARTMENT_ICONS: {missing}"


def test_category_icons_has_at_least_30_entries():
    """Spec calls for 30+ specific category overrides."""
    assert len(CATEGORY_ICONS) >= 30


def test_keyword_icons_covers_common_produce():
    """A handful of obvious produce keywords must be present."""
    expected = {"banana", "apple", "strawberry", "blueberry", "avocado", "broccoli"}
    missing = expected - set(KEYWORD_ICONS.keys())
    assert not missing, f"Missing keyword icons: {missing}"


# ---------------------------------------------------------------------------
# build_emoji_map on a tiny synthetic catalog
# ---------------------------------------------------------------------------


def test_build_emoji_map_synthetic():
    """One emoji per row, keyed by product_id, all non-empty."""
    df = pd.DataFrame(
        [
            {
                "product_id": 1,
                "product_name": "Banana",
                "category": "fresh fruits",
                "department": "produce",
            },
            {
                "product_id": 2,
                "product_name": "Whole Milk",
                "category": "milk",
                "department": "dairy eggs",
            },
            {
                "product_id": 3,
                "product_name": "Mystery Item",
                "category": "unknown",
                "department": "produce",
            },
            {
                "product_id": 4,
                "product_name": "",
                "category": "",
                "department": "",
            },
        ]
    )
    result = build_emoji_map(df)
    assert set(result.keys()) == {1, 2, 3, 4}
    assert result[1] == "🍌"
    assert result[2] == "🥛"
    assert result[3] == DEPARTMENT_ICONS["produce"]
    assert result[4] == FALLBACK_EMOJI


def test_build_emoji_map_empty_dataframe():
    """Empty in, empty out — no exceptions."""
    df = pd.DataFrame(
        columns=["product_id", "product_name", "category", "department"]
    )
    assert build_emoji_map(df) == {}


# ---------------------------------------------------------------------------
# Sanity check on the real catalog
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_catalog() -> pd.DataFrame:
    path = DATA_DIR / "processed" / "product_catalog.parquet"
    if not path.exists():
        pytest.skip("Product catalog not found. Run data pipeline first.")
    return pd.read_parquet(path)


def test_real_catalog_runs_under_5s_and_70pct_coverage(real_catalog):
    """End-to-end: full 50k catalog should map in < 5s, and at least 70% of
    products should get a more-specific emoji than the generic 📦 box."""
    t0 = time.perf_counter()
    emoji_map = build_emoji_map(real_catalog)
    elapsed = time.perf_counter() - t0

    assert elapsed < 5.0, f"build_emoji_map took {elapsed:.2f}s (>= 5s budget)"
    assert len(emoji_map) == len(real_catalog)

    specific = sum(1 for e in emoji_map.values() if e != FALLBACK_EMOJI)
    coverage = specific / len(emoji_map)
    assert coverage >= 0.70, (
        f"Only {coverage:.1%} of products got a specific emoji "
        f"(need >= 70%)."
    )
