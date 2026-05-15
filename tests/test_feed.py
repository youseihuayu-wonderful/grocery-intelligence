"""Tests for the feed-based discovery module.

We use a small synthetic catalog so the unit tests stay fast. One
sanity test loads the real catalog + real user store to make sure the
module works end-to-end against production data.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.recommend.feed import (
    FEED_TYPES,
    get_bestsellers,
    get_for_you,
    get_healthy_picks,
    get_trending_in_department,
    list_departments,
)
from src.recommend.personalization import (
    UserPersonalizationStore,
    UserProfile,
)


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def tiny_catalog() -> pd.DataFrame:
    """Six-product synthetic catalog covering all the assertion paths.

    Departments: produce (3), dairy (2), snacks (1).
    Nutrition grades cover a/b (healthy) and c/d/e/NaN.
    Order counts are unique so sort ties never make assertions flaky.
    """
    return pd.DataFrame(
        [
            {
                "product_id": 1,
                "product_name": "Banana",
                "category": "fresh fruits",
                "department": "produce",
                "brand": "Dole",
                "nutrition_grade": "a",
                "order_count": 5000,
            },
            {
                "product_id": 2,
                "product_name": "Apple",
                "category": "fresh fruits",
                "department": "produce",
                "brand": "Gala",
                "nutrition_grade": "a",
                "order_count": 4000,
            },
            {
                "product_id": 3,
                "product_name": "Lettuce",
                "category": "fresh vegetables",
                "department": "produce",
                "brand": "Earthbound",
                "nutrition_grade": "b",
                "order_count": 1500,
            },
            {
                "product_id": 4,
                "product_name": "Whole Milk",
                "category": "milk",
                "department": "dairy",
                "brand": "Horizon",
                "nutrition_grade": "c",
                "order_count": 6000,
            },
            {
                "product_id": 5,
                "product_name": "Greek Yogurt",
                "category": "yogurt",
                "department": "dairy",
                "brand": "Chobani",
                "nutrition_grade": "b",
                "order_count": 3000,
            },
            {
                "product_id": 6,
                "product_name": "Potato Chips",
                "category": "chips",
                "department": "snacks",
                "brand": "Lay's",
                "nutrition_grade": "e",
                "order_count": 8000,
            },
        ]
    )


@pytest.fixture
def store_with_dairy_user() -> UserPersonalizationStore:
    """User 42 favors dairy + milk; user 7 favors snacks + chips."""
    profiles = {
        42: UserProfile(
            user_id=42,
            total_orders=20,
            avg_basket_size=5.0,
            favorite_products=[4],  # already loves whole milk
            favorite_categories={"milk": 1.0, "yogurt": 0.8},
            favorite_departments={"dairy": 1.0},
            favorite_brands={"Horizon": 1.0, "Chobani": 0.5},
        ),
        7: UserProfile(
            user_id=7,
            total_orders=15,
            avg_basket_size=4.0,
            favorite_products=[6],  # already loves chips
            favorite_categories={"chips": 1.0},
            favorite_departments={"snacks": 1.0},
            favorite_brands={"Lay's": 1.0},
        ),
    }
    return UserPersonalizationStore(profiles)


# ---------------------------------------------------------------------------
# get_bestsellers
# ---------------------------------------------------------------------------
def test_bestsellers_sorted_by_order_count(tiny_catalog: pd.DataFrame) -> None:
    out = get_bestsellers(tiny_catalog, top_k=3)
    assert len(out) == 3
    # 8000 chips, 6000 milk, 5000 banana
    assert [p["product_id"] for p in out] == [6, 4, 1]
    assert [p["feed_score"] for p in out] == [8000, 6000, 5000]


def test_bestsellers_top_k_caps_at_catalog_size(tiny_catalog: pd.DataFrame) -> None:
    out = get_bestsellers(tiny_catalog, top_k=100)
    assert len(out) == 6  # there are only 6 products


def test_bestsellers_empty_catalog_does_not_crash() -> None:
    empty = pd.DataFrame(columns=["product_id", "order_count"])
    assert get_bestsellers(empty, top_k=5) == []


# ---------------------------------------------------------------------------
# get_healthy_picks
# ---------------------------------------------------------------------------
def test_healthy_picks_filters_to_a_and_b(tiny_catalog: pd.DataFrame) -> None:
    out = get_healthy_picks(tiny_catalog, top_k=10)
    grades = {p["nutrition_grade"] for p in out}
    assert grades.issubset({"a", "b"})
    # Should include all four a/b products in the synthetic set.
    assert {p["product_id"] for p in out} == {1, 2, 3, 5}


def test_healthy_picks_sorted_by_popularity_within_healthy(
    tiny_catalog: pd.DataFrame,
) -> None:
    out = get_healthy_picks(tiny_catalog, top_k=10)
    # Banana(5000) > Apple(4000) > Yogurt(3000) > Lettuce(1500).
    assert [p["product_id"] for p in out] == [1, 2, 5, 3]
    assert [p["feed_score"] for p in out] == [5000, 4000, 3000, 1500]


def test_healthy_picks_empty_catalog() -> None:
    empty = pd.DataFrame(columns=["product_id", "nutrition_grade", "order_count"])
    assert get_healthy_picks(empty, top_k=5) == []


def test_healthy_picks_missing_grade_column_returns_empty() -> None:
    no_grade = pd.DataFrame(
        [{"product_id": 1, "product_name": "x", "order_count": 5}]
    )
    assert get_healthy_picks(no_grade, top_k=5) == []


# ---------------------------------------------------------------------------
# get_trending_in_department
# ---------------------------------------------------------------------------
def test_trending_in_department_only_returns_products_in_dept(
    tiny_catalog: pd.DataFrame,
) -> None:
    out = get_trending_in_department(tiny_catalog, "produce", top_k=10)
    assert {p["department"] for p in out} == {"produce"}
    # produce: banana(5000), apple(4000), lettuce(1500)
    assert [p["product_id"] for p in out] == [1, 2, 3]


def test_trending_in_department_case_insensitive(
    tiny_catalog: pd.DataFrame,
) -> None:
    upper = get_trending_in_department(tiny_catalog, "DAIRY", top_k=10)
    lower = get_trending_in_department(tiny_catalog, "dairy", top_k=10)
    assert [p["product_id"] for p in upper] == [p["product_id"] for p in lower]


def test_trending_in_department_unknown_dept(tiny_catalog: pd.DataFrame) -> None:
    assert get_trending_in_department(tiny_catalog, "nonexistent", top_k=5) == []


def test_trending_in_department_empty_catalog() -> None:
    empty = pd.DataFrame(columns=["product_id", "department", "order_count"])
    assert get_trending_in_department(empty, "produce", top_k=5) == []


def test_trending_in_department_blank_dept(tiny_catalog: pd.DataFrame) -> None:
    assert get_trending_in_department(tiny_catalog, "", top_k=5) == []
    assert get_trending_in_department(tiny_catalog, "   ", top_k=5) == []


# ---------------------------------------------------------------------------
# get_for_you
# ---------------------------------------------------------------------------
def test_for_you_unknown_user_falls_back_to_bestsellers(
    tiny_catalog: pd.DataFrame,
    store_with_dairy_user: UserPersonalizationStore,
) -> None:
    fallback = get_for_you(tiny_catalog, store_with_dairy_user, user_id=999, top_k=3)
    expected = get_bestsellers(tiny_catalog, top_k=3)
    assert [p["product_id"] for p in fallback] == [p["product_id"] for p in expected]
    assert [p["feed_score"] for p in fallback] == [p["feed_score"] for p in expected]


def test_for_you_none_user_falls_back(
    tiny_catalog: pd.DataFrame,
    store_with_dairy_user: UserPersonalizationStore,
) -> None:
    fallback = get_for_you(tiny_catalog, store_with_dairy_user, user_id=None, top_k=2)
    assert len(fallback) == 2
    assert fallback[0]["product_id"] == 6  # chips, the most popular


def test_for_you_excludes_user_favorites(
    tiny_catalog: pd.DataFrame,
    store_with_dairy_user: UserPersonalizationStore,
) -> None:
    out = get_for_you(tiny_catalog, store_with_dairy_user, user_id=42, top_k=10)
    pids = {p["product_id"] for p in out}
    # Product 4 (whole milk) is in user 42's favorite_products list.
    assert 4 not in pids
    # The remaining five products should all be present.
    assert pids == {1, 2, 3, 5, 6}


def test_for_you_sorted_by_feed_score_desc(
    tiny_catalog: pd.DataFrame,
    store_with_dairy_user: UserPersonalizationStore,
) -> None:
    out = get_for_you(tiny_catalog, store_with_dairy_user, user_id=42, top_k=10)
    scores = [p["feed_score"] for p in out]
    assert scores == sorted(scores, reverse=True)
    # User 42 loves dairy/yogurt -- Greek Yogurt should beat snacks/chips
    # once we exclude their existing favorite (milk).
    top_pid = out[0]["product_id"]
    assert top_pid == 5  # Greek Yogurt


def test_for_you_personalization_differentiates_users(
    tiny_catalog: pd.DataFrame,
    store_with_dairy_user: UserPersonalizationStore,
) -> None:
    """Two users with different affinities should get different #1 items."""
    dairy_user = get_for_you(tiny_catalog, store_with_dairy_user, 42, top_k=5)
    # User 7 has only one favorite (chips, pid 6) so excluding it means
    # the next-best non-favorite item should still reflect their snacks
    # preference -- nothing in the synthetic catalog matches snacks
    # except chips, so user 7 should rely heavily on popularity. The
    # ordered lists themselves should differ.
    snack_user = get_for_you(tiny_catalog, store_with_dairy_user, 7, top_k=5)
    assert [p["product_id"] for p in dairy_user] != [
        p["product_id"] for p in snack_user
    ]


def test_for_you_empty_catalog(
    store_with_dairy_user: UserPersonalizationStore,
) -> None:
    empty = pd.DataFrame(columns=["product_id", "order_count"])
    assert get_for_you(empty, store_with_dairy_user, user_id=42, top_k=5) == []


# ---------------------------------------------------------------------------
# list_departments
# ---------------------------------------------------------------------------
def test_list_departments_sorted_by_popularity(tiny_catalog: pd.DataFrame) -> None:
    out = list_departments(tiny_catalog)
    # produce total = 5000+4000+1500 = 10500
    # dairy total  = 6000+3000 = 9000
    # snacks total = 8000
    assert out == ["produce", "dairy", "snacks"]


def test_list_departments_empty_catalog() -> None:
    empty = pd.DataFrame(columns=["product_id", "department", "order_count"])
    assert list_departments(empty) == []


def test_list_departments_drops_nan_and_blank() -> None:
    cat = pd.DataFrame(
        [
            {"product_id": 1, "department": "produce", "order_count": 10},
            {"product_id": 2, "department": None, "order_count": 20},
            {"product_id": 3, "department": "   ", "order_count": 30},
            {"product_id": 4, "department": "dairy", "order_count": 5},
        ]
    )
    out = list_departments(cat)
    assert out == ["produce", "dairy"]


# ---------------------------------------------------------------------------
# Feed registry
# ---------------------------------------------------------------------------
def test_feed_types_registry_has_expected_keys() -> None:
    assert set(FEED_TYPES.keys()) == {"bestsellers", "healthy-picks", "for-you"}


# ---------------------------------------------------------------------------
# Sanity test against the real production data
# ---------------------------------------------------------------------------
REAL_CATALOG = Path("data/processed/product_catalog.parquet")
REAL_PROFILES = Path("data/processed/user_profiles.parquet")


@pytest.mark.skipif(
    not REAL_CATALOG.exists() or not REAL_PROFILES.exists(),
    reason="real catalog / profiles parquet missing",
)
def test_for_you_against_real_data_returns_sorted_top_k() -> None:
    catalog = pd.read_parquet(REAL_CATALOG)
    store = UserPersonalizationStore.load(REAL_PROFILES)
    demo_users = store.list_demo_users(20)
    assert demo_users, "expected at least one demo user from the real store"

    uid = demo_users[-1]["user_id"]  # heaviest buyer in the spaced sample
    feed = get_for_you(catalog, store, user_id=uid, top_k=20)
    assert len(feed) == 20
    scores = [p["feed_score"] for p in feed]
    assert scores == sorted(scores, reverse=True)
    # Every item must have a personalization score field too.
    assert all("personalization_score" in p for p in feed)
