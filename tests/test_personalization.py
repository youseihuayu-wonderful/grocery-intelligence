"""Tests for the user personalization module.

Synthetic dataset:
    user 1 buys only dairy products (milk, yogurt)
    user 2 buys only snacks (chips, pretzels)
    user 3 buys a balanced mix
    user 4 has too few orders -> should be filtered out by min_orders=5

Each (user, product) pair is repeated across multiple orders so we can
verify both totals and the favorite_products ranking.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.recommend.personalization import (
    UserPersonalizationStore,
    UserProfile,
    compute_user_profiles,
    rerank_with_personalization,
)


PROD_MILK = 101
PROD_YOGURT = 102
PROD_CHIPS = 201
PROD_PRETZELS = 202
PROD_APPLE = 301


@pytest.fixture
def synthetic_catalog() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "product_id": PROD_MILK,
                "product_name": "Whole Milk",
                "category": "milk",
                "department": "dairy",
                "brand": "Horizon",
                "order_count": 10_000,
            },
            {
                "product_id": PROD_YOGURT,
                "product_name": "Greek Yogurt",
                "category": "yogurt",
                "department": "dairy",
                "brand": "Chobani",
                "order_count": 8_000,
            },
            {
                "product_id": PROD_CHIPS,
                "product_name": "Potato Chips",
                "category": "chips",
                "department": "snacks",
                "brand": "Lay's",
                "order_count": 7_000,
            },
            {
                "product_id": PROD_PRETZELS,
                "product_name": "Pretzels",
                "category": "chips",
                "department": "snacks",
                "brand": "Snyder's",
                "order_count": 3_000,
            },
            {
                "product_id": PROD_APPLE,
                "product_name": "Apple",
                "category": "fresh fruits",
                "department": "produce",
                "brand": None,
                "order_count": 20_000,
            },
        ]
    )


@pytest.fixture
def synthetic_orders() -> pd.DataFrame:
    """user 1: orders 1..6 (dairy only)
    user 2: orders 7..11 (snacks only)
    user 3: orders 12..16 (mix)
    user 4: orders 17..18 (only 2 orders -- filtered by min_orders=5)
    """
    rows = [
        (1, 1, "prior"),
        (2, 1, "prior"),
        (3, 1, "prior"),
        (4, 1, "prior"),
        (5, 1, "prior"),
        (6, 1, "prior"),
        (7, 2, "prior"),
        (8, 2, "prior"),
        (9, 2, "prior"),
        (10, 2, "prior"),
        (11, 2, "prior"),
        (12, 3, "prior"),
        (13, 3, "prior"),
        (14, 3, "prior"),
        (15, 3, "prior"),
        (16, 3, "prior"),
        (17, 4, "prior"),
        (18, 4, "prior"),
    ]
    return pd.DataFrame(rows, columns=["order_id", "user_id", "eval_set"])


@pytest.fixture
def synthetic_order_products() -> pd.DataFrame:
    rows = [
        # user 1 -- 8 milk + 4 yogurt = 12 items across 6 orders
        (1, PROD_MILK, 0),
        (1, PROD_YOGURT, 0),
        (2, PROD_MILK, 1),
        (2, PROD_YOGURT, 1),
        (3, PROD_MILK, 1),
        (3, PROD_YOGURT, 1),
        (4, PROD_MILK, 1),
        (4, PROD_YOGURT, 1),
        (5, PROD_MILK, 1),
        (5, PROD_MILK, 1),  # ignore duplicate for now
        (6, PROD_MILK, 1),
        (6, PROD_MILK, 1),
        # user 2 -- chips & pretzels
        (7, PROD_CHIPS, 0),
        (7, PROD_PRETZELS, 0),
        (8, PROD_CHIPS, 1),
        (8, PROD_PRETZELS, 1),
        (9, PROD_CHIPS, 1),
        (10, PROD_CHIPS, 1),
        (10, PROD_PRETZELS, 1),
        (11, PROD_CHIPS, 1),
        # user 3 -- balanced mix
        (12, PROD_MILK, 0),
        (12, PROD_CHIPS, 0),
        (13, PROD_MILK, 1),
        (13, PROD_CHIPS, 1),
        (14, PROD_YOGURT, 0),
        (14, PROD_PRETZELS, 0),
        (15, PROD_MILK, 1),
        (15, PROD_CHIPS, 1),
        (16, PROD_APPLE, 0),
        # user 4 -- below threshold
        (17, PROD_APPLE, 0),
        (18, PROD_APPLE, 1),
    ]
    return pd.DataFrame(
        rows, columns=["order_id", "product_id", "reordered"]
    )


# ---------------------------------------------------------------------------
# compute_user_profiles
# ---------------------------------------------------------------------------
def test_filters_users_below_min_orders(
    synthetic_orders, synthetic_order_products, synthetic_catalog
):
    profiles = compute_user_profiles(
        synthetic_orders,
        synthetic_order_products,
        synthetic_catalog,
        min_orders=5,
    )
    # user 4 only has 2 orders -> filtered out.
    assert set(profiles.keys()) == {1, 2, 3}


def test_user1_favorite_department_normalized(
    synthetic_orders, synthetic_order_products, synthetic_catalog
):
    profiles = compute_user_profiles(
        synthetic_orders,
        synthetic_order_products,
        synthetic_catalog,
        min_orders=5,
    )
    user1 = profiles[1]
    # User 1 only buys dairy -> dairy is the top label with score 1.0.
    assert user1.favorite_departments == {"dairy": 1.0}
    # Categories: milk and yogurt both present, milk should be top (more
    # purchases) and normalized to 1.0; yogurt < 1.0.
    assert "milk" in user1.favorite_categories
    assert "yogurt" in user1.favorite_categories
    assert user1.favorite_categories["milk"] == pytest.approx(1.0)
    assert 0.0 < user1.favorite_categories["yogurt"] < 1.0


def test_user2_favorite_department_is_snacks(
    synthetic_orders, synthetic_order_products, synthetic_catalog
):
    profiles = compute_user_profiles(
        synthetic_orders,
        synthetic_order_products,
        synthetic_catalog,
        min_orders=5,
    )
    user2 = profiles[2]
    assert user2.favorite_departments == {"snacks": 1.0}


def test_totals_and_basket_size(
    synthetic_orders, synthetic_order_products, synthetic_catalog
):
    profiles = compute_user_profiles(
        synthetic_orders,
        synthetic_order_products,
        synthetic_catalog,
        min_orders=5,
    )
    user1 = profiles[1]
    # 6 prior orders.
    assert user1.total_orders == 6
    # 12 items in user 1's history -> 12/6 = 2.0 avg basket size.
    assert user1.avg_basket_size == pytest.approx(12 / 6)


def test_favorite_products_milk_first(
    synthetic_orders, synthetic_order_products, synthetic_catalog
):
    profiles = compute_user_profiles(
        synthetic_orders,
        synthetic_order_products,
        synthetic_catalog,
        min_orders=5,
    )
    user1 = profiles[1]
    # Milk appears 8 times, yogurt 4. Milk should rank first.
    assert user1.favorite_products[0] == PROD_MILK
    assert PROD_YOGURT in user1.favorite_products


def test_favorite_products_capped_at_top_50(synthetic_catalog):
    """Synthetic catalog with 100 products -> only top 50 kept."""
    # Build a wider synthetic dataset: one user who bought 100 distinct
    # products, each appearing 1..100 times.
    catalog_rows = []
    op_rows = []
    user_id = 999
    for pid in range(1, 101):
        catalog_rows.append(
            {
                "product_id": pid,
                "category": "misc",
                "department": "misc",
                "brand": None,
            }
        )
        # Each product appears `pid` times across `pid` orders.
        for k in range(pid):
            op_rows.append((10_000 + pid * 200 + k, pid, 0))

    cat = pd.DataFrame(catalog_rows)
    op = pd.DataFrame(op_rows, columns=["order_id", "product_id", "reordered"])
    orders = pd.DataFrame(
        [(o, user_id, "prior") for (o, _, _) in op_rows],
        columns=["order_id", "user_id", "eval_set"],
    )

    profiles = compute_user_profiles(orders, op, cat, min_orders=5)
    assert user_id in profiles
    user = profiles[user_id]
    assert len(user.favorite_products) == 50
    # Top product should be the one with the most purchases (pid=100).
    assert user.favorite_products[0] == 100


# ---------------------------------------------------------------------------
# UserPersonalizationStore.score_for_user
# ---------------------------------------------------------------------------
def test_score_higher_for_aligned_product(
    synthetic_orders, synthetic_order_products, synthetic_catalog
):
    profiles = compute_user_profiles(
        synthetic_orders,
        synthetic_order_products,
        synthetic_catalog,
        min_orders=5,
    )
    store = UserPersonalizationStore(profiles)

    dairy_product = {
        "product_id": PROD_YOGURT,
        "category": "yogurt",
        "department": "dairy",
        "brand": "Chobani",
    }
    snack_product = {
        "product_id": PROD_CHIPS,
        "category": "chips",
        "department": "snacks",
        "brand": "Lay's",
    }

    dairy_score = store.score_for_user(1, dairy_product)
    snack_score = store.score_for_user(1, snack_product)
    # User 1 only buys dairy -> dairy product scores higher.
    assert dairy_score > snack_score
    # And scores are bounded [0, 1].
    assert 0.0 <= dairy_score <= 1.0
    assert 0.0 <= snack_score <= 1.0


def test_score_favorite_product_bonus(
    synthetic_orders, synthetic_order_products, synthetic_catalog
):
    profiles = compute_user_profiles(
        synthetic_orders,
        synthetic_order_products,
        synthetic_catalog,
        min_orders=5,
    )
    store = UserPersonalizationStore(profiles)
    # Milk is user 1's #1 favorite -> the favorite_products term (0.40)
    # plus the category/dept/brand terms should give a high score.
    milk_product = {
        "product_id": PROD_MILK,
        "category": "milk",
        "department": "dairy",
        "brand": "Horizon",
    }
    score = store.score_for_user(1, milk_product)
    # 0.40 favorite bonus + 0.30 * 1.0 cat + 0.20 * 1.0 dept + 0.10 * brand
    # The brand share is bounded by 1.0 so the total is at least 0.90.
    assert score >= 0.9


def test_score_cold_start_returns_zero(
    synthetic_orders, synthetic_order_products, synthetic_catalog
):
    profiles = compute_user_profiles(
        synthetic_orders,
        synthetic_order_products,
        synthetic_catalog,
        min_orders=5,
    )
    store = UserPersonalizationStore(profiles)
    score = store.score_for_user(99999, {"product_id": PROD_MILK, "category": "milk"})
    assert score == 0.0


def test_get_profile_returns_none_for_unknown(
    synthetic_orders, synthetic_order_products, synthetic_catalog
):
    profiles = compute_user_profiles(
        synthetic_orders,
        synthetic_order_products,
        synthetic_catalog,
        min_orders=5,
    )
    store = UserPersonalizationStore(profiles)
    assert store.get_profile(99999) is None
    assert store.get_profile(1) is not None


# ---------------------------------------------------------------------------
# rerank_with_personalization
# ---------------------------------------------------------------------------
def test_rerank_favors_user_preferences(
    synthetic_orders, synthetic_order_products, synthetic_catalog
):
    profiles = compute_user_profiles(
        synthetic_orders,
        synthetic_order_products,
        synthetic_catalog,
        min_orders=5,
    )
    store = UserPersonalizationStore(profiles)

    products = [
        {
            "product_id": PROD_CHIPS,
            "category": "chips",
            "department": "snacks",
            "brand": "Lay's",
            "order_count": 7_000,
            # Marginally higher base relevance.
            "relevance_score": 0.6,
        },
        {
            "product_id": PROD_MILK,
            "category": "milk",
            "department": "dairy",
            "brand": "Horizon",
            "order_count": 10_000,
            "relevance_score": 0.5,
        },
    ]
    ranked = rerank_with_personalization(
        products, user_id=1, store=store, alpha=0.5
    )
    # User 1 is dairy-only -> milk should jump to top despite lower
    # base relevance.
    assert ranked[0]["product_id"] == PROD_MILK
    assert ranked[1]["product_id"] == PROD_CHIPS
    # And each product picks up the new fields.
    assert "personalization_score" in ranked[0]
    assert "final_score" in ranked[0]


def test_rerank_cold_start_fallback():
    """user_id=None -> popularity + relevance only, no crash."""
    products = [
        {
            "product_id": PROD_MILK,
            "relevance_score": 0.3,
            "order_count": 10_000,
        },
        {
            "product_id": PROD_CHIPS,
            "relevance_score": 0.9,
            "order_count": 7_000,
        },
    ]
    ranked = rerank_with_personalization(
        products, user_id=None, store=None, alpha=0.3, popularity_weight=0.2
    )
    # Higher relevance + comparable popularity -> chips wins.
    assert ranked[0]["product_id"] == PROD_CHIPS
    for p in ranked:
        assert "personalization_score" in p
        assert p["personalization_score"] == 0.0
        assert "final_score" in p


def test_rerank_unknown_user_uses_fallback(
    synthetic_orders, synthetic_order_products, synthetic_catalog
):
    """user_id is set but not in store -> same fallback as anonymous."""
    profiles = compute_user_profiles(
        synthetic_orders,
        synthetic_order_products,
        synthetic_catalog,
        min_orders=5,
    )
    store = UserPersonalizationStore(profiles)
    products = [
        {"product_id": PROD_MILK, "relevance_score": 0.4, "order_count": 5_000},
        {"product_id": PROD_CHIPS, "relevance_score": 0.8, "order_count": 5_000},
    ]
    ranked = rerank_with_personalization(
        products, user_id=99999, store=store, alpha=0.3, popularity_weight=0.2
    )
    # Equal popularity, so higher relevance wins.
    assert ranked[0]["product_id"] == PROD_CHIPS


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def test_save_load_roundtrip(
    tmp_path, synthetic_orders, synthetic_order_products, synthetic_catalog
):
    profiles = compute_user_profiles(
        synthetic_orders,
        synthetic_order_products,
        synthetic_catalog,
        min_orders=5,
    )
    store = UserPersonalizationStore(profiles)

    out = tmp_path / "profiles.parquet"
    store.save(out)
    assert out.exists()

    loaded = UserPersonalizationStore.load(out)
    assert set(loaded.profiles.keys()) == set(store.profiles.keys())

    for uid, original in store.profiles.items():
        restored = loaded.profiles[uid]
        assert isinstance(restored, UserProfile)
        assert restored.user_id == original.user_id
        assert restored.total_orders == original.total_orders
        assert restored.avg_basket_size == pytest.approx(original.avg_basket_size)
        assert restored.favorite_products == original.favorite_products
        assert restored.favorite_categories == original.favorite_categories
        assert restored.favorite_departments == original.favorite_departments
        assert restored.favorite_brands == original.favorite_brands


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        UserPersonalizationStore.load(tmp_path / "nope.parquet")


# ---------------------------------------------------------------------------
# Demo users
# ---------------------------------------------------------------------------
def test_list_demo_users_spans_distribution(synthetic_catalog):
    """Build many users with varied order counts and check the demo list
    spans low-, mid-, and high-volume buckets.
    """
    # Build N=30 users with order counts 5..34.
    orders_rows = []
    op_rows = []
    next_order_id = 1
    for user_id in range(1, 31):
        num_orders = 5 + user_id  # 6..35
        for _ in range(num_orders):
            orders_rows.append((next_order_id, user_id, "prior"))
            # Each order has 1 milk item.
            op_rows.append((next_order_id, PROD_MILK, 0))
            next_order_id += 1
    orders = pd.DataFrame(orders_rows, columns=["order_id", "user_id", "eval_set"])
    op = pd.DataFrame(op_rows, columns=["order_id", "product_id", "reordered"])

    profiles = compute_user_profiles(orders, op, synthetic_catalog, min_orders=5)
    store = UserPersonalizationStore(profiles)

    demos = store.list_demo_users(n=5)
    assert len(demos) == 5

    counts = [d["total_orders"] for d in demos]
    # Spread across the distribution: min is from the low end, max from high.
    assert min(counts) <= 10  # one of the low-volume users
    assert max(counts) >= 30  # and one of the high-volume users
    # Sanity: counts should be non-decreasing because we sample sorted indices.
    assert counts == sorted(counts)
    # Each entry has the required shape.
    for d in demos:
        assert "user_id" in d
        assert "total_orders" in d
        assert "summary" in d
        assert isinstance(d["summary"], str)
        assert str(d["user_id"]) in d["summary"]


def test_list_demo_users_respects_n_larger_than_store(synthetic_catalog):
    """Asking for more demo users than exist returns all of them."""
    orders = pd.DataFrame(
        [
            (1, 1, "prior"),
            (2, 1, "prior"),
            (3, 1, "prior"),
            (4, 1, "prior"),
            (5, 1, "prior"),
        ],
        columns=["order_id", "user_id", "eval_set"],
    )
    op = pd.DataFrame(
        [(o, PROD_MILK, 0) for o in [1, 2, 3, 4, 5]],
        columns=["order_id", "product_id", "reordered"],
    )
    profiles = compute_user_profiles(orders, op, synthetic_catalog, min_orders=5)
    store = UserPersonalizationStore(profiles)
    demos = store.list_demo_users(n=50)
    assert len(demos) == 1
