"""Tests for category-level browsing."""

from __future__ import annotations

import pandas as pd
import pytest

from src.search.category_browse import (
    CATEGORY_KEYWORDS,
    browse_category,
    is_category_query,
)


def _fake_catalog() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"product_id": 1, "product_name": "Banana", "department": "produce", "category": "fresh fruits", "order_count": 1000},
            {"product_id": 2, "product_name": "Apple", "department": "produce", "category": "fresh fruits", "order_count": 800},
            {"product_id": 3, "product_name": "Strawberry", "department": "produce", "category": "fresh fruits", "order_count": 700},
            {"product_id": 4, "product_name": "Lemon", "department": "produce", "category": "fresh fruits", "order_count": 600},
            {"product_id": 5, "product_name": "Blueberry pack", "department": "produce", "category": "packaged vegetables fruits", "order_count": 500},
            {"product_id": 6, "product_name": "Carrot", "department": "produce", "category": "fresh vegetables", "order_count": 900},
            {"product_id": 7, "product_name": "Spinach", "department": "produce", "category": "fresh vegetables", "order_count": 850},
            {"product_id": 8, "product_name": "Broccoli", "department": "produce", "category": "fresh vegetables", "order_count": 750},
            {"product_id": 9, "product_name": "Cola", "department": "beverages", "category": "soft drinks", "order_count": 1200},
            {"product_id": 10, "product_name": "Sparkling Water", "department": "beverages", "category": "water seltzer sparkling water", "order_count": 1100},
            {"product_id": 11, "product_name": "Orange Juice", "department": "beverages", "category": "juice nectars", "order_count": 950},
            {"product_id": 12, "product_name": "Chicken Breast", "department": "meat seafood", "category": "packaged poultry", "order_count": 600},
            {"product_id": 13, "product_name": "Ground Beef", "department": "meat seafood", "category": "meat counter", "order_count": 550},
        ]
    )


class TestIsCategoryQuery:
    def test_known_keyword_singular(self):
        assert is_category_query("fruit") is True

    def test_known_keyword_plural(self):
        assert is_category_query("fruits") is True

    def test_unknown_keyword(self):
        assert is_category_query("banana") is False

    def test_case_and_whitespace_insensitive(self):
        assert is_category_query("  Fruit  ") is True
        assert is_category_query("VEGETABLES") is True

    def test_all_listed_keywords_resolve(self):
        for kw in CATEGORY_KEYWORDS:
            assert is_category_query(kw) is True


class TestBrowseCategory:
    def test_fruit_returns_only_fruits(self):
        catalog = _fake_catalog()
        results = browse_category(catalog, "fruit", top_k=5)
        assert len(results) > 0
        for r in results:
            assert r["category"] in ("fresh fruits", "packaged vegetables fruits")

    def test_unknown_query_returns_empty(self):
        catalog = _fake_catalog()
        assert browse_category(catalog, "computers", top_k=5) == []

    def test_sorted_by_order_count(self):
        catalog = _fake_catalog()
        # for fruits the top by order_count is Banana (1000)
        results = browse_category(catalog, "fruit", top_k=3)
        assert results[0]["product_name"] == "Banana"

    def test_diversification_cap_respected_when_top_k_small_enough(self):
        catalog = _fake_catalog()
        # top_k=3, cap=2 -> 2 from fresh fruits + 1 from packaged = 3, no relaxation needed
        results = browse_category(catalog, "fruit", top_k=3, max_per_subcategory=2)
        from_fresh = [r for r in results if r["category"] == "fresh fruits"]
        assert len(from_fresh) <= 2

    def test_relaxes_cap_when_under_top_k(self):
        catalog = _fake_catalog()
        # only 'fresh fruits' has 4 items, 'packaged vegetables fruits' has 1
        # with cap=2, first pass yields 3 (2 fresh + 1 packaged); ask for 5 -> should fill more
        results = browse_category(catalog, "fruit", top_k=5, max_per_subcategory=2)
        assert len(results) == 5

    def test_drinks_diversified(self):
        catalog = _fake_catalog()
        results = browse_category(catalog, "drinks", top_k=3)
        cats = {r["category"] for r in results}
        # 3 different drink sub-categories in fake catalog
        assert len(cats) >= 2

    def test_meat_routes_to_meat_seafood_department(self):
        catalog = _fake_catalog()
        results = browse_category(catalog, "meat", top_k=10)
        for r in results:
            assert r["department"] == "meat seafood"
