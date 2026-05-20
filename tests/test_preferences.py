"""Tests for the dietary-preferences persistence layer.

Covers :class:`src.users.preferences.PreferenceStore` (round-trip,
clearing, listing) and the search-filter helper
:func:`src.users.preferences.apply_preferences_to_search`.

All SQLite-backed tests use the ``tmp_path`` fixture for an isolated
DB file per test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.users.preferences import (
    PreferenceStore,
    apply_preferences_to_search,
)


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def store(tmp_path: Path) -> PreferenceStore:
    """A fresh PreferenceStore backed by a per-test temp SQLite file."""
    db_path = tmp_path / "prefs.db"
    s = PreferenceStore(db_path)
    yield s
    s.close()


# ======================================================================
# PreferenceStore — round-trip & clearing
# ======================================================================


def test_set_and_get_preferences_round_trip(store: PreferenceStore) -> None:
    store.set_preferences(
        user_id=42,
        dietary_attributes=["organic", "low-sugar"],
        excluded_attributes=["high-sugar"],
    )
    got = store.get_preferences(user_id=42)
    assert got == {
        "dietary_attributes": ["organic", "low-sugar"],
        "excluded_attributes": ["high-sugar"],
    }


def test_get_preferences_unknown_user_returns_empty(
    store: PreferenceStore,
) -> None:
    got = store.get_preferences(user_id=99999)
    assert got == {"dietary_attributes": [], "excluded_attributes": []}


def test_set_preferences_overwrites(store: PreferenceStore) -> None:
    """Calling set_preferences again replaces (not merges) the old list."""
    store.set_preferences(
        user_id=1,
        dietary_attributes=["organic"],
        excluded_attributes=[],
    )
    store.set_preferences(
        user_id=1,
        dietary_attributes=["vegan", "gluten-free"],
        excluded_attributes=["high-sugar"],
    )
    got = store.get_preferences(user_id=1)
    assert got["dietary_attributes"] == ["vegan", "gluten-free"]
    assert got["excluded_attributes"] == ["high-sugar"]


def test_set_preferences_none_clears_existing(store: PreferenceStore) -> None:
    """Passing None for both lists (or empty lists) clears the row."""
    store.set_preferences(
        user_id=1,
        dietary_attributes=["organic"],
        excluded_attributes=["high-sugar"],
    )
    assert store.get_preferences(1)["dietary_attributes"] == ["organic"]
    # None clears.
    store.set_preferences(user_id=1, dietary_attributes=None, excluded_attributes=None)
    assert store.get_preferences(1) == {
        "dietary_attributes": [],
        "excluded_attributes": [],
    }
    # And the user no longer counts as having preferences.
    assert 1 not in store.list_users_with_prefs()


def test_set_preferences_empty_lists_clears(store: PreferenceStore) -> None:
    store.set_preferences(
        user_id=7,
        dietary_attributes=["vegan"],
        excluded_attributes=[],
    )
    assert store.get_preferences(7)["dietary_attributes"] == ["vegan"]
    store.set_preferences(user_id=7, dietary_attributes=[], excluded_attributes=[])
    assert store.get_preferences(7) == {
        "dietary_attributes": [],
        "excluded_attributes": [],
    }


def test_set_preferences_normalizes_input(store: PreferenceStore) -> None:
    """Whitespace and duplicates are cleaned up before storage."""
    store.set_preferences(
        user_id=3,
        dietary_attributes=["  organic ", "organic", "vegan", ""],
        excluded_attributes=["high-sugar", "high-sugar"],
    )
    got = store.get_preferences(3)
    assert got["dietary_attributes"] == ["organic", "vegan"]
    assert got["excluded_attributes"] == ["high-sugar"]


def test_list_users_with_prefs(store: PreferenceStore) -> None:
    """list_users_with_prefs returns sorted user_ids with non-default prefs."""
    assert store.list_users_with_prefs() == []
    store.set_preferences(7, ["organic"], [])
    store.set_preferences(2, ["vegan"], [])
    store.set_preferences(11, [], ["high-sugar"])
    assert store.list_users_with_prefs() == [2, 7, 11]

    # Clearing removes from the list.
    store.set_preferences(2, [], [])
    assert store.list_users_with_prefs() == [7, 11]


def test_context_manager_persists(tmp_path: Path) -> None:
    db_path = tmp_path / "ctx.db"
    with PreferenceStore(db_path) as s:
        s.set_preferences(5, ["organic"], ["high-sugar"])
    # After context exit, reopen and read back.
    with PreferenceStore(db_path) as s2:
        assert s2.get_preferences(5) == {
            "dietary_attributes": ["organic"],
            "excluded_attributes": ["high-sugar"],
        }


# ======================================================================
# apply_preferences_to_search
# ======================================================================


def _make_products() -> list[dict]:
    """A small synthetic catalog with explicit ``attributes`` fields."""
    return [
        {
            "product_id": 1,
            "product_name": "Organic Greek Yogurt",
            "attributes": ["organic", "high-protein"],
        },
        {
            "product_id": 2,
            "product_name": "Regular Yogurt",
            "attributes": ["high-sugar"],
        },
        {
            "product_id": 3,
            "product_name": "Organic Vegan Yogurt",
            "attributes": ["organic", "vegan", "dairy-free"],
        },
        {
            "product_id": 4,
            "product_name": "Yogurt without attributes",
            # No 'attributes' key.
        },
        {
            "product_id": 5,
            "product_name": "Empty attributes",
            "attributes": [],
        },
    ]


def test_apply_required_attributes_passes_when_all_present() -> None:
    products = _make_products()
    out = apply_preferences_to_search(
        products,
        {"dietary_attributes": ["organic"], "excluded_attributes": []},
    )
    ids = [p["product_id"] for p in out]
    # Products 1 and 3 have 'organic'. Others don't (or no attrs at all).
    assert ids == [1, 3]


def test_apply_required_attributes_filters_when_missing() -> None:
    products = _make_products()
    out = apply_preferences_to_search(
        products,
        {
            "dietary_attributes": ["organic", "vegan"],
            "excluded_attributes": [],
        },
    )
    ids = [p["product_id"] for p in out]
    # Only product 3 has BOTH organic AND vegan.
    assert ids == [3]


def test_apply_excluded_attributes_filters() -> None:
    products = _make_products()
    out = apply_preferences_to_search(
        products,
        {
            "dietary_attributes": [],
            "excluded_attributes": ["high-sugar"],
        },
    )
    ids = [p["product_id"] for p in out]
    # Product 2 is dropped (has 'high-sugar'). Everything else stays
    # because excluded is the only constraint and others don't match it.
    assert 2 not in ids
    assert set(ids) == {1, 3, 4, 5}


def test_apply_required_and_excluded_combined() -> None:
    products = _make_products()
    out = apply_preferences_to_search(
        products,
        {
            "dietary_attributes": ["organic"],
            "excluded_attributes": ["vegan"],
        },
    )
    ids = [p["product_id"] for p in out]
    # Need organic AND not vegan: product 1 stays, product 3 dropped.
    assert ids == [1]


def test_apply_empty_preferences_returns_all() -> None:
    products = _make_products()
    out = apply_preferences_to_search(
        products,
        {"dietary_attributes": [], "excluded_attributes": []},
    )
    ids = [p["product_id"] for p in out]
    assert ids == [1, 2, 3, 4, 5]


def test_apply_missing_attributes_field_filters_when_required_nonempty() -> None:
    """Product with no 'attributes' key cannot satisfy a required attribute."""
    products = _make_products()
    out = apply_preferences_to_search(
        products,
        {"dietary_attributes": ["organic"], "excluded_attributes": []},
    )
    ids = [p["product_id"] for p in out]
    # Product 4 has no 'attributes' field at all -> filtered out.
    assert 4 not in ids
    # Product 5 has empty attributes -> also filtered out (doesn't have 'organic').
    assert 5 not in ids


def test_apply_missing_attributes_field_passes_when_no_constraints() -> None:
    """With no constraints at all, missing 'attributes' field is fine."""
    products = _make_products()
    out = apply_preferences_to_search(
        products,
        {"dietary_attributes": [], "excluded_attributes": []},
    )
    # Product 4 (no attributes) survives.
    assert any(p["product_id"] == 4 for p in out)
    # Product 5 (empty attributes) also survives.
    assert any(p["product_id"] == 5 for p in out)


def test_apply_preserves_order() -> None:
    products = _make_products()
    out = apply_preferences_to_search(
        products,
        {"dietary_attributes": ["organic"], "excluded_attributes": []},
    )
    # Original order preserved (1 comes before 3).
    assert [p["product_id"] for p in out] == [1, 3]


def test_apply_handles_unusual_inputs() -> None:
    """Non-dict preferences or malformed attributes are tolerated."""
    products = _make_products()
    # Non-dict preferences -> no filtering applied (returns input).
    assert apply_preferences_to_search(products, None) == products  # type: ignore[arg-type]
    # Missing keys in preferences dict treated as empty lists.
    out = apply_preferences_to_search(products, {})
    assert [p["product_id"] for p in out] == [1, 2, 3, 4, 5]


def test_round_trip_with_filter(store: PreferenceStore) -> None:
    """End-to-end: store prefs, fetch, filter a product list."""
    store.set_preferences(
        user_id=42,
        dietary_attributes=["organic"],
        excluded_attributes=["high-sugar"],
    )
    prefs = store.get_preferences(42)
    products = _make_products()
    out = apply_preferences_to_search(products, prefs)
    # Need 'organic' AND not 'high-sugar': products 1 and 3 survive.
    assert [p["product_id"] for p in out] == [1, 3]
