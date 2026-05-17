"""Category-level browsing.

When the user types a high-level category keyword like "fruit",
"vegetable", "meat", "drinks", etc., the regular semantic search
returns ~5 similar products (e.g. 5 different brands of carrots).
Users expect a diverse mix instead — a few apples, some bananas,
some strawberries, etc. This module detects category queries and
returns a diversified result set, picking from across the
sub-categories in the catalog.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd


# Map from user-typed keyword (lowercased) to the catalog `department`s
# and `category`s that count as that keyword. Both lists can be empty;
# a product matches if it belongs to ANY of the listed departments
# OR ANY of the listed categories.
CATEGORY_KEYWORDS: dict[str, dict[str, list[str]]] = {
    # Fruits & vegetables
    "fruit":      {"departments": [], "categories": ["fresh fruits", "packaged vegetables fruits"]},
    "fruits":     {"departments": [], "categories": ["fresh fruits", "packaged vegetables fruits"]},
    "vegetable":  {"departments": [], "categories": ["fresh vegetables", "packaged vegetables fruits", "frozen produce"]},
    "vegetables": {"departments": [], "categories": ["fresh vegetables", "packaged vegetables fruits", "frozen produce"]},
    "veggie":     {"departments": [], "categories": ["fresh vegetables", "packaged vegetables fruits", "frozen produce"]},
    "veggies":    {"departments": [], "categories": ["fresh vegetables", "packaged vegetables fruits", "frozen produce"]},
    "produce":    {"departments": ["produce"], "categories": []},

    # Meat & seafood
    "meat":     {"departments": ["meat seafood"], "categories": []},
    "seafood":  {"departments": [], "categories": ["seafood counter", "packaged seafood"]},
    "fish":     {"departments": [], "categories": ["seafood counter", "packaged seafood"]},
    "poultry":  {"departments": [], "categories": ["poultry counter", "packaged poultry"]},

    # Dairy & eggs
    "dairy":   {"departments": ["dairy eggs"], "categories": []},

    # Beverages & alcohol
    "drink":      {"departments": ["beverages"], "categories": []},
    "drinks":     {"departments": ["beverages"], "categories": []},
    "beverage":   {"departments": ["beverages"], "categories": []},
    "beverages":  {"departments": ["beverages"], "categories": []},
    "alcohol":    {"departments": ["alcohol"], "categories": []},
    "liquor":     {"departments": ["alcohol"], "categories": []},

    # Snacks & sweets
    "snack":      {"departments": ["snacks"], "categories": []},
    "snacks":     {"departments": ["snacks"], "categories": []},
    "dessert":    {"departments": [], "categories": ["frozen dessert", "ice cream ice", "cookies cakes"]},
    "desserts":   {"departments": [], "categories": ["frozen dessert", "ice cream ice", "cookies cakes"]},
    "sweet":      {"departments": [], "categories": ["candy chocolate", "cookies cakes", "ice cream ice"]},
    "sweets":     {"departments": [], "categories": ["candy chocolate", "cookies cakes", "ice cream ice"]},

    # Bakery & grains
    "bakery":   {"departments": ["bakery"], "categories": []},
    "grain":    {"departments": [], "categories": ["grains rice dried goods", "cereal"]},
    "grains":   {"departments": [], "categories": ["grains rice dried goods", "cereal"]},

    # Other departments
    "frozen":     {"departments": ["frozen"], "categories": []},
    "breakfast":  {"departments": ["breakfast"], "categories": []},
    "pantry":     {"departments": ["pantry"], "categories": []},
    "deli":       {"departments": ["deli"], "categories": []},
    "household":  {"departments": ["household"], "categories": []},
    "baby":       {"departments": ["babies"], "categories": []},
    "babies":     {"departments": ["babies"], "categories": []},
    "pets":       {"departments": ["pets"], "categories": []},
    "pet":        {"departments": ["pets"], "categories": []},
}


def is_category_query(query: str) -> bool:
    """Return True if the query is a high-level category keyword."""
    return query.strip().lower() in CATEGORY_KEYWORDS


def browse_category(
    catalog: pd.DataFrame,
    query: str,
    top_k: int = 15,
    max_per_subcategory: int = 3,
) -> list[dict]:
    """Return a diversified list of top products in the requested category.

    Algorithm:
        1. Find products matching ANY of the mapped departments OR categories.
        2. Sort by order_count descending.
        3. Greedily pick products, but cap at `max_per_subcategory` per
           `category` so the result spans many sub-categories instead of
           returning 10 brands of one item.

    Returns an empty list if the query isn't a known category keyword
    or the catalog has no matching products.
    """
    q = query.strip().lower()
    if q not in CATEGORY_KEYWORDS:
        return []

    mapping = CATEGORY_KEYWORDS[q]
    departments = set(mapping.get("departments", []))
    categories = set(mapping.get("categories", []))

    mask = pd.Series([False] * len(catalog), index=catalog.index)
    if departments:
        mask = mask | catalog["department"].isin(departments)
    if categories:
        mask = mask | catalog["category"].isin(categories)

    filtered = catalog[mask]
    if filtered.empty:
        return []

    filtered = filtered.sort_values("order_count", ascending=False, na_position="last")
    filtered_rows = filtered.to_dict("records")

    # Two-pass: first do diversified pick with the cap, then if we
    # still don't have enough, fill the remainder from the same
    # candidate pool (relaxing the cap).
    picked: list[dict] = []
    picked_ids: set = set()
    per_subcat: dict[str, int] = {}
    for row in filtered_rows:
        subcat = str(row.get("category", "unknown") or "unknown")
        if per_subcat.get(subcat, 0) >= max_per_subcategory:
            continue
        picked.append(row)
        picked_ids.add(row.get("product_id"))
        per_subcat[subcat] = per_subcat.get(subcat, 0) + 1
        if len(picked) >= top_k:
            break

    if len(picked) < top_k:
        for row in filtered_rows:
            if row.get("product_id") in picked_ids:
                continue
            picked.append(row)
            if len(picked) >= top_k:
                break

    return picked
