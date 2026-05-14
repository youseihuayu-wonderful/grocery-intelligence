"""Algorithmic product badge computation.

Computes Amazon-style algorithmic badges ("Bestseller", "Customer Favorite",
"Healthy Choice", etc.) for every product in the catalog. Thresholds for
popularity-based badges are derived from the actual data distribution so the
module adapts as the catalog grows.
"""

from __future__ import annotations

import pandas as pd

# Human-readable labels for each badge identifier. The frontend / API can use
# these directly for display.
BADGE_LABELS: dict[str, str] = {
    "bestseller": "🏆 Bestseller",
    "popular": "🔥 Popular",
    "customer-favorite": "❤️ Customer Favorite",
    "healthy-choice": "🥗 Healthy Choice",
    "high-protein": "💪 High Protein",
    "low-sugar": "🍯 Low Sugar",
    "low-calorie": "⚖️ Low Calorie",
    "high-fiber": "🌾 High Fiber",
}

# Static nutrition thresholds. Popularity thresholds are computed per-catalog.
_HIGH_PROTEIN_G = 15.0
_LOW_SUGAR_G = 5.0
_LOW_CALORIE_KCAL = 100.0
_HIGH_FIBER_G = 6.0
_CUSTOMER_FAVORITE_REORDER_RATE = 0.75
_CUSTOMER_FAVORITE_MIN_ORDERS = 30
_HEALTHY_GRADES = {"a", "b"}


def compute_badges(catalog: pd.DataFrame) -> dict[int, list[str]]:
    """Compute algorithmic badges for every product in the catalog.

    Args:
        catalog: Product catalog with columns:
            product_id, product_name, category, department, brand,
            calories_100g, protein_100g, sugar_100g, fat_100g, fiber_100g,
            nutrition_grade, order_count, reorder_rate

    Returns:
        Dict mapping product_id -> list of badge string identifiers, e.g.:
            42 -> ["bestseller", "customer-favorite"]
            7  -> ["healthy-choice"]
            12 -> []   # no badges
    """
    if catalog.empty:
        return {}

    # Compute popularity thresholds from the current catalog distribution so
    # the badges stay meaningful even if the data scale changes.
    bestseller_threshold = catalog["order_count"].quantile(0.95)
    popular_threshold = catalog["order_count"].quantile(0.75)

    # Vectorized boolean masks per badge. Using `&` rather than chained Python
    # logic keeps this O(n) over the full catalog instead of per-row.
    order_count = catalog["order_count"]
    reorder_rate = catalog["reorder_rate"]

    is_bestseller = order_count >= bestseller_threshold
    # Popular = top 25% but not already a bestseller, so the two badges are
    # mutually exclusive on a single product.
    is_popular = (order_count >= popular_threshold) & ~is_bestseller

    is_customer_favorite = (
        (reorder_rate >= _CUSTOMER_FAVORITE_REORDER_RATE)
        & (order_count >= _CUSTOMER_FAVORITE_MIN_ORDERS)
    )

    # Nutrition grade is a string; lowercase + isin guards against NaN and
    # case inconsistencies. NaN values become False naturally.
    nutrition_grade = catalog["nutrition_grade"].astype("string").str.lower()
    is_healthy_choice = nutrition_grade.isin(_HEALTHY_GRADES).fillna(False)

    # Macro-based badges: NaN -> False so missing nutrition data never earns
    # a badge by accident.
    is_high_protein = (catalog["protein_100g"] >= _HIGH_PROTEIN_G).fillna(False)
    is_low_sugar = (
        catalog["sugar_100g"].notna() & (catalog["sugar_100g"] <= _LOW_SUGAR_G)
    )
    is_low_calorie = (
        catalog["calories_100g"].notna()
        & (catalog["calories_100g"] <= _LOW_CALORIE_KCAL)
    )
    is_high_fiber = (catalog["fiber_100g"] >= _HIGH_FIBER_G).fillna(False)

    # Stack each mask as a column on a small frame keyed by product_id so we
    # can build the per-product badge list in a single pass.
    flags = pd.DataFrame(
        {
            "bestseller": is_bestseller.values,
            "popular": is_popular.values,
            "customer-favorite": is_customer_favorite.values,
            "healthy-choice": is_healthy_choice.values,
            "high-protein": is_high_protein.values,
            "low-sugar": is_low_sugar.values,
            "low-calorie": is_low_calorie.values,
            "high-fiber": is_high_fiber.values,
        },
        index=catalog["product_id"].values,
    )

    badge_columns = list(flags.columns)
    result: dict[int, list[str]] = {}
    # itertuples is ~10x faster than iterrows for this many rows.
    for row in flags.itertuples(index=True, name=None):
        product_id = int(row[0])
        # row[0] is the index (product_id), the rest are the flag columns in
        # the same order as badge_columns.
        badges = [badge_columns[i] for i, flag in enumerate(row[1:]) if flag]
        result[product_id] = badges

    return result
