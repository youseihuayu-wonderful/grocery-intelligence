"""Automatic attribute extraction for grocery products.

Scans every product's name, ingredients, nutrition macros and allergen
information to produce Amazon-style filterable attributes (organic,
gluten-free, vegan, high-protein, ...). Pure string matching plus numeric
thresholds — no ML.

Two public entry points:

* `extract_attributes(product)` — single product (dict-like) -> sorted list
  of attribute IDs.
* `extract_attributes_bulk(catalog)` — pandas DataFrame -> dict mapping
  product_id to attribute lists.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import pandas as pd

# ---------------------------------------------------------------------------
# Attribute catalogue
# ---------------------------------------------------------------------------

# Each entry: attribute_id -> tuple of lower-case substrings to look for in
# the combined (product_name + " " + ingredients) text.
_KEYWORD_RULES: dict[str, tuple[str, ...]] = {
    "organic": ("organic",),
    "gluten-free": ("gluten free", "gluten-free", "gf "),
    "vegan": ("vegan",),
    "vegetarian": ("vegetarian",),
    "kosher": ("kosher",),
    "dairy-free": ("dairy free", "dairy-free", "non-dairy", "lactose free"),
    "sugar-free": ("sugar free", "sugar-free", "no sugar added", "unsweetened"),
    "keto-friendly": ("keto",),
    "low-carb": ("low carb", "low-carb"),
    "whole-grain": ("whole grain", "whole-grain", "whole wheat"),
    "non-gmo": ("non gmo", "non-gmo", "no gmo"),
}

# Nutrition-threshold rules: (field name, comparator, threshold). The
# comparator is a callable that accepts (value, threshold) and returns bool.
_GE = lambda value, threshold: value >= threshold  # noqa: E731
_LE = lambda value, threshold: value <= threshold  # noqa: E731

_THRESHOLD_RULES: tuple[tuple[str, str, Any, Any], ...] = (
    ("high-protein", "protein_100g", _GE, 15.0),
    ("low-sugar", "sugar_100g", _LE, 5.0),
    ("low-calorie", "calories_100g", _LE, 100.0),
    ("low-fat", "fat_100g", _LE, 3.0),
    ("high-fiber", "fiber_100g", _GE, 6.0),
)

# NOTE: an allergen-derived `nut-free` attribute was intentionally removed.
# The Instacart catalog ships no populated `allergens_en` field (0% coverage),
# and inferring "nut-free" from the mere absence of nut words in a sparsely
# populated ingredients list would be an UNSAFE false claim. We only assert
# attributes we can actually substantiate from real data.


# Human-readable labels (with emoji) for the UI. Order roughly groups by
# dietary -> nutrition.
ATTRIBUTE_LABELS: dict[str, str] = {
    "organic": "🌿 Organic",
    "gluten-free": "🌾 Gluten-Free",
    "vegan": "🌱 Vegan",
    "vegetarian": "🥬 Vegetarian",
    "kosher": "✡️ Kosher",
    "dairy-free": "🥛 Dairy-Free",
    "sugar-free": "🍯 Sugar-Free",
    "keto-friendly": "🥑 Keto",
    "low-carb": "🥗 Low Carb",
    "whole-grain": "🌾 Whole Grain",
    "non-gmo": "🌽 Non-GMO",
    "high-protein": "💪 High Protein",
    "low-sugar": "🍯 Low Sugar",
    "low-calorie": "⚖️ Low Calorie",
    "low-fat": "🥦 Low Fat",
    "high-fiber": "🌾 High Fiber",
}

# Master list of every supported attribute ID, sorted for stable iteration.
ALL_ATTRIBUTES: list[str] = sorted(ATTRIBUTE_LABELS.keys())


# ---------------------------------------------------------------------------
# Natural-language nutrition / dietary intent
# ---------------------------------------------------------------------------

# Phrases that signal the user wants results constrained to an attribute, e.g.
# "high protein low sugar breakfast" -> {"high-protein", "low-sugar"}. Kept
# conservative and directional: a bare "sugar" must NOT trigger "low-sugar"
# (the shopper may want sugar). Longer phrases are matched as substrings on the
# lower-cased query. Each attribute resolves to the SAME id used by
# extract_attributes, which is only assigned when the product actually has the
# backing data — so filtering on detected intent never lets unknown-nutrition
# products masquerade as qualifying.
_INTENT_PHRASES: dict[str, tuple[str, ...]] = {
    "high-protein": ("high protein", "high-protein", "protein rich", "rich in protein", "more protein"),
    "low-sugar": ("low sugar", "low-sugar", "less sugar", "reduced sugar", "no sugar", "sugarless"),
    "sugar-free": ("sugar free", "sugar-free", "no sugar added", "unsweetened"),
    "low-calorie": ("low calorie", "low-calorie", "low cal", "fewer calories", "diet ", "light "),
    "low-fat": ("low fat", "low-fat", "fat free", "fat-free", "reduced fat"),
    "high-fiber": ("high fiber", "high-fiber", "high fibre", "fiber rich"),
    "vegan": ("vegan", "plant based", "plant-based"),
    "vegetarian": ("vegetarian",),
    "gluten-free": ("gluten free", "gluten-free", "no gluten"),
    "dairy-free": ("dairy free", "dairy-free", "non-dairy", "non dairy", "lactose free", "lactose-free"),
    "keto-friendly": ("keto", "ketogenic"),
    "low-carb": ("low carb", "low-carb"),
    "organic": ("organic",),
    "whole-grain": ("whole grain", "whole-grain", "whole wheat"),
    "non-gmo": ("non gmo", "non-gmo", "no gmo"),
}


def parse_nutrition_intent(query: str) -> list[str]:
    """Extract dietary / nutrition attribute filters implied by a search query.

    Returns a sorted list of attribute IDs (subset of ``ALL_ATTRIBUTES``).
    Conservative: only fires on directional phrases so ordinary product
    searches ("sugar", "milk") are unaffected. ``low-sugar`` and ``sugar-free``
    can both fire; callers may de-duplicate semantically if desired.
    """
    if not query:
        return []
    # Pad so leading/trailing single-word phrases like "diet " / "light " match.
    text = f" {query.lower()} "
    found = {
        attr_id
        for attr_id, phrases in _INTENT_PHRASES.items()
        if any(phrase in text for phrase in phrases)
    }
    return sorted(found)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_missing(value: Any) -> bool:
    """True when a value is None, NaN, or pandas-NA. Plain strings/numbers
    always come back False even if empty (empty string is handled by the
    string normalizer separately)."""
    if value is None:
        return True
    # NaN is the only float that isn't equal to itself; this also catches
    # numpy.float64 NaN safely.
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        # pandas.isna handles pd.NA, numpy.nan, and friends. Wrap in try
        # because it raises on certain array-like inputs we don't care about.
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _clean_text(value: Any) -> str:
    """Normalize a possibly-missing text field to a lowercased string. Missing
    or non-string values become "" so downstream `in` checks short-circuit."""
    if _is_missing(value):
        return ""
    return str(value).lower()


def _clean_number(value: Any) -> float | None:
    """Normalize a numeric field. Returns None when missing or non-numeric so
    threshold rules can skip cleanly (the spec requires we only emit a
    threshold attribute when the relevant field is NOT null)."""
    if _is_missing(value):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result):
        return None
    return result


def _text_contains_any(text: str, needles: Iterable[str]) -> bool:
    """Lower-case substring scan: True if any needle appears in `text`."""
    return any(needle in text for needle in needles)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_attributes(product: dict) -> list[str]:
    """Extract dietary and nutritional attributes from a single product.

    Args:
        product: Mapping with keys like ``product_name``, ``ingredients``,
            ``calories_100g``, ``protein_100g``, ``sugar_100g``, ``fat_100g``,
            ``fiber_100g``, ``nutrition_grade``. Any field may be ``None`` /
            ``NaN`` — missing values are handled silently.

    Returns:
        Sorted list of attribute IDs the product matches, e.g.
        ``["dairy-free", "gluten-free", "low-sugar", "organic"]``.
    """
    found: set[str] = set()

    # --- Keyword attributes ------------------------------------------------
    # Per spec, skip the keyword scan entirely if both name and ingredients
    # are missing (we have no text to match against).
    name = _clean_text(product.get("product_name"))
    ingredients = _clean_text(product.get("ingredients"))
    if name or ingredients:
        # Pad with spaces so a trailing token like "gf " can match even at
        # the end of a field. Joining both fields means a hit in either one
        # is enough.
        haystack = f" {name} {ingredients} "
        for attr_id, needles in _KEYWORD_RULES.items():
            if _text_contains_any(haystack, needles):
                found.add(attr_id)

    # --- Threshold attributes ---------------------------------------------
    for attr_id, field, comparator, threshold in _THRESHOLD_RULES:
        value = _clean_number(product.get(field))
        if value is None:
            continue
        if comparator(value, threshold):
            found.add(attr_id)

    return sorted(found)


def extract_attributes_bulk(catalog: pd.DataFrame) -> dict[int, list[str]]:
    """Run :func:`extract_attributes` on every row of ``catalog``.

    Args:
        catalog: Product catalog DataFrame containing at minimum a
            ``product_id`` column plus whatever subset of the attribute
            source columns happens to be present.

    Returns:
        Dict mapping ``product_id`` -> list of attribute IDs.
    """
    if catalog.empty:
        return {}

    # Restrict the column set to known sources to keep the per-row dict
    # construction cheap. Anything missing on the DataFrame is simulated as
    # NaN so extract_attributes still treats it as "missing".
    source_columns = (
        "product_name",
        "ingredients",
        "calories_100g",
        "protein_100g",
        "sugar_100g",
        "fat_100g",
        "fiber_100g",
        "nutrition_grade",
        "allergens_en",
    )

    available = [c for c in source_columns if c in catalog.columns]
    # Pull product_id values and the relevant fields as plain Python objects
    # so per-row dict construction sees real Nones for NaN where possible.
    ids = catalog["product_id"].tolist()

    # Use itertuples for speed; index=False so attribute access is cleaner.
    sub = catalog[available]
    records = sub.to_dict(orient="records")

    result: dict[int, list[str]] = {}
    for product_id, record in zip(ids, records):
        result[product_id] = extract_attributes(record)
    return result


__all__ = [
    "ALL_ATTRIBUTES",
    "ATTRIBUTE_LABELS",
    "extract_attributes",
    "extract_attributes_bulk",
]
