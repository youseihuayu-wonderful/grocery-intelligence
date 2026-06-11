"""Deterministic synthetic-price generation for the grocery catalog.

The Instacart catalog ships without price data. To make the cart feel
real, this module computes a plausible price for every product based on
its department, name keywords, nutrition grade, and order popularity.

The algorithm is fully deterministic — no randomness, no I/O, no
external state — so the same product always gets the same price.

Public functions
----------------
- :func:`generate_synthetic_price` — price one product dict.
- :func:`build_price_map` — vectorize over a catalog DataFrame.
- :func:`attach_prices` — mutate a list of product dicts in place.
"""

from __future__ import annotations

import math
from typing import Iterable

import pandas as pd


# --------------------------------------------------------------------
# Base-price table (USD). Tuned to feel right for a US grocery shop.
# --------------------------------------------------------------------
BASE_PRICES: dict[str, float] = {
    "produce": 2.50,
    "dairy eggs": 3.50,
    "meat seafood": 9.00,
    "bakery": 4.00,
    "snacks": 3.00,
    "beverages": 3.50,
    "frozen": 5.00,
    "pantry": 4.00,
    "household": 6.00,
    "babies": 12.00,
    "personal care": 7.00,
    "alcohol": 14.00,
    # Reasonable defaults for the rest of the Instacart departments.
    "canned goods": 3.00,
    "dry goods pasta": 3.50,
    "deli": 7.50,
    "international": 5.50,
    "breakfast": 4.50,
    "pets": 8.00,
    "other": 4.00,
    "missing": 4.00,
}

# Anything not in the table above falls back to this.
DEFAULT_BASE_PRICE = 4.00

# Multiplier knobs.
_ORGANIC_MULT = 1.30
_PREMIUM_MULT = 1.40            # "grass-fed", "grass fed", "wild"
_NUTRITION_A_MULT = 1.10
_NUTRITION_LOW_MULT = 0.90      # grades d, e

# Popularity boost.
_POPULARITY_PER_LOG10 = 0.50    # +$0.50 per log10(1 + order_count)
_POPULARITY_CAP = 5.00          # ...but never more than +$5

# Floor.
_MIN_PRICE = 0.99

# Keyword sets, lower-cased to compare against `name.lower()`.
_PREMIUM_KEYWORDS = ("grass-fed", "grass fed", "wild")


def _safe_float(value, default: float = 0.0) -> float:
    """Coerce ``value`` to a finite float, falling back to ``default``.

    Handles ``None``, NaN, and stringy numeric values that occasionally
    appear in real catalog data.
    """
    if value is None:
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(f) or math.isinf(f):
        return default
    return f


def _normalize_text(value) -> str:
    """Lower-case a string-ish value; empty for ``None`` / NaN."""
    if value is None:
        return ""
    # Pandas may hand us a float NaN where it expected a string.
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).lower()


def generate_synthetic_price(product: dict) -> float:
    """Compute a deterministic synthetic price for a single product.

    The algorithm starts from a department-based base price, applies a
    handful of multiplicative adjustments for premium / organic /
    nutrition markers, and then adds a logarithmic popularity boost.

    Parameters
    ----------
    product:
        A dict-like representation of one catalog row. The function
        looks up ``department``, ``product_name``, ``nutrition_grade``,
        and ``order_count`` if present, ignoring anything else.

    Returns
    -------
    float
        A USD price rounded to two decimals, never less than $0.99.
    """
    department = _normalize_text(product.get("department")).strip()
    base = BASE_PRICES.get(department, DEFAULT_BASE_PRICE)

    name = _normalize_text(product.get("product_name"))
    nutrition_grade = _normalize_text(product.get("nutrition_grade")).strip()
    order_count = _safe_float(product.get("order_count"), 0.0)
    if order_count < 0:
        order_count = 0.0

    # Multiplicative adjustments. Applied in sequence so "organic
    # grass-fed grade-a" all stack.
    multiplier = 1.0
    if "organic" in name:
        multiplier *= _ORGANIC_MULT
    if any(kw in name for kw in _PREMIUM_KEYWORDS):
        multiplier *= _PREMIUM_MULT
    if nutrition_grade == "a":
        multiplier *= _NUTRITION_A_MULT
    elif nutrition_grade in ("d", "e"):
        multiplier *= _NUTRITION_LOW_MULT

    # Popularity boost: log10(1 + N) * 0.50, capped at $5.
    popularity_boost = math.log10(1.0 + order_count) * _POPULARITY_PER_LOG10
    if popularity_boost > _POPULARITY_CAP:
        popularity_boost = _POPULARITY_CAP

    price = base * multiplier + popularity_boost
    if price < _MIN_PRICE:
        price = _MIN_PRICE
    return round(price, 2)


def build_price_map(catalog: pd.DataFrame) -> dict[int, float]:
    """Build a ``{product_id: price}`` map for an entire catalog.

    This is the hot path called once at startup to attach prices to all
    49k+ products. The implementation is vectorized via pandas to keep
    the full build well under 3 seconds.

    Parameters
    ----------
    catalog:
        DataFrame with at least ``product_id`` and ``department``.
        Optional but used: ``product_name``, ``nutrition_grade``,
        ``order_count``.

    Returns
    -------
    dict[int, float]
        Product id to price. Order is not guaranteed.
    """
    if catalog is None or catalog.empty:
        return {}

    # Pull the columns we need; default missing ones to "neutral" values.
    # This works whether the column is present or absent.
    df = catalog
    n = len(df)

    if "department" in df.columns:
        dept = df["department"].astype("string").str.lower().fillna("")
    else:
        dept = pd.Series([""] * n, index=df.index, dtype="string")
    dept = dept.str.strip()

    if "product_name" in df.columns:
        name = df["product_name"].astype("string").str.lower().fillna("")
    else:
        name = pd.Series([""] * n, index=df.index, dtype="string")

    if "nutrition_grade" in df.columns:
        grade = df["nutrition_grade"].astype("string").str.lower().fillna("")
    else:
        grade = pd.Series([""] * n, index=df.index, dtype="string")
    grade = grade.str.strip()

    if "order_count" in df.columns:
        order_count = pd.to_numeric(df["order_count"], errors="coerce").fillna(0.0)
        order_count = order_count.clip(lower=0.0)
    else:
        order_count = pd.Series([0.0] * n, index=df.index, dtype="float64")

    # Base price by department, with fallback.
    base = dept.map(BASE_PRICES).astype("float64").fillna(DEFAULT_BASE_PRICE)

    # Multipliers, vectorized over Series.
    is_organic = name.str.contains("organic", regex=False, na=False)
    # contains() supports a single substring, so OR three calls together.
    is_premium = (
        name.str.contains("grass-fed", regex=False, na=False)
        | name.str.contains("grass fed", regex=False, na=False)
        | name.str.contains("wild", regex=False, na=False)
    )
    is_grade_a = grade == "a"
    is_grade_low = grade.isin(["d", "e"])

    multiplier = pd.Series(1.0, index=df.index, dtype="float64")
    multiplier = multiplier.where(~is_organic, multiplier * _ORGANIC_MULT)
    multiplier = multiplier.where(~is_premium, multiplier * _PREMIUM_MULT)
    multiplier = multiplier.where(~is_grade_a, multiplier * _NUTRITION_A_MULT)
    multiplier = multiplier.where(~is_grade_low, multiplier * _NUTRITION_LOW_MULT)

    # Popularity boost = log10(1 + N) * 0.5, capped at +$5.
    popularity = (
        (1.0 + order_count).apply(math.log10) * _POPULARITY_PER_LOG10
    ).clip(upper=_POPULARITY_CAP)

    price = (base * multiplier + popularity).clip(lower=_MIN_PRICE).round(2)

    return dict(zip(df["product_id"].astype("int64"), price.astype("float64")))


def attach_prices(
    products: list[dict], price_map: dict[int, float]
) -> list[dict]:
    """Add a ``price`` field to every product dict found in ``price_map``.

    Mutates ``products`` in place. Products whose ``product_id`` is not
    in ``price_map`` are left untouched. Returns the same list for
    chaining convenience.

    Parameters
    ----------
    products:
        List of product dicts (e.g. cart items, recommendations).
    price_map:
        Output of :func:`build_price_map`.
    """
    if not products or not price_map:
        return products

    for product in products:
        pid = product.get("product_id")
        if pid is None:
            continue
        try:
            pid_int = int(pid)
        except (TypeError, ValueError):
            continue
        if pid_int in price_map:
            product["price"] = float(price_map[pid_int])
    return products


__all__ = [
    "BASE_PRICES",
    "DEFAULT_BASE_PRICE",
    "generate_synthetic_price",
    "build_price_map",
    "attach_prices",
]
