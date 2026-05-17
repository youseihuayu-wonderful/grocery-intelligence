"""Side-by-side product comparison.

Builds the data backing an Amazon-style "Compare with similar items"
table. Given 2-6 product ids the module returns:

* a row per product with all comparable attributes pulled from the
  catalog (name / brand / category / nutrition / popularity), and
* a row per attribute with the values lined up across products and
  a ``best_index`` pointer to the winner (or ``None`` when there is
  no meaningful winner -- e.g. text fields, NaNs, or all-equal).

The set of comparable attributes is intentionally small and explicit
(:data:`COMPARE_ATTRS`) so the API/UI layer can render a fixed table
schema without inspecting the catalog itself.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


# Attributes presented in the comparison table, in display order.
COMPARE_ATTRS: tuple[str, ...] = (
    "product_name",
    "brand",
    "category",
    "department",
    "calories_100g",
    "protein_100g",
    "sugar_100g",
    "fat_100g",
    "fiber_100g",
    "nutrition_grade",
    "order_count",
    "reorder_rate",
)


# Per-attribute display labels.
_LABELS: dict[str, str] = {
    "product_name": "Product",
    "brand": "Brand",
    "category": "Category",
    "department": "Department",
    "calories_100g": "Calories per 100g",
    "protein_100g": "Protein per 100g",
    "sugar_100g": "Sugar per 100g",
    "fat_100g": "Fat per 100g",
    "fiber_100g": "Fiber per 100g",
    "nutrition_grade": "Nutrition grade",
    "order_count": "Total orders",
    "reorder_rate": "Reorder rate",
}


# Attributes where a higher numeric value is better.
_HIGHER_IS_BETTER: frozenset[str] = frozenset({
    "order_count",
    "reorder_rate",
    "protein_100g",
    "fiber_100g",
})

# Attributes where a lower numeric value is better.
_LOWER_IS_BETTER: frozenset[str] = frozenset({
    "sugar_100g",
    "fat_100g",
    "calories_100g",
})

# Text fields where no winner is meaningful.
_TEXT_FIELDS: frozenset[str] = frozenset({
    "product_name",
    "brand",
    "category",
    "department",
})

# Nutrition grade ranking: a is best, e is worst.
_GRADE_RANK: dict[str, int] = {"a": 5, "b": 4, "c": 3, "d": 2, "e": 1}


def _is_missing(value: Any) -> bool:
    """True when a catalog value should be treated as missing.

    Catches NaN floats, ``None``, pandas/numpy NaT, and the
    pandas-arrow ``pd.NA`` sentinel.
    """
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        # Some non-scalar types (e.g. arrays) blow up pd.isna.
        return False
    return False


def _normalize_value(value: Any) -> Any:
    """Map missing values to ``None`` and unwrap numpy/pandas scalars.

    This produces JSON-friendly types so the result of compare_products
    can be returned directly from an HTTP handler.
    """
    if _is_missing(value):
        return None
    # numpy scalars expose .item() to convert to Python builtins.
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except (ValueError, TypeError):
            return value
    return value


def _best_index_numeric(values: list[Any], higher_is_better: bool) -> int | None:
    """Index of the winning value for a numeric attribute.

    Returns ``None`` if zero or one non-missing values are present
    (no comparison is meaningful) or if every non-missing value is
    equal. Missing positions are skipped entirely.
    """
    non_missing = [
        (idx, float(v))
        for idx, v in enumerate(values)
        if not _is_missing(v)
    ]
    if len(non_missing) < 2:
        return None
    target = (
        max(non_missing, key=lambda iv: iv[1])
        if higher_is_better
        else min(non_missing, key=lambda iv: iv[1])
    )
    # If every non-missing value ties, there's no winner.
    target_val = target[1]
    distinct = {v for _, v in non_missing}
    if len(distinct) == 1:
        return None
    # Tie-break on first-occurrence wins (matches max/min default).
    for idx, v in non_missing:
        if math.isclose(v, target_val):
            return idx
    return target[0]


def _best_index_grade(values: list[Any]) -> int | None:
    """Index of the best nutrition grade among the values."""
    ranked = []
    for idx, v in enumerate(values):
        if _is_missing(v):
            continue
        # Lower-case the grade so 'A' and 'a' compare equal.
        rank = _GRADE_RANK.get(str(v).strip().lower())
        if rank is None:
            continue
        ranked.append((idx, rank))
    if len(ranked) < 2:
        return None
    distinct = {r for _, r in ranked}
    if len(distinct) == 1:
        return None
    return max(ranked, key=lambda iv: iv[1])[0]


def compare_products(
    catalog: pd.DataFrame,
    product_ids: list[int],
) -> dict:
    """Build a side-by-side comparison of 2-6 products.

    Args:
        catalog: Full product catalog DataFrame; must contain a
            ``product_id`` column and every attribute in
            :data:`COMPARE_ATTRS`.
        product_ids: 2 to 6 product ids to compare. Order is preserved
            in the output.

    Returns:
        A dict with three keys:

        - ``product_ids`` -- the input ids (echoed).
        - ``products`` -- list (same order) of dicts containing every
          ``COMPARE_ATTRS`` field for each product. Missing values
          are emitted as ``None``.
        - ``attributes`` -- list of per-attribute dicts with keys
          ``key`` (attribute name), ``label`` (display label),
          ``values`` (list aligned to ``product_ids``), ``best_index``
          (int or ``None``) and ``lower_is_better`` (bool, ``False``
          for text fields).

    Raises:
        ValueError: If fewer than 2 or more than 6 ids are supplied,
            or if any id is missing from the catalog.
    """
    if len(product_ids) < 2:
        raise ValueError(
            f"compare_products requires at least 2 product_ids, got {len(product_ids)}"
        )
    if len(product_ids) > 6:
        raise ValueError(
            f"compare_products supports at most 6 product_ids, got {len(product_ids)}"
        )

    # Validate required catalog columns up front -- friendlier than
    # KeyError'ing out in the middle of building the table.
    missing_cols = [c for c in COMPARE_ATTRS if c not in catalog.columns]
    if missing_cols:
        raise ValueError(
            f"catalog is missing required columns: {missing_cols}"
        )
    if "product_id" not in catalog.columns:
        raise ValueError("catalog must have a 'product_id' column")

    # Index for fast lookup; preserve input order in output.
    indexed = catalog.set_index("product_id")
    pids_int = [int(p) for p in product_ids]
    missing_pids = [p for p in pids_int if p not in indexed.index]
    if missing_pids:
        raise ValueError(
            f"product_id(s) not found in catalog: {missing_pids}"
        )

    # ---- Per-product rows ------------------------------------------------
    products: list[dict[str, Any]] = []
    for pid in pids_int:
        row = indexed.loc[pid]
        # If duplicate product_ids ever leak into the catalog, .loc
        # returns a DataFrame; defensively take the first row.
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        record: dict[str, Any] = {"product_id": int(pid)}
        for attr in COMPARE_ATTRS:
            record[attr] = _normalize_value(row[attr])
        products.append(record)

    # ---- Per-attribute rows ---------------------------------------------
    attributes: list[dict[str, Any]] = []
    for attr in COMPARE_ATTRS:
        values = [p[attr] for p in products]
        if attr in _TEXT_FIELDS:
            best_index: int | None = None
            lower_is_better = False
        elif attr == "nutrition_grade":
            best_index = _best_index_grade(values)
            lower_is_better = False
        elif attr in _LOWER_IS_BETTER:
            best_index = _best_index_numeric(values, higher_is_better=False)
            lower_is_better = True
        elif attr in _HIGHER_IS_BETTER:
            best_index = _best_index_numeric(values, higher_is_better=True)
            lower_is_better = False
        else:
            # Unknown attr -- treat as no winner.
            best_index = None
            lower_is_better = False

        attributes.append(
            {
                "key": attr,
                "label": _LABELS.get(attr, attr),
                "values": values,
                "best_index": best_index,
                "lower_is_better": lower_is_better,
            }
        )

    return {
        "product_ids": pids_int,
        "products": products,
        "attributes": attributes,
    }
