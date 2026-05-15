"""Feed-based discovery module.

Surface curated and personalized product feeds for the home page so
shoppers can browse without typing a query (analogous to TikTok Shop's
For-You feed or Amazon's homepage carousels).

Each public ``get_*`` function returns ``list[dict]`` (NOT a DataFrame)
so the API layer can pass results straight to ``json.dumps``. Every
returned product carries a ``feed_score`` field that documents the
ranking signal used for that particular feed.

Feed types:
    * ``bestsellers`` -- most-ordered products overall.
    * ``healthy-picks`` -- nutrition-grade A/B products, sorted by
      popularity within the healthy set.
    * ``for-you`` -- personalized ranking that blends per-user affinity
      with global popularity, *excluding* a user's existing favorites
      so we surface new relevant items instead of regurgitating their
      regular cart.
    * ``trending_in_department`` -- popular products inside a single
      department like ``produce`` or ``dairy eggs``.

The for-you feed leans on
:class:`src.recommend.personalization.UserPersonalizationStore` for the
per-user score; this module never mutates that store.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover -- import only for type hints
    from src.recommend.personalization import UserPersonalizationStore


# ---------------------------------------------------------------------------
# Feed registry exposed to the API/UI.
# ---------------------------------------------------------------------------
FEED_TYPES: dict[str, str] = {
    "bestsellers": "🏆 Bestsellers",
    "healthy-picks": "🥗 Healthy Picks",
    "for-you": "✨ For You",
}


# Personalization vs popularity blend for the For-You feed.
_FOR_YOU_PERSONAL_WEIGHT = 0.7
_FOR_YOU_POPULARITY_WEIGHT = 0.3

# Same popularity normalization cap as personalization.rerank so the
# scales are comparable when blending. log1p(1M) ~ 13.8.
_MAX_ORDER_COUNT = 1_000_000.0
_LOG_MAX_ORDER_COUNT = math.log1p(_MAX_ORDER_COUNT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _records_with_feed_score(
    frame: pd.DataFrame, score_col: str
) -> list[dict]:
    """Convert ``frame`` to ``list[dict]`` with a ``feed_score`` field.

    NaN/Inf values are converted to ``None`` so the result is safe to
    JSON-serialize. ``score_col`` is the column whose value should be
    copied into ``feed_score`` for each row.
    """
    if frame.empty:
        return []

    # ``to_dict("records")`` preserves NaN as float('nan') which is not
    # JSON-safe. Convert object-typed NaNs to None first; numeric NaN is
    # handled per-record below.
    records = frame.to_dict("records")
    cleaned: list[dict] = []
    for rec in records:
        clean: dict[str, Any] = {}
        for k, v in rec.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                clean[k] = None
            elif v is pd.NaT:
                clean[k] = None
            else:
                clean[k] = v
        score = clean.get(score_col, 0)
        # ``score_col`` may itself have been nulled above; fall back to 0.
        if score is None:
            score = 0
        clean["feed_score"] = score
        cleaned.append(clean)
    return cleaned


def _normalized_popularity(order_count: float) -> float:
    """log1p(order_count) / log1p(MAX_ORDER_COUNT), clamped to [0, 1]."""
    try:
        oc = float(order_count)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(oc) or oc <= 0:
        return 0.0
    return min(math.log1p(oc) / _LOG_MAX_ORDER_COUNT, 1.0)


# ---------------------------------------------------------------------------
# Public feeds
# ---------------------------------------------------------------------------
def get_bestsellers(catalog: pd.DataFrame, top_k: int = 20) -> list[dict]:
    """Return top-K products by ``order_count`` overall.

    Args:
        catalog: Product catalog with at least ``order_count`` column.
        top_k: Maximum number of products to return.

    Returns:
        ``list[dict]`` of products sorted by ``order_count`` desc, each
        with a ``feed_score`` field equal to ``order_count``.
    """
    if catalog is None or catalog.empty or top_k <= 0:
        return []
    if "order_count" not in catalog.columns:
        return []

    sorted_df = catalog.sort_values(
        "order_count", ascending=False, kind="mergesort"
    ).head(top_k)
    return _records_with_feed_score(sorted_df, "order_count")


def get_healthy_picks(catalog: pd.DataFrame, top_k: int = 20) -> list[dict]:
    """Return top-K products with nutrition grade ``a`` or ``b``.

    The healthy subset is sorted by ``order_count`` so we surface popular
    healthy items rather than obscure ones. If the catalog lacks
    ``nutrition_grade``, falls back to an empty list (we don't want to
    silently advertise unverified items as healthy).
    """
    if catalog is None or catalog.empty or top_k <= 0:
        return []
    if "nutrition_grade" not in catalog.columns:
        return []
    if "order_count" not in catalog.columns:
        return []

    grades = catalog["nutrition_grade"].astype("string").str.lower()
    healthy = catalog[grades.isin(["a", "b"])]
    if healthy.empty:
        return []

    sorted_df = healthy.sort_values(
        "order_count", ascending=False, kind="mergesort"
    ).head(top_k)
    return _records_with_feed_score(sorted_df, "order_count")


def get_trending_in_department(
    catalog: pd.DataFrame,
    department: str,
    top_k: int = 20,
) -> list[dict]:
    """Top-K most-ordered products in a given department.

    Args:
        catalog: Catalog with ``department`` and ``order_count`` columns.
        department: Department label, e.g. ``"produce"`` or
            ``"dairy eggs"``. Matched case-insensitively after trimming.
        top_k: Maximum number of products to return.

    Returns:
        ``list[dict]`` of products in the department, sorted by
        ``order_count`` desc. Empty list if the department is missing,
        unknown, or the catalog has no matching rows.
    """
    if catalog is None or catalog.empty or top_k <= 0:
        return []
    if not isinstance(department, str) or not department.strip():
        return []
    if "department" not in catalog.columns or "order_count" not in catalog.columns:
        return []

    needle = department.strip().lower()
    departments = catalog["department"].astype("string").str.strip().str.lower()
    matching = catalog[departments == needle]
    if matching.empty:
        return []

    sorted_df = matching.sort_values(
        "order_count", ascending=False, kind="mergesort"
    ).head(top_k)
    return _records_with_feed_score(sorted_df, "order_count")


def get_for_you(
    catalog: pd.DataFrame,
    store: "UserPersonalizationStore",
    user_id: int,
    top_k: int = 20,
) -> list[dict]:
    """Personalized 'For You' feed for ``user_id``.

    Algorithm:
        1. If the user has no profile, fall back to
           :func:`get_bestsellers` so cold-start users still see a feed.
        2. Otherwise drop products in ``profile.favorite_products`` --
           we want to surface NEW relevant items, not regurgitate the
           regular cart.
        3. Score every remaining product with
           ``store.score_for_user(user_id, product)`` and combine with
           popularity:

               feed_score = 0.7 * personalization_score
                          + 0.3 * normalized_order_count

        4. Return the top-K combined records.

    Args:
        catalog: Full product catalog.
        store: Loaded :class:`UserPersonalizationStore`.
        user_id: Instacart user id. ``None`` or unknown ids fall back
            to bestsellers.
        top_k: Number of items to return.

    Returns:
        ``list[dict]`` sorted by ``feed_score`` desc. Each record also
        carries a ``personalization_score`` field for transparency.
    """
    if catalog is None or catalog.empty or top_k <= 0:
        return []

    profile = None
    if store is not None and user_id is not None:
        profile = store.get_profile(user_id)

    # Cold start -- no profile means no signal beyond popularity.
    if profile is None:
        return get_bestsellers(catalog, top_k=top_k)

    # Drop the user's existing favorites so the feed surfaces NEW items.
    fav_set = set(int(pid) for pid in (profile.favorite_products or []))
    if fav_set and "product_id" in catalog.columns:
        candidates = catalog[~catalog["product_id"].astype("int64").isin(fav_set)]
    else:
        candidates = catalog

    if candidates.empty:
        return []

    # Score everything. ``score_for_user`` operates on plain dicts; using
    # ``to_dict("records")`` here keeps the per-product call shape stable
    # even though it's a touch slower than a vectorized path -- the feed
    # gets cached upstream so this runs rarely.
    records = candidates.to_dict("records")

    scored: list[dict] = []
    for rec in records:
        # Sanitize NaNs once per record so JSON serialization downstream
        # never trips, and so the personalization scorer doesn't see
        # ``float('nan')`` for category/department/brand.
        clean: dict[str, Any] = {}
        for k, v in rec.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                clean[k] = None
            else:
                clean[k] = v

        personal = float(store.score_for_user(user_id, clean))
        popularity = _normalized_popularity(clean.get("order_count", 0) or 0)
        feed_score = (
            _FOR_YOU_PERSONAL_WEIGHT * personal
            + _FOR_YOU_POPULARITY_WEIGHT * popularity
        )
        clean["personalization_score"] = personal
        clean["feed_score"] = feed_score
        scored.append(clean)

    scored.sort(key=lambda p: p.get("feed_score", 0.0), reverse=True)
    return scored[:top_k]


def list_departments(catalog: pd.DataFrame) -> list[str]:
    """Return department names sorted by total ``order_count`` desc.

    Empty/missing departments are dropped. Returns ``[]`` for an empty
    catalog or one without a ``department`` column.
    """
    if catalog is None or catalog.empty:
        return []
    if "department" not in catalog.columns:
        return []

    has_orders = "order_count" in catalog.columns
    frame = catalog[["department"] + (["order_count"] if has_orders else [])].copy()
    frame = frame.dropna(subset=["department"])
    if frame.empty:
        return []

    frame["department"] = frame["department"].astype(str).str.strip()
    frame = frame[frame["department"] != ""]
    if frame.empty:
        return []

    if has_orders:
        totals = (
            frame.groupby("department", sort=False)["order_count"]
            .sum()
            .sort_values(ascending=False)
        )
    else:
        totals = (
            frame.groupby("department", sort=False)
            .size()
            .sort_values(ascending=False)
        )
    return totals.index.astype(str).tolist()


__all__ = [
    "FEED_TYPES",
    "get_bestsellers",
    "get_healthy_picks",
    "get_trending_in_department",
    "get_for_you",
    "list_departments",
]
