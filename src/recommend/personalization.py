"""User personalization module.

Builds per-user preference profiles from Instacart-style order history and
re-ranks product lists for individual users.

A profile captures:
  1. Aggregate stats (total orders, avg basket size).
  2. Top-50 favorite product_ids by order count.
  3. Category / department / brand affinity dicts -- counts normalized
     to [0, 1] by dividing by each user's max-count label.

Ranking combines (a) base relevance score from search, (b) a global
popularity boost via log1p(order_count), and (c) per-user affinity.
Anonymous users (no profile) skip (c) and fall back to relevance plus
the popularity bump so the API/UI can call the same code path.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class UserProfile:
    """Per-user preference profile.

    Attributes:
        user_id: Instacart user identifier.
        total_orders: Number of distinct orders this user has placed.
        avg_basket_size: Mean number of items per order.
        favorite_products: Top 50 product_ids ranked by purchase count.
        favorite_categories: Mapping of category -> score in [0, 1],
            normalized so the most-bought category for this user is 1.0.
        favorite_departments: Same shape as favorite_categories, for
            the coarser department label.
        favorite_brands: Same shape, for brand. May be empty when the
            user mostly bought unbranded items.
    """

    user_id: int
    total_orders: int
    avg_basket_size: float
    favorite_products: list[int] = field(default_factory=list)
    favorite_categories: dict[str, float] = field(default_factory=dict)
    favorite_departments: dict[str, float] = field(default_factory=dict)
    favorite_brands: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Profile build
# ---------------------------------------------------------------------------
_TOP_N_FAVORITE_PRODUCTS = 50
_TOP_N_CATEGORY_LABELS = 15


def compute_user_profiles(
    orders: pd.DataFrame,
    order_products: pd.DataFrame,
    catalog: pd.DataFrame,
    min_orders: int = 5,
) -> dict[int, UserProfile]:
    """Compute a ``UserProfile`` for every user with >= ``min_orders``.

    Algorithm:
        1. Filter ``orders`` to ``eval_set == 'prior'`` (if column present).
        2. Join orders <-> order_products on order_id to get user-product rows.
        3. Join with ``catalog`` to attach category / department / brand.
        4. For each user:
           - total_orders = unique order_id count
           - avg_basket_size = total items / total_orders
           - top-50 products by frequency
           - normalize category / department / brand counts to [0, 1]
             by dividing each label's count by the user's max-count label.

    Args:
        orders: DataFrame with columns ``order_id``, ``user_id``, and
            optionally ``eval_set``.
        order_products: DataFrame with columns ``order_id``, ``product_id``,
            and optionally ``reordered``.
        catalog: DataFrame with ``product_id`` and optionally
            ``category`` / ``department`` / ``brand`` columns.
        min_orders: Minimum total prior orders for a user to be included.

    Returns:
        Dict mapping ``user_id`` -> ``UserProfile``.
    """
    if not {"order_id", "user_id"}.issubset(orders.columns):
        raise ValueError("orders must have 'order_id' and 'user_id' columns")
    if not {"order_id", "product_id"}.issubset(order_products.columns):
        raise ValueError(
            "order_products must have 'order_id' and 'product_id' columns"
        )
    if "product_id" not in catalog.columns:
        raise ValueError("catalog must have a 'product_id' column")

    # ---- 1. Restrict to prior orders, drop users below threshold ------
    prior = orders
    if "eval_set" in orders.columns:
        prior = orders[orders["eval_set"] == "prior"]
    prior = prior[["order_id", "user_id"]].copy()

    # Count distinct orders per user using the prior table directly.
    user_order_counts = (
        prior.groupby("user_id", sort=False)["order_id"]
        .nunique()
        .rename("total_orders")
    )
    eligible_users = user_order_counts.index[user_order_counts >= min_orders]
    if len(eligible_users) == 0:
        return {}
    eligible_set = set(eligible_users.tolist())

    prior_eligible = prior[prior["user_id"].isin(eligible_set)]

    # ---- 2. Join order-product rows back to user_id -------------------
    op = order_products[
        ["order_id", "product_id"]
        + (["reordered"] if "reordered" in order_products.columns else [])
    ]
    # Inner join via merge -- the prior_eligible table is the filter.
    user_products = op.merge(
        prior_eligible[["order_id", "user_id"]],
        on="order_id",
        how="inner",
    )

    # avg basket size = total items / total orders (both restricted to
    # the prior set). We compute total items per user from user_products
    # and divide by the order count we already have.
    items_per_user = (
        user_products.groupby("user_id", sort=False).size().rename("total_items")
    )

    # ---- 3. Join product attributes -----------------------------------
    cat_cols = ["product_id"]
    for col in ("category", "department", "brand"):
        if col in catalog.columns:
            cat_cols.append(col)
    cat_slim = catalog[cat_cols].drop_duplicates(subset="product_id")
    user_products = user_products.merge(cat_slim, on="product_id", how="left")

    # ---- 4. Favorite products (top 50 by order count) -----------------
    upp = (
        user_products.groupby(["user_id", "product_id"], sort=False)
        .size()
        .rename("count")
        .reset_index()
    )
    upp = upp.sort_values(["user_id", "count"], ascending=[True, False])
    fav_products: dict[int, list[int]] = (
        upp.groupby("user_id", sort=False)["product_id"]
        .apply(lambda s: s.head(_TOP_N_FAVORITE_PRODUCTS).astype(int).tolist())
        .to_dict()
    )

    # ---- 5. Favorite categories / departments / brands ----------------
    fav_categories = _top_normalized_per_user(
        user_products, "category", _TOP_N_CATEGORY_LABELS
    )
    fav_departments = _top_normalized_per_user(
        user_products, "department", _TOP_N_CATEGORY_LABELS
    )
    fav_brands = _top_normalized_per_user(
        user_products, "brand", _TOP_N_CATEGORY_LABELS
    )

    # ---- 6. Materialize profiles --------------------------------------
    totals = (
        user_order_counts.loc[user_order_counts >= min_orders]
        .to_frame()
        .join(items_per_user, how="left")
    )
    totals["total_items"] = totals["total_items"].fillna(0).astype(np.int64)
    totals["total_orders"] = totals["total_orders"].astype(np.int64)
    totals["avg_basket_size"] = np.where(
        totals["total_orders"] > 0,
        totals["total_items"] / totals["total_orders"],
        0.0,
    )

    profiles: dict[int, UserProfile] = {}
    for user_id, row in totals.iterrows():
        uid = int(user_id)
        profiles[uid] = UserProfile(
            user_id=uid,
            total_orders=int(row["total_orders"]),
            avg_basket_size=float(row["avg_basket_size"]),
            favorite_products=fav_products.get(uid, []),
            favorite_categories=fav_categories.get(uid, {}),
            favorite_departments=fav_departments.get(uid, {}),
            favorite_brands=fav_brands.get(uid, {}),
        )
    return profiles


def _top_normalized_per_user(
    user_products: pd.DataFrame, column: str, top_n: int
) -> dict[int, dict[str, float]]:
    """For each user, return top-N labels of ``column`` normalized to [0, 1].

    Normalization: count / max_count_for_that_user. So a user's most-bought
    label has score 1.0 and everything else is a fraction of that.

    NaN labels are dropped first so a user with mostly unbranded items
    returns an empty dict for brand instead of an all-NaN entry.
    """
    if column not in user_products.columns:
        return {}

    sub = user_products[["user_id", column]].dropna(subset=[column])
    if sub.empty:
        return {}

    counts = (
        sub.groupby(["user_id", column], sort=False)
        .size()
        .rename("count")
        .reset_index()
    )

    # Normalize by each user's max count -> top label = 1.0.
    max_per_user = counts.groupby("user_id", sort=False)["count"].transform("max")
    counts["score"] = counts["count"] / max_per_user

    counts = counts.sort_values(["user_id", "score"], ascending=[True, False])

    out: dict[int, dict[str, float]] = {}
    for uid, group in counts.groupby("user_id", sort=False):
        head = group.head(top_n)
        out[int(uid)] = dict(
            zip(
                head[column].astype(str).tolist(),
                head["score"].astype(float).tolist(),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Personalization store
# ---------------------------------------------------------------------------
class UserPersonalizationStore:
    """In-memory store of user profiles with personalized scoring."""

    def __init__(self, profiles: dict[int, UserProfile]):
        # Keep an int-keyed dict so callers don't have to think about
        # numpy int types from pandas indexes.
        self.profiles: dict[int, UserProfile] = {
            int(uid): p for uid, p in profiles.items()
        }

    # -- Persistence ----------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Serialize profiles to parquet (one row per user).

        The nested dict columns (favorite_*) are JSON-encoded strings so
        the file round-trips cleanly through pyarrow without complex
        nested types.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        records = []
        for uid, p in self.profiles.items():
            records.append(
                {
                    "user_id": int(uid),
                    "total_orders": int(p.total_orders),
                    "avg_basket_size": float(p.avg_basket_size),
                    "favorite_products": json.dumps(
                        [int(x) for x in p.favorite_products]
                    ),
                    "favorite_categories": json.dumps(
                        {str(k): float(v) for k, v in p.favorite_categories.items()}
                    ),
                    "favorite_departments": json.dumps(
                        {str(k): float(v) for k, v in p.favorite_departments.items()}
                    ),
                    "favorite_brands": json.dumps(
                        {str(k): float(v) for k, v in p.favorite_brands.items()}
                    ),
                }
            )
        frame = pd.DataFrame.from_records(records)
        # Stable column order for downstream consumers.
        frame = frame[
            [
                "user_id",
                "total_orders",
                "avg_basket_size",
                "favorite_products",
                "favorite_categories",
                "favorite_departments",
                "favorite_brands",
            ]
        ]
        frame.to_parquet(path, index=False)

    @classmethod
    def load(cls, path: str | Path) -> "UserPersonalizationStore":
        """Load from a parquet file written by ``save()``."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path)

        profiles: dict[int, UserProfile] = {}
        for row in frame.itertuples(index=False):
            uid = int(row.user_id)
            profiles[uid] = UserProfile(
                user_id=uid,
                total_orders=int(row.total_orders),
                avg_basket_size=float(row.avg_basket_size),
                favorite_products=list(json.loads(row.favorite_products)),
                favorite_categories=dict(json.loads(row.favorite_categories)),
                favorite_departments=dict(json.loads(row.favorite_departments)),
                favorite_brands=dict(json.loads(row.favorite_brands)),
            )
        return cls(profiles)

    # -- Lookup ---------------------------------------------------------
    def get_profile(self, user_id: int) -> UserProfile | None:
        """Return profile or None if user not in store."""
        if user_id is None:
            return None
        try:
            return self.profiles.get(int(user_id))
        except (TypeError, ValueError):
            return None

    # -- Scoring --------------------------------------------------------
    def score_for_user(self, user_id: int, product: dict) -> float:
        """Personalization score in [0, 1] for a (user, product) pair.

        Weighted blend:
            0.40 * (1 if product_id in favorite_products else 0)
            0.30 * favorite_categories.get(category, 0)
            0.20 * favorite_departments.get(department, 0)
            0.10 * favorite_brands.get(brand, 0)

        Returns 0.0 if the user has no profile (cold-start fallback).
        """
        profile = self.get_profile(user_id)
        if profile is None:
            return 0.0

        # Favorite-product term.
        fav_term = 0.0
        pid = product.get("product_id")
        if pid is not None and profile.favorite_products:
            try:
                pid_int = int(pid)
            except (TypeError, ValueError):
                pid_int = None
            if pid_int is not None and pid_int in profile.favorite_products:
                fav_term = 1.0

        # Affinity terms -- look up by label (cast to str to match the
        # store keys, which are always strings).
        category = product.get("category")
        department = product.get("department")
        brand = product.get("brand")

        cat_term = (
            profile.favorite_categories.get(str(category), 0.0)
            if category is not None
            else 0.0
        )
        dept_term = (
            profile.favorite_departments.get(str(department), 0.0)
            if department is not None
            else 0.0
        )
        brand_term = (
            profile.favorite_brands.get(str(brand), 0.0)
            if brand is not None
            else 0.0
        )

        score = (
            0.40 * fav_term
            + 0.30 * cat_term
            + 0.20 * dept_term
            + 0.10 * brand_term
        )
        # Each component is in [0, 1] and the weights sum to 1, so the
        # blend is already bounded -- clamp defensively against floats.
        return max(0.0, min(1.0, score))

    # -- Demo helpers ---------------------------------------------------
    def list_demo_users(self, n: int = 20) -> list[dict]:
        """Return ``n`` diverse demo users for the UI dropdown.

        Pick users spanning the order-count distribution. For each user
        we emit a short, human-readable summary like:
            "User 42 -- 87 orders, prefers Dairy & Produce"
        """
        if not self.profiles or n <= 0:
            return []

        # Sort users by total_orders so we can sample evenly across
        # the distribution. Ties broken by user_id for determinism.
        sorted_users = sorted(
            self.profiles.values(),
            key=lambda p: (p.total_orders, p.user_id),
        )
        total = len(sorted_users)
        if n >= total:
            picked = sorted_users
        else:
            # Evenly spaced indices from low- to high-volume.
            # endpoint=True so the highest-volume user is included.
            idxs = np.linspace(0, total - 1, num=n, dtype=int)
            picked = [sorted_users[i] for i in idxs]

        out: list[dict] = []
        for p in picked:
            top_depts = _format_top_labels(p.favorite_departments, limit=2)
            if not top_depts:
                top_depts = _format_top_labels(p.favorite_categories, limit=2)
            if top_depts:
                summary = (
                    f"User {p.user_id} -- {p.total_orders} orders, "
                    f"prefers {top_depts}"
                )
            else:
                summary = (
                    f"User {p.user_id} -- {p.total_orders} orders"
                )
            out.append(
                {
                    "user_id": int(p.user_id),
                    "total_orders": int(p.total_orders),
                    "summary": summary,
                }
            )
        return out


def _format_top_labels(labels: dict[str, float], limit: int = 2) -> str:
    """Format top-N labels as a human-friendly ``A & B`` string."""
    if not labels:
        return ""
    # Already roughly sorted but be defensive -- some load paths might
    # not preserve dict order across versions.
    top = sorted(labels.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    names = [_titlecase_label(k) for k, _ in top]
    return " & ".join(names)


def _titlecase_label(label: str) -> str:
    """Titlecase a label like 'fresh fruits' -> 'Fresh Fruits'."""
    if not label:
        return label
    return " ".join(part.capitalize() for part in label.split())


# ---------------------------------------------------------------------------
# Re-ranking
# ---------------------------------------------------------------------------
# Conservative cap so the most popular product can't completely
# dominate. log1p(1M) ~ 13.8 is the normalization denominator.
_MAX_ORDER_COUNT = 1_000_000.0
_LOG_MAX_ORDER_COUNT = math.log1p(_MAX_ORDER_COUNT)


def _normalized_popularity(product: dict) -> float:
    """log1p(order_count) / log1p(MAX_ORDER_COUNT), clamped to [0, 1]."""
    raw = product.get("order_count", 0) or 0
    try:
        oc = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if oc <= 0:
        return 0.0
    return min(math.log1p(oc) / _LOG_MAX_ORDER_COUNT, 1.0)


def rerank_with_personalization(
    products: list[dict],
    user_id: int | None,
    store: "UserPersonalizationStore | None",
    alpha: float = 0.3,
    popularity_weight: float = 0.2,
) -> list[dict]:
    """Re-rank a search result list with personalization + popularity.

    final_score =
        (1 - alpha - popularity_weight) * relevance_score
      + alpha                          * personalization_score
      + popularity_weight              * normalized_popularity

    Adds ``personalization_score`` and ``final_score`` fields to each
    product. Sorts in place by ``final_score`` descending.

    Cold-start: if ``user_id`` is None or not in the store (or no store
    was provided), the personalization term is dropped and reapportioned
    to relevance, so popularity still helps:
        final_score = (1 - popularity_weight) * relevance_score
                    + popularity_weight       * normalized_popularity
    """
    profile: UserProfile | None = None
    if store is not None and user_id is not None:
        profile = store.get_profile(user_id)

    relevance_weight = max(0.0, 1.0 - alpha - popularity_weight)
    cold_relevance_weight = max(0.0, 1.0 - popularity_weight)

    for product in products:
        relevance = float(product.get("relevance_score", 0.0) or 0.0)
        popularity = _normalized_popularity(product)

        if profile is None:
            personalization = 0.0
            product["personalization_score"] = 0.0
            product["final_score"] = (
                cold_relevance_weight * relevance
                + popularity_weight * popularity
            )
        else:
            personalization = store.score_for_user(user_id, product) if store else 0.0
            product["personalization_score"] = personalization
            product["final_score"] = (
                relevance_weight * relevance
                + alpha * personalization
                + popularity_weight * popularity
            )

    products.sort(key=lambda p: p.get("final_score", 0.0), reverse=True)
    return products


# ---------------------------------------------------------------------------
# Convenience re-exports for callers that want module-level helpers.
# ---------------------------------------------------------------------------
__all__ = [
    "UserProfile",
    "UserPersonalizationStore",
    "compute_user_profiles",
    "rerank_with_personalization",
]
