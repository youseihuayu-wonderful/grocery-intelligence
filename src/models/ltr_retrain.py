"""Behavioral LTR retraining pipeline.

Takes the behavioral events logged in ``data/processed/behavior.db`` and
uses them to retrain the existing :class:`~src.models.ltr.LTRModel`
(XGBoost XGBRanker).

The idea closes the behavioral feedback loop the same way the real
Amazon / Walmart ranking stack does it:

1. Pull the per-(user, product) feature matrix from the behavior log.
2. Synthesize realistic queries from each user's purchase categories.
3. For each synthetic query, run the existing search engine to collect
   the candidates the live system would actually surface.
4. Label each candidate by how much the user has bought it (or its
   category) in the past.
5. Train ``XGBRanker`` on the resulting (features, labels, group)
   tuples and save the model.

This module does **not** modify the base :class:`LTRModel` - it only
generates training data and calls the existing ``train`` / ``save``
methods.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.models.ltr import LTRModel


# ---------------------------------------------------------------------------
# Tunables (callers can override via build_kwargs)
# ---------------------------------------------------------------------------

# Minimum number of distinct (product, n_purchases > 0) rows a user
# must have before we consider them as a training source. Users below
# this threshold do not have enough signal to drive an LTR objective.
_MIN_PURCHASE_ROWS_PER_USER = 5

# How many *favorite* categories per user we consider when synthesizing
# queries. 5 is a balance between breadth (different shopping contexts)
# and noise (going too deep into the user's long tail).
_TOP_CATEGORIES_PER_USER = 5

# Maximum top-1 product name length to use as a synthetic query before
# truncation. Long names dilute the BM25 signal and look unlike real
# user queries.
_QUERY_MAX_TOKENS = 6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _category_to_query(category: str, catalog: pd.DataFrame) -> str | None:
    """Pick a representative product name from a category to use as a query.

    Strategy: take the most-ordered product in the category and trim
    its name down to a short BM25-friendly query.
    """
    rows = catalog[catalog["category"] == category]
    if rows.empty:
        return None

    # Prefer the popular product (more likely to be a recognizable query).
    if "order_count" in rows.columns:
        rows = rows.sort_values("order_count", ascending=False)

    name = str(rows.iloc[0].get("product_name", "")).strip()
    if not name:
        return None

    tokens = name.split()
    if len(tokens) > _QUERY_MAX_TOKENS:
        tokens = tokens[:_QUERY_MAX_TOKENS]
    return " ".join(tokens).lower()


def _user_favorite_categories(
    user_feature_rows: pd.DataFrame,
    catalog: pd.DataFrame,
    top_n: int,
) -> list[str]:
    """Return up to ``top_n`` categories ranked by the user's total purchases."""
    # Join behavior rows (user, product, n_purchases) with the catalog
    # so we can aggregate by category.
    merged = user_feature_rows.merge(
        catalog[["product_id", "category"]],
        on="product_id",
        how="inner",
    )
    if merged.empty:
        return []
    cat_purchases = (
        merged.groupby("category")["n_purchases"]
        .sum()
        .sort_values(ascending=False)
    )
    # Drop the "missing"/empty categories - they're noise.
    cat_purchases = cat_purchases[
        cat_purchases.index.notna()
        & (cat_purchases.index != "")
        & (cat_purchases.index != "missing")
    ]
    return cat_purchases.head(top_n).index.tolist()


def _label_candidate(
    product_id: int,
    user_purchases: dict[int, int],
    favorite_categories: set[str],
    product_category: str,
) -> int:
    """Apply the labeling rule for a single (user, candidate) pair.

    Labels:
        3 -- user purchased the product 5+ times
        2 -- user purchased the product 1-4 times
        1 -- not purchased, but product is in one of the user's
             favorite categories
        0 -- otherwise
    """
    n = user_purchases.get(int(product_id), 0)
    if n >= 5:
        return 3
    if n >= 1:
        return 2
    if product_category in favorite_categories:
        return 1
    return 0


def _candidate_indices_from_results(
    results: list[dict],
    catalog: pd.DataFrame,
) -> list[int]:
    """Map search-engine result dicts back to catalog row indices.

    The search engine returns dicts containing the original catalog
    row (plus a ``relevance_score``). We need the integer positional
    indices because ``LTRModel.extract_features`` looks them up via
    ``catalog.iloc[idx]``.
    """
    # Build a product_id -> positional-index map once.
    if not hasattr(_candidate_indices_from_results, "_cache"):
        _candidate_indices_from_results._cache = {}  # type: ignore[attr-defined]
    cache: dict[int, dict[int, int]] = (
        _candidate_indices_from_results._cache  # type: ignore[attr-defined]
    )
    cat_id = id(catalog)
    if cat_id not in cache:
        cache[cat_id] = {
            int(pid): pos
            for pos, pid in enumerate(catalog["product_id"].tolist())
        }
    pid_to_pos = cache[cat_id]

    indices: list[int] = []
    for item in results:
        pid = item.get("product_id")
        if pid is None:
            continue
        pos = pid_to_pos.get(int(pid))
        if pos is not None:
            indices.append(pos)
    return indices


def _bm25_and_semantic_scores(
    search_engine: Any,
    query: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the full BM25 + semantic score arrays for a query.

    Used directly so we can reuse the existing
    :meth:`LTRModel.extract_features` API which expects per-product
    score arrays.
    """
    bm25_scores = search_engine.bm25.get_scores(query.lower().split())
    query_emb = search_engine.embedder.embed_query(query)
    semantic_scores = np.dot(search_engine.embeddings, query_emb)
    return np.asarray(bm25_scores), np.asarray(semantic_scores)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_training_data_from_behavior(
    behavior_logger: Any,
    catalog: pd.DataFrame,
    search_engine: Any,
    max_users: int = 1000,
    queries_per_user: int = 5,
    candidates_per_query: int = 50,
) -> tuple[pd.DataFrame, np.ndarray, list[int]]:
    """Generate ``(features, labels, group)`` for XGBRanker training.

    See module docstring for the high-level algorithm.

    Args:
        behavior_logger: A :class:`BehaviorLogger`-compatible object.
        catalog: Product catalog DataFrame (must include
            ``product_id``, ``product_name``, ``category``).
        search_engine: Search engine exposing ``.search(query, top_k)``
            and the underlying ``.bm25``, ``.embedder``, ``.embeddings``
            attributes used by :meth:`LTRModel.extract_features`.
        max_users: Maximum number of users to sample.
        queries_per_user: Number of synthetic queries to generate per user.
        candidates_per_query: Number of search candidates to score per query.

    Returns:
        A ``(features, labels, group)`` triple.

        * ``features`` -- :class:`pandas.DataFrame` whose columns are
          exactly :attr:`LTRModel.feature_names`.
        * ``labels`` -- 1D :class:`numpy.ndarray` of integer relevance
          labels with one entry per feature row.
        * ``group`` -- list of integers, one per emitted query, giving
          the row-count of that query's candidate block. Sum equals
          ``len(features)``.

    Notes:
        - Users without any purchase history are skipped (cold-start
          safety).
        - Queries that surface zero candidates are dropped.
    """
    # ---- 1. Pull the behavioral feature matrix --------------------------
    fm = behavior_logger.get_feature_matrix()
    if fm.empty:
        # Empty case: callers get well-typed empty outputs.
        empty_features = pd.DataFrame(columns=LTRModel().feature_names)
        return empty_features, np.array([], dtype=np.int64), []

    # Restrict to rows where the user actually purchased the product.
    # Views/clicks alone don't tell us much about preference, and the
    # seed script in this repo only logs ``purchase`` events anyway.
    purchase_rows = fm[fm["n_purchases"] > 0]
    if purchase_rows.empty:
        empty_features = pd.DataFrame(columns=LTRModel().feature_names)
        return empty_features, np.array([], dtype=np.int64), []

    # ---- 2. Sample max_users users with at least N purchase rows --------
    rows_per_user = purchase_rows.groupby("user_id").size()
    eligible_users = rows_per_user[rows_per_user >= _MIN_PURCHASE_ROWS_PER_USER].index.tolist()
    if not eligible_users:
        empty_features = pd.DataFrame(columns=LTRModel().feature_names)
        return empty_features, np.array([], dtype=np.int64), []

    # Deterministic sampling: take the heaviest users first. They give
    # the densest training signal for a fixed compute budget.
    eligible_sorted = (
        rows_per_user.loc[eligible_users].sort_values(ascending=False).index.tolist()
    )
    sampled_users = eligible_sorted[:max_users]

    # Pre-load the LTR model just to grab the canonical feature-name list
    # for the returned DataFrame.
    feature_names = LTRModel().feature_names

    # Category lookup for the labeling rule.
    cat_lookup = dict(
        zip(
            catalog["product_id"].astype(int).tolist(),
            catalog["category"].fillna("").astype(str).tolist(),
        )
    )

    # ---- 3. Iterate users + 4. Iterate queries -------------------------
    all_features_arrays: list[np.ndarray] = []
    all_labels: list[int] = []
    group_sizes: list[int] = []

    queries_seen = 0
    users_with_data: set[int] = set()

    for user_id in sampled_users:
        user_rows = purchase_rows[purchase_rows["user_id"] == user_id]
        if user_rows.empty:
            continue

        # Map of product_id -> count of purchases by this user. Used by
        # the labeling rule and the favorite-category detection.
        user_purchase_counts: dict[int, int] = dict(
            zip(
                user_rows["product_id"].astype(int).tolist(),
                user_rows["n_purchases"].astype(int).tolist(),
            )
        )

        fav_categories = _user_favorite_categories(
            user_rows, catalog, _TOP_CATEGORIES_PER_USER
        )
        if not fav_categories:
            continue

        # Build up to ``queries_per_user`` synthetic queries from the
        # user's favorite categories. We iterate in order so the most
        # important category is queried first.
        synthetic_queries: list[str] = []
        for cat in fav_categories:
            q = _category_to_query(cat, catalog)
            if q is None or q in synthetic_queries:
                continue
            synthetic_queries.append(q)
            if len(synthetic_queries) >= queries_per_user:
                break

        if not synthetic_queries:
            continue

        fav_set = set(fav_categories)
        user_emitted = False
        for query in synthetic_queries:
            # Run the search to get candidates the *live* system would
            # have returned. We turn off the reranker here because we
            # only need the candidate set + raw retrieval scores; the
            # cross-encoder rerank would be wasted work.
            try:
                results = search_engine.search(
                    query,
                    top_k=candidates_per_query,
                    use_reranker=False,
                )
            except TypeError:
                # Minimal fakes used in unit tests may not accept
                # ``use_reranker``.
                results = search_engine.search(query, top_k=candidates_per_query)

            candidate_idx = _candidate_indices_from_results(results, catalog)
            if not candidate_idx:
                continue

            # Compute raw BM25 / semantic arrays for feature extraction.
            try:
                bm25_scores, semantic_scores = _bm25_and_semantic_scores(
                    search_engine, query
                )
            except AttributeError:
                # Minimal fakes that expose only ``search`` but not the
                # underlying BM25 / embedder won't be used in real
                # retraining; we skip them gracefully.
                continue

            # Extract LTR features through the existing model API so
            # the schema stays in lockstep with serving.
            ltr_helper = LTRModel()
            X = ltr_helper.extract_features(
                query=query,
                candidate_indices=candidate_idx,
                bm25_scores=bm25_scores,
                semantic_scores=semantic_scores,
                catalog=catalog,
            )

            # Build labels for the same candidates.
            labels: list[int] = []
            for pos in candidate_idx:
                row = catalog.iloc[pos]
                pid = int(row["product_id"])
                cat = cat_lookup.get(pid, "")
                labels.append(
                    _label_candidate(
                        product_id=pid,
                        user_purchases=user_purchase_counts,
                        favorite_categories=fav_set,
                        product_category=cat,
                    )
                )

            all_features_arrays.append(X)
            all_labels.extend(labels)
            group_sizes.append(len(candidate_idx))
            queries_seen += 1
            user_emitted = True

        if user_emitted:
            users_with_data.add(int(user_id))

    if not all_features_arrays:
        empty_features = pd.DataFrame(columns=feature_names)
        return empty_features, np.array([], dtype=np.int64), []

    features_array = np.vstack(all_features_arrays)
    features_df = pd.DataFrame(features_array, columns=feature_names)
    labels_array = np.asarray(all_labels, dtype=np.int64)

    # Stash the user count on the DataFrame so the orchestrator can
    # report the actual figure (not just the cap).
    features_df.attrs["n_users_with_data"] = len(users_with_data)

    logger.info(
        f"Built training data: {len(features_df):,} rows, "
        f"{queries_seen} queries, {len(users_with_data)} users"
    )
    return features_df, labels_array, group_sizes


def retrain_ltr_from_behavior(
    behavior_logger: Any,
    catalog: pd.DataFrame,
    search_engine: Any,
    output_path: Path,
    **build_kwargs: Any,
) -> dict:
    """Build training data, train XGBRanker, save to disk, return metrics.

    Args:
        behavior_logger: :class:`BehaviorLogger`-compatible instance.
        catalog: Product catalog DataFrame.
        search_engine: :class:`GrocerySearchEngine`-compatible instance.
        output_path: Where to save the trained model.
        **build_kwargs: Forwarded to :func:`build_training_data_from_behavior`
            (``max_users``, ``queries_per_user``, ``candidates_per_query``).

    Returns:
        Summary dict with the keys documented in the module-level
        contract (n_training_samples, n_queries, n_users,
        feature_importance, label_distribution, wall_clock_seconds,
        model_path).
    """
    t0 = time.time()

    features_df, labels, group = build_training_data_from_behavior(
        behavior_logger=behavior_logger,
        catalog=catalog,
        search_engine=search_engine,
        **build_kwargs,
    )

    if features_df.empty or not group:
        raise ValueError(
            "No training data could be built from the behavior log. "
            "Check that the behavior log has purchase events and that "
            "the catalog/search engine are wired correctly."
        )

    # XGBRanker can be invoked directly with our pre-built feature
    # matrix - we don't need to round-trip through ``LTRModel.train``
    # (which re-runs feature extraction). Instantiating LTRModel here
    # gives us identical hyperparameters and the .save() helper.
    #
    # ``n_jobs=1`` is intentional: on macOS, XGBoost's OpenMP runtime
    # can clash with the OpenMP loaded by sentence-transformers /
    # torch (both bring their own libomp). The result is a silent
    # SIGABRT inside the native thread pool. With ~45K rows the
    # single-threaded training still finishes in seconds, so the
    # cost is negligible.
    import xgboost as xgb

    ltr = LTRModel()
    ltr.model = xgb.XGBRanker(
        objective="rank:ndcg",
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        tree_method="hist",
        n_jobs=1,
        lambdarank_num_pair_per_sample=8,
        lambdarank_pair_method="topk",
    )
    ltr.model.fit(features_df.to_numpy(), labels, group=group)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ltr.save(output_path)

    # ---- Summary metrics ------------------------------------------------
    importance_arr = ltr.model.feature_importances_
    feature_importance = {
        name: float(score)
        for name, score in zip(ltr.feature_names, importance_arr)
    }

    label_distribution: dict[int, int] = {}
    unique, counts = np.unique(labels, return_counts=True)
    for lbl, cnt in zip(unique.tolist(), counts.tolist()):
        label_distribution[int(lbl)] = int(cnt)

    # Actual users that contributed at least one query (stashed by the
    # builder). Falls back to the cap if missing for any reason.
    n_users_actual = int(
        features_df.attrs.get(
            "n_users_with_data", build_kwargs.get("max_users", 1000)
        )
    )

    summary = {
        "n_training_samples": int(len(features_df)),
        "n_queries": int(len(group)),
        "n_users": n_users_actual,
        "feature_importance": feature_importance,
        "label_distribution": label_distribution,
        "wall_clock_seconds": float(time.time() - t0),
        "model_path": str(output_path),
    }
    return summary
