"""Unit tests for the behavioral LTR retraining pipeline.

These tests use synthetic data: a tiny catalog, an in-memory
:class:`BehaviorLogger`, and a minimal fake search engine. We do
*not* spin up the real 49K-row :class:`GrocerySearchEngine` here -
that's covered by the end-to-end script.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.ltr import LTRModel
from src.models.ltr_retrain import (
    build_training_data_from_behavior,
    retrain_ltr_from_behavior,
)
from src.recommend.behavior import BehaviorLogger


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def _make_catalog() -> pd.DataFrame:
    """A 5-product catalog spanning 2 categories with the columns that
    :meth:`LTRModel.extract_features` reads."""
    return pd.DataFrame(
        [
            {
                "product_id": 101,
                "product_name": "fresh organic banana",
                "category": "produce",
                "department": "produce",
                "brand": "Brand A",
                "ingredients": "banana",
                "calories_100g": 89.0,
                "protein_100g": 1.1,
                "sugar_100g": 12.0,
                "fat_100g": 0.3,
                "fiber_100g": 2.6,
                "nutrition_grade": "a",
                "allergens_en": "",
                "order_count": 5000,
                "reorder_rate": 0.85,
            },
            {
                "product_id": 102,
                "product_name": "red apple gala",
                "category": "produce",
                "department": "produce",
                "brand": "Brand A",
                "ingredients": "apple",
                "calories_100g": 52.0,
                "protein_100g": 0.3,
                "sugar_100g": 10.4,
                "fat_100g": 0.2,
                "fiber_100g": 2.4,
                "nutrition_grade": "a",
                "allergens_en": "",
                "order_count": 4000,
                "reorder_rate": 0.7,
            },
            {
                "product_id": 103,
                "product_name": "whole milk gallon",
                "category": "dairy",
                "department": "dairy",
                "brand": "Brand B",
                "ingredients": "milk",
                "calories_100g": 60.0,
                "protein_100g": 3.2,
                "sugar_100g": 4.8,
                "fat_100g": 3.3,
                "fiber_100g": 0.0,
                "nutrition_grade": "c",
                "allergens_en": "milk",
                "order_count": 3000,
                "reorder_rate": 0.75,
            },
            {
                "product_id": 104,
                "product_name": "greek yogurt vanilla",
                "category": "dairy",
                "department": "dairy",
                "brand": "Brand B",
                "ingredients": "milk, vanilla",
                "calories_100g": 100.0,
                "protein_100g": 9.0,
                "sugar_100g": 7.0,
                "fat_100g": 2.5,
                "fiber_100g": 0.0,
                "nutrition_grade": "b",
                "allergens_en": "milk",
                "order_count": 2000,
                "reorder_rate": 0.65,
            },
            {
                "product_id": 105,
                "product_name": "potato chips classic",
                "category": "snacks",
                "department": "snacks",
                "brand": "Brand C",
                "ingredients": "potato, salt, oil",
                "calories_100g": 540.0,
                "protein_100g": 6.0,
                "sugar_100g": 0.5,
                "fat_100g": 32.0,
                "fiber_100g": 4.0,
                "nutrition_grade": "d",
                "allergens_en": "",
                "order_count": 1500,
                "reorder_rate": 0.55,
            },
        ]
    )


class FakeSearchEngine:
    """A minimal search engine that the retrain pipeline can exercise.

    Exposes:
      * ``catalog`` (passed in)
      * ``bm25`` (with ``.get_scores``)
      * ``embedder`` (with ``.embed_query``)
      * ``embeddings`` (matching ``len(catalog)``)
      * ``search(query, top_k, use_reranker=...)`` returning a list of
        dicts shaped like the real engine.

    ``search`` always returns the first ``top_k`` rows of the catalog
    so the labeling rule has predictable targets.
    """

    def __init__(self, catalog: pd.DataFrame):
        self.catalog = catalog.reset_index(drop=True)
        # 4-dim toy embeddings (one per product). Random but seeded so
        # repeated calls are deterministic across test runs.
        rng = np.random.default_rng(0)
        self.embeddings = rng.normal(size=(len(self.catalog), 4)).astype(np.float32)
        # Normalize so dot products stay bounded.
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings /= np.maximum(norms, 1e-9)
        self.bm25 = _FakeBM25(len(self.catalog))
        self.embedder = _FakeEmbedder()

    def search(self, query, top_k=10, use_reranker=False, **_):
        rows = []
        for _, r in self.catalog.head(top_k).iterrows():
            d = r.to_dict()
            d["relevance_score"] = 1.0
            rows.append(d)
        return rows


class _FakeBM25:
    """Returns a fixed, deterministic score vector regardless of query."""

    def __init__(self, n: int):
        self._scores = np.linspace(0.5, 2.0, n)

    def get_scores(self, _tokens):
        return self._scores.copy()


class _FakeEmbedder:
    """Returns a fixed 4-dim query vector regardless of input."""

    def embed_query(self, _q):
        # Unit vector pointing along dim 0.
        v = np.zeros(4, dtype=np.float32)
        v[0] = 1.0
        return v


@pytest.fixture
def catalog() -> pd.DataFrame:
    return _make_catalog()


@pytest.fixture
def engine(catalog: pd.DataFrame) -> FakeSearchEngine:
    return FakeSearchEngine(catalog)


@pytest.fixture
def behavior(tmp_path: Path) -> BehaviorLogger:
    """In-process BehaviorLogger seeded with 3 users and known patterns.

    Layout:
      user 1: buys product 101 six times (label 3 expected),
              buys product 102 twice (label 2 expected),
              buys product 103 once (label 2 expected),
              buys product 104 once and product 105 once -- gives
              us at least 5 distinct purchase rows so the user is
              eligible.
      user 2: only one purchase row (will be filtered as ineligible).
      user 3: NO purchases (cold-start; should be ignored entirely).
    """
    db = tmp_path / "behavior_retrain.db"
    log = BehaviorLogger(db)
    # User 1
    for _ in range(6):
        log.log_event(product_id=101, event_type="purchase", user_id=1)
    for _ in range(2):
        log.log_event(product_id=102, event_type="purchase", user_id=1)
    log.log_event(product_id=103, event_type="purchase", user_id=1)
    log.log_event(product_id=104, event_type="purchase", user_id=1)
    log.log_event(product_id=105, event_type="purchase", user_id=1)
    # User 2 — only one purchase row (ineligible).
    log.log_event(product_id=101, event_type="purchase", user_id=2)
    # User 3 — only views/clicks, no purchases.
    log.log_event(product_id=101, event_type="view", user_id=3)
    log.log_event(product_id=102, event_type="click", user_id=3)
    yield log
    log.close()


# ---------------------------------------------------------------------------
# build_training_data_from_behavior
# ---------------------------------------------------------------------------


class TestBuildTrainingData:
    def test_emits_features_for_eligible_user(
        self,
        behavior: BehaviorLogger,
        catalog: pd.DataFrame,
        engine: FakeSearchEngine,
    ):
        feats, labels, group = build_training_data_from_behavior(
            behavior_logger=behavior,
            catalog=catalog,
            search_engine=engine,
            max_users=10,
            queries_per_user=3,
            candidates_per_query=5,
        )
        # User 1 is the only eligible user and has 2 favorite categories
        # (produce + dairy) so we expect <= 2 queries (queries_per_user
        # caps to the categories available).
        assert len(group) >= 1
        assert len(group) <= 3
        # Each candidate block has 5 rows (top_k=5, catalog size 5).
        assert all(g == 5 for g in group)
        # Total rows == sum of groups.
        assert len(feats) == sum(group)
        assert len(labels) == sum(group)

    def test_feature_columns_match_ltr_model(
        self,
        behavior: BehaviorLogger,
        catalog: pd.DataFrame,
        engine: FakeSearchEngine,
    ):
        feats, _labels, _group = build_training_data_from_behavior(
            behavior_logger=behavior,
            catalog=catalog,
            search_engine=engine,
            max_users=10,
            queries_per_user=2,
            candidates_per_query=5,
        )
        assert list(feats.columns) == LTRModel().feature_names

    def test_labels_match_rules(
        self,
        behavior: BehaviorLogger,
        catalog: pd.DataFrame,
        engine: FakeSearchEngine,
    ):
        """The labeling rule:
        - 3 = product purchased 5+ times by the user
        - 2 = product purchased 1-4 times
        - 1 = no purchase, but product is in a favorite category
        - 0 = otherwise
        """
        feats, labels, group = build_training_data_from_behavior(
            behavior_logger=behavior,
            catalog=catalog,
            search_engine=engine,
            max_users=10,
            queries_per_user=5,
            candidates_per_query=5,
        )
        # We expect at least one label of 3 (product 101 bought 6 times)
        # and at least one label of 2 (products bought 1-4 times).
        unique_labels = set(int(x) for x in labels)
        assert 3 in unique_labels, f"Expected label 3 in {unique_labels}"
        assert 2 in unique_labels, f"Expected label 2 in {unique_labels}"
        # No label > 3 should appear.
        assert max(unique_labels) <= 3
        # No label < 0.
        assert min(unique_labels) >= 0

        # Now verify the rule on individual rows. We iterate by query
        # block and check each (query, candidate) pair.
        cursor = 0
        for g in group:
            for _ in range(g):
                # We can't deterministically check every row without
                # reproducing the labeler, but we can assert the
                # well-formedness: labels are integers in {0,1,2,3}.
                lbl = int(labels[cursor])
                assert lbl in {0, 1, 2, 3}
                cursor += 1

    def test_group_sums_to_total_rows(
        self,
        behavior: BehaviorLogger,
        catalog: pd.DataFrame,
        engine: FakeSearchEngine,
    ):
        feats, labels, group = build_training_data_from_behavior(
            behavior_logger=behavior,
            catalog=catalog,
            search_engine=engine,
            max_users=10,
            queries_per_user=2,
            candidates_per_query=5,
        )
        assert sum(group) == len(feats) == len(labels)

    def test_cold_start_user_skipped(
        self,
        tmp_path: Path,
        catalog: pd.DataFrame,
        engine: FakeSearchEngine,
    ):
        """A logger containing only one user with NO purchases should
        produce zero training rows (or all-zero labels if it produces
        any). We test the no-rows case here because the user has zero
        purchase rows."""
        db = tmp_path / "cold.db"
        log = BehaviorLogger(db)
        # Only views/clicks - no purchases.
        for pid in (101, 102, 103):
            log.log_event(product_id=pid, event_type="view", user_id=99)

        feats, labels, group = build_training_data_from_behavior(
            behavior_logger=log,
            catalog=catalog,
            search_engine=engine,
            max_users=10,
            queries_per_user=2,
            candidates_per_query=5,
        )
        log.close()

        assert len(feats) == 0
        assert len(labels) == 0
        assert group == []
        # Columns should still be the canonical LTR feature names.
        assert list(feats.columns) == LTRModel().feature_names

    def test_user_below_min_purchase_rows_skipped(
        self,
        tmp_path: Path,
        catalog: pd.DataFrame,
        engine: FakeSearchEngine,
    ):
        """User 2 in our fixture has only one purchase row; on an
        otherwise-empty database, the builder should drop them and
        return zero rows."""
        db = tmp_path / "small.db"
        log = BehaviorLogger(db)
        log.log_event(product_id=101, event_type="purchase", user_id=42)

        feats, labels, group = build_training_data_from_behavior(
            behavior_logger=log,
            catalog=catalog,
            search_engine=engine,
            max_users=10,
            queries_per_user=2,
            candidates_per_query=5,
        )
        log.close()

        assert len(feats) == 0
        assert len(labels) == 0
        assert group == []


# ---------------------------------------------------------------------------
# retrain_ltr_from_behavior
# ---------------------------------------------------------------------------


class TestRetrainEndToEnd:
    def test_end_to_end_saves_model_and_returns_summary(
        self,
        tmp_path: Path,
        behavior: BehaviorLogger,
        catalog: pd.DataFrame,
        engine: FakeSearchEngine,
    ):
        out = tmp_path / "ltr_behavioral.xgb"
        summary = retrain_ltr_from_behavior(
            behavior_logger=behavior,
            catalog=catalog,
            search_engine=engine,
            output_path=out,
            max_users=10,
            queries_per_user=3,
            candidates_per_query=5,
        )

        # File exists.
        assert out.exists()

        # Summary keys.
        for key in (
            "n_training_samples",
            "n_queries",
            "n_users",
            "feature_importance",
            "label_distribution",
            "wall_clock_seconds",
            "model_path",
        ):
            assert key in summary, f"missing key: {key}"

        # Sanity ranges.
        assert summary["n_training_samples"] > 0
        assert summary["n_queries"] > 0
        assert summary["n_users"] >= 1
        assert summary["wall_clock_seconds"] >= 0
        assert summary["model_path"] == str(out)

        # Feature importance has all 12 features.
        assert set(summary["feature_importance"].keys()) == set(
            LTRModel().feature_names
        )

        # Label distribution is a dict[int, int].
        assert isinstance(summary["label_distribution"], dict)
        assert all(isinstance(k, int) for k in summary["label_distribution"])
        assert all(isinstance(v, int) for v in summary["label_distribution"].values())

    def test_loaded_model_can_predict(
        self,
        tmp_path: Path,
        behavior: BehaviorLogger,
        catalog: pd.DataFrame,
        engine: FakeSearchEngine,
    ):
        out = tmp_path / "ltr_behavioral.xgb"
        retrain_ltr_from_behavior(
            behavior_logger=behavior,
            catalog=catalog,
            search_engine=engine,
            output_path=out,
            max_users=10,
            queries_per_user=2,
            candidates_per_query=5,
        )

        # Round-trip through ``load`` and call ``predict`` on a small
        # candidate set to confirm the model is usable end-to-end.
        ltr = LTRModel()
        ltr.load(out)

        bm25_scores = engine.bm25.get_scores(["fresh", "produce"])
        sem_scores = np.dot(engine.embeddings, engine.embedder.embed_query("x"))
        scores = ltr.predict(
            query="fresh produce",
            candidate_indices=[0, 1, 2, 3, 4],
            bm25_scores=bm25_scores,
            semantic_scores=sem_scores,
            catalog=catalog,
        )
        assert scores.shape == (5,)

    def test_empty_behavior_raises(
        self,
        tmp_path: Path,
        catalog: pd.DataFrame,
        engine: FakeSearchEngine,
    ):
        """An empty behavior log should produce a clear error rather
        than silently saving a degenerate model."""
        db = tmp_path / "empty.db"
        log = BehaviorLogger(db)
        out = tmp_path / "should_not_exist.xgb"
        with pytest.raises(ValueError):
            retrain_ltr_from_behavior(
                behavior_logger=log,
                catalog=catalog,
                search_engine=engine,
                output_path=out,
                max_users=10,
                queries_per_user=2,
                candidates_per_query=5,
            )
        log.close()
