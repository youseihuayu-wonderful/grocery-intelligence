"""Integration tests for the full search and recommendation pipeline.

These tests use the real product catalog and pre-computed embeddings.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture(scope="module")
def catalog():
    """Load the real product catalog."""
    path = DATA_DIR / "processed" / "product_catalog.parquet"
    if not path.exists():
        pytest.skip("Product catalog not found. Run data pipeline first.")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def embeddings():
    """Load pre-computed product embeddings."""
    path = DATA_DIR / "embeddings" / "product_embeddings.npy"
    if not path.exists():
        pytest.skip("Embeddings not found. Run embedding pipeline first.")
    return np.load(path)


@pytest.fixture(scope="module")
def embedder():
    """Load the embedding model."""
    from src.models.embeddings import ProductEmbedder
    return ProductEmbedder()


@pytest.fixture(scope="module")
def search_engine(catalog, embeddings, embedder):
    """Initialize the search engine with real data."""
    from src.search.engine import GrocerySearchEngine
    return GrocerySearchEngine(catalog=catalog, embeddings=embeddings, embedder=embedder)


@pytest.fixture(scope="module")
def recommender(catalog, embeddings, embedder):
    """Initialize the substitute recommender with real data."""
    from src.recommend.substitute import SubstituteRecommender
    return SubstituteRecommender(catalog=catalog, embedder=embedder, embeddings=embeddings)


# ── Search Engine Tests ──────────────────────────────────────────────────────


class TestSearchEngine:
    def test_catalog_loaded(self, search_engine):
        assert len(search_engine.catalog) > 40000

    def test_embeddings_shape(self, search_engine):
        n_products = len(search_engine.catalog)
        assert search_engine.embeddings.shape == (n_products, 384)

    def test_search_returns_results(self, search_engine):
        results = search_engine.search("yogurt", top_k=5, use_reranker=False)
        assert len(results) == 5
        assert all("product_name" in r for r in results)

    def test_search_yogurt_relevance(self, search_engine):
        results = search_engine.search("greek yogurt", top_k=10, use_reranker=False)
        names = [r["product_name"].lower() for r in results]
        yogurt_count = sum(1 for n in names if "yogurt" in n)
        assert yogurt_count >= 5, f"Expected >=5 yogurt results, got {yogurt_count}"

    def test_search_bread(self, search_engine):
        results = search_engine.search("whole wheat bread", top_k=10, use_reranker=False)
        categories = [r.get("category", "").lower() for r in results]
        bread_count = sum(1 for c in categories if "bread" in c)
        assert bread_count >= 5

    def test_search_by_category(self, search_engine):
        results = search_engine.search_by_category("yogurt", top_k=5)
        assert len(results) > 0
        assert all("yogurt" in r.get("category", "").lower() for r in results)

    def test_get_categories(self, search_engine):
        cats = search_engine.get_categories()
        assert len(cats) > 50
        assert "yogurt" in cats

    def test_empty_query(self, search_engine):
        results = search_engine.search("", top_k=5, use_reranker=False)
        assert isinstance(results, list)


# ── Substitute Recommender Tests ─────────────────────────────────────────────


class TestSubstituteRecommender:
    def test_find_substitutes(self, recommender, catalog):
        # Pick a yogurt product
        yogurts = catalog[catalog["category"] == "yogurt"]
        if yogurts.empty:
            pytest.skip("No yogurt products")
        pid = yogurts.iloc[0]["product_id"]

        subs = recommender.find_substitutes(pid, top_k=5, same_category=True)
        assert len(subs) > 0
        assert all(s.get("category") == "yogurt" for s in subs)
        assert all("similarity_score" in s for s in subs)

    def test_substitutes_exclude_original(self, recommender, catalog):
        pid = catalog.iloc[0]["product_id"]
        subs = recommender.find_substitutes(pid, top_k=5, same_category=False)
        sub_ids = [s["product_id"] for s in subs]
        assert pid not in sub_ids

    def test_find_healthier_alternatives(self, recommender, catalog):
        # Pick a product with nutrition data
        with_nutrition = catalog[catalog["sugar_100g"].notna()]
        if with_nutrition.empty:
            pytest.skip("No products with nutrition data")
        pid = with_nutrition.iloc[0]["product_id"]

        subs = recommender.find_healthier_alternatives(pid, top_k=3)
        assert isinstance(subs, list)

    def test_invalid_product_id(self, recommender):
        subs = recommender.find_substitutes(product_id=999999999, top_k=5)
        assert subs == []

    def test_substitution_reasons(self, recommender, catalog):
        yogurts = catalog[catalog["category"] == "yogurt"]
        if yogurts.empty:
            pytest.skip("No yogurt products")
        pid = yogurts.iloc[0]["product_id"]

        subs = recommender.find_substitutes(pid, top_k=3, same_category=True)
        if subs:
            assert "substitution_reasons" in subs[0]
            assert isinstance(subs[0]["substitution_reasons"], list)


# ── API Tests ────────────────────────────────────────────────────────────────


class TestAPIModels:
    """Test the API Pydantic models without running the server."""

    def test_search_request_defaults(self):
        from src.api.main import SearchRequest
        req = SearchRequest(query="yogurt")
        assert req.top_k == 10
        assert req.use_reranker is False

    def test_substitute_request_defaults(self):
        from src.api.main import SubstituteRequest
        req = SubstituteRequest(product_id=1)
        assert req.top_k == 5
        assert req.substitute_type == "similar"

    def test_product_result_optional_fields(self):
        from src.api.main import ProductResult
        pr = ProductResult(product_id=1, product_name="Test")
        assert pr.category is None
        assert pr.relevance_score is None


# ── LTR Model Tests ──────────────────────────────────────────────────────────


class TestLTRModel:
    def test_feature_names(self):
        from src.models.ltr import LTRModel
        ltr = LTRModel()
        assert "bm25_score" in ltr.feature_names
        assert "semantic_score" in ltr.feature_names
        assert "order_count_log" in ltr.feature_names
        assert len(ltr.feature_names) == 12

    def test_grade_map(self):
        from src.models.ltr import GRADE_MAP
        assert GRADE_MAP["a"] == 5
        assert GRADE_MAP["e"] == 1
        assert len(GRADE_MAP) == 5

    def test_model_not_trained(self):
        from src.models.ltr import LTRModel
        ltr = LTRModel()
        assert ltr.model is None
        assert ltr.feature_importance() == {}
