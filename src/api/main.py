"""FastAPI application for Grocery Intelligence search and recommendation API."""

from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger

DATA_DIR = Path(__file__).parent.parent.parent / "data"


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    use_reranker: bool = False


class SubstituteRequest(BaseModel):
    product_id: int
    top_k: int = 5
    same_category: bool = True
    substitute_type: str = "similar"  # "similar", "healthier", "cheaper"


class ProductResult(BaseModel):
    product_id: int
    product_name: str
    category: str | None = None
    department: str | None = None
    brand: str | None = None
    calories_100g: float | None = None
    protein_100g: float | None = None
    sugar_100g: float | None = None
    nutrition_grade: str | None = None
    order_count: int | None = None
    reorder_rate: float | None = None
    relevance_score: float | None = None
    similarity_score: float | None = None
    substitution_reasons: list[str] | None = None


class SearchResponse(BaseModel):
    query: str
    results: list[ProductResult]
    total_results: int


class SubstituteResponse(BaseModel):
    original_product: ProductResult
    substitutes: list[ProductResult]


# Global engine instances
_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load data and initialize engines on startup."""
    logger.info("Loading product catalog...")
    catalog = pd.read_parquet(DATA_DIR / "processed" / "product_catalog.parquet")
    embeddings = np.load(DATA_DIR / "embeddings" / "product_embeddings.npy")
    logger.info(f"Loaded {len(catalog)} products, embeddings shape: {embeddings.shape}")

    from src.search.engine import GrocerySearchEngine
    from src.recommend.substitute import SubstituteRecommender

    logger.info("Initializing search engine...")
    _state["search_engine"] = GrocerySearchEngine(
        catalog=catalog, embeddings=embeddings
    )

    logger.info("Initializing substitute recommender...")
    _state["recommender"] = SubstituteRecommender(
        catalog=catalog, embeddings=embeddings
    )

    _state["catalog"] = catalog
    logger.info("All engines initialized. API ready.")
    yield
    _state.clear()


app = FastAPI(
    title="Grocery Intelligence API",
    description="AI-powered grocery product search and substitute recommendation",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    ready = "search_engine" in _state
    return {
        "status": "healthy" if ready else "loading",
        "version": "0.1.0",
        "products_loaded": len(_state.get("catalog", [])),
    }


@app.post("/search", response_model=SearchResponse)
async def search_products(request: SearchRequest):
    """Search for grocery products using hybrid semantic + keyword search."""
    engine = _state.get("search_engine")
    if engine is None:
        raise HTTPException(503, "Search engine not initialized yet")

    results = engine.search(
        query=request.query,
        top_k=request.top_k,
        use_reranker=request.use_reranker,
    )

    return SearchResponse(
        query=request.query,
        results=[ProductResult(**_clean_product(r)) for r in results],
        total_results=len(results),
    )


@app.post("/substitute", response_model=SubstituteResponse)
async def get_substitutes(request: SubstituteRequest):
    """Get substitute recommendations for an out-of-stock product."""
    recommender = _state.get("recommender")
    catalog = _state.get("catalog")
    if recommender is None:
        raise HTTPException(503, "Recommender not initialized yet")

    # Get original product info
    original = catalog[catalog["product_id"] == request.product_id]
    if original.empty:
        raise HTTPException(404, f"Product {request.product_id} not found")
    original_dict = original.iloc[0].to_dict()

    if request.substitute_type == "healthier":
        subs = recommender.find_healthier_alternatives(
            request.product_id, request.top_k
        )
    elif request.substitute_type == "cheaper":
        subs = recommender.find_cheaper_alternatives(
            request.product_id, request.top_k
        )
    else:
        subs = recommender.find_substitutes(
            request.product_id,
            top_k=request.top_k,
            same_category=request.same_category,
        )

    return SubstituteResponse(
        original_product=ProductResult(**_clean_product(original_dict)),
        substitutes=[ProductResult(**_clean_product(s)) for s in subs],
    )


@app.get("/products")
async def list_products(
    category: str | None = None,
    limit: int = Query(default=20, le=100),
):
    """List products, optionally filtered by category."""
    engine = _state.get("search_engine")
    if engine is None:
        raise HTTPException(503, "Search engine not initialized")

    if category:
        results = engine.search_by_category(category, top_k=limit)
    else:
        results = engine.catalog.head(limit).to_dict("records")

    return {"products": [_clean_product(r) for r in results], "total": len(results)}


@app.get("/categories")
async def list_categories():
    """List all product categories."""
    engine = _state.get("search_engine")
    if engine is None:
        raise HTTPException(503, "Search engine not initialized")
    return {"categories": engine.get_categories()}


def _clean_product(d: dict) -> dict:
    """Clean a product dict for JSON serialization (handle NaN values)."""
    cleaned = {}
    for k, v in d.items():
        if isinstance(v, float) and np.isnan(v):
            cleaned[k] = None
        elif isinstance(v, (np.integer,)):
            cleaned[k] = int(v)
        elif isinstance(v, (np.floating,)):
            cleaned[k] = float(v)
        else:
            cleaned[k] = v
    return cleaned
