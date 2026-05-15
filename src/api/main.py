"""FastAPI application for Grocery Intelligence search and recommendation API."""

from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger

from src.models.llm import answer_question

DATA_DIR = Path(__file__).parent.parent.parent / "data"


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    use_reranker: bool = False
    attributes: list[str] | None = None  # filter results to only products with ALL of these attributes
    user_id: int | None = None  # personalize ranking if user has a profile


class EventRequest(BaseModel):
    product_id: int
    event_type: str  # "view" | "click" | "add_to_cart" | "purchase"
    user_id: int | None = None
    query: str | None = None
    position: int | None = None


class FeedResponse(BaseModel):
    feed_type: str
    products: list["ProductResult"]
    user_id: int | None = None


class SubstituteRequest(BaseModel):
    product_id: int
    top_k: int = 5
    same_category: bool = True
    substitute_type: str = "similar"  # "similar", "healthier", "cheaper"


class QARequest(BaseModel):
    question: str
    top_k: int = 5  # how many products to retrieve as context


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
    badges: list[str] | None = None
    attributes: list[str] | None = None
    personalization_score: float | None = None
    final_score: float | None = None
    feed_score: float | None = None


class SearchResponse(BaseModel):
    query: str
    results: list[ProductResult]
    total_results: int


class SubstituteResponse(BaseModel):
    original_product: ProductResult
    substitutes: list[ProductResult]


class QAResponse(BaseModel):
    question: str
    answer: str
    referenced_products: list[ProductResult]
    model: str


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
    from src.recommend.badges import compute_badges
    from src.search.attributes import extract_attributes_bulk

    logger.info("Initializing search engine...")
    _state["search_engine"] = GrocerySearchEngine(
        catalog=catalog, embeddings=embeddings
    )

    logger.info("Initializing substitute recommender...")
    _state["recommender"] = SubstituteRecommender(
        catalog=catalog, embeddings=embeddings
    )

    logger.info("Computing product badges...")
    _state["badges"] = compute_badges(catalog)
    n_with_badges = sum(1 for b in _state["badges"].values() if b)
    logger.info(f"Badges computed for {n_with_badges} products")

    logger.info("Extracting product attributes...")
    _state["attributes"] = extract_attributes_bulk(catalog)
    n_with_attrs = sum(1 for a in _state["attributes"].values() if a)
    logger.info(f"Attributes extracted for {n_with_attrs} products")

    fbt_path = DATA_DIR / "processed" / "fbt_model.parquet"
    if fbt_path.exists():
        from src.recommend.fbt import FrequentlyBoughtTogether
        logger.info("Loading FBT model...")
        _state["fbt"] = FrequentlyBoughtTogether().load(fbt_path)
        logger.info("FBT model loaded")
    else:
        logger.info("FBT model not found — skipping. Run scripts/build_fbt.py to enable.")
        _state["fbt"] = None

    profiles_path = DATA_DIR / "processed" / "user_profiles.parquet"
    if profiles_path.exists():
        from src.recommend.personalization import UserPersonalizationStore
        logger.info("Loading user personalization store...")
        _state["personalization"] = UserPersonalizationStore.load(profiles_path)
        logger.info(f"Loaded {len(_state['personalization'].profiles)} user profiles")
    else:
        logger.info("User profiles not found — skipping. Run scripts/build_user_profiles.py to enable.")
        _state["personalization"] = None

    behavior_path = DATA_DIR / "processed" / "behavior.db"
    if behavior_path.exists():
        from src.recommend.behavior import BehaviorLogger
        logger.info("Connecting to behavior log...")
        _state["behavior"] = BehaviorLogger(behavior_path)
        logger.info(f"Behavior log connected ({_state['behavior'].count_events():,} events)")
    else:
        from src.recommend.behavior import BehaviorLogger
        logger.info("Behavior log will be created on first event.")
        _state["behavior"] = BehaviorLogger(behavior_path)

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
    """Search for grocery products using hybrid semantic + keyword search.

    Final ranking blends relevance + popularity + (optional) personalization.
    """
    engine = _state.get("search_engine")
    if engine is None:
        raise HTTPException(503, "Search engine not initialized yet")

    fetch_k = max(request.top_k * 5, 50)

    results = engine.search(
        query=request.query,
        top_k=fetch_k,
        use_reranker=request.use_reranker,
    )

    cleaned = [_clean_product(r) for r in results]

    if request.attributes:
        required = set(request.attributes)
        cleaned = [
            r for r in cleaned
            if r.get("attributes") and required.issubset(set(r["attributes"]))
        ]

    from src.recommend.personalization import rerank_with_personalization
    store = _state.get("personalization")
    cleaned = rerank_with_personalization(
        cleaned,
        user_id=request.user_id,
        store=store,
        alpha=0.30 if request.user_id else 0.0,
        popularity_weight=0.25,
    )

    cleaned = cleaned[: request.top_k]

    return SearchResponse(
        query=request.query,
        results=[ProductResult(**r) for r in cleaned],
        total_results=len(cleaned),
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


@app.post("/qa", response_model=QAResponse)
async def shopping_qa(request: QARequest):
    """Answer shopping questions grounded in the product catalog (RAG)."""
    engine = _state.get("search_engine")
    if engine is None:
        raise HTTPException(503, "Search engine not initialized yet")

    results = engine.search(
        query=request.question,
        top_k=request.top_k,
        use_reranker=False,
    )

    cleaned_products = [_clean_product(r) for r in results]

    try:
        llm_result = answer_question(request.question, cleaned_products)
    except Exception as e:
        raise HTTPException(500, f"LLM error: {str(e)}")

    return QAResponse(
        question=request.question,
        answer=llm_result["answer"],
        referenced_products=[ProductResult(**p) for p in cleaned_products],
        model=llm_result["model"],
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


@app.get("/attributes")
async def list_attributes():
    """List all supported attribute filters with human-readable labels."""
    from src.search.attributes import ALL_ATTRIBUTES, ATTRIBUTE_LABELS
    return {
        "attributes": [
            {"id": a, "label": ATTRIBUTE_LABELS.get(a, a)} for a in ALL_ATTRIBUTES
        ]
    }


@app.get("/users/demo")
async def list_demo_users(n: int = Query(default=20, le=100)):
    """List demo users spanning the order-count distribution.

    Returns users the frontend can show in a 'Sign in as ...' dropdown.
    """
    store = _state.get("personalization")
    if store is None:
        return {"users": []}
    return {"users": store.list_demo_users(n=n)}


@app.get("/users/{user_id}/profile")
async def get_user_profile(user_id: int):
    """Return a user's profile (favorite categories, departments, brands, top products)."""
    store = _state.get("personalization")
    if store is None:
        raise HTTPException(503, "Personalization not loaded")
    profile = store.get_profile(user_id)
    if profile is None:
        raise HTTPException(404, f"User {user_id} not in profile store")
    return {
        "user_id": profile.user_id,
        "total_orders": profile.total_orders,
        "avg_basket_size": profile.avg_basket_size,
        "favorite_products": profile.favorite_products[:10],
        "favorite_categories": dict(list(profile.favorite_categories.items())[:5]),
        "favorite_departments": dict(list(profile.favorite_departments.items())[:5]),
        "favorite_brands": dict(list(profile.favorite_brands.items())[:5]),
    }


@app.get("/feed/{feed_type}", response_model=FeedResponse)
async def get_feed(
    feed_type: str,
    top_k: int = Query(default=20, le=50),
    user_id: int | None = None,
    department: str | None = None,
):
    """Get a discovery feed (no query needed).

    feed_type: one of 'bestsellers', 'healthy-picks', 'for-you', 'department'
    """
    catalog = _state.get("catalog")
    if catalog is None:
        raise HTTPException(503, "Catalog not loaded")

    from src.recommend.feed import (
        get_bestsellers, get_healthy_picks, get_for_you, get_trending_in_department
    )

    if feed_type == "bestsellers":
        products = get_bestsellers(catalog, top_k=top_k)
    elif feed_type == "healthy-picks":
        products = get_healthy_picks(catalog, top_k=top_k)
    elif feed_type == "for-you":
        store = _state.get("personalization")
        if store is None or user_id is None:
            products = get_bestsellers(catalog, top_k=top_k)
        else:
            products = get_for_you(catalog, store, user_id, top_k=top_k)
    elif feed_type == "department":
        if not department:
            raise HTTPException(400, "department query param required for 'department' feed")
        products = get_trending_in_department(catalog, department, top_k=top_k)
    else:
        raise HTTPException(400, f"Unknown feed type: {feed_type}")

    cleaned = [_clean_product(p) for p in products]
    return FeedResponse(
        feed_type=feed_type,
        products=[ProductResult(**c) for c in cleaned],
        user_id=user_id,
    )


@app.get("/departments")
async def list_departments_endpoint():
    """List departments sorted by total order_count."""
    catalog = _state.get("catalog")
    if catalog is None:
        raise HTTPException(503, "Catalog not loaded")
    from src.recommend.feed import list_departments
    return {"departments": list_departments(catalog)}


@app.post("/events")
async def log_event(request: EventRequest):
    """Log a user behavioral event (view/click/add_to_cart/purchase)."""
    behavior = _state.get("behavior")
    if behavior is None:
        raise HTTPException(503, "Behavior logger not initialized")
    try:
        event_id = behavior.log_event(
            product_id=request.product_id,
            event_type=request.event_type,
            user_id=request.user_id,
            query=request.query,
            position=request.position,
        )
        return {"event_id": event_id, "status": "logged"}
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/related/{product_id}")
async def get_related_products(product_id: int, top_k: int = 10):
    """Frequently bought together — Amazon-style item-item recommendations."""
    fbt = _state.get("fbt")
    catalog = _state.get("catalog")
    if fbt is None:
        raise HTTPException(503, "FBT model not loaded. Run scripts/build_fbt.py to build it.")

    related = fbt.get_related(product_id, top_k=top_k)
    if not related:
        return {"product_id": product_id, "related": []}

    related_ids = [pid for pid, _ in related]
    scores = {pid: score for pid, score in related}
    products = catalog[catalog["product_id"].isin(related_ids)]

    rows = []
    for _, row in products.iterrows():
        d = _clean_product(row.to_dict())
        d["similarity_score"] = scores.get(d["product_id"])
        rows.append(d)

    rows.sort(key=lambda r: r.get("similarity_score") or 0, reverse=True)
    return {"product_id": product_id, "related": rows}


def _clean_product(d: dict) -> dict:
    """Clean a product dict for JSON serialization (handle NaN values).

    Also merges in precomputed badges and attributes by product_id.
    """
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

    pid = cleaned.get("product_id")
    if pid is not None:
        badges = _state.get("badges", {}).get(pid, []) or []
        attrs = _state.get("attributes", {}).get(pid, []) or []
        cleaned["badges"] = badges if badges else None
        cleaned["attributes"] = attrs if attrs else None

    return cleaned
