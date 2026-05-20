"""FastAPI application for Grocery Intelligence search and recommendation API."""

# macOS libomp conflict workaround — sentence-transformers (PyTorch), faiss-cpu,
# and xgboost each bundle their own libomp; loading all three in one process
# can SIGABRT. These env vars must be set BEFORE numpy / torch / xgboost import.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

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
    emoji: str | None = None
    image_url: str | None = None
    price: float | None = None


class SearchResponse(BaseModel):
    query: str
    results: list[ProductResult]
    total_results: int
    corrected_query: str | None = None  # populated if spelling correction changed the query


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

    from src.recommend.images import build_emoji_map
    logger.info("Building product emoji map...")
    _state["emoji_map"] = build_emoji_map(catalog)
    logger.info(f"Emoji map built for {len(_state['emoji_map'])} products")

    images_path = DATA_DIR / "processed" / "product_images.parquet"
    if images_path.exists():
        images_df = pd.read_parquet(images_path)
        _state["image_map"] = dict(zip(images_df["product_id"], images_df["image_url"]))
        logger.info(f"Loaded {len(_state['image_map'])} OFF product images")
    else:
        _state["image_map"] = {}

    experiments_path = Path(__file__).parent.parent.parent / "experiments" / "ranking_v1.yaml"
    if experiments_path.exists():
        from src.experiments.ab_testing import ExperimentRegistry
        _state["experiments"] = ExperimentRegistry.load(experiments_path)
        active = _state["experiments"].list_active()
        logger.info(f"Loaded experiments: {[e.name for e in active]}")
    else:
        _state["experiments"] = None

    from src.search.query_understanding import (
        QueryVocabulary, SpellingCorrector, AutocompleteSuggester,
    )
    logger.info("Building query understanding vocabulary...")
    _state["vocab"] = QueryVocabulary(catalog)
    _state["spelling"] = SpellingCorrector(_state["vocab"])
    _state["autocomplete"] = AutocompleteSuggester(_state["vocab"])
    logger.info(
        f"Vocab built: {len(_state['vocab'].unigrams):,} unigrams, "
        f"{len(_state['vocab'].full_names):,} full names"
    )

    ann_flat_path = DATA_DIR / "embeddings" / "ann_flat" / "index.faiss"
    if ann_flat_path.exists():
        from src.search.ann_index import ANNIndex
        try:
            _state["ann_index"] = ANNIndex.load(ann_flat_path.parent)
            logger.info(f"FAISS ANN index loaded ({len(_state['ann_index']):,} vectors)")
        except Exception as e:
            logger.warning(f"FAISS index load failed: {e}")
            _state["ann_index"] = None
    else:
        _state["ann_index"] = None

    ltr_behavioral_path = Path(__file__).parent.parent.parent / "models" / "ltr_behavioral.xgb"
    if ltr_behavioral_path.exists():
        from src.models.ltr import LTRModel
        try:
            _state["ltr_behavioral"] = LTRModel(ltr_behavioral_path)
            logger.info(f"Behavioral LTR model loaded from {ltr_behavioral_path.name}")
        except Exception as e:
            logger.warning(f"Behavioral LTR load failed: {e}")
            _state["ltr_behavioral"] = None
    else:
        _state["ltr_behavioral"] = None

    from src.shopping.cart import CartStore
    _state["cart"] = CartStore()
    logger.info("Cart/wishlist store ready")

    orders_csv = DATA_DIR / "raw" / "instacart" / "orders.csv"
    order_products_csv = DATA_DIR / "raw" / "instacart" / "order_products__prior.csv"
    if orders_csv.exists() and order_products_csv.exists():
        from src.shopping.orders import OrderHistoryStore
        logger.info("Loading order history (user_cap=10000)...")
        try:
            _state["order_history"] = OrderHistoryStore(
                orders_csv, order_products_csv, user_cap=10_000
            )
            logger.info("Order history loaded")
        except Exception as e:
            logger.warning(f"Order history load failed: {e}")
            _state["order_history"] = None
    else:
        logger.info("Instacart CSVs not found — order history unavailable")
        _state["order_history"] = None

    coview_path = DATA_DIR / "processed" / "coview_model.parquet"
    if coview_path.exists():
        from src.recommend.coview import CustomersAlsoViewed
        try:
            _state["coview"] = CustomersAlsoViewed.load(coview_path)
            logger.info(f"Coview model loaded ({len(_state['coview'].related_map)} products)")
        except Exception as e:
            logger.warning(f"Coview load failed: {e}")
            _state["coview"] = None
    else:
        _state["coview"] = None

    from src.infra.cache import CacheClient
    _state["cache"] = CacheClient()
    if _state["cache"].is_available():
        logger.info("Redis cache connected")
    else:
        logger.info("Redis not reachable — running without cache layer")

    from src.pricing.mock_prices import build_price_map
    logger.info("Generating mock prices for all products...")
    _state["price_map"] = build_price_map(catalog)
    logger.info(f"Price map built for {len(_state['price_map'])} products")

    from src.users.preferences import PreferenceStore
    _state["preferences"] = PreferenceStore()
    logger.info("Dietary preferences store ready")

    from src.shopping.subscriptions import SubscriptionStore
    _state["subscriptions"] = SubscriptionStore()
    logger.info("Subscription store ready")

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

from src.infra.metrics import install_metrics
install_metrics(app)


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
    Applies spelling correction transparently — if the corrected query has
    enough vocabulary support to differ from the input, we search the
    corrected version and return the suggestion alongside results.
    """
    engine = _state.get("search_engine")
    if engine is None:
        raise HTTPException(503, "Search engine not initialized yet")

    spelling = _state.get("spelling")
    effective_query = request.query
    corrected_query = None
    if spelling is not None:
        candidate = spelling.correct(request.query)
        if candidate and candidate != request.query:
            corrected_query = candidate
            effective_query = candidate

    from src.search.category_browse import is_category_query, browse_category
    catalog_df = _state.get("catalog")
    if catalog_df is not None and is_category_query(effective_query):
        browse_rows = browse_category(catalog_df, effective_query, top_k=request.top_k)
        cleaned = [_clean_product(r) for r in browse_rows]
        if request.attributes:
            required = set(request.attributes)
            cleaned = [
                r for r in cleaned
                if r.get("attributes") and required.issubset(set(r["attributes"]))
            ]
        cleaned = cleaned[: request.top_k]
        return SearchResponse(
            query=request.query,
            results=[ProductResult(**r) for r in cleaned],
            total_results=len(cleaned),
            corrected_query=corrected_query,
        )

    fetch_k = max(request.top_k * 5, 50)

    results = engine.search(
        query=effective_query,
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
    from src.experiments.ab_testing import get_variant_config
    store = _state.get("personalization")

    default_cfg = {
        "alpha": 0.30 if request.user_id else 0.0,
        "popularity_weight": 0.25,
    }
    registry = _state.get("experiments")
    if registry is not None:
        variant_name, variant_cfg = get_variant_config(
            registry,
            experiment_name="ranking_v1",
            user_id=request.user_id,
            default_config=default_cfg,
        )
    else:
        variant_name, variant_cfg = "control", default_cfg

    cleaned = rerank_with_personalization(
        cleaned,
        user_id=request.user_id,
        store=store,
        alpha=variant_cfg.get("alpha", default_cfg["alpha"]),
        popularity_weight=variant_cfg.get("popularity_weight", default_cfg["popularity_weight"]),
    )

    cleaned = cleaned[: request.top_k]

    return SearchResponse(
        query=request.query,
        results=[ProductResult(**r) for r in cleaned],
        total_results=len(cleaned),
        corrected_query=corrected_query,
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


@app.get("/suggest")
async def suggest_query(prefix: str = "", top_k: int = Query(default=8, le=20)):
    """Autocomplete suggestions for a query prefix.

    Returns up to top_k suggestion objects: {"text", "type", "score"}.
    Empty prefix returns popular categories.
    """
    autocomplete = _state.get("autocomplete")
    if autocomplete is None:
        return {"suggestions": []}
    return {"suggestions": autocomplete.suggest(prefix, top_k=top_k)}


@app.get("/correct")
async def correct_query(query: str):
    """Spelling correction for a query. Returns the corrected version
    plus a few alternative options for a 'did you mean?' UI."""
    spelling = _state.get("spelling")
    if spelling is None:
        return {"query": query, "corrected": query, "alternatives": []}
    return {
        "query": query,
        "corrected": spelling.correct(query),
        "alternatives": spelling.suggest_corrections(query, top_k=3),
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


@app.get("/experiments")
async def list_experiments(user_id: int | None = None):
    """List active A/B experiments + (optionally) which variant a user is in."""
    registry = _state.get("experiments")
    if registry is None:
        return {"experiments": []}

    from src.experiments.ab_testing import assign_variant
    output = []
    for exp in registry.list_active():
        item = {
            "name": exp.name,
            "description": exp.description,
            "variants": [
                {"name": v.name, "traffic_weight": v.traffic_weight, "config": v.config}
                for v in exp.variants
            ],
        }
        if user_id is not None:
            variant = assign_variant(exp, user_id)
            item["assigned_variant"] = variant.name
            item["assigned_config"] = variant.config
        output.append(item)
    return {"experiments": output}


@app.get("/experiments/{experiment_name}/metrics")
async def experiment_metrics(
    experiment_name: str,
    sample_users: int = Query(default=500, le=5000),
):
    """Compute per-variant metrics for an experiment from the behavior log.

    Samples `sample_users` random user_ids from the behavior log,
    assigns them to variants, then aggregates events per variant.
    """
    registry = _state.get("experiments")
    behavior = _state.get("behavior")
    if registry is None:
        raise HTTPException(404, "Experiments not loaded")
    if behavior is None:
        raise HTTPException(503, "Behavior log not initialized")

    experiment = registry.get(experiment_name)
    if experiment is None:
        raise HTTPException(404, f"Experiment {experiment_name} not found")

    from src.experiments.ab_testing import assign_variant
    from src.evaluation.online_metrics import compute_variant_metrics
    import sqlite3

    conn = sqlite3.connect(behavior.db_path) if hasattr(behavior, "db_path") else None
    user_ids = []
    try:
        if conn:
            rows = conn.execute(
                "SELECT DISTINCT user_id FROM events WHERE user_id IS NOT NULL "
                f"ORDER BY RANDOM() LIMIT {int(sample_users)}"
            ).fetchall()
            user_ids = [r[0] for r in rows]
    finally:
        if conn:
            conn.close()

    user_variant_map = {uid: assign_variant(experiment, uid).name for uid in user_ids}
    metrics = compute_variant_metrics(behavior, user_variant_map)
    return {
        "experiment": experiment_name,
        "sample_size": len(user_ids),
        "variants": {
            name: {
                "n_users": vm.n_users,
                "n_views": vm.n_views,
                "n_clicks": vm.n_clicks,
                "n_purchases": vm.n_purchases,
                "ctr": vm.ctr,
                "conversion_rate": vm.conversion_rate,
                "mrr_at_10": vm.mrr_at_10,
                "avg_click_position": vm.avg_click_position,
            }
            for name, vm in metrics.items()
        },
    }


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


class CartItemRequest(BaseModel):
    user_id: int
    product_id: int
    qty: int = 1


@app.post("/cart/add")
async def cart_add(req: CartItemRequest):
    """Add an item to the user's cart (increments qty if already present)."""
    _state["cart"].add_to_cart(req.user_id, req.product_id, req.qty)
    return {"status": "added", "cart_count": _state["cart"].cart_count(req.user_id)}


@app.post("/cart/update")
async def cart_update(req: CartItemRequest):
    """Set the qty for an item (removes when qty<=0)."""
    _state["cart"].update_cart_qty(req.user_id, req.product_id, req.qty)
    return {"status": "ok", "cart_count": _state["cart"].cart_count(req.user_id)}


@app.post("/cart/remove")
async def cart_remove(req: CartItemRequest):
    _state["cart"].remove_from_cart(req.user_id, req.product_id)
    return {"status": "removed"}


@app.get("/cart/{user_id}")
async def cart_get(user_id: int):
    """Return the user's full cart with product details merged from catalog."""
    catalog = _state.get("catalog")
    cart_rows = _state["cart"].get_cart(user_id)
    if not cart_rows or catalog is None:
        return {"items": [], "total_items": 0}
    pids = [r["product_id"] for r in cart_rows]
    products = catalog[catalog["product_id"].isin(pids)].set_index("product_id")
    items = []
    for row in cart_rows:
        if row["product_id"] in products.index:
            p = _clean_product(products.loc[row["product_id"]].to_dict())
            p["qty"] = row["qty"]
            items.append(p)
    return {"items": items, "total_items": sum(r["qty"] for r in cart_rows)}


@app.post("/cart/clear/{user_id}")
async def cart_clear(user_id: int):
    _state["cart"].clear_cart(user_id)
    return {"status": "cleared"}


class WishlistRequest(BaseModel):
    user_id: int
    product_id: int


@app.post("/wishlist/add")
async def wishlist_add(req: WishlistRequest):
    _state["cart"].add_to_wishlist(req.user_id, req.product_id)
    return {"status": "added"}


@app.post("/wishlist/remove")
async def wishlist_remove(req: WishlistRequest):
    _state["cart"].remove_from_wishlist(req.user_id, req.product_id)
    return {"status": "removed"}


@app.get("/wishlist/{user_id}")
async def wishlist_get(user_id: int):
    catalog = _state.get("catalog")
    rows = _state["cart"].get_wishlist(user_id)
    if not rows or catalog is None:
        return {"items": []}
    pids = [r["product_id"] for r in rows]
    products = catalog[catalog["product_id"].isin(pids)]
    items = [_clean_product(r) for _, r in products.iterrows()]
    return {"items": items}


@app.get("/orders/{user_id}")
async def orders_for_user(user_id: int, limit: int = Query(default=10, le=50)):
    """Return the user's prior orders, newest first."""
    store = _state.get("order_history")
    if store is None:
        raise HTTPException(503, "Order history not loaded")
    orders = store.get_user_orders(user_id)[:limit]
    return {"user_id": user_id, "order_count": len(orders), "orders": orders}


@app.get("/orders/buy-again/{user_id}")
async def buy_again(user_id: int, top_k: int = Query(default=10, le=30)):
    """Return user's most-frequently-purchased products with full product info."""
    store = _state.get("order_history")
    catalog = _state.get("catalog")
    if store is None:
        raise HTTPException(503, "Order history not loaded")
    pids = store.buy_again_top(user_id, top_k=top_k)
    if not pids:
        return {"user_id": user_id, "products": []}
    df = catalog[catalog["product_id"].isin(pids)].set_index("product_id")
    products = []
    for pid in pids:
        if pid in df.index:
            products.append(_clean_product(df.loc[pid].to_dict()))
    return {"user_id": user_id, "products": products}


@app.get("/coview/{product_id}")
async def coview_for_product(product_id: int, top_k: int = 10):
    """Customers also viewed — user-level item co-occurrence."""
    coview = _state.get("coview")
    catalog = _state.get("catalog")
    if coview is None:
        raise HTTPException(503, "Coview model not loaded. Run scripts/build_coview.py")
    related = coview.get_related(product_id, top_k=top_k)
    if not related:
        return {"product_id": product_id, "related": []}
    related_ids = [pid for pid, _ in related]
    scores = {pid: s for pid, s in related}
    rows = []
    df = catalog[catalog["product_id"].isin(related_ids)]
    for _, row in df.iterrows():
        d = _clean_product(row.to_dict())
        d["similarity_score"] = scores.get(d["product_id"])
        rows.append(d)
    rows.sort(key=lambda r: r.get("similarity_score") or 0, reverse=True)
    return {"product_id": product_id, "related": rows}


@app.get("/compare")
async def compare_endpoint(ids: str):
    """Side-by-side product comparison. ids is comma-separated product_ids."""
    catalog = _state.get("catalog")
    try:
        product_ids = [int(x) for x in ids.split(",") if x.strip()]
    except ValueError:
        raise HTTPException(400, "ids must be a comma-separated list of integers")
    if len(product_ids) < 2 or len(product_ids) > 6:
        raise HTTPException(400, "compare requires 2-6 product ids")
    from src.recommend.compare import compare_products
    try:
        result = compare_products(catalog, product_ids)
    except Exception as e:
        raise HTTPException(400, str(e))
    return result


class PreferencesRequest(BaseModel):
    user_id: int
    dietary_attributes: list[str] | None = None
    excluded_attributes: list[str] | None = None


class FollowupRequest(BaseModel):
    previous_query: str
    previous_filters: dict | None = None
    user_followup: str
    user_id: int | None = None
    top_k: int = 10


@app.post("/search/followup")
async def search_followup(req: FollowupRequest):
    """Multi-turn conversational search: interpret follow-up and re-run search."""
    from src.models.llm import conversational_search_followup
    try:
        interpretation = conversational_search_followup(
            previous_query=req.previous_query,
            previous_filters=req.previous_filters or {},
            user_followup=req.user_followup,
        )
    except Exception as e:
        raise HTTPException(500, f"LLM follow-up interpretation failed: {e}")

    new_query = interpretation.get("interpreted_query", req.previous_query)
    new_filters = interpretation.get("interpreted_filters", {}) or {}
    new_attrs = new_filters.get("attributes") or None

    search_req = SearchRequest(
        query=new_query,
        top_k=req.top_k,
        attributes=new_attrs,
        user_id=req.user_id,
    )
    search_resp = await search_products(search_req)
    return {
        "interpreted_query": new_query,
        "interpreted_filters": new_filters,
        "clarification": interpretation.get("clarification", ""),
        "search_response": search_resp.model_dump() if hasattr(search_resp, "model_dump") else search_resp.dict(),
    }


@app.get("/preferences/{user_id}")
async def get_user_preferences(user_id: int):
    prefs = _state["preferences"].get_preferences(user_id)
    return prefs


@app.post("/preferences")
async def set_user_preferences(req: PreferencesRequest):
    _state["preferences"].set_preferences(
        req.user_id,
        dietary_attributes=req.dietary_attributes or [],
        excluded_attributes=req.excluded_attributes or [],
    )
    return {"status": "saved", "preferences": _state["preferences"].get_preferences(req.user_id)}


class RecipeRequest(BaseModel):
    recipe: str
    user_id: int | None = None


@app.post("/agent/recipe-to-cart")
async def recipe_to_cart(req: RecipeRequest):
    """LLM extracts ingredients from a recipe, matches each to a catalog product."""
    engine = _state.get("search_engine")
    catalog = _state.get("catalog")
    if engine is None:
        raise HTTPException(503, "Search engine not initialized")

    user_prefs = None
    if req.user_id and _state.get("preferences"):
        prefs = _state["preferences"].get_preferences(req.user_id)
        user_prefs = prefs.get("dietary_attributes") or None

    from src.agents.shopping_agent import recipe_to_cart_plan
    try:
        plan = recipe_to_cart_plan(
            req.recipe,
            search_engine=engine,
            catalog=catalog,
            user_dietary_preferences=user_prefs,
        )
    except Exception as e:
        raise HTTPException(500, f"Recipe planning failed: {e}")

    matches_out = []
    for m in plan["matches"]:
        product = m.matched_product
        if product is not None:
            cleaned = _clean_product(product)
            matches_out.append({
                "requested_name": m.requested_name,
                "quantity": m.quantity,
                "confidence": m.confidence,
                "product": cleaned,
            })
        else:
            matches_out.append({
                "requested_name": m.requested_name,
                "quantity": m.quantity,
                "confidence": 0.0,
                "product": None,
            })

    return {
        "recipe": plan["recipe"],
        "summary": plan["summary"],
        "ingredients": plan["ingredients"],
        "matches": matches_out,
    }


class GoalRequest(BaseModel):
    goal: str
    user_id: int | None = None
    max_products: int = 15


@app.post("/agent/plan-shopping")
async def plan_shopping(req: GoalRequest):
    """Higher-level goal interpretation into a shopping plan."""
    engine = _state.get("search_engine")
    catalog = _state.get("catalog")
    if engine is None:
        raise HTTPException(503, "Search engine not initialized")

    user_prefs = None
    if req.user_id and _state.get("preferences"):
        prefs = _state["preferences"].get_preferences(req.user_id)
        user_prefs = prefs.get("dietary_attributes") or None

    from src.agents.shopping_agent import plan_to_cart_plan
    try:
        plan = plan_to_cart_plan(
            req.goal,
            search_engine=engine,
            catalog=catalog,
            max_products=req.max_products,
            user_dietary_preferences=user_prefs,
        )
    except Exception as e:
        raise HTTPException(500, f"Goal planning failed: {e}")

    categories_out = []
    for cat in plan["categories"]:
        items_out = []
        for m in cat["items"]:
            if m.matched_product is not None:
                items_out.append({
                    "requested_name": m.requested_name,
                    "confidence": m.confidence,
                    "product": _clean_product(m.matched_product),
                })
        categories_out.append({"category": cat["category"], "items": items_out})

    return {
        "goal": plan["goal"],
        "interpretation": plan["interpretation"],
        "notes": plan["notes"],
        "categories": categories_out,
    }


@app.get("/pricing/{product_id}/history")
async def price_history_endpoint(product_id: int, days: int = 90):
    """Generate a 90-day price history chart for a product."""
    price_map = _state.get("price_map", {})
    if product_id not in price_map:
        raise HTTPException(404, "Product not found in price map")
    from src.pricing.history import generate_price_history, detect_price_drop
    history = generate_price_history(product_id, price_map[product_id], days=days)
    drop = detect_price_drop(history)
    return {"product_id": product_id, "current_price": price_map[product_id],
            "history": history, "drop_signal": drop}


@app.get("/cart/{user_id}/pricing")
async def cart_pricing_endpoint(user_id: int):
    """Returns subtotal, applied/available promotions, total."""
    cart_response = await cart_get(user_id)
    items = cart_response["items"]
    from src.pricing.promotions import cart_pricing_summary
    from dataclasses import asdict
    summary = cart_pricing_summary(items)
    summary["promotions_applied"] = [asdict(p) for p in summary["promotions_applied"]]
    summary["promotions_available"] = [asdict(p) for p in summary["promotions_available"]]
    return summary


class SubscribeRequest(BaseModel):
    user_id: int
    product_id: int
    frequency: str
    qty: int = 1


@app.post("/subscriptions/add")
async def subscriptions_add(req: SubscribeRequest):
    try:
        sid = _state["subscriptions"].subscribe(req.user_id, req.product_id, req.frequency, req.qty)
        return {"subscription_id": sid, "status": "active"}
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/subscriptions/{subscription_id}/cancel")
async def subscriptions_cancel(subscription_id: int):
    _state["subscriptions"].cancel(subscription_id)
    return {"status": "cancelled"}


@app.get("/subscriptions/{user_id}")
async def subscriptions_for_user(user_id: int):
    catalog = _state.get("catalog")
    subs = _state["subscriptions"].get_user_subscriptions(user_id)
    if not subs:
        return {"subscriptions": [], "estimated_monthly_value": 0.0}
    pids = [s["product_id"] for s in subs]
    df = catalog[catalog["product_id"].isin(pids)].set_index("product_id")
    enriched = []
    for s in subs:
        item = dict(s)
        if s["product_id"] in df.index:
            item["product"] = _clean_product(df.loc[s["product_id"]].to_dict())
        enriched.append(item)
    monthly_value = _state["subscriptions"].estimated_monthly_value(
        user_id, _state.get("price_map", {})
    )
    return {"subscriptions": enriched, "estimated_monthly_value": monthly_value}


@app.post("/subscriptions/fulfill-due")
async def subscriptions_fulfill():
    """Demo: simulate the cron firing right now — fulfills all due subs into carts."""
    from src.shopping.subscriptions import fulfill_due_subscriptions
    fired = fulfill_due_subscriptions(_state["subscriptions"], _state["cart"])
    return {"fired": fired, "count": len(fired)}


@app.get("/users/{user_id}/recently-viewed")
async def recently_viewed_endpoint(user_id: int, limit: int = Query(default=10, le=30)):
    from src.recommend.recently_viewed import get_recently_viewed
    catalog = _state.get("catalog")
    pids = get_recently_viewed(_state["behavior"], user_id, limit=limit)
    if not pids:
        return {"user_id": user_id, "products": []}
    df = catalog[catalog["product_id"].isin(pids)].set_index("product_id")
    products = []
    for pid in pids:
        if pid in df.index:
            products.append(_clean_product(df.loc[pid].to_dict()))
    return {"user_id": user_id, "products": products}


@app.get("/analytics/overview")
async def analytics_overview():
    """Aggregated analytics for an admin dashboard."""
    from src.analytics.dashboard import (
        top_queries, funnel_metrics, hot_products, daily_event_counts,
        category_breakdown, search_quality_signals,
    )
    from dataclasses import asdict
    logger_b = _state.get("behavior")
    catalog = _state.get("catalog")
    if logger_b is None:
        raise HTTPException(503, "Behavior log not initialized")
    funnel = funnel_metrics(logger_b)
    return {
        "top_queries": [asdict(q) for q in top_queries(logger_b, limit=10)],
        "funnel": asdict(funnel),
        "hot_products": [
            {"product_id": pid, "count": c}
            for pid, c in hot_products(logger_b, limit=10)
        ],
        "daily_counts": daily_event_counts(logger_b, days_back=14),
        "category_breakdown": category_breakdown(logger_b, catalog),
        "search_quality": search_quality_signals(logger_b),
    }


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
        cleaned["emoji"] = _state.get("emoji_map", {}).get(pid)
        cleaned["image_url"] = _state.get("image_map", {}).get(pid)
        cleaned["price"] = _state.get("price_map", {}).get(pid)

    return cleaned
