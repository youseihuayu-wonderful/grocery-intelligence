"""FastAPI application for Grocery Intelligence search and recommendation API."""

from fastapi import FastAPI, Query
from pydantic import BaseModel

app = FastAPI(
    title="Grocery Intelligence API",
    description="AI-powered grocery product search and substitute recommendation",
    version="0.1.0",
)


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    use_reranker: bool = True


class SubstituteRequest(BaseModel):
    product_id: int
    top_k: int = 5
    same_category: bool = True
    substitute_type: str = "similar"  # "similar", "healthier", "cheaper"


class ProductResult(BaseModel):
    product_id: int
    product_name: str
    category: str | None = None
    brand: str | None = None
    price: float | None = None
    relevance_score: float | None = None
    similarity_score: float | None = None
    substitution_reasons: list[str] | None = None


class SearchResponse(BaseModel):
    query: str
    rewritten_query: str | None = None
    results: list[ProductResult]
    explanation: str | None = None
    total_results: int


class SubstituteResponse(BaseModel):
    original_product: ProductResult
    substitutes: list[ProductResult]
    explanation: str | None = None


# Global engine instances (initialized on startup)
search_engine = None
substitute_recommender = None


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "0.1.0"}


@app.post("/search", response_model=SearchResponse)
async def search_products(request: SearchRequest):
    """Search for grocery products using hybrid semantic + keyword search."""
    if search_engine is None:
        return {"error": "Search engine not initialized. Run setup first."}

    results = search_engine.search(
        query=request.query,
        top_k=request.top_k,
        use_reranker=request.use_reranker,
    )

    return SearchResponse(
        query=request.query,
        results=[ProductResult(**r) for r in results],
        total_results=len(results),
    )


@app.post("/substitute", response_model=SubstituteResponse)
async def get_substitutes(request: SubstituteRequest):
    """Get substitute recommendations for an out-of-stock product."""
    if substitute_recommender is None:
        return {"error": "Recommender not initialized. Run setup first."}

    if request.substitute_type == "healthier":
        subs = substitute_recommender.find_healthier_alternatives(
            request.product_id, request.top_k
        )
    elif request.substitute_type == "cheaper":
        subs = substitute_recommender.find_cheaper_alternatives(
            request.product_id, request.top_k
        )
    else:
        subs = substitute_recommender.find_substitutes(
            request.product_id,
            top_k=request.top_k,
            same_category=request.same_category,
        )

    return SubstituteResponse(
        original_product=ProductResult(
            product_id=request.product_id,
            product_name="",  # Will be filled from catalog
        ),
        substitutes=[ProductResult(**s) for s in subs],
    )


@app.get("/products")
async def list_products(
    category: str | None = None,
    limit: int = Query(default=20, le=100),
):
    """List products, optionally filtered by category."""
    if search_engine is None:
        return {"error": "Search engine not initialized."}

    if category:
        results = search_engine.search_by_category(category, top_k=limit)
    else:
        results = search_engine.catalog.head(limit).to_dict("records")

    return {"products": results, "total": len(results)}
