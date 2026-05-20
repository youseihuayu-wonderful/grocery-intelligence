"""Sanity check for ``src.agents.shopping_agent``.

* If ``OPENAI_API_KEY`` is set in the environment, runs the full
  ``recipe_to_cart_plan`` pipeline against the real LLM + real catalog
  for the query ``"Korean beef bowls"``.
* Otherwise stubs the ingredient list manually and runs
  ``match_ingredients_to_catalog`` to demo the matching step alone.

Either way, prints 5-8 matched products with their relevance scores.

Run with::

    cd /Users/shihuayu/grocery-intelligence
    source venv/bin/activate
    python scripts/test_shopping_agent.py
"""

# macOS libomp conflict workaround — must be set BEFORE numpy / torch /
# sentence-transformers / xgboost import. Loading the search engine pulls
# in sentence-transformers which can otherwise SIGABRT on macOS.
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
from pathlib import Path

# Make ``src.*`` importable when run as a plain script.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.agents.shopping_agent import (  # noqa: E402
    match_ingredients_to_catalog,
    recipe_to_cart_plan,
)
from src.search.engine import GrocerySearchEngine  # noqa: E402


CATALOG_PARQUET = ROOT / "data" / "processed" / "product_catalog.parquet"
EMBEDDINGS_NPY = ROOT / "data" / "embeddings" / "product_embeddings.npy"


# Fallback ingredient list used when OPENAI_API_KEY is unavailable.
_STUBBED_KOREAN_BEEF_BOWL_INGREDIENTS: list[dict] = [
    {"name": "ground beef",     "quantity": "1 lb",     "category_hint": "meat seafood"},
    {"name": "white rice",      "quantity": "2 cups",   "category_hint": "pantry"},
    {"name": "soy sauce",       "quantity": "1/4 cup",  "category_hint": "pantry"},
    {"name": "garlic",          "quantity": "3 cloves", "category_hint": "produce"},
    {"name": "green onion",     "quantity": "2 stalks", "category_hint": "produce"},
    {"name": "sesame oil",      "quantity": "1 tbsp",   "category_hint": "pantry"},
    {"name": "brown sugar",     "quantity": "2 tbsp",   "category_hint": "pantry"},
    {"name": "ginger",          "quantity": "1 tsp",    "category_hint": "produce"},
]


def _print_matches(matches, header: str) -> None:
    print()
    print("=" * 80)
    print(header)
    print("=" * 80)
    for m in matches:
        if m.matched_product is None:
            print(f"  [no match] {m.requested_name!r} (qty: {m.quantity})")
            continue
        p = m.matched_product
        pname = p.get("product_name", "?")
        pid = p.get("product_id", "?")
        dept = p.get("department", "?")
        cat = p.get("category", "?")
        print(
            f"  [conf {m.confidence:.3f}] {m.requested_name!r:30s} "
            f"-> #{pid} {pname!r} ({dept} / {cat}, qty: {m.quantity})"
        )


def main() -> int:
    if not CATALOG_PARQUET.exists():
        print(f"[error] catalog not found at {CATALOG_PARQUET}", file=sys.stderr)
        return 1

    print(f"[load] reading catalog from {CATALOG_PARQUET}")
    catalog = pd.read_parquet(CATALOG_PARQUET)
    print(f"[load] catalog has {len(catalog):,} products")

    embeddings = None
    if EMBEDDINGS_NPY.exists():
        print(f"[load] reading embeddings from {EMBEDDINGS_NPY}")
        embeddings = np.load(EMBEDDINGS_NPY)
        print(f"[load] embeddings shape: {embeddings.shape}")
    else:
        print("[load] no pre-computed embeddings, will be built on the fly")

    print("[load] initializing search engine ...")
    engine = GrocerySearchEngine(catalog=catalog, embeddings=embeddings)
    print("[load] search engine ready")

    has_api_key = bool(os.getenv("OPENAI_API_KEY"))

    if has_api_key:
        print()
        print("OPENAI_API_KEY is set — running full pipeline with real LLM.")
        result = recipe_to_cart_plan(
            recipe_query="Korean beef bowls",
            search_engine=engine,
            catalog=catalog,
        )
        print()
        print(f"recipe: {result['recipe']!r}")
        print(f"summary: {result['summary']}")
        print(f"ingredients extracted: {len(result['ingredients'])}")
        for ing in result["ingredients"]:
            print(
                f"  - {ing.get('name'):30s} qty={ing.get('quantity')!s:>12s} "
                f"hint={ing.get('category_hint')}"
            )
        _print_matches(result["matches"], "matched products (real LLM)")
    else:
        print()
        print(
            "OPENAI_API_KEY not set — using stubbed ingredient list "
            "to demo the matching step."
        )
        matches = match_ingredients_to_catalog(
            ingredients=_STUBBED_KOREAN_BEEF_BOWL_INGREDIENTS,
            search_engine=engine,
            catalog=catalog,
        )
        _print_matches(matches, "matched products (stubbed ingredients)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
