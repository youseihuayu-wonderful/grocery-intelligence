"""High-level shopping agent that orchestrates LLM + search to fulfill goals.

This module sits one layer above :mod:`src.search.engine`. It uses the
LLM (see :mod:`src.models.llm`) to interpret natural-language shopping
goals — a recipe (\"Korean beef bowls\") or a higher-level objective
(\"Healthy lunches for the work week, under $50 total\") — into a
structured ingredient list, then resolves every ingredient to a real
product from the catalog via :class:`GrocerySearchEngine`.

Public API (in order of increasing abstraction):

* :func:`match_ingredients_to_catalog` — list of ingredient dicts ->
  list of :class:`MatchedIngredient` (one per requested ingredient).
* :func:`recipe_to_cart_plan` — recipe text -> ingredient extraction ->
  catalog matches.
* :func:`plan_to_cart_plan` — goal text -> structured plan -> category-
  grouped catalog matches, capped at ``max_products``.

All functions accept an optional ``user_dietary_preferences`` list
(e.g. ``[\"dairy-free\", \"vegan\"]``); when supplied, ambiguous
ingredient queries (\"milk\", \"yogurt\") prefer search results whose
``attributes`` field matches.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import pandas as pd
from loguru import logger

from src.models.llm import extract_recipe_ingredients, plan_shopping_goal


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class MatchedIngredient:
    """One requested ingredient resolved (or not) to a catalog product."""

    requested_name: str
    quantity: str | None
    matched_product: dict | None  # full product row from catalog, or None
    confidence: float              # 0..1 — uses the search relevance_score

    def to_dict(self) -> dict:
        """JSON-serializable representation."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------


# Ingredients where it's worth nudging the result toward the user's
# dietary preference (e.g. lactose-free user searching for "milk").
# We keep this small and obvious — most ingredients don't need it.
_AMBIGUOUS_TOKENS: frozenset[str] = frozenset(
    {
        "milk", "yogurt", "yoghurt", "cheese", "butter", "cream",
        "ice cream", "icecream",
        "chicken", "beef", "pork", "turkey", "ham", "bacon", "sausage",
        "bread", "pasta", "noodles", "tortilla", "tortillas",
        "rice", "cereal", "oats", "oatmeal",
        "sauce", "dressing", "mayo", "mayonnaise",
        "snack", "snacks", "cracker", "crackers", "cookie", "cookies",
        "juice", "soda", "drink",
    }
)


def _ingredient_is_ambiguous(name: str) -> bool:
    """Cheap heuristic: does the ingredient name hint at a category broad
    enough that a dietary preference might matter?"""
    if not name:
        return False
    lowered = name.lower().strip()
    if lowered in _AMBIGUOUS_TOKENS:
        return True
    # Also catch multi-word names that contain an ambiguous token, e.g.
    # "whole milk" or "shredded cheese".
    return any(token in lowered.split() for token in _AMBIGUOUS_TOKENS)


def _attribute_matches_preference(
    product_attrs: Any,
    preferences: list[str],
) -> bool:
    """True iff any user preference is present in the product's attributes."""
    if not preferences or not product_attrs:
        return False
    try:
        attr_set = set(product_attrs)
    except TypeError:
        return False
    return any(pref in attr_set for pref in preferences)


def _pick_best_match(
    candidates: list[dict],
    requested_name: str,
    preferences: list[str] | None,
) -> dict | None:
    """Return the single best product from a list of search candidates.

    When ``preferences`` is provided AND the ingredient is ambiguous,
    prefer the highest-scoring candidate whose ``attributes`` include
    one of the preferred attributes. Otherwise return the top result.
    """
    if not candidates:
        return None

    if preferences and _ingredient_is_ambiguous(requested_name):
        for cand in candidates:
            if _attribute_matches_preference(cand.get("attributes"), preferences):
                return cand

    return candidates[0]


# ---------------------------------------------------------------------------
# Core: ingredient -> product matching
# ---------------------------------------------------------------------------


def match_ingredients_to_catalog(
    ingredients: list[dict],
    search_engine,
    catalog: pd.DataFrame,
    user_dietary_preferences: list[str] | None = None,
) -> list[MatchedIngredient]:
    """For each ingredient, run the search engine to find the best catalog match.

    Steps per ingredient:
        1. Build a query from the ingredient name.
        2. Call ``search_engine.search(query, top_k=5)``.
        3. If ``user_dietary_preferences`` provided AND the requested item
           is ambiguous (e.g. ``milk``), prefer a result with a matching
           ``attributes`` entry (e.g. ``dairy-free``).
        4. Return the chosen match with its ``relevance_score`` as
           confidence (clipped to ``[0, 1]``).

    Returns a list of :class:`MatchedIngredient` in the same order as
    the input. Empty input -> empty output. No match -> ``matched_product
    is None`` and ``confidence == 0.0``.
    """
    # ``catalog`` is accepted for API consistency / future use (e.g. a
    # second-pass lookup); the search engine already references it. The
    # parameter is retained so callers can keep one source of truth.
    del catalog

    results: list[MatchedIngredient] = []
    if not ingredients:
        return results

    for ing in ingredients:
        if not isinstance(ing, dict):
            continue
        name = (ing.get("name") or "").strip()
        quantity = ing.get("quantity")
        if not name:
            results.append(
                MatchedIngredient(
                    requested_name="",
                    quantity=quantity,
                    matched_product=None,
                    confidence=0.0,
                )
            )
            continue

        try:
            candidates = search_engine.search(
                query=name,
                top_k=5,
                use_reranker=False,
            )
        except TypeError:
            # Tolerate fake search engines whose ``search`` signature
            # doesn't accept ``use_reranker`` keyword arg.
            candidates = search_engine.search(name, 5)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"search failed for ingredient '{name}': {exc}")
            candidates = []

        match = _pick_best_match(
            candidates, requested_name=name,
            preferences=user_dietary_preferences,
        )

        if match is None:
            results.append(
                MatchedIngredient(
                    requested_name=name,
                    quantity=quantity,
                    matched_product=None,
                    confidence=0.0,
                )
            )
            continue

        raw_score = match.get("relevance_score")
        try:
            score = float(raw_score) if raw_score is not None else 0.0
        except (TypeError, ValueError):
            score = 0.0
        confidence = max(0.0, min(1.0, score))

        results.append(
            MatchedIngredient(
                requested_name=name,
                quantity=quantity,
                matched_product=match,
                confidence=confidence,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Recipe pipeline
# ---------------------------------------------------------------------------


def recipe_to_cart_plan(
    recipe_query: str,
    search_engine,
    catalog: pd.DataFrame,
    llm_client=None,
    user_dietary_preferences: list[str] | None = None,
) -> dict:
    """Full pipeline: recipe text -> ingredient list -> catalog matches.

    Returns::

        {
            "recipe": str,                          # original query
            "ingredients": list[dict],              # raw LLM output
            "matches": list[MatchedIngredient],     # one per ingredient
            "summary": str,                         # 1-sentence summary
        }
    """
    ingredients = extract_recipe_ingredients(recipe_query, client=llm_client)
    matches = match_ingredients_to_catalog(
        ingredients=ingredients,
        search_engine=search_engine,
        catalog=catalog,
        user_dietary_preferences=user_dietary_preferences,
    )

    matched_count = sum(1 for m in matches if m.matched_product is not None)
    summary = (
        f"Found {matched_count} of {len(ingredients)} ingredients for "
        f"'{recipe_query}'."
    )

    return {
        "recipe": recipe_query,
        "ingredients": ingredients,
        "matches": matches,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Goal-level planning pipeline
# ---------------------------------------------------------------------------


def plan_to_cart_plan(
    goal: str,
    search_engine,
    catalog: pd.DataFrame,
    llm_client=None,
    max_products: int = 20,
    user_dietary_preferences: list[str] | None = None,
) -> dict:
    """Higher-level goal -> shopping plan.

    Uses :func:`plan_shopping_goal` from :mod:`src.models.llm` to
    interpret the goal, then runs catalog matching on the structured
    items. Limits to ``max_products`` total to keep responses bounded.

    Returns::

        {
            "goal": str,
            "interpretation": str,
            "categories": list[dict],   # [{"category": ..., "items": [MatchedIngredient]}]
            "total_products": int,
            "notes": str,
        }
    """
    plan = plan_shopping_goal(goal, client=llm_client)

    interpretation = plan.get("interpretation", "")
    notes = plan.get("notes", "")
    raw_categories = plan.get("shopping_categories", []) or []

    out_categories: list[dict] = []
    total_products = 0
    budget_remaining = max(0, int(max_products))

    for cat in raw_categories:
        if budget_remaining <= 0:
            break
        if not isinstance(cat, dict):
            continue
        label = cat.get("category", "")
        items = cat.get("items", []) or []
        if not isinstance(items, list):
            continue

        # Convert plain item names into the ingredient-dict shape that
        # match_ingredients_to_catalog understands.
        truncated = items[:budget_remaining]
        ingredient_dicts = [
            {"name": str(item), "quantity": None, "category_hint": label}
            for item in truncated
            if item
        ]
        matched = match_ingredients_to_catalog(
            ingredients=ingredient_dicts,
            search_engine=search_engine,
            catalog=catalog,
            user_dietary_preferences=user_dietary_preferences,
        )

        out_categories.append({"category": label, "items": matched})
        total_products += len(matched)
        budget_remaining -= len(matched)

    return {
        "goal": goal,
        "interpretation": interpretation,
        "categories": out_categories,
        "total_products": total_products,
        "notes": notes,
    }
