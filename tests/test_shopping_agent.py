"""Tests for src.agents.shopping_agent and the two LLM functions
``extract_recipe_ingredients`` and ``plan_shopping_goal``.

All LLM calls are mocked — no live OpenAI traffic. The search engine
is a tiny in-process fake.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.agents.shopping_agent import (
    MatchedIngredient,
    match_ingredients_to_catalog,
    plan_to_cart_plan,
    recipe_to_cart_plan,
)
from src.models import llm as llm_module


# ---------------------------------------------------------------------------
# Fakes / fixtures
# ---------------------------------------------------------------------------


class FakeSearchEngine:
    """Minimal stand-in for :class:`GrocerySearchEngine`.

    Constructed with a dict ``query -> list[product_dict]``. Returns the
    pre-canned results for an exact query match (case-insensitive),
    truncated to ``top_k``. Unknown queries return ``[]``.

    Records every call on ``self.calls`` as ``(query, top_k)`` tuples
    for assertions.
    """

    def __init__(self, responses: dict[str, list[dict]] | None = None):
        self.responses: dict[str, list[dict]] = {
            k.lower(): list(v) for k, v in (responses or {}).items()
        }
        self.calls: list[tuple[str, int]] = []

    def search(
        self,
        query: str,
        top_k: int = 10,
        use_reranker: bool = False,
    ) -> list[dict]:
        self.calls.append((query, top_k))
        results = self.responses.get(query.lower(), [])
        return results[:top_k]


def _product(
    product_id: int,
    name: str,
    relevance_score: float = 0.9,
    attributes: list[str] | None = None,
    **extras: Any,
) -> dict:
    """Build a product dict in the catalog-row shape."""
    row: dict[str, Any] = {
        "product_id": product_id,
        "product_name": name,
        "category": "test",
        "department": "test",
        "relevance_score": relevance_score,
    }
    if attributes is not None:
        row["attributes"] = attributes
    row.update(extras)
    return row


@pytest.fixture
def tiny_catalog() -> pd.DataFrame:
    """A 4-row dataframe just so we have something to pass through."""
    return pd.DataFrame(
        [
            {"product_id": 1, "product_name": "Ground Beef"},
            {"product_id": 2, "product_name": "White Rice"},
            {"product_id": 3, "product_name": "Whole Milk"},
            {"product_id": 4, "product_name": "Almond Milk"},
        ]
    )


# ---------------------------------------------------------------------------
# match_ingredients_to_catalog
# ---------------------------------------------------------------------------


def test_match_ingredients_three_items_preserves_order(
    tiny_catalog: pd.DataFrame,
) -> None:
    """3 ingredients in => 3 matches out, same order, confidence = score."""
    engine = FakeSearchEngine(
        {
            "ground beef": [_product(1, "Ground Beef 80/20", relevance_score=0.92)],
            "white rice": [_product(2, "Long Grain White Rice", relevance_score=0.85)],
            "garlic":     [_product(3, "Fresh Garlic Bulb",     relevance_score=0.78)],
        }
    )

    ingredients = [
        {"name": "ground beef", "quantity": "1 lb", "category_hint": "meat"},
        {"name": "white rice",  "quantity": "2 cups", "category_hint": "pantry"},
        {"name": "garlic",      "quantity": "3 cloves", "category_hint": "produce"},
    ]

    matches = match_ingredients_to_catalog(
        ingredients, search_engine=engine, catalog=tiny_catalog,
    )

    assert len(matches) == 3
    assert all(isinstance(m, MatchedIngredient) for m in matches)
    # Order preserved.
    assert [m.requested_name for m in matches] == ["ground beef", "white rice", "garlic"]
    # Confidence pulled from relevance_score.
    assert matches[0].confidence == pytest.approx(0.92)
    assert matches[1].confidence == pytest.approx(0.85)
    assert matches[2].confidence == pytest.approx(0.78)
    # Quantities passed through.
    assert matches[0].quantity == "1 lb"
    assert matches[2].quantity == "3 cloves"
    # Matched product is the top search result.
    assert matches[0].matched_product["product_id"] == 1
    assert matches[1].matched_product["product_name"] == "Long Grain White Rice"
    # The search engine was called once per ingredient with top_k=5.
    assert len(engine.calls) == 3
    assert all(call[1] == 5 for call in engine.calls)


def test_match_ingredients_empty_input(tiny_catalog: pd.DataFrame) -> None:
    engine = FakeSearchEngine()
    matches = match_ingredients_to_catalog(
        ingredients=[], search_engine=engine, catalog=tiny_catalog,
    )
    assert matches == []
    assert engine.calls == []


def test_match_ingredients_no_results(tiny_catalog: pd.DataFrame) -> None:
    """If search returns nothing, matched_product is None and confidence is 0."""
    engine = FakeSearchEngine({})  # every query returns []
    matches = match_ingredients_to_catalog(
        ingredients=[{"name": "unobtanium", "quantity": None, "category_hint": None}],
        search_engine=engine,
        catalog=tiny_catalog,
    )
    assert len(matches) == 1
    assert matches[0].requested_name == "unobtanium"
    assert matches[0].matched_product is None
    assert matches[0].confidence == 0.0


def test_match_ingredients_dietary_preference_prefers_attribute(
    tiny_catalog: pd.DataFrame,
) -> None:
    """For an ambiguous ingredient (``milk``), prefer the dairy-free option
    when the user has lactose-free preferences — even when its relevance
    score is lower than the top result."""
    engine = FakeSearchEngine(
        {
            # Note: top relevance_score wins by default, but the test
            # asserts the dietary preference overrides this.
            "milk": [
                _product(10, "Whole Milk", relevance_score=0.99, attributes=[]),
                _product(11, "Almond Milk", relevance_score=0.80, attributes=["dairy-free"]),
            ]
        }
    )
    matches = match_ingredients_to_catalog(
        ingredients=[{"name": "milk", "quantity": "1 gal", "category_hint": "dairy eggs"}],
        search_engine=engine,
        catalog=tiny_catalog,
        user_dietary_preferences=["dairy-free"],
    )
    assert len(matches) == 1
    assert matches[0].matched_product["product_id"] == 11
    assert matches[0].matched_product["product_name"] == "Almond Milk"
    # Confidence comes from the chosen product's relevance_score.
    assert matches[0].confidence == pytest.approx(0.80)


def test_match_ingredients_no_preference_takes_top_result(
    tiny_catalog: pd.DataFrame,
) -> None:
    """Without dietary preferences, the top-scoring result wins even for
    ambiguous ingredients."""
    engine = FakeSearchEngine(
        {
            "milk": [
                _product(10, "Whole Milk", relevance_score=0.99, attributes=[]),
                _product(11, "Almond Milk", relevance_score=0.80, attributes=["dairy-free"]),
            ]
        }
    )
    matches = match_ingredients_to_catalog(
        ingredients=[{"name": "milk", "quantity": "1 gal", "category_hint": "dairy eggs"}],
        search_engine=engine,
        catalog=tiny_catalog,
    )
    assert matches[0].matched_product["product_id"] == 10  # whole milk


# ---------------------------------------------------------------------------
# recipe_to_cart_plan
# ---------------------------------------------------------------------------


def test_recipe_to_cart_plan_full_pipeline(tiny_catalog: pd.DataFrame) -> None:
    """The pipeline: LLM extracts 4 ingredients -> 4 search calls -> 4 matches."""
    fake_ingredients = [
        {"name": "ground beef", "quantity": "1 lb",   "category_hint": "meat seafood"},
        {"name": "white rice",  "quantity": "2 cups", "category_hint": "pantry"},
        {"name": "soy sauce",   "quantity": "1/4 cup", "category_hint": "pantry"},
        {"name": "garlic",      "quantity": "3 cloves", "category_hint": "produce"},
    ]
    engine = FakeSearchEngine(
        {
            "ground beef": [_product(1, "Ground Beef 80/20", relevance_score=0.91)],
            "white rice":  [_product(2, "Jasmine Rice",      relevance_score=0.83)],
            "soy sauce":   [_product(3, "Kikkoman Soy Sauce", relevance_score=0.88)],
            "garlic":      [_product(4, "Garlic Bulb",       relevance_score=0.79)],
        }
    )

    with patch(
        "src.agents.shopping_agent.extract_recipe_ingredients",
        return_value=fake_ingredients,
    ) as mock_extract:
        result = recipe_to_cart_plan(
            recipe_query="Korean beef bowls",
            search_engine=engine,
            catalog=tiny_catalog,
            llm_client=MagicMock(),
        )

    # LLM was called once with the recipe.
    assert mock_extract.call_count == 1
    args, kwargs = mock_extract.call_args
    # Recipe text is the first positional or in the kwargs.
    assert "Korean beef bowls" in (args[0] if args else kwargs.get("recipe_query", ""))

    # Top-level shape.
    assert result["recipe"] == "Korean beef bowls"
    assert result["ingredients"] == fake_ingredients
    assert isinstance(result["summary"], str)
    assert len(result["summary"]) > 0

    # 4 matches, one per ingredient, in order.
    matches = result["matches"]
    assert len(matches) == 4
    assert [m.requested_name for m in matches] == [
        "ground beef", "white rice", "soy sauce", "garlic",
    ]
    assert all(m.matched_product is not None for m in matches)

    # Engine called once per ingredient.
    assert len(engine.calls) == 4
    queried = [c[0] for c in engine.calls]
    assert queried == ["ground beef", "white rice", "soy sauce", "garlic"]


# ---------------------------------------------------------------------------
# plan_to_cart_plan
# ---------------------------------------------------------------------------


def test_plan_to_cart_plan_respects_max_products(
    tiny_catalog: pd.DataFrame,
) -> None:
    """LLM returns 2 categories x 3 items = 6 items, but max_products=4
    truncates the output."""
    fake_plan = {
        "interpretation": "Healthy work-week lunches for one person",
        "shopping_categories": [
            {
                "category": "proteins",
                "items": ["grilled chicken breast", "boiled eggs", "tofu"],
            },
            {
                "category": "vegetables",
                "items": ["spinach", "tomatoes", "cucumber"],
            },
        ],
        "notes": "Meal-prep on Sunday for the whole week.",
    }
    engine = FakeSearchEngine(
        {
            "grilled chicken breast": [_product(101, "Chicken Breast",  relevance_score=0.9)],
            "boiled eggs":             [_product(102, "Large Eggs",      relevance_score=0.88)],
            "tofu":                    [_product(103, "Firm Tofu",       relevance_score=0.86)],
            "spinach":                 [_product(104, "Baby Spinach",    relevance_score=0.91)],
            "tomatoes":                [_product(105, "Vine Tomatoes",   relevance_score=0.87)],
            "cucumber":                [_product(106, "Hot House Cucumber", relevance_score=0.83)],
        }
    )

    with patch(
        "src.agents.shopping_agent.plan_shopping_goal",
        return_value=fake_plan,
    ) as mock_plan:
        result = plan_to_cart_plan(
            goal="Healthy lunches for the work week, under $50 total",
            search_engine=engine,
            catalog=tiny_catalog,
            llm_client=MagicMock(),
            max_products=4,
        )

    assert mock_plan.call_count == 1
    assert "Healthy lunches" in mock_plan.call_args[0][0]

    # Output structure.
    assert result["goal"] == "Healthy lunches for the work week, under $50 total"
    assert result["interpretation"] == "Healthy work-week lunches for one person"
    assert result["notes"] == "Meal-prep on Sunday for the whole week."

    # max_products=4 should be respected — 6 candidates -> 4 matches.
    assert result["total_products"] == 4
    # Categories should reflect the truncation: first 3 in proteins,
    # then 1 in vegetables.
    cats = result["categories"]
    assert len(cats) == 2
    assert cats[0]["category"] == "proteins"
    assert len(cats[0]["items"]) == 3
    assert cats[1]["category"] == "vegetables"
    assert len(cats[1]["items"]) == 1
    # All items are MatchedIngredient instances.
    for cat in cats:
        for it in cat["items"]:
            assert isinstance(it, MatchedIngredient)

    # Search called exactly 4 times.
    assert len(engine.calls) == 4


def test_plan_to_cart_plan_basic_structure(tiny_catalog: pd.DataFrame) -> None:
    """Smaller plan that fits well under max_products."""
    fake_plan = {
        "interpretation": "Quick weeknight pasta dinner for two",
        "shopping_categories": [
            {"category": "pasta",    "items": ["spaghetti"]},
            {"category": "produce",  "items": ["garlic", "basil"]},
        ],
        "notes": "",
    }
    engine = FakeSearchEngine(
        {
            "spaghetti": [_product(201, "Barilla Spaghetti", relevance_score=0.92)],
            "garlic":    [_product(202, "Garlic Bulb",       relevance_score=0.80)],
            "basil":     [_product(203, "Fresh Basil",        relevance_score=0.75)],
        }
    )
    with patch(
        "src.agents.shopping_agent.plan_shopping_goal",
        return_value=fake_plan,
    ):
        result = plan_to_cart_plan(
            goal="Pasta dinner",
            search_engine=engine,
            catalog=tiny_catalog,
            llm_client=MagicMock(),
            max_products=20,
        )
    assert result["total_products"] == 3
    assert len(result["categories"]) == 2
    assert result["categories"][0]["items"][0].matched_product["product_id"] == 201


# ---------------------------------------------------------------------------
# extract_recipe_ingredients — LLM wrapper, mocked
# ---------------------------------------------------------------------------


def _mock_chat_response(json_payload: dict) -> SimpleNamespace:
    """Build a fake OpenAI chat.completions response that returns json_payload."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=json.dumps(json_payload))
            )
        ]
    )


def test_extract_recipe_ingredients_calls_openai_with_recipe_text() -> None:
    """The recipe query must appear somewhere in the prompt sent to OpenAI."""
    payload = {
        "ingredients": [
            {"name": "ground beef", "quantity": "1 lb", "category_hint": "meat seafood"},
            {"name": "white rice",  "quantity": "2 cups", "category_hint": "pantry"},
        ]
    }
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_chat_response(payload)

    ingredients = llm_module.extract_recipe_ingredients(
        recipe_query="Korean beef bowls", client=client,
    )

    assert len(ingredients) == 2
    assert ingredients[0]["name"] == "ground beef"
    assert ingredients[0]["quantity"] == "1 lb"
    assert ingredients[0]["category_hint"] == "meat seafood"

    # Verify the prompt includes the recipe and uses json_object mode.
    call = client.chat.completions.create.call_args
    kwargs = call.kwargs
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["temperature"] == 0
    assert kwargs["response_format"] == {"type": "json_object"}
    messages = kwargs["messages"]
    # System + user message at minimum.
    assert any("ingredient" in m["content"].lower() for m in messages if m["role"] == "system")
    user_messages = [m["content"] for m in messages if m["role"] == "user"]
    assert any("Korean beef bowls" in m for m in user_messages)


def test_extract_recipe_ingredients_handles_bare_list_response() -> None:
    """The LLM might respond with a bare list rather than {ingredients: [...]}."""
    bare_dict_with_list = {"items": [{"name": "milk", "quantity": "1 gal"}]}
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_chat_response(bare_dict_with_list)

    out = llm_module.extract_recipe_ingredients(recipe_query="cereal", client=client)
    assert len(out) == 1
    assert out[0]["name"] == "milk"
    assert out[0]["quantity"] == "1 gal"
    assert out[0]["category_hint"] is None


def test_extract_recipe_ingredients_skips_invalid_items() -> None:
    payload = {
        "ingredients": [
            {"name": "bread", "quantity": "1 loaf"},
            {"name": "", "quantity": "ignored"},   # empty name -> skipped
            "not a dict",                          # non-dict -> skipped
            {"quantity": "no name field"},         # missing name -> skipped
        ]
    }
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_chat_response(payload)
    out = llm_module.extract_recipe_ingredients(recipe_query="toast", client=client)
    assert len(out) == 1
    assert out[0]["name"] == "bread"


# ---------------------------------------------------------------------------
# plan_shopping_goal — LLM wrapper, mocked
# ---------------------------------------------------------------------------


def test_plan_shopping_goal_returns_structure_and_uses_goal_in_prompt() -> None:
    payload = {
        "interpretation": "Healthy week-day lunches for one, ~$50",
        "shopping_categories": [
            {"category": "proteins",  "items": ["chicken breast", "eggs"]},
            {"category": "vegetables", "items": ["spinach", "tomatoes"]},
        ],
        "notes": "Meal-prep on Sunday saves time.",
    }
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_chat_response(payload)

    plan = llm_module.plan_shopping_goal(
        goal="Healthy lunches for the work week, under $50 total",
        client=client,
    )

    assert plan["interpretation"].startswith("Healthy")
    assert len(plan["shopping_categories"]) == 2
    assert plan["shopping_categories"][0]["category"] == "proteins"
    assert "chicken breast" in plan["shopping_categories"][0]["items"]
    assert plan["notes"].startswith("Meal-prep")

    # The goal text shows up in the user prompt.
    user_messages = [
        m["content"] for m in client.chat.completions.create.call_args.kwargs["messages"]
        if m["role"] == "user"
    ]
    assert any("Healthy lunches for the work week" in m for m in user_messages)


def test_plan_shopping_goal_handles_malformed_response() -> None:
    """A poorly-formed LLM response should still return the expected keys
    rather than crashing."""
    payload = {"interpretation": "ok"}  # missing other fields
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_chat_response(payload)
    plan = llm_module.plan_shopping_goal(goal="dinner", client=client)
    assert plan["interpretation"] == "ok"
    assert plan["shopping_categories"] == []
    assert plan["notes"] == ""


def test_plan_shopping_goal_includes_user_profile_when_provided() -> None:
    payload = {"interpretation": "x", "shopping_categories": [], "notes": ""}
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_chat_response(payload)

    llm_module.plan_shopping_goal(
        goal="dinner for two",
        user_profile_summary="Vegetarian, avoids gluten, favorite cuisine Italian",
        client=client,
    )
    user_messages = [
        m["content"] for m in client.chat.completions.create.call_args.kwargs["messages"]
        if m["role"] == "user"
    ]
    assert any("Vegetarian" in m for m in user_messages)
