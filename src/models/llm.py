"""LLM integration for query understanding and response generation.

Uses OpenAI GPT-4o-mini via API for:
- Query rewriting and intent extraction
- Generating recommendation explanations
- Shopping Q&A with product grounding
"""

import os
import json

from openai import OpenAI
from loguru import logger


def get_client() -> OpenAI:
    """Get OpenAI client using API key from environment."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not set. Add it to .env file or environment."
        )
    return OpenAI(api_key=api_key)


def rewrite_query(query: str, client: OpenAI | None = None) -> dict:
    """Rewrite a natural language grocery query into structured intent.

    Example:
        Input: "cheap healthy snack for kids under $5"
        Output: {
            "rewritten_query": "healthy snack for children",
            "filters": {"max_price": 5.0, "dietary": ["healthy"]},
            "intent": "search"
        }
    """
    if client is None:
        client = get_client()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a grocery search query parser. "
                    "Extract the user's intent, rewrite their query for "
                    "semantic search, and extract any filters.\n\n"
                    "Return JSON with:\n"
                    '- "rewritten_query": optimized search text\n'
                    '- "filters": {max_price, min_protein, max_sugar, '
                    "dietary, brand, category}\n"
                    '- "intent": one of "search", "substitute", "question"'
                ),
            },
            {"role": "user", "content": query},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )

    result = json.loads(response.choices[0].message.content)
    logger.info(f"Query rewritten: '{query}' -> {result}")
    return result


def generate_explanation(
    query: str,
    products: list[dict],
    client: OpenAI | None = None,
) -> str:
    """Generate a natural language explanation for search results.

    Args:
        query: Original user query
        products: List of recommended products with their attributes
    """
    if client is None:
        client = get_client()

    product_list = "\n".join(
        f"- {p['product_name']} (${p.get('price', 'N/A')}, "
        f"{p.get('calories_100g', 'N/A')} cal)"
        for p in products[:5]
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful grocery shopping assistant. "
                    "Explain why these products match the user's search. "
                    "Be concise (2-3 sentences). Reference specific "
                    "product attributes like nutrition, price, or dietary info."
                ),
            },
            {
                "role": "user",
                "content": f"Query: {query}\n\nResults:\n{product_list}",
            },
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content


def generate_substitute_explanation(
    original_product: dict,
    substitutes: list[dict],
    client: OpenAI | None = None,
) -> str:
    """Explain why substitute products are good alternatives."""
    if client is None:
        client = get_client()

    sub_list = "\n".join(
        f"- {s['product_name']} ({s.get('brand', 'N/A')}, "
        f"${s.get('price', 'N/A')})"
        for s in substitutes[:5]
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a grocery shopping assistant. The user's "
                    "desired product is unavailable. Explain why each "
                    "substitute is a good alternative. Be concise."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Original: {original_product['product_name']}\n\n"
                    f"Substitutes:\n{sub_list}"
                ),
            },
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content


def answer_question(
    question: str,
    products: list[dict],
    client: OpenAI | None = None,
) -> dict:
    """RAG-style Q&A grounded in retrieved product catalog.

    Args:
        question: User's natural language question (e.g. "What can I use
            instead of heavy cream?")
        products: Top-K relevant products retrieved by the search engine.
            Each dict has keys like product_name, category, department,
            brand, calories_100g, protein_100g, sugar_100g, nutrition_grade.

    Returns:
        {
            "answer": str,         # 2-4 sentence answer grounded in products
            "model": str,          # which model was used (e.g. "gpt-4o-mini")
        }
    """
    if client is None:
        client = get_client()

    model = "gpt-4o-mini"

    product_lines = []
    for p in products[:8]:
        attrs = []
        if p.get("brand"):
            attrs.append(f"brand: {p['brand']}")
        if p.get("category"):
            attrs.append(f"category: {p['category']}")
        if p.get("department"):
            attrs.append(f"department: {p['department']}")
        if p.get("calories_100g") is not None:
            attrs.append(f"{p['calories_100g']} cal/100g")
        if p.get("protein_100g") is not None:
            attrs.append(f"protein: {p['protein_100g']}g/100g")
        if p.get("sugar_100g") is not None:
            attrs.append(f"sugar: {p['sugar_100g']}g/100g")
        if p.get("nutrition_grade"):
            attrs.append(f"grade: {p['nutrition_grade']}")
        name = p.get("product_name", "Unknown")
        if attrs:
            product_lines.append(f"- {name} ({', '.join(attrs)})")
        else:
            product_lines.append(f"- {name}")

    product_context = (
        "\n".join(product_lines) if product_lines else "(no products available)"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful grocery shopping assistant. "
                    "Answer the user's question using ONLY the provided "
                    "products as evidence. Cite specific product names from "
                    "the list. Be concise (2-4 sentences). If the products "
                    "do not actually answer the question, respond with "
                    '"I don\'t have enough information".'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Available products:\n{product_context}"
                ),
            },
        ],
        temperature=0.3,
    )

    answer = response.choices[0].message.content
    logger.info(f"Q&A: '{question}' -> '{answer}'")
    return {"answer": answer, "model": model}


def extract_recipe_ingredients(
    recipe_query: str,
    client: "OpenAI | None" = None,
) -> list[dict]:
    """LLM extracts a list of ingredients from a recipe / dish name / meal description.

    Examples:
        Input: "Korean beef bowls"
        Output: [
            {"name": "ground beef", "quantity": "1 lb", "category_hint": "meat"},
            {"name": "white rice", "quantity": "2 cups", "category_hint": "grains"},
            {"name": "soy sauce", "quantity": "1/4 cup", "category_hint": "pantry"},
            {"name": "garlic", "quantity": "3 cloves", "category_hint": "produce"},
            {"name": "green onion", "quantity": "2 stalks", "category_hint": "produce"},
            ...
        ]

    The category_hint is a hint to help the catalog matcher narrow down
    (typical departments: produce, dairy eggs, meat seafood, pantry,
    bakery, frozen, beverages, snacks). Use null if unsure.
    """
    if client is None:
        client = get_client()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a culinary assistant for a grocery shopping app. "
                    "Given a recipe, dish name, or meal description, produce a "
                    "complete shopping ingredient list.\n\n"
                    "Return JSON with a top-level key \"ingredients\" whose value "
                    "is an array of objects. Each object MUST have:\n"
                    "- \"name\": short generic grocery item name "
                    "(e.g. \"ground beef\", \"white rice\", \"soy sauce\")\n"
                    "- \"quantity\": estimated quantity as a string "
                    "(e.g. \"1 lb\", \"2 cups\", \"3 cloves\") "
                    "or null if unknown\n"
                    "- \"category_hint\": one of "
                    "[\"produce\", \"dairy eggs\", \"meat seafood\", "
                    "\"pantry\", \"bakery\", \"frozen\", \"beverages\", "
                    "\"snacks\"] or null if unsure\n\n"
                    "Use simple generic names that match grocery catalogs. "
                    "Avoid duplicates. Omit very common pantry items "
                    "(salt, pepper, water) unless central to the dish."
                ),
            },
            {
                "role": "user",
                "content": f"Recipe / dish / meal: {recipe_query}",
            },
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )

    raw = json.loads(response.choices[0].message.content)
    # Be lenient — the model might wrap the list under different keys.
    if isinstance(raw, dict):
        for key in ("ingredients", "items", "list", "result"):
            if key in raw and isinstance(raw[key], list):
                ingredients = raw[key]
                break
        else:
            # Fall back: collect any list-valued field.
            ingredients = next(
                (v for v in raw.values() if isinstance(v, list)), []
            )
    elif isinstance(raw, list):
        ingredients = raw
    else:
        ingredients = []

    # Normalize each entry — ensure keys exist.
    normalized = []
    for item in ingredients:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not name:
            continue
        normalized.append(
            {
                "name": str(name).strip(),
                "quantity": item.get("quantity"),
                "category_hint": item.get("category_hint"),
            }
        )

    logger.info(
        f"extract_recipe_ingredients: '{recipe_query}' -> "
        f"{len(normalized)} ingredients"
    )
    return normalized


def plan_shopping_goal(
    goal: str,
    user_profile_summary: str | None = None,
    client: "OpenAI | None" = None,
) -> dict:
    """Convert a goal statement into a shopping plan structure.

    Example:
        Input: "Healthy lunches for the work week, under $50 total"
        Output: {
            "interpretation": "5 healthy work lunches for one person, budget ~$50",
            "shopping_categories": [
                {"category": "proteins", "items": ["grilled chicken breast", "boiled eggs", ...]},
                {"category": "vegetables", "items": ["spinach", "tomatoes", ...]},
                ...
            ],
            "notes": "Optional advice about meal-prep, budget, etc."
        }
    """
    if client is None:
        client = get_client()

    user_context = ""
    if user_profile_summary:
        user_context = (
            f"\n\nUser context (for personalization):\n{user_profile_summary}"
        )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an AI shopping planner. Given a high-level "
                    "shopping goal, produce a structured grocery plan.\n\n"
                    "Return JSON with EXACTLY these keys:\n"
                    "- \"interpretation\": 1 sentence explaining how you "
                    "understood the goal (servings, occasion, constraints).\n"
                    "- \"shopping_categories\": array of objects, each with "
                    "\"category\" (a short label like \"proteins\", "
                    "\"vegetables\", \"grains\", \"dairy\", \"pantry\", "
                    "\"snacks\", \"beverages\") and \"items\" (an array of "
                    "short generic grocery item names).\n"
                    "- \"notes\": 1-2 sentence advice about meal-prep, "
                    "budget, substitutions, or storage. May be empty string.\n\n"
                    "Use simple grocery-catalog-friendly item names. "
                    "Respect any explicit dietary, budget, or serving "
                    "constraints in the goal."
                ),
            },
            {"role": "user", "content": f"Goal: {goal}{user_context}"},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )

    raw = json.loads(response.choices[0].message.content)
    # Normalize — guarantee the three keys exist.
    plan = {
        "interpretation": raw.get("interpretation", "") if isinstance(raw, dict) else "",
        "shopping_categories": [],
        "notes": raw.get("notes", "") if isinstance(raw, dict) else "",
    }
    cats = raw.get("shopping_categories") if isinstance(raw, dict) else None
    if isinstance(cats, list):
        for entry in cats:
            if not isinstance(entry, dict):
                continue
            label = entry.get("category") or entry.get("name") or ""
            items = entry.get("items") or []
            if not isinstance(items, list):
                continue
            cleaned_items = [str(it).strip() for it in items if it]
            plan["shopping_categories"].append(
                {"category": str(label).strip(), "items": cleaned_items}
            )

    logger.info(
        f"plan_shopping_goal: '{goal}' -> {len(plan['shopping_categories'])} categories"
    )
    return plan


# ---------------------------------------------------------------------------
# Multi-turn conversational search
# ---------------------------------------------------------------------------

# Attribute IDs the model is allowed to emit. Kept in sync with
# ``src/search/attributes.py::ALL_ATTRIBUTES`` but inlined here so this
# module stays cheap to import (no pandas pull-through).
_KNOWN_ATTRIBUTE_IDS: tuple[str, ...] = (
    "dairy-free",
    "gluten-free",
    "high-fiber",
    "high-protein",
    "keto-friendly",
    "kosher",
    "low-calorie",
    "low-carb",
    "low-fat",
    "low-sugar",
    "non-gmo",
    "nut-free",
    "organic",
    "sugar-free",
    "vegan",
    "vegetarian",
    "whole-grain",
)

# Sort hints the model is allowed to emit. Search engine maps these.
_KNOWN_SORT_BY: tuple[str, ...] = (
    "price",          # cheaper / cheapest / lowest price
    "price_desc",     # most expensive
    "rating",         # best rated / healthier (proxy)
    "popularity",     # popular / most ordered
    "relevance",      # default
)

_CONVERSATIONAL_SYSTEM_PROMPT = (
    "You are a multi-turn grocery search assistant. The user is having "
    "a conversation with a search engine. You receive:\n"
    "  - previous_query: the user's last search text (string)\n"
    "  - previous_filters: filters active on that last search (object, "
    "may contain keys like 'attributes' (list of IDs) or 'sort_by').\n"
    "  - user_followup: the new turn the user just typed.\n\n"
    "Your job is to classify the follow-up and produce the NEXT search:\n"
    "  1. TOPIC SHIFT: the user is asking for a different product "
    "(e.g. previous_query='yogurt', followup='show me bread instead' "
    "or just 'bread'). In this case set interpreted_query to the new "
    "product and RESET interpreted_filters to {} unless the follow-up "
    "explicitly keeps a filter.\n"
    "  2. REFINEMENT: the user wants the same product but with a new "
    "constraint (e.g. 'make it organic', 'cheaper one', 'gluten free "
    "version'). In this case keep interpreted_query == previous_query "
    "and MERGE the new filters into previous_filters.\n"
    "  3. SORT HINT: words like 'cheaper', 'cheapest', 'lowest price' "
    "-> sort_by='price'. 'most expensive' -> 'price_desc'. 'popular', "
    "'most popular', 'best selling' -> 'popularity'. 'healthier', "
    "'best rated' -> 'rating'.\n"
    "  4. AMBIGUOUS / unclear: prefer KEEPING previous_query and "
    "previous_filters and put a short note in clarification.\n\n"
    "Known attribute IDs (use ONLY these in 'attributes', lowercase, "
    "hyphenated): "
    + ", ".join(_KNOWN_ATTRIBUTE_IDS)
    + ".\n"
    "Known sort_by values: " + ", ".join(_KNOWN_SORT_BY) + ".\n\n"
    "Return JSON with EXACTLY these keys:\n"
    '  - "interpreted_query": string (the search text for the next turn)\n'
    '  - "interpreted_filters": object with optional keys "attributes" '
    '(array of known attribute IDs) and "sort_by" (one of the known '
    "sort values). Omit keys you don't need; do not invent new keys.\n"
    '  - "clarification": one short sentence describing what you did, '
    "in natural English (shown to the user).\n\n"
    "Rules:\n"
    "  - Never invent attribute IDs. If the user mentions a constraint "
    "you don't recognize, leave it out of attributes (you may still "
    "mention it in clarification).\n"
    "  - Deduplicate the attributes list.\n"
    "  - When in doubt, default to keeping previous_query."
)


def conversational_search_followup(
    previous_query: str,
    previous_filters: dict,
    user_followup: str,
    client: "OpenAI | None" = None,
) -> dict:
    """Interpret a follow-up turn in a multi-turn search conversation.

    Examples:
        prev_query='yogurt', prev_filters={}, followup='make it organic'
        -> {
            "interpreted_query": "yogurt",
            "interpreted_filters": {"attributes": ["organic"]},
            "clarification": "Searching yogurt with the organic filter."
          }

        prev_query='almond milk',
        prev_filters={"attributes":["organic"]},
        followup='cheaper one'
        -> {
            "interpreted_query": "almond milk",
            "interpreted_filters": {
                "attributes": ["organic"],
                "sort_by": "price",
            },
            "clarification": "Sorting organic almond milk by price ascending."
          }

        followup='show me yogurt instead'  (topic change!)
        -> {
            "interpreted_query": "yogurt",
            "interpreted_filters": {},  # reset
            "clarification": "Switching to yogurt search."
          }

    Uses ``gpt-4o-mini``, ``response_format=json_object``, ``temperature=0``.

    Robust to malformed model output: on JSON parse failure, returns
    ``{"interpreted_query": previous_query, "interpreted_filters": {},
    "clarification": "Could not interpret follow-up; reusing previous query."}``.
    """
    if client is None:
        client = get_client()

    # Defensive: ensure previous_filters is a dict we can dump to JSON.
    prev_filters_payload: dict = (
        previous_filters if isinstance(previous_filters, dict) else {}
    )

    user_payload = json.dumps(
        {
            "previous_query": str(previous_query or ""),
            "previous_filters": prev_filters_payload,
            "user_followup": str(user_followup or ""),
        },
        ensure_ascii=False,
    )

    fallback = {
        "interpreted_query": str(previous_query or ""),
        "interpreted_filters": {},
        "clarification": (
            "Could not interpret follow-up; reusing previous query."
        ),
    }

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _CONVERSATIONAL_SYSTEM_PROMPT},
                {"role": "user", "content": user_payload},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        content = response.choices[0].message.content
        raw = json.loads(content) if content else {}
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning(
            f"conversational_search_followup: malformed response "
            f"({exc}); returning fallback."
        )
        return fallback
    except Exception as exc:  # noqa: BLE001 — network / client errors
        logger.warning(
            f"conversational_search_followup: client error "
            f"({type(exc).__name__}: {exc}); returning fallback."
        )
        return fallback

    if not isinstance(raw, dict):
        logger.warning(
            f"conversational_search_followup: response was not a dict "
            f"({type(raw).__name__}); returning fallback."
        )
        return fallback

    # Normalize interpreted_query — default to previous_query when missing.
    interpreted_query = raw.get("interpreted_query")
    if not isinstance(interpreted_query, str) or not interpreted_query.strip():
        interpreted_query = str(previous_query or "")
    else:
        interpreted_query = interpreted_query.strip()

    # Normalize interpreted_filters — keep only known keys with valid shapes.
    raw_filters = raw.get("interpreted_filters")
    interpreted_filters: dict = {}
    if isinstance(raw_filters, dict):
        attrs = raw_filters.get("attributes")
        if isinstance(attrs, list):
            known = set(_KNOWN_ATTRIBUTE_IDS)
            cleaned_attrs: list[str] = []
            seen: set[str] = set()
            for a in attrs:
                if not isinstance(a, str):
                    continue
                a_norm = a.strip().lower()
                # Keep both known IDs and (defensively) any non-empty
                # hyphenated string the engine might recognize. We
                # intentionally allow unknown IDs through so the engine
                # can decide; this matches the docstring promise that
                # the function is forgiving.
                if not a_norm:
                    continue
                if a_norm in seen:
                    continue
                seen.add(a_norm)
                if a_norm in known or all(
                    ch.isalnum() or ch in "-_" for ch in a_norm
                ):
                    cleaned_attrs.append(a_norm)
            if cleaned_attrs:
                interpreted_filters["attributes"] = cleaned_attrs

        sort_by = raw_filters.get("sort_by")
        if isinstance(sort_by, str) and sort_by.strip():
            sb = sort_by.strip().lower()
            if sb in _KNOWN_SORT_BY:
                interpreted_filters["sort_by"] = sb

    # Clarification — short natural-language description.
    clarification = raw.get("clarification")
    if not isinstance(clarification, str):
        clarification = ""
    clarification = clarification.strip()

    result = {
        "interpreted_query": interpreted_query,
        "interpreted_filters": interpreted_filters,
        "clarification": clarification,
    }
    logger.info(
        f"conversational_search_followup: prev='{previous_query}' "
        f"followup='{user_followup}' -> {result}"
    )
    return result
