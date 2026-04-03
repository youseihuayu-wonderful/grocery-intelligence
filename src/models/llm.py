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
