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
