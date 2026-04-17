"""Streamlit frontend for Grocery Intelligence.

A demo UI for semantic grocery search and substitute recommendations.
"""

import os
import streamlit as st
import requests

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Grocery Intelligence",
    page_icon="🛒",
    layout="wide",
)

st.title("🛒 Grocery Intelligence")
st.caption("AI-Powered Semantic Search & Substitute Recommendation — 49,688 real products")

tab1, tab2 = st.tabs(["🔍 Smart Search", "🔄 Substitute Finder"])


def render_product_card(product, rank=None, show_score=True, score_key="relevance_score"):
    """Render a product as a styled card."""
    name = product.get("product_name", "Unknown")
    category = product.get("category", "")
    department = product.get("department", "")
    brand = product.get("brand", "")
    score = product.get(score_key)

    # Nutrition info
    cal = product.get("calories_100g")
    protein = product.get("protein_100g")
    sugar = product.get("sugar_100g")
    grade = product.get("nutrition_grade")
    popularity = product.get("order_count")
    reorder = product.get("reorder_rate")

    with st.container(border=True):
        cols = st.columns([0.5, 4, 2, 1.5])

        if rank is not None:
            cols[0].markdown(f"### #{rank}")

        with cols[1]:
            st.markdown(f"**{name}**")
            meta_parts = []
            if brand:
                meta_parts.append(f"🏷️ {brand}")
            if category:
                meta_parts.append(f"📁 {category}")
            if department:
                meta_parts.append(f"🏬 {department}")
            if meta_parts:
                st.caption(" · ".join(meta_parts))

        with cols[2]:
            nutrition_parts = []
            if cal is not None:
                nutrition_parts.append(f"{cal:.0f} cal")
            if protein is not None:
                nutrition_parts.append(f"{protein:.1f}g protein")
            if sugar is not None:
                nutrition_parts.append(f"{sugar:.1f}g sugar")
            if nutrition_parts:
                st.caption(" · ".join(nutrition_parts))
            if grade:
                grade_colors = {"a": "🟢", "b": "🟡", "c": "🟠", "d": "🔴", "e": "⚫"}
                st.caption(f"Nutri-Score: {grade_colors.get(grade, '⚪')} {grade.upper()}")

        with cols[3]:
            if show_score and score is not None:
                st.metric("Score", f"{score:.3f}")
            if popularity:
                st.caption(f"📊 {popularity:,} orders")

        # Substitution reasons
        reasons = product.get("substitution_reasons", [])
        if reasons:
            st.success(" · ".join(f"✓ {r}" for r in reasons))


with tab1:
    st.markdown("Search naturally — the engine combines keyword matching, semantic understanding, and AI reranking.")

    query = st.text_input(
        "What are you looking for?",
        placeholder="e.g., low-sugar yogurt, organic almond milk, Korean BBQ ingredients",
    )

    col1, col2, col3 = st.columns([2, 1, 1])
    with col2:
        top_k = st.slider("Results", 5, 20, 10)
    with col3:
        use_reranker = st.checkbox("AI Reranking (cross-encoder)", value=False)

    if st.button("Search", type="primary", use_container_width=True) and query:
        with st.spinner("Searching across 49,688 products..."):
            try:
                response = requests.post(
                    f"{API_URL}/search",
                    json={
                        "query": query,
                        "top_k": top_k,
                        "use_reranker": use_reranker,
                    },
                    timeout=30,
                )
                data = response.json()

                st.subheader(f"Results for \"{query}\" ({data.get('total_results', 0)} found)")

                for i, product in enumerate(data.get("results", []), 1):
                    render_product_card(product, rank=i)

            except requests.ConnectionError:
                st.error(
                    "Cannot connect to API. "
                    "Start the backend: `uvicorn src.api.main:app --reload`"
                )
            except requests.Timeout:
                st.warning("Search timed out. Try a simpler query or disable reranking.")

with tab2:
    st.markdown("Find alternatives when a product is out of stock.")

    col1, col2 = st.columns([2, 1])
    with col1:
        product_id = st.number_input("Product ID", min_value=1, value=1, step=1)
    with col2:
        sub_type = st.selectbox(
            "Substitute type",
            ["similar", "healthier", "cheaper"],
            format_func=lambda x: {"similar": "🔄 Most Similar", "healthier": "💪 Healthier", "cheaper": "💰 Cheaper"}[x],
        )

    num_subs = st.slider("Number of substitutes", 3, 10, 5)

    if st.button("Find Substitutes", type="primary", use_container_width=True):
        with st.spinner("Finding the best alternatives..."):
            try:
                response = requests.post(
                    f"{API_URL}/substitute",
                    json={
                        "product_id": int(product_id),
                        "top_k": num_subs,
                        "substitute_type": sub_type,
                    },
                    timeout=30,
                )

                if response.status_code == 404:
                    st.error(f"Product ID {product_id} not found in catalog.")
                else:
                    data = response.json()

                    original = data.get("original_product", {})
                    st.subheader("Original Product")
                    render_product_card(original, show_score=False)

                    st.subheader(f"Top {sub_type.title()} Substitutes")
                    for i, sub in enumerate(data.get("substitutes", []), 1):
                        render_product_card(sub, rank=i, score_key="similarity_score")

            except requests.ConnectionError:
                st.error(
                    "Cannot connect to API. "
                    "Start the backend: `uvicorn src.api.main:app --reload`"
                )

# Sidebar with project info
with st.sidebar:
    st.markdown("### About")
    st.markdown(
        "**Grocery Intelligence** uses hybrid search "
        "(BM25 + semantic embeddings + cross-encoder reranking) "
        "over 49,688 real Instacart products enriched with "
        "Open Food Facts nutrition data."
    )
    st.markdown("---")
    st.markdown("### Tech Stack")
    st.markdown(
        "- **Embeddings**: all-MiniLM-L6-v2 (384d)\n"
        "- **Reranker**: ms-marco-MiniLM-L-6-v2\n"
        "- **Search**: BM25 + Semantic + RRF\n"
        "- **Backend**: FastAPI\n"
        "- **Data**: Instacart + Open Food Facts"
    )

    # Quick health check
    try:
        health = requests.get(f"{API_URL}/health", timeout=2).json()
        status = health.get("status", "unknown")
        count = health.get("products_loaded", 0)
        if status == "healthy":
            st.success(f"API: {count:,} products loaded")
        else:
            st.warning(f"API: {status}")
    except Exception:
        st.error("API offline")
