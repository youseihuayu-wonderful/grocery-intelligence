"""Streamlit frontend for Grocery Intelligence.

A demo UI for semantic grocery search and substitute recommendations.
"""

import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Grocery Intelligence",
    page_icon="🛒",
    layout="wide",
)

st.title("🛒 Grocery Intelligence")
st.subheader("AI-Powered Grocery Search & Recommendation")

tab1, tab2 = st.tabs(["🔍 Smart Search", "🔄 Substitute Finder"])

with tab1:
    st.markdown("Search for grocery products using natural language.")
    query = st.text_input(
        "What are you looking for?",
        placeholder="e.g., low-sugar yogurt, cheap high-protein breakfast",
    )

    col1, col2 = st.columns([3, 1])
    with col2:
        top_k = st.slider("Results", 5, 20, 10)
        use_reranker = st.checkbox("Use AI reranking", value=True)

    if st.button("Search", type="primary") and query:
        with st.spinner("Searching..."):
            try:
                response = requests.post(
                    f"{API_URL}/search",
                    json={
                        "query": query,
                        "top_k": top_k,
                        "use_reranker": use_reranker,
                    },
                )
                data = response.json()

                if data.get("explanation"):
                    st.info(data["explanation"])

                for i, product in enumerate(data.get("results", []), 1):
                    with st.container():
                        col_rank, col_name, col_cat, col_score = st.columns(
                            [0.5, 3, 2, 1]
                        )
                        col_rank.write(f"**#{i}**")
                        col_name.write(f"**{product['product_name']}**")
                        col_cat.write(product.get("category", ""))
                        score = product.get("relevance_score", 0)
                        col_score.write(f"Score: {score:.3f}")
            except requests.ConnectionError:
                st.error(
                    "Cannot connect to API. "
                    "Start the backend: `uvicorn src.api.main:app --reload`"
                )

with tab2:
    st.markdown("Find alternatives when a product is out of stock.")

    product_id = st.number_input("Product ID", min_value=1, value=1)
    sub_type = st.selectbox(
        "Substitute type",
        ["similar", "healthier", "cheaper"],
    )

    if st.button("Find Substitutes", type="primary"):
        with st.spinner("Finding substitutes..."):
            try:
                response = requests.post(
                    f"{API_URL}/substitute",
                    json={
                        "product_id": product_id,
                        "top_k": 5,
                        "substitute_type": sub_type,
                    },
                )
                data = response.json()

                if data.get("explanation"):
                    st.info(data["explanation"])

                for sub in data.get("substitutes", []):
                    with st.container():
                        st.write(f"**{sub['product_name']}**")
                        if sub.get("substitution_reasons"):
                            for reason in sub["substitution_reasons"]:
                                st.write(f"  ✓ {reason}")
                        st.divider()
            except requests.ConnectionError:
                st.error(
                    "Cannot connect to API. "
                    "Start the backend: `uvicorn src.api.main:app --reload`"
                )
