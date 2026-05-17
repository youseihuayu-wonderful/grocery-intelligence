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


@st.cache_data(ttl=600)
def fetch_demo_users():
    try:
        r = requests.get(f"{API_URL}/users/demo", params={"n": 20}, timeout=5)
        return r.json().get("users", [])
    except requests.RequestException:
        return []


_demo_users = fetch_demo_users()
_user_options = {"Guest (no personalization)": None}
for u in _demo_users:
    label = u.get("summary") or f"User {u['user_id']} — {u.get('total_orders', 0)} orders"
    _user_options[label] = u["user_id"]

_selected_label = st.selectbox(
    "Sign in as a demo user",
    options=list(_user_options.keys()),
    help="Search results, badges, and the Home feed personalize to the selected user's order history.",
)
ACTIVE_USER_ID = _user_options[_selected_label]
if ACTIVE_USER_ID is not None:
    st.caption(f"✨ Personalization active for User {ACTIVE_USER_ID}")

tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🏠 Home", "🔍 Smart Search", "🛒 Cart", "📦 Your Orders", "🔄 Substitute Finder", "🤖 Shopping Assistant"]
)


BADGE_LABELS = {
    "bestseller": "🏆 Bestseller",
    "popular": "🔥 Popular",
    "customer-favorite": "❤️ Customer Favorite",
    "healthy-choice": "🥗 Healthy Choice",
    "high-protein": "💪 High Protein",
    "low-sugar": "🍯 Low Sugar",
    "low-calorie": "⚖️ Low Calorie",
    "high-fiber": "🌾 High Fiber",
}

ATTRIBUTE_LABELS = {
    "organic": "🌿 Organic",
    "gluten-free": "🌾 Gluten-Free",
    "vegan": "🌱 Vegan",
    "vegetarian": "🥬 Vegetarian",
    "kosher": "✡️ Kosher",
    "dairy-free": "🥛 Dairy-Free",
    "sugar-free": "🍯 Sugar-Free",
    "keto-friendly": "🥑 Keto",
    "low-carb": "🥗 Low Carb",
    "whole-grain": "🌾 Whole Grain",
    "non-gmo": "🌽 Non-GMO",
    "high-protein": "💪 High Protein",
    "low-sugar": "🍯 Low Sugar",
    "low-calorie": "⚖️ Low Calorie",
    "low-fat": "🥦 Low Fat",
    "high-fiber": "🌾 High Fiber",
    "nut-free": "🥜 Nut-Free",
}


def render_product_card(product, rank=None, show_score=True, score_key="relevance_score"):
    """Render a product as a styled card."""
    name = product.get("product_name", "Unknown")
    category = product.get("category", "")
    department = product.get("department", "")
    brand = product.get("brand", "")
    score = product.get(score_key)
    badges = product.get("badges") or []
    attributes = product.get("attributes") or []

    # Nutrition info
    cal = product.get("calories_100g")
    protein = product.get("protein_100g")
    sugar = product.get("sugar_100g")
    grade = product.get("nutrition_grade")
    popularity = product.get("order_count")
    reorder = product.get("reorder_rate")

    emoji = product.get("emoji", "📦")
    image_url = product.get("image_url")

    with st.container(border=True):
        cols = st.columns([0.7, 4, 2, 1.5])

        with cols[0]:
            if image_url:
                st.image(image_url, width=70)
            else:
                st.markdown(
                    f"<div style='font-size:48px; line-height:1; text-align:center; padding-top:6px'>{emoji}</div>",
                    unsafe_allow_html=True,
                )
            if rank is not None:
                st.caption(f"#{rank}")

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
            if popularity:
                st.markdown(f"**📊 {popularity:,}**")
                st.caption("orders")
            personalization = product.get("personalization_score")
            if personalization and personalization > 0.2:
                st.caption(f"🎯 Match: {int(personalization * 100)}%")

        # Badges row
        if badges:
            badge_text = "  ".join(BADGE_LABELS.get(b, b) for b in badges)
            st.markdown(f"**{badge_text}**")

        # Attributes row
        if attributes:
            attr_text = " · ".join(ATTRIBUTE_LABELS.get(a, a) for a in attributes)
            st.caption(attr_text)

        # Substitution reasons
        reasons = product.get("substitution_reasons", [])
        if reasons:
            st.success(" · ".join(f"✓ {r}" for r in reasons))

        # Cart / Wishlist actions (only shown when signed in)
        pid = product.get("product_id")
        if pid is not None and ACTIVE_USER_ID is not None:
            action_cols = st.columns([1, 1, 3])
            with action_cols[0]:
                if st.button("🛒 Add", key=f"add_cart_{pid}_{rank}"):
                    requests.post(
                        f"{API_URL}/cart/add",
                        json={"user_id": ACTIVE_USER_ID, "product_id": pid, "qty": 1},
                        timeout=5,
                    )
                    st.toast(f"Added '{name}' to cart") if hasattr(st, "toast") else None
            with action_cols[1]:
                if st.button("❤️ Save", key=f"save_wl_{pid}_{rank}"):
                    requests.post(
                        f"{API_URL}/wishlist/add",
                        json={"user_id": ACTIVE_USER_ID, "product_id": pid},
                        timeout=5,
                    )


with tab0:
    st.markdown("Discovery feeds — no query needed. Bestsellers, healthy picks, and personalized recommendations.")

    @st.cache_data(ttl=300)
    def fetch_departments():
        try:
            r = requests.get(f"{API_URL}/departments", timeout=5)
            return r.json().get("departments", [])
        except requests.RequestException:
            return []

    def _fetch_feed(feed_type, user_id=None, department=None, top_k=8):
        params = {"top_k": top_k}
        if user_id is not None:
            params["user_id"] = user_id
        if department is not None:
            params["department"] = department
        try:
            r = requests.get(f"{API_URL}/feed/{feed_type}", params=params, timeout=15)
            if r.status_code == 200:
                return r.json().get("products", [])
        except requests.RequestException:
            pass
        return []

    if ACTIVE_USER_ID is not None:
        st.subheader("✨ For You")
        st.caption(f"Personalized based on User {ACTIVE_USER_ID}'s order history")
        for_you_products = _fetch_feed("for-you", user_id=ACTIVE_USER_ID, top_k=6)
        if for_you_products:
            for i, p in enumerate(for_you_products, 1):
                render_product_card(p, rank=i, show_score=False)
        else:
            st.info("No personalized recommendations available yet.")
        st.markdown("---")
    else:
        st.info("💡 Sign in as a demo user above to see a personalized For-You feed.")

    st.subheader("🏆 Bestsellers")
    st.caption("Top products by total order count across all users")
    bestsellers = _fetch_feed("bestsellers", top_k=6)
    for i, p in enumerate(bestsellers, 1):
        render_product_card(p, rank=i, show_score=False)
    st.markdown("---")

    st.subheader("🥗 Healthy Picks")
    st.caption("Top-rated products with Nutri-Score A or B")
    healthy = _fetch_feed("healthy-picks", top_k=6)
    for i, p in enumerate(healthy, 1):
        render_product_card(p, rank=i, show_score=False)
    st.markdown("---")

    departments = fetch_departments()
    if departments:
        st.subheader("🏬 Browse by Department")
        default_dept = "produce" if "produce" in departments else departments[0]
        selected_dept = st.selectbox(
            "Department",
            options=departments,
            index=departments.index(default_dept),
        )
        dept_products = _fetch_feed("department", department=selected_dept, top_k=6)
        for i, p in enumerate(dept_products, 1):
            render_product_card(p, rank=i, show_score=False)


POPULAR_CATEGORIES = {
    "🍎 Fruits": [
        ("🍌", "banana"),
        ("🍎", "apple"),
        ("🍊", "orange"),
        ("🍓", "strawberry"),
        ("🥑", "avocado"),
        ("🍇", "grapes"),
        ("🍋", "lemon"),
        ("🫐", "blueberry"),
    ],
    "🥬 Vegetables": [
        ("🥬", "lettuce"),
        ("🥦", "broccoli"),
        ("🧅", "onion"),
        ("🍅", "tomato"),
        ("🥕", "carrot"),
        ("🥔", "potato"),
        ("🌿", "spinach"),
        ("🍄", "mushroom"),
    ],
    "🥩 Meat & Seafood": [
        ("🍗", "chicken"),
        ("🥩", "beef"),
        ("🐟", "salmon"),
        ("🦐", "shrimp"),
        ("🥓", "bacon"),
        ("🦃", "turkey"),
    ],
    "🥛 Dairy & Eggs": [
        ("🥛", "milk"),
        ("🥚", "eggs"),
        ("🧀", "cheese"),
        ("🍦", "yogurt"),
        ("🧈", "butter"),
        ("🍨", "ice cream"),
    ],
    "🍞 Bread, Pasta & Grains": [
        ("🍞", "bread"),
        ("🍝", "pasta"),
        ("🍚", "rice"),
        ("🥣", "cereal"),
        ("🥯", "bagel"),
        ("🌽", "corn"),
    ],
    "🍷 Beverages & Alcohol": [
        ("☕", "coffee"),
        ("🍵", "tea"),
        ("🍷", "wine"),
        ("🍺", "beer"),
        ("💧", "water"),
        ("🧃", "juice"),
        ("🥤", "soda"),
    ],
    "🍫 Snacks & Sweets": [
        ("🍫", "chocolate"),
        ("🍪", "cookies"),
        ("🍟", "chips"),
        ("🍿", "popcorn"),
        ("🍬", "candy"),
        ("🥨", "pretzel"),
    ],
}


with tab1:
    st.markdown(
        "Search naturally — the engine combines keyword matching, semantic understanding, and AI reranking. "
        "Type a specific product (e.g. `banana`, `oat milk`, `gluten-free bread`) "
        "or a whole category (e.g. `fruit`, `vegetable`, `meat`, `drinks`, `dessert`) to browse."
    )

    if "current_query" not in st.session_state:
        st.session_state["current_query"] = ""

    st.caption("🔥 **Popular searches** — click a category to expand, then click any product to search instantly. Or click \"Browse all\" to see all items in that category:")
    CATEGORY_BROWSE_QUERIES = {
        "🍎 Fruits": "fruit",
        "🥬 Vegetables": "vegetable",
        "🥩 Meat & Seafood": "meat",
        "🥛 Dairy & Eggs": "dairy",
        "🍞 Bread, Pasta & Grains": "bakery",
        "🍷 Beverages & Alcohol": "drinks",
        "🍫 Snacks & Sweets": "snacks",
    }

    n_cols = 6
    for category_label, terms in POPULAR_CATEGORIES.items():
        with st.expander(category_label, expanded=False):
            browse_query = CATEGORY_BROWSE_QUERIES.get(category_label)
            if browse_query and st.button(
                f"📂 Browse all {category_label.split(' ', 1)[1]} (diverse mix)",
                key=f"browse_{category_label}",
                use_container_width=True,
            ):
                st.session_state["current_query"] = browse_query
                st.session_state["auto_search"] = True

            cols = st.columns(n_cols)
            for i, (emoji, term) in enumerate(terms):
                with cols[i % n_cols]:
                    if st.button(f"{emoji} {term}", key=f"chip_{category_label}_{term}", use_container_width=True):
                        st.session_state["current_query"] = term
                        st.session_state["auto_search"] = True

    query = st.text_input(
        "What are you looking for? (type here, then click Search below)",
        key="current_query",
        placeholder="e.g., low-sugar yogurt, organic almond milk, Korean BBQ ingredients",
    )

    if query and len(query) >= 2:
        try:
            sug_response = requests.get(
                f"{API_URL}/suggest", params={"prefix": query, "top_k": 6}, timeout=2,
            )
            if sug_response.status_code == 200:
                suggestions = sug_response.json().get("suggestions", [])
                if suggestions:
                    pretty = " · ".join(s.get("text", "") for s in suggestions[:6])
                    st.caption(f"💡 Suggestions: {pretty}")
        except requests.RequestException:
            pass

    col1, col2, col3 = st.columns([2, 1, 1])
    with col2:
        top_k = st.slider("Results", 5, 20, 10)
    with col3:
        use_reranker = st.checkbox("AI Reranking (cross-encoder)", value=False)

    with st.expander("🏷️ Filter by attributes (optional)", expanded=False):
        st.caption(
            "Tick boxes to narrow results to products matching ALL selected attributes. "
            "These are dietary/nutritional tags, not product names — e.g. organic, gluten-free."
        )
        attr_cols = st.columns(3)
        selected_attrs = []
        for i, attr_id in enumerate(list(ATTRIBUTE_LABELS.keys())):
            col = attr_cols[i % 3]
            with col:
                if st.checkbox(ATTRIBUTE_LABELS[attr_id], key=f"attr_{attr_id}"):
                    selected_attrs.append(attr_id)

    search_clicked = st.button("🔍 Search", type="primary", use_container_width=True, key="smart_search_btn")
    auto_search = st.session_state.pop("auto_search", False)

    if search_clicked or auto_search:
        if not query:
            st.warning("⚠️ Please type something in the search box first.")
        else:
            with st.spinner(f"Searching across 49,688 products for '{query}'..."):
                try:
                    response = requests.post(
                        f"{API_URL}/search",
                        json={
                            "query": query,
                            "top_k": top_k,
                            "use_reranker": use_reranker,
                            "attributes": selected_attrs if selected_attrs else None,
                            "user_id": ACTIVE_USER_ID,
                        },
                        timeout=30,
                    )
                    if response.status_code == 200:
                        st.session_state["search_data"] = response.json()
                        st.session_state["search_input"] = query
                        st.success(f"✅ Got {st.session_state['search_data'].get('total_results', 0)} results — see below ↓")
                    else:
                        st.error(f"API returned {response.status_code}: {response.text[:200]}")
                except requests.ConnectionError:
                    st.error(
                        "Cannot connect to API. "
                        "Start the backend: `uvicorn src.api.main:app --reload`"
                    )
                except requests.Timeout:
                    st.warning("Search timed out. Try a simpler query or disable reranking.")
                except Exception as exc:
                    st.error(f"Unexpected error: {exc}")

    data = st.session_state.get("search_data")
    last_query = st.session_state.get("search_input", "")
    if data:
        corrected = data.get("corrected_query")
        if corrected and corrected != last_query:
            st.info(
                f"🔤 Searching for **{corrected}** instead of \"{last_query}\" "
                "(auto-corrected)."
            )

        results = data.get("results", [])
        st.subheader(f"📦 Results for \"{corrected or last_query}\" ({data.get('total_results', 0)} found)")

        if not results:
            st.warning("No products matched. Try a broader query or clear filters.")
        for i, product in enumerate(results, 1):
            render_product_card(product, rank=i)

with tab2:
    st.markdown("Your shopping cart and saved-for-later list.")
    if ACTIVE_USER_ID is None:
        st.info("💡 Sign in as a demo user above to use the cart.")
    else:
        try:
            cart_response = requests.get(f"{API_URL}/cart/{ACTIVE_USER_ID}", timeout=5)
            cart_data = cart_response.json() if cart_response.status_code == 200 else {"items": []}
        except requests.RequestException:
            cart_data = {"items": []}

        items = cart_data.get("items", [])
        total = cart_data.get("total_items", 0)
        st.subheader(f"🛒 Cart ({total} items)")
        if not items:
            st.info("Your cart is empty. Add items from Smart Search results.")
        else:
            for item in items:
                with st.container(border=True):
                    cols = st.columns([0.7, 4, 2, 1])
                    with cols[0]:
                        st.markdown(
                            f"<div style='font-size:36px; text-align:center'>{item.get('emoji', '📦')}</div>",
                            unsafe_allow_html=True,
                        )
                    with cols[1]:
                        st.markdown(f"**{item['product_name']}**")
                        st.caption(f"{item.get('category', '')} · {item.get('department', '')}")
                    with cols[2]:
                        st.markdown(f"Qty: **{item['qty']}**")
                    with cols[3]:
                        if st.button("Remove", key=f"cart_rm_{item['product_id']}"):
                            requests.post(
                                f"{API_URL}/cart/remove",
                                json={"user_id": ACTIVE_USER_ID, "product_id": item["product_id"]},
                                timeout=5,
                            )
                            st.experimental_rerun()
            if st.button("🗑️ Clear cart", type="secondary"):
                requests.post(f"{API_URL}/cart/clear/{ACTIVE_USER_ID}", timeout=5)
                st.experimental_rerun()
            st.success(f"✅ Ready to checkout — {total} items selected (demo only, no payment)")

        st.markdown("---")
        st.subheader("❤️ Wishlist")
        try:
            wl_response = requests.get(f"{API_URL}/wishlist/{ACTIVE_USER_ID}", timeout=5)
            wl_items = wl_response.json().get("items", []) if wl_response.status_code == 200 else []
        except requests.RequestException:
            wl_items = []
        if not wl_items:
            st.caption("Nothing saved for later yet.")
        else:
            for it in wl_items:
                render_product_card(it, show_score=False)


with tab3:
    st.markdown("Your prior orders + one-click reorder using real Instacart history.")
    if ACTIVE_USER_ID is None:
        st.info("💡 Sign in as a demo user above to see your order history.")
    else:
        st.subheader("⭐ Buy It Again")
        st.caption("Your most-frequently-purchased items")
        try:
            ba_response = requests.get(
                f"{API_URL}/orders/buy-again/{ACTIVE_USER_ID}", params={"top_k": 6}, timeout=10,
            )
            buy_again_items = ba_response.json().get("products", []) if ba_response.status_code == 200 else []
        except requests.RequestException:
            buy_again_items = []
        if not buy_again_items:
            st.caption("No prior order history found for this user (history loaded only for the top 10K most active users — try a heavy buyer like User 206105).")
        else:
            for p in buy_again_items:
                render_product_card(p, show_score=False)

        st.markdown("---")
        st.subheader("📦 Recent Orders")
        try:
            orders_response = requests.get(
                f"{API_URL}/orders/{ACTIVE_USER_ID}", params={"limit": 5}, timeout=10,
            )
            orders_data = orders_response.json() if orders_response.status_code == 200 else {"orders": []}
        except requests.RequestException:
            orders_data = {"orders": []}
        recent_orders = orders_data.get("orders", [])
        if not recent_orders:
            st.caption("No orders found.")
        else:
            for o in recent_orders[:5]:
                with st.expander(f"Order #{o['order_number']} — {o['n_items']} items"):
                    st.caption(
                        f"Day-of-week: {o['order_dow']} · Hour: {o['order_hour_of_day']}:00 · "
                        f"Days since prior: {o.get('days_since_prior_order') or 'N/A'}"
                    )
                    if st.button(f"🔁 Reorder all {o['n_items']} items", key=f"reorder_{o['order_id']}"):
                        for entry in o["items"]:
                            try:
                                requests.post(
                                    f"{API_URL}/cart/add",
                                    json={
                                        "user_id": ACTIVE_USER_ID,
                                        "product_id": entry["product_id"],
                                        "qty": 1,
                                    },
                                    timeout=5,
                                )
                            except requests.RequestException:
                                pass
                        st.success(f"Added {o['n_items']} items to your cart. Switch to 🛒 Cart tab to view.")


with tab4:
    st.markdown("Find alternatives when a product is out of stock — search by name to find your product.")

    product_query = st.text_input(
        "Search for a product",
        placeholder="e.g., chocolate cookies, greek yogurt, almond milk",
        key="sub_search",
    )

    if st.button("Search Products", type="secondary", use_container_width=True) and product_query:
        with st.spinner("Searching..."):
            try:
                response = requests.post(
                    f"{API_URL}/search",
                    json={"query": product_query, "top_k": 20, "use_reranker": False},
                    timeout=15,
                )
                results = response.json().get("results", [])
                st.session_state["sub_products"] = results
            except requests.ConnectionError:
                st.error("Cannot connect to API.")

    sub_products = st.session_state.get("sub_products", [])

    if sub_products:
        product_options = {
            f"{p['product_name']}  —  {p.get('category', '')}  ({p.get('order_count', 0) or 0:,} orders)": p["product_id"]
            for p in sub_products
        }
        selected_label = st.selectbox(
            "Select a product",
            options=list(product_options.keys()),
        )
        selected_pid = product_options[selected_label]

        col1, col2 = st.columns([1, 1])
        with col1:
            sub_type = st.selectbox(
                "Substitute type",
                ["similar", "healthier", "cheaper"],
                format_func=lambda x: {"similar": "🔄 Most Similar", "healthier": "💪 Healthier", "cheaper": "💰 Cheaper"}[x],
            )
        with col2:
            num_subs = st.slider("Number of substitutes", 3, 10, 5)

        if st.button("Find Substitutes", type="primary", use_container_width=True):
            with st.spinner("Finding the best alternatives..."):
                try:
                    response = requests.post(
                        f"{API_URL}/substitute",
                        json={
                            "product_id": selected_pid,
                            "top_k": num_subs,
                            "substitute_type": sub_type,
                        },
                        timeout=30,
                    )

                    if response.status_code == 404:
                        st.error("Product not found in catalog.")
                    else:
                        data = response.json()

                        original = data.get("original_product", {})
                        st.subheader("Original Product")
                        render_product_card(original, show_score=False)

                        st.subheader(f"Top {sub_type.title()} Substitutes")
                        for i, sub in enumerate(data.get("substitutes", []), 1):
                            render_product_card(sub, rank=i, score_key="similarity_score")

                        # Frequently Bought Together (Amazon-style)
                        try:
                            fbt_response = requests.get(
                                f"{API_URL}/related/{selected_pid}",
                                params={"top_k": 5},
                                timeout=10,
                            )
                            if fbt_response.status_code == 200:
                                fbt_data = fbt_response.json()
                                related = fbt_data.get("related", [])
                                if related:
                                    st.subheader("Frequently Bought Together")
                                    st.caption(
                                        "Other shoppers who bought this also bought "
                                        "(from real Instacart order history)"
                                    )
                                    for i, item in enumerate(related, 1):
                                        render_product_card(item, rank=i, score_key="similarity_score")
                            elif fbt_response.status_code == 503:
                                st.info(
                                    "💡 FBT model not loaded. "
                                    "Run `python3 scripts/build_fbt.py` to enable "
                                    "Frequently Bought Together recommendations."
                                )
                        except requests.RequestException:
                            pass  # FBT is optional — silently skip

                except requests.ConnectionError:
                    st.error(
                        "Cannot connect to API. "
                        "Start the backend: `uvicorn src.api.main:app --reload`"
                    )
    elif product_query:
        st.info("No products found. Try a different search term.")

with tab5:
    st.markdown(
        "Ask shopping questions in natural language — answers are grounded in the product catalog "
        "using retrieval-augmented generation (RAG)."
    )

    question = st.text_area(
        "Your question",
        placeholder="e.g., What can I use instead of heavy cream?",
        key="qa_question",
    )

    qa_top_k = st.slider("Products to ground the answer", 3, 10, 5, key="qa_top_k")

    if st.button("Ask", type="primary", use_container_width=True, key="qa_ask") and question:
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_URL}/qa",
                    json={"question": question, "top_k": qa_top_k},
                    timeout=60,
                )

                if response.status_code == 200:
                    data = response.json()

                    st.info(data.get("answer", ""))

                    model_name = data.get("model", "")
                    if model_name:
                        st.caption(f"Powered by {model_name}")

                    referenced = data.get("referenced_products", [])
                    if referenced:
                        st.subheader("Referenced Products")
                        for i, product in enumerate(referenced, 1):
                            render_product_card(product, rank=i, show_score=False)
                else:
                    try:
                        err = response.json()
                        message = err.get("detail") or err.get("message") or response.text
                    except Exception:
                        message = response.text
                    st.error(f"Error from server: {message}")

            except requests.ConnectionError:
                st.error(
                    "Cannot connect to API. "
                    "Start the backend: `uvicorn src.api.main:app --reload`"
                )
            except requests.Timeout:
                st.warning("Request timed out. Try a simpler question.")

# Sidebar with project info
with st.sidebar:
    st.markdown("### 🧪 A/B Experiments")
    try:
        exp_response = requests.get(
            f"{API_URL}/experiments",
            params={"user_id": ACTIVE_USER_ID} if ACTIVE_USER_ID else {},
            timeout=2,
        )
        if exp_response.status_code == 200:
            experiments_data = exp_response.json().get("experiments", [])
            if experiments_data:
                for exp in experiments_data:
                    variant = exp.get("assigned_variant", "—")
                    st.markdown(f"**{exp['name']}**")
                    st.caption(exp["description"])
                    if ACTIVE_USER_ID:
                        st.success(f"Variant: `{variant}`")
                    else:
                        st.caption("(sign in to see your variant)")
            else:
                st.caption("No active experiments")
    except Exception:
        st.caption("Experiments unavailable")

    st.markdown("---")
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
