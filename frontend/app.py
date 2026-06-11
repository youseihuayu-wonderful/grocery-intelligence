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

tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["🏠 Home", "🔍 Smart Search", "🛒 Cart", "📦 Your Orders",
     "🔁 Subscribe & Save", "🔄 Substitute Finder", "🤖 Shopping Assistant", "📊 Analytics"]
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
            price = product.get("price")
            if price is not None:
                st.markdown(f"### ${price:,.2f}")
                st.caption("est.")
            if popularity:
                st.caption(f"📊 {popularity:,} orders")
            personalization = product.get("personalization_score")
            if personalization and personalization > 0.2:
                st.caption(f"🎯 Match: {int(personalization * 100)}%")
            if show_score and score is not None:
                st.caption(f"🔢 Score: {score:.3f}")

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

    def _fetch_recommend(user_id, top_k=6, exclude_purchased=False):
        try:
            r = requests.get(
                f"{API_URL}/recommend",
                params={
                    "user_id": user_id,
                    "top_k": top_k,
                    "exclude_purchased": exclude_purchased,
                },
                timeout=15,
            )
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        return {}

    if ACTIVE_USER_ID is not None:
        st.subheader("🎯 Recommended for You")
        st.caption(
            "Two-tower neural retrieval model trained on real purchases — "
            "Recall@10 **+56%** vs popularity (FAISS candidate generation)"
        )
        novel_only = st.toggle(
            "Show only new products (exclude past purchases)",
            value=False,
            key="rec_novel_only",
            help="Off = grocery reorder task (69% of next purchases are repurchases). "
                 "On = novel-item discovery.",
        )
        rec = _fetch_recommend(
            ACTIVE_USER_ID, top_k=6, exclude_purchased=novel_only
        )
        rec_products = rec.get("products", [])
        if rec.get("source") == "popularity":
            st.info(
                "❄️ Cold-start user (no usable purchase history) — "
                "showing popularity-based picks instead."
            )
        if rec_products:
            for i, p in enumerate(rec_products, 1):
                render_product_card(
                    p, rank=i, show_score=True, score_key="feed_score"
                )
        else:
            st.info("No recommendations available yet.")
        st.markdown("---")

        st.subheader("✨ For You")
        st.caption(f"Content-based profile from User {ACTIVE_USER_ID}'s order history")
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
            # Merge sidebar dietary prefs into the search attribute filter
            applied_attrs = list(selected_attrs)
            if ACTIVE_USER_ID is not None:
                try:
                    pr = requests.get(f"{API_URL}/preferences/{ACTIVE_USER_ID}", timeout=2)
                    if pr.status_code == 200:
                        saved = pr.json().get("dietary_attributes", []) or []
                        for s in saved:
                            if s not in applied_attrs:
                                applied_attrs.append(s)
                        if saved:
                            st.caption(f"🍃 Auto-applied your saved preferences: {', '.join(saved)}")
                except requests.RequestException:
                    pass

            with st.spinner(f"Searching across 49,688 products for '{query}'..."):
                try:
                    response = requests.post(
                        f"{API_URL}/search",
                        json={
                            "query": query,
                            "top_k": top_k,
                            "use_reranker": use_reranker,
                            "attributes": applied_attrs if applied_attrs else None,
                            "user_id": ACTIVE_USER_ID,
                        },
                        timeout=30,
                    )
                    if response.status_code == 200:
                        st.session_state["search_data"] = response.json()
                        st.session_state["last_search_attrs"] = applied_attrs
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

        applied = data.get("applied_filters")
        if applied:
            chips = " ".join(ATTRIBUTE_LABELS.get(a, a) for a in applied)
            st.success(f"🥗 Auto-detected health filters from your query: {chips}")
            st.caption(
                "Only products with **verified** Open Food Facts nutrition data "
                "that meet these criteria are shown — nutrition data covers ~15% "
                "of the catalog, so some relevant items may be hidden."
            )

        if not results:
            st.warning("No products matched. Try a broader query or clear filters.")
        for i, product in enumerate(results, 1):
            render_product_card(product, rank=i)

        # Multi-turn conversational follow-up
        st.markdown("---")
        st.subheader("💬 Refine your search")
        st.caption("Type a follow-up like 'make it organic' or 'cheaper one' or 'show bread instead'")
        followup = st.text_input(
            "Follow-up",
            placeholder="e.g., make it organic · cheaper one · show me yogurt instead",
            key="followup_input",
            label_visibility="collapsed",
        )
        if st.button("🎤 Apply follow-up", key="followup_btn") and followup:
            with st.spinner("Interpreting follow-up..."):
                try:
                    fr = requests.post(
                        f"{API_URL}/search/followup",
                        json={
                            "previous_query": corrected or last_query,
                            "previous_filters": {"attributes": st.session_state.get("last_search_attrs", [])},
                            "user_followup": followup,
                            "user_id": ACTIVE_USER_ID,
                            "top_k": top_k,
                        },
                        timeout=30,
                    )
                    if fr.status_code == 200:
                        fdata = fr.json()
                        st.info(f"🤖 {fdata.get('clarification', '')}")
                        st.session_state["search_data"] = fdata.get("search_response", {})
                        st.session_state["search_input"] = fdata.get("interpreted_query", followup)
                        st.session_state["last_search_attrs"] = (
                            fdata.get("interpreted_filters", {}).get("attributes", [])
                        )
                        st.experimental_rerun()
                    else:
                        st.error(f"Error: {fr.text[:200]}")
                except requests.RequestException as exc:
                    st.error(str(exc))

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
                    cols = st.columns([0.7, 4, 1.5, 1.5, 1])
                    with cols[0]:
                        st.markdown(
                            f"<div style='font-size:36px; text-align:center'>{item.get('emoji', '📦')}</div>",
                            unsafe_allow_html=True,
                        )
                    with cols[1]:
                        st.markdown(f"**{item['product_name']}**")
                        st.caption(f"{item.get('category', '')} · {item.get('department', '')}")
                    with cols[2]:
                        unit_price = item.get("price")
                        if unit_price is not None:
                            line_total = unit_price * item["qty"]
                            st.markdown(f"**${line_total:,.2f}**")
                            st.caption(f"${unit_price:,.2f} × {item['qty']}")
                        else:
                            st.markdown(f"Qty: **{item['qty']}**")
                    with cols[3]:
                        st.markdown(f"Qty: **{item['qty']}**")
                    with cols[4]:
                        if st.button("Remove", key=f"cart_rm_{item['product_id']}"):
                            requests.post(
                                f"{API_URL}/cart/remove",
                                json={"user_id": ACTIVE_USER_ID, "product_id": item["product_id"]},
                                timeout=5,
                            )
                            st.experimental_rerun()

            # Pricing summary + promotions
            try:
                pricing_response = requests.get(
                    f"{API_URL}/cart/{ACTIVE_USER_ID}/pricing", timeout=5,
                )
                pricing = pricing_response.json() if pricing_response.status_code == 200 else {}
            except requests.RequestException:
                pricing = {}

            if pricing:
                st.markdown("---")
                subtotal = pricing.get("subtotal", 0)
                total_discount = pricing.get("total_discount", 0)
                final_total = pricing.get("total", 0)

                # Applied promotions
                applied = pricing.get("promotions_applied", [])
                if applied:
                    st.subheader("🎉 Promotions Applied")
                    for p in applied:
                        st.success(f"✅ **{p['title']}** — {p['description']} (Save ${p['discount_amount']:.2f})")

                # Incentive promotions (progress bars)
                available = pricing.get("promotions_available", [])
                if available:
                    st.subheader("💡 Unlock More Savings")
                    for p in available[:3]:
                        progress = p.get("progress", 0)
                        st.markdown(f"**{p['title']}** — {p['description']}")
                        st.progress(min(progress, 1.0))

                st.markdown("---")
                bottom_cols = st.columns(3)
                bottom_cols[0].metric("Subtotal", f"${subtotal:,.2f}")
                bottom_cols[1].metric("Discount", f"-${total_discount:,.2f}")
                bottom_cols[2].metric("**Total**", f"${final_total:,.2f}")
                st.caption("💡 Prices are synthetic estimates — Instacart ships no price data.")

            cart_button_cols = st.columns([2, 1])
            with cart_button_cols[0]:
                if st.button("💳 Checkout (demo)", type="primary", use_container_width=True):
                    st.balloons()
                    st.success(f"✅ Order placed! {total} items, total ${pricing.get('total', 0):,.2f} (demo only, no payment processed).")
            with cart_button_cols[1]:
                if st.button("🗑️ Clear cart", use_container_width=True):
                    requests.post(f"{API_URL}/cart/clear/{ACTIVE_USER_ID}", timeout=5)
                    st.experimental_rerun()

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
    st.markdown("Subscribe to products you reorder regularly — auto-fills your cart on a schedule.")
    if ACTIVE_USER_ID is None:
        st.info("💡 Sign in as a demo user above to manage subscriptions.")
    else:
        try:
            subs_resp = requests.get(f"{API_URL}/subscriptions/{ACTIVE_USER_ID}", timeout=5)
            subs_data = subs_resp.json() if subs_resp.status_code == 200 else {"subscriptions": [], "estimated_monthly_value": 0.0}
        except requests.RequestException:
            subs_data = {"subscriptions": [], "estimated_monthly_value": 0.0}

        active_subs = subs_data.get("subscriptions", [])
        monthly_value = subs_data.get("estimated_monthly_value", 0.0)

        m1, m2 = st.columns(2)
        m1.metric("Active Subscriptions", len(active_subs))
        m2.metric("Estimated Monthly Cost", f"${monthly_value:,.2f}")

        st.markdown("---")
        st.subheader("📅 Your Active Subscriptions")
        if not active_subs:
            st.info("No active subscriptions yet. Use the Smart Search results' 🔁 button to subscribe to a product.")
        else:
            for s in active_subs:
                prod = s.get("product") or {}
                with st.container(border=True):
                    cols = st.columns([0.7, 4, 2, 1.5, 1])
                    with cols[0]:
                        st.markdown(
                            f"<div style='font-size:36px; text-align:center'>{prod.get('emoji', '📦')}</div>",
                            unsafe_allow_html=True,
                        )
                    with cols[1]:
                        st.markdown(f"**{prod.get('product_name', '?')}**")
                        st.caption(f"{s['frequency']} · qty {s['qty']}")
                    with cols[2]:
                        days_until = s.get("days_until_next", 0)
                        if days_until <= 0:
                            st.markdown("**⏰ Due today**")
                        else:
                            st.markdown(f"Next: in **{days_until:.0f}** days")
                    with cols[3]:
                        if prod.get("price"):
                            st.markdown(f"${prod['price']:,.2f}")
                    with cols[4]:
                        if st.button("❌", key=f"sub_cancel_{s['id']}"):
                            requests.post(f"{API_URL}/subscriptions/{s['id']}/cancel", timeout=5)
                            st.experimental_rerun()

            if st.button("🚚 Simulate weekly delivery (fulfill due subs)", type="primary"):
                r = requests.post(f"{API_URL}/subscriptions/fulfill-due", timeout=10)
                if r.status_code == 200:
                    count = r.json().get("count", 0)
                    if count:
                        st.success(f"✅ Fulfilled {count} subscriptions — items added to your 🛒 Cart.")
                    else:
                        st.info("No subscriptions are due right now.")

        st.markdown("---")
        st.subheader("➕ Subscribe to a new product")
        st.caption("Type a product name to find and subscribe.")
        sub_search = st.text_input("Product name", key="sub_new_search")
        if sub_search:
            try:
                sub_resp = requests.post(
                    f"{API_URL}/search",
                    json={"query": sub_search, "top_k": 5},
                    timeout=10,
                )
                sub_candidates = sub_resp.json().get("results", []) if sub_resp.status_code == 200 else []
            except requests.RequestException:
                sub_candidates = []
            if sub_candidates:
                for c in sub_candidates:
                    cc = st.columns([0.5, 3, 2, 2])
                    with cc[0]:
                        st.markdown(f"<div style='font-size:24px'>{c.get('emoji', '📦')}</div>", unsafe_allow_html=True)
                    with cc[1]:
                        st.markdown(f"**{c['product_name']}**")
                    with cc[2]:
                        freq = st.selectbox(
                            "Frequency", ["weekly", "biweekly", "monthly"],
                            key=f"sub_freq_{c['product_id']}",
                        )
                    with cc[3]:
                        if st.button("🔁 Subscribe", key=f"sub_btn_{c['product_id']}"):
                            requests.post(
                                f"{API_URL}/subscriptions/add",
                                json={
                                    "user_id": ACTIVE_USER_ID,
                                    "product_id": c["product_id"],
                                    "frequency": freq,
                                    "qty": 1,
                                },
                                timeout=5,
                            )
                            st.success(f"Subscribed to {c['product_name']} ({freq})")


with tab5:
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

with tab6:
    assistant_mode = st.radio(
        "Mode",
        ["💬 Q&A", "🍳 Recipe → Cart", "🎯 Goal-based Shopping"],
        horizontal=True,
        key="assistant_mode",
    )

    if assistant_mode == "🍳 Recipe → Cart":
        st.markdown(
            "Type a recipe or dish name — the AI extracts ingredients, matches them to real products, "
            "and lets you add the whole shopping list to your cart with one click."
        )
        recipe = st.text_input(
            "Recipe / dish",
            placeholder="e.g., Korean beef bowls · vegetarian pad thai · chocolate chip cookies",
            key="recipe_query",
        )
        if st.button("🧑‍🍳 Generate shopping list", type="primary", use_container_width=True) and recipe:
            with st.spinner("Asking the AI to decompose the recipe..."):
                try:
                    r = requests.post(
                        f"{API_URL}/agent/recipe-to-cart",
                        json={"recipe": recipe, "user_id": ACTIVE_USER_ID},
                        timeout=60,
                    )
                    if r.status_code == 200:
                        st.session_state["recipe_plan"] = r.json()
                    else:
                        st.error(f"Error: {r.text[:200]}")
                except requests.RequestException as exc:
                    st.error(f"Request failed: {exc}")
                except requests.Timeout:
                    st.warning("Timed out. Try a simpler recipe.")

        plan = st.session_state.get("recipe_plan")
        if plan:
            st.success(f"📋 {plan.get('summary', '')}")
            matches = plan.get("matches", [])
            st.subheader(f"🛒 Shopping list ({len(matches)} items)")
            for m in matches:
                product = m.get("product")
                with st.container(border=True):
                    cols = st.columns([0.7, 4, 1.5, 1.5])
                    with cols[0]:
                        emoji = product.get("emoji", "📦") if product else "❓"
                        st.markdown(
                            f"<div style='font-size:36px; text-align:center'>{emoji}</div>",
                            unsafe_allow_html=True,
                        )
                    with cols[1]:
                        st.markdown(f"**{m['requested_name']}**")
                        if m.get("quantity"):
                            st.caption(f"Needed: {m['quantity']}")
                        if product:
                            st.caption(f"Matched: {product['product_name']}")
                    with cols[2]:
                        if product and product.get("price"):
                            st.markdown(f"**${product['price']:.2f}**")
                    with cols[3]:
                        if product and ACTIVE_USER_ID and st.button(
                            "🛒 Add", key=f"recipe_add_{product['product_id']}"
                        ):
                            requests.post(
                                f"{API_URL}/cart/add",
                                json={"user_id": ACTIVE_USER_ID, "product_id": product["product_id"], "qty": 1},
                                timeout=5,
                            )

            if ACTIVE_USER_ID and st.button(
                "🛒 Add ALL to cart", type="primary", use_container_width=True
            ):
                added = 0
                for m in matches:
                    if m.get("product"):
                        try:
                            requests.post(
                                f"{API_URL}/cart/add",
                                json={
                                    "user_id": ACTIVE_USER_ID,
                                    "product_id": m["product"]["product_id"],
                                    "qty": 1,
                                },
                                timeout=5,
                            )
                            added += 1
                        except requests.RequestException:
                            pass
                st.success(f"✅ Added {added} items to your cart. Switch to 🛒 Cart tab to view.")
            elif not ACTIVE_USER_ID:
                st.info("💡 Sign in as a demo user above to add items to your cart.")

    elif assistant_mode == "🎯 Goal-based Shopping":
        st.markdown(
            "Tell the AI your shopping goal in plain language — it interprets and returns a structured plan."
        )
        goal = st.text_input(
            "Your goal",
            placeholder="e.g., Healthy lunches for the work week under $50 · Snacks for kids party",
            key="goal_query",
        )
        if st.button("🎯 Plan", type="primary", use_container_width=True, key="goal_plan") and goal:
            with st.spinner("Planning..."):
                try:
                    r = requests.post(
                        f"{API_URL}/agent/plan-shopping",
                        json={"goal": goal, "user_id": ACTIVE_USER_ID, "max_products": 20},
                        timeout=60,
                    )
                    if r.status_code == 200:
                        st.session_state["goal_plan"] = r.json()
                    else:
                        st.error(f"Error: {r.text[:200]}")
                except requests.RequestException as exc:
                    st.error(str(exc))

        gplan = st.session_state.get("goal_plan")
        if gplan:
            st.info(f"💡 {gplan.get('interpretation', '')}")
            if gplan.get("notes"):
                st.caption(gplan["notes"])
            for cat in gplan.get("categories", []):
                st.subheader(f"📦 {cat['category']}")
                for item in cat.get("items", []):
                    if item.get("product"):
                        render_product_card(item["product"], show_score=False)

    else:  # 💬 Q&A mode
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

with tab7:
    st.markdown("Admin dashboard — aggregate behavior log analytics across all users.")
    try:
        a_resp = requests.get(f"{API_URL}/analytics/overview", timeout=15)
        a_data = a_resp.json() if a_resp.status_code == 200 else {}
    except requests.RequestException as exc:
        a_data = {}
        st.error(f"Analytics API unavailable: {exc}")

    if a_data:
        funnel = a_data.get("funnel", {})
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Views", f"{funnel.get('n_views', 0):,}")
        m2.metric("Clicks", f"{funnel.get('n_clicks', 0):,}")
        m3.metric("Add to Cart", f"{funnel.get('n_add_to_cart', 0):,}")
        m4.metric("Purchases", f"{funnel.get('n_purchases', 0):,}")

        st.markdown("---")
        st.subheader("🔻 Conversion Funnel Rates")
        fc1, fc2, fc3, fc4 = st.columns(4)
        fc1.metric("V → C", f"{funnel.get('view_to_click_rate', 0):.2%}")
        fc2.metric("C → Cart", f"{funnel.get('click_to_cart_rate', 0):.2%}")
        fc3.metric("Cart → Buy", f"{funnel.get('cart_to_purchase_rate', 0):.2%}")
        fc4.metric("Overall", f"{funnel.get('overall_conversion', 0):.2%}")

        st.markdown("---")
        st.subheader("🔥 Top Products")
        hp = a_data.get("hot_products", [])
        if hp:
            import pandas as _pd
            st.dataframe(_pd.DataFrame(hp), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("🥬 Category Breakdown")
        cb = a_data.get("category_breakdown", [])
        if cb:
            import pandas as _pd
            cb_df = _pd.DataFrame(cb)
            if not cb_df.empty:
                cb_df["share"] = cb_df["share"].apply(lambda v: f"{v:.1%}")
                st.dataframe(cb_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("📈 Daily Activity (last 14 days)")
        daily = a_data.get("daily_counts", [])
        if daily:
            import pandas as _pd
            ddf = _pd.DataFrame(daily).set_index("date")
            st.line_chart(ddf)

        st.markdown("---")
        st.subheader("🔍 Top Search Queries")
        tq = a_data.get("top_queries", [])
        if tq:
            import pandas as _pd
            st.dataframe(_pd.DataFrame(tq), use_container_width=True, hide_index=True)
        else:
            st.caption("No search queries logged yet — make a few searches in the 🔍 tab to populate this.")

        st.markdown("---")
        sq = a_data.get("search_quality", {})
        st.subheader("✨ Search Quality Signals")
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Total searches", f"{sq.get('total_search_events', 0):,}")
        sc2.metric("Distinct queries", f"{sq.get('distinct_queries', 0):,}")
        sc3.metric("CTR", f"{sq.get('click_through_rate', 0):.2%}")


# Sidebar with project info
with st.sidebar:
    st.markdown("### 🍃 Dietary Preferences")
    if ACTIVE_USER_ID is None:
        st.caption("Sign in to set persistent preferences")
    else:
        try:
            prefs_resp = requests.get(f"{API_URL}/preferences/{ACTIVE_USER_ID}", timeout=2)
            saved_prefs = prefs_resp.json() if prefs_resp.status_code == 200 else {}
        except requests.RequestException:
            saved_prefs = {}

        DIETARY_OPTIONS = [
            ("organic", "🌿 Organic"),
            ("gluten-free", "🌾 Gluten-Free"),
            ("vegan", "🌱 Vegan"),
            ("vegetarian", "🥬 Vegetarian"),
            ("dairy-free", "🥛 Dairy-Free"),
            ("low-sugar", "🍯 Low Sugar"),
            ("high-protein", "💪 High Protein"),
            ("low-fat", "🥦 Low Fat"),
            ("keto-friendly", "🥑 Keto"),
        ]
        current = set(saved_prefs.get("dietary_attributes", []))
        st.caption("Select dietary requirements to auto-filter your searches.")
        selected = []
        for attr_id, label in DIETARY_OPTIONS:
            if st.checkbox(label, value=(attr_id in current), key=f"pref_{attr_id}"):
                selected.append(attr_id)
        if selected != list(current):
            if st.button("💾 Save preferences"):
                requests.post(
                    f"{API_URL}/preferences",
                    json={
                        "user_id": ACTIVE_USER_ID,
                        "dietary_attributes": selected,
                        "excluded_attributes": [],
                    },
                    timeout=5,
                )
                st.success("Saved!")

    st.markdown("---")
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
        "- **Recommender**: two-tower + FAISS retrieval\n"
        "- **Backend**: FastAPI\n"
        "- **Data**: Instacart + Open Food Facts"
    )
    st.markdown("---")
    st.caption(
        "💡 **Note on prices**: Instacart ships no price data, so prices shown "
        "are *deterministic synthetic estimates* (from department, nutrition, and "
        "popularity) — not real market prices. Everything else is real data."
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
