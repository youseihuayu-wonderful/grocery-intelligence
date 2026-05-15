# Grocery Intelligence — Roadmap

This project covers the **core search and recommendation system** for an e-commerce platform. Below are the three concrete directions we could take it next, each tied to a clear career / learning goal.

## Current scope (already built)

The project today implements the search/ranking system in depth:

- **Hybrid retrieval**: BM25 + semantic embeddings + Reciprocal Rank Fusion
- **Cross-encoder reranking** (`ms-marco-MiniLM-L-6-v2`)
- **XGBoost Learn-to-Rank** trained on 1M real Instacart purchase events
- **Substitute recommendation** (similar / healthier / cheaper)
- **LLM Q&A** with GPT-4o-mini, grounded in the catalog (RAG)
- **Algorithmic badges** (Bestseller, Healthy Choice, etc.)
- **Auto-extracted attribute filters** (organic, gluten-free, high-protein, ...)
- **Frequently Bought Together** from real basket data (lift-based)
- **User personalization** built from 162K user profiles
- **Feed-based discovery** (For You, Bestsellers, Healthy Picks, by Department)
- **Behavior tracking** in SQLite with 1M seeded events
- **Multimodal product imagery** (98.5% emoji coverage + partial Open Food Facts images)
- **A/B testing framework** with consistent-hash bucket assignment
- **Online metrics** (CTR / Conversion / MRR with variant comparison)
- **Spelling correction + autocomplete** over the catalog vocabulary
- **FAISS ANN index** (8x faster than the numpy baseline, 98.3% recall)
- **Behavioral LTR retraining pipeline** that closes the feedback loop

**Test coverage**: 208+ passing tests across all modules.

---

## Three directions for next phase

### Direction 1 — Deepen search algorithms

**Goal**: become a Search / Ranking engineer.

| Task | Why it matters | Effort |
|------|----------------|--------|
| Two-Tower neural ranking model | Replaces XGBoost with deep learning (user tower + product tower). Industry standard at large platforms. | 1-2 days |
| Multi-turn conversational search | "Show me cheaper alternatives" — context preservation. | 4-6h |
| Multi-objective ranking | Balance relevance / margin / freshness / diversity in one score. | 6-8h |
| Real-time LTR retraining | Auto-retrain from live behavioral events on a schedule. | 4-6h |
| Online learning / bandits | Update ranker weights from clicks within minutes, not days. | 1-2 days |

**Why this**: This is the most technically deep direction. Every senior search/ranking interview tests these. Real Amazon and TikTok ranking teams spend most of their time on the algorithms in this list.

---

### Direction 2 — Complete the e-commerce platform

**Goal**: become a full-stack e-commerce engineer.

| Task | Why it matters | Effort |
|------|----------------|--------|
| Shopping cart + checkout flow | Without a cart, this is a search demo, not a store. | 1 day |
| Order history + reorder | Repeat purchase is the #1 grocery use case. | 4-6h |
| Wishlist / Save for later | Standard retention feature. | 2-3h |
| Compare items side-by-side | Amazon's "Compare with similar items" feature. | 3-4h |
| Subscriptions (auto-repurchase) | "Subscribe & Save" — high-LTV revenue stream. | 4-6h |
| Promotions / coupons / loyalty | Drives engagement; pricing logic is non-trivial. | 1 day |
| Multi-seller marketplace | One product, many sellers, different prices. | 1-2 days |
| Real reviews + ratings | Requires user-generated content + moderation. | 1-2 days |

**Data limitations**: Instacart data has **no prices, no reviews**. To make these features real, we'd need to either mock those fields or join another dataset.

**Why this**: Hands-on full-stack e-commerce experience. Better fit if the goal is to build a product end-to-end vs deepen one specialized stack.

---

### Direction 3 — Productionize the system

**Goal**: become an ML Systems / Platform engineer.

| Task | Why it matters | Effort |
|------|----------------|--------|
| Redis caching layer | Cache hot queries + user profiles → 10× lower p99. | 3-4h |
| Docker Compose deployment | One-command bring-up of API + Streamlit + Redis. | 2-3h |
| Prometheus + Grafana | Real metrics dashboards, latency / error / QPS tracking. | 4-6h |
| OpenTelemetry distributed tracing | See where time is spent across BM25 / semantic / rerank. | 4-6h |
| Model monitoring + drift detection | Auto-alert when ranking quality drops. | 6-8h |
| CI/CD with canary deploys | Test new ranking model on 5% traffic first. | 1 day |
| Airflow / Dagster for retraining | Scheduled jobs to rebuild profiles, FBT, LTR. | 6-8h |
| Privacy / GDPR / user data deletion | Real systems must comply. | 4-6h |

**Why this**: This is the boring-but-essential layer. Every senior MLE/platform interview asks about it. It's also where the system goes from "cool demo" to "I'd actually trust this with traffic."

---

## Decision matrix

| If your goal is... | Pick |
|--------------------|------|
| **Ranking / search engineer at Amazon / Doordash / Instacart / TikTok Shop** | Direction 1 |
| **Full-stack engineer building an e-commerce product** | Direction 2 |
| **ML systems / platform engineer** | Direction 3 |
| **Generalist who wants strongest portfolio piece** | Direction 3 (most visible impact for least domain expertise required) |
| **You want the most learning per hour** | Direction 1 — algorithm depth pays the most long-term |

---

## Stack reference

| Layer | What we use today | What "production" would use |
|-------|-------------------|------------------------------|
| Vector search | FAISS (in-process) | Pinecone / Weaviate / Qdrant cluster |
| Caching | None | Redis cluster |
| Database | Parquet files + SQLite | Postgres + Redis + S3 |
| Model serving | uvicorn (single process) | TorchServe / Triton + autoscaling |
| Frontend | Streamlit (single user) | Next.js + CDN |
| Auth | None (demo users only) | Auth0 / Cognito |
| Observability | Loguru text logs | Prometheus + Grafana + Sentry |
| Orchestration | Bash scripts | Airflow / Dagster |
| Deployment | Local Python | Kubernetes + Helm |
