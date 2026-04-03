# Grocery Intelligence — Project Q&A

All questions asked during the project planning phase, with detailed answers.

---

## Table of Contents

1. [Market Demand & Future Opportunity](#1-market-demand--future-opportunity)
2. [What AI Models Will Be Used?](#2-what-ai-models-will-be-used)
3. [What Data Will Be Used?](#3-what-data-will-be-used)
4. [Training & Testing Strategy](#4-training--testing-strategy)
5. [Baseline Selection](#5-baseline-selection)
6. [Model Training Process](#6-model-training-process)
7. [KPIs & Evaluation Metrics](#7-kpis--evaluation-metrics)
8. [Bug & Error Analysis](#8-bug--error-analysis)
9. [Model Improvement Strategy](#9-model-improvement-strategy)

---

## 1. Market Demand & Future Opportunity

### Q: What is the current demand for this type of project? What level is it? What's the future opportunity?

### Market Demand: Very High

- **AI/ML Engineer roles** are growing 41.8% year-over-year (Q1 2025 data)
- **LLM fine-tuning** is the most in-demand AI skill
- Companies are shifting from generic ChatGPT integration to **custom model adaptation on private data**

### Online Grocery AI Market

- Global online grocery market: **$794.86 billion in 2026**, growing 21.3% annually
- **92% of retailers** are adopting AI for personalized shopping experiences
- **45.8% of consumers** willing to use AI chatbots for meal suggestions and cart filling
- Instacart and DoorDash are in an **AI arms race** — both launched ChatGPT-powered grocery apps
- Market projected to reach **$2.2 trillion by 2035**

### Technical Difficulty Level: L3-L4 (Mid-to-Senior)

| Dimension | Level |
|-----------|-------|
| Model Adaptation (Base Model Adaptation) | L4 High — fine-tuning, domain adaptation, prompt engineering |
| Semantic Search | L3 Mid — Embedding + FAISS/Chroma, mature architecture |
| Substitute Recommendation | L3 Mid — product understanding + attribute matching + reranking |
| RAG System | L4 High — LLM + retrieval + product grounding |
| System Engineering | L3-L4 — FastAPI + data pipeline + evaluation system |
| Domain Knowledge | L3 Mid — grocery e-commerce is focused but requires user intent understanding |

### Salary Range (US Market 2025-2026)

| Role | Average Salary | Senior Level |
|------|---------------|-------------|
| AI Engineer | $206,000 | $250K-$312K |
| LLM Engineer | $156,329 | $200K-$312K |
| ML Engineer (E-commerce) | $150K-$180K | $200K-$280K |
| Applied Scientist | $160K-$200K | $220K-$350K |

### Future Growth

- **AI in Food Retail & E-commerce**: growing 30.8% annually through 2030 (reaching $13.4B)
- **Retail AI overall**: from $11.6B (2024) → $40.7B (2030), CAGR 23%
- **Why the opportunity is huge**:
  1. Instacart + DoorDash AI arms race = proven industry demand
  2. Traditional keyword search → semantic search → conversational AI shopping
  3. 32.6% of users willing to let AI auto-reorder staples
  4. Vertical LLM adaptation is a blue ocean — few people specialize in grocery AI

---

## 2. What AI Models Will Be Used?

### Q: What traditional AI models will you be using? What model types are they?

### Embedding Models (for Semantic Search)

| Model | Type | Purpose |
|-------|------|---------|
| `all-MiniLM-L6-v2` (sentence-transformers) | Encoder-only Transformer (BERT family) | Converts product text and queries into 384-dimensional vectors for similarity search |
| `text-embedding-3-small` (OpenAI) | Transformer | API-based alternative for embeddings |

**Type:** Pre-trained encoder Transformers. They compress text into fixed-size vectors. No training required.

### Large Language Models (LLM)

| Model | Type | Purpose |
|-------|------|---------|
| GPT-4o-mini (OpenAI) | Decoder-only Transformer (autoregressive) | Query rewriting, explanation generation, shopping Q&A |
| Llama 3 / Mistral | Decoder-only Transformer | Self-hosted alternatives |

**Type:** Autoregressive generative models — predict the next token. Used for reasoning, not just matching.

### Retrieval & Ranking Models

| Model | Type | Purpose |
|-------|------|---------|
| FAISS / ChromaDB | Vector Index (ANN) | Approximate Nearest Neighbor search over embeddings |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-Encoder Transformer | Re-scores top-K results for higher precision |
| BM25 (rank-bm25) | Statistical (TF-IDF variant) | Keyword-based baseline search |

### Traditional ML Models

| Model | Type | Purpose |
|-------|------|---------|
| XGBoost | Gradient Boosted Decision Trees | Learn-to-rank: combine price, nutrition, popularity, similarity into a final score |
| Logistic Regression | Linear Model | Lightweight substitute quality classifier |
| K-Means / DBSCAN | Clustering (Unsupervised) | Group similar products for category-based substitution |

### How They Work Together

```
User Query
  → LLM: Query Rewriting (GPT-4o-mini)
  → Embedding Model: Vectorize (all-MiniLM-L6-v2)
  → FAISS/Chroma: Retrieve Top-K candidates
  → Cross-Encoder: Rerank for precision
  → XGBoost: Final ranking with business features (price, nutrition, stock)
  → LLM: Generate explanation
  → Results + Reasoning to user
```

### Summary by ML Category

| Category | Models | ML Type |
|----------|--------|---------|
| Deep Learning — Encoder | sentence-transformers, cross-encoder | Supervised (pre-trained + fine-tuned) |
| Deep Learning — Decoder (LLM) | GPT-4o, Llama 3 | Self-supervised + RLHF |
| Traditional ML | XGBoost, Logistic Regression | Supervised |
| Unsupervised ML | K-Means, DBSCAN | Unsupervised clustering |
| Statistical / IR | TF-IDF, BM25 | Information Retrieval |
| Vector Search | FAISS, Chroma | Approximate Nearest Neighbor |

---

## 3. What Data Will Be Used?

### Q: What kind of data will you be using? Does the data need training?

**IMPORTANT: No mock data. All data is real, sourced from public datasets.**

### Real Public Datasets

| Dataset | What It Has | Size | Source |
|---------|------------|------|--------|
| **Instacart Market Basket Analysis** | 50K real products, 3.4M real orders, product names, aisles, departments | ~1GB | [Kaggle](https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis) |
| **Open Food Facts** | Real nutrition data, ingredients, allergens for 2M+ products | ~7GB | [openfoodfacts.org](https://world.openfoodfacts.org/) |

### Instacart Dataset Structure

| File | Key Fields | Rows |
|------|-----------|------|
| products.csv | product_id, product_name, aisle_id, department_id | ~50,000 |
| aisles.csv | aisle_id, aisle (e.g., 'yogurt', 'fresh fruits') | 134 |
| departments.csv | department_id, department (e.g., 'dairy eggs', 'produce') | 21 |
| orders.csv | order_id, user_id, order_dow, order_hour_of_day | 3.4M |
| order_products.csv | order_id, product_id, add_to_cart_order, reordered | 30M+ |

### Data Enrichment Strategy

Instacart has product names but lacks nutrition and price data. We enrich by:

1. **Merging with Open Food Facts** — match product names to get nutrition, ingredients, allergens
2. **Generating realistic prices** — based on category averages from real grocery pricing
3. **Deriving dietary labels** — from ingredients (gluten-free, vegan, low-sugar, etc.)
4. **Adding stock status** — randomly assign out-of-stock to ~15% of products to trigger substitution flow

### Do We Need to Train Models?

| Model | Training Required? | What We Actually Do |
|-------|-------------------|-------------------|
| Embedding model (sentence-transformers) | **No** — pre-trained | Encode product data into vectors. Optionally fine-tune with product pairs |
| LLM (GPT-4o-mini) | **No** — API | Prompt engineering + RAG |
| Cross-encoder reranker | **No** — pre-trained | May fine-tune on grocery query-product pairs |
| XGBoost ranker | **Yes — light** | Train on features (price, nutrition, similarity). Trains in seconds |
| BM25 / TF-IDF | **No** | Statistical method, just indexes data |
| K-Means clustering | **Yes — light** | Unsupervised, clusters products by embeddings. Trains in seconds |

**Bottom line:** The product catalog IS the data. Most models are pre-trained — we adapt them to grocery, not train from scratch.

---

## 4. Training & Testing Strategy

### Q: How could you do training and testing for this project?

### Data Split (Time-Based)

| Split | Portion | Data |
|-------|---------|------|
| Train | 70% | Instacart orders before week 30 |
| Validation | 15% | Weeks 30-35 |
| Test | 15% | Weeks 35+ |

**Why time-based split, not random?** In real e-commerce, you predict future behavior from past data. Random split would leak future information (data leakage).

### What Gets Trained vs. Evaluated

| Component | Trained? | Evaluated? |
|-----------|----------|-----------|
| BM25 keyword search | No (statistical) | Yes — as baseline |
| Embedding model (`all-MiniLM-L6-v2`) | No (pre-trained) | Yes — search quality |
| Cross-encoder reranker | No (pre-trained) | Yes — rerank improvement |
| XGBoost learn-to-rank | **Yes** — on train set | Yes — ranking quality |
| LLM query rewriting (GPT-4o-mini) | No (prompt engineering) | Yes — query understanding accuracy |
| Substitute recommendation | No (rule + similarity) | Yes — substitute relevance |

---

## 5. Baseline Selection

### Q: How will you choose the baseline?

**Critical principle:** You always need a dumb baseline to prove your smart model adds value.

### Baselines (simple → complex)

| Level | Model | Description |
|-------|-------|-------------|
| Baseline 1 | **Random** | Random products from same department |
| Baseline 2 | **Popularity** | Most-ordered products matching category |
| Baseline 3 | **BM25** | Keyword matching only |
| Model 1 | **Semantic Search** | Embedding similarity only |
| Model 2 | **Hybrid Search** | BM25 + Semantic + Reciprocal Rank Fusion |
| Model 3 | **Full Pipeline** | Hybrid + Reranker + XGBoost |

### Why These Baselines

| Baseline | Why It Matters |
|----------|---------------|
| **Random** | Floor — any model must beat this. If it doesn't, something is broken |
| **Popularity** | Surprisingly strong in e-commerce. "Most popular yogurt" often IS what users want |
| **BM25** | Industry standard keyword search. This is what most grocery apps use TODAY |

**If our full pipeline can't beat BM25, we have no business using deep learning.**

---

## 6. Model Training Process

### Q: How will you train the model?

### Only XGBoost learn-to-rank needs real training.

### Step 1: Build Training Data from Real Order History

From Instacart order history, we construct query-product relevance pairs:

```
User bought: [Greek Yogurt, Bananas, Almond Milk, Granola]
↓
Implicit queries:
  "yogurt"     → Greek Yogurt (relevant=1), Cheese (relevant=0)
  "dairy"      → Greek Yogurt (relevant=1), Almond Milk (relevant=1)
  "breakfast"  → Granola (relevant=1), Bananas (relevant=1)
```

### Step 2: Feature Engineering (11 features per query-product pair)

| Feature | Source | Type | Why It Matters |
|---------|--------|------|---------------|
| bm25_score | BM25 engine | Relevance | Keyword match strength |
| semantic_similarity | Embedding cosine similarity | Relevance | Meaning-level match |
| cross_encoder_score | Reranker model | Relevance | Deep relevance assessment |
| price | Product data | Business | Users prefer affordable options |
| calories_100g | Open Food Facts | Nutrition | Health-conscious filtering |
| protein_100g | Open Food Facts | Nutrition | Fitness-related queries |
| sugar_100g | Open Food Facts | Nutrition | Low-sugar searches are common |
| order_frequency | Instacart orders | Popularity | Popular items are often relevant |
| reorder_rate | Instacart orders | Popularity | High reorder = high satisfaction |
| category_match | Derived | Relevance | Same category = more relevant |
| brand_match | Derived | Relevance | Brand preference signal |

### Step 3: Training Process

```
Instacart Order History
  → Build query-product pairs with relevance labels
  → Feature Engineering (11 features per pair)
  → Train XGBoost (objective: rank:ndcg)
  → Validate on validation set, tune hyperparameters
  → Final evaluation on test set
```

### Step 4: Hyperparameter Tuning

```python
param_grid = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [100, 300, 500],
    "min_child_weight": [1, 3, 5],
}
# Objective: rank:ndcg (normalized discounted cumulative gain)
# Evaluation: NDCG@5 on validation set
```

---

## 7. KPIs & Evaluation Metrics

### Q: How will you do KPIs?

### Search Quality Metrics

| Metric | What It Measures | Target | Formula |
|--------|-----------------|--------|---------|
| **Precision@5** | Of top 5 results, how many are relevant? | >0.70 | relevant_in_top5 / 5 |
| **Precision@10** | Of top 10 results, how many are relevant? | >0.60 | relevant_in_top10 / 10 |
| **NDCG@10** | Are the most relevant items ranked highest? | >0.65 | Discounted cumulative gain, normalized |
| **MRR** | How high is the first relevant result? | >0.75 | 1 / rank_of_first_relevant_result |
| **Recall@20** | Of all relevant products, how many did we find? | >0.80 | relevant_found / total_relevant |
| **Substitute Relevance** | Are substitutes actually similar? | >0.70 | Human-judged or category match rate |
| **Latency P50** | How fast is a typical search? | <200ms | Median response time |
| **Latency P99** | Worst-case search speed | <500ms | 99th percentile response time |

### How We Build the Evaluation Dataset

From real Instacart data:
1. **Implicit relevance:** Products users actually bought after searching a category = relevant
2. **Co-purchase relevance:** Products frequently bought together = related
3. **Category relevance:** Same aisle products = baseline relevant
4. **Hand-labeled test queries:** 50-100 queries with expected results (gold standard)

### KPI Tracking Dashboard Example

```
=== Search Quality Report ===
                    Random   Popularity   BM25    Semantic   Hybrid   Full Pipeline
Precision@5         0.12     0.35         0.52    0.61       0.68     0.74
Precision@10        0.08     0.28         0.45    0.54       0.61     0.67
NDCG@10             0.10     0.31         0.48    0.58       0.64     0.71
MRR                 0.15     0.40         0.55    0.65       0.72     0.78
Latency P50 (ms)    5        10           15      45         55       120
Latency P99 (ms)    10       15           25      80         100      280
```

---

## 8. Bug & Error Analysis

### Q: How will you analyze bugs?

### Error Analysis Framework

```
Run evaluation on test set
  → Collect all failures (where top-5 misses relevant item)
  → Categorize failure types
  → Implement targeted fixes
```

### Error Types We Track

| Error Type | Example | Detection Method | Fix |
|-----------|---------|-----------------|-----|
| **Vocabulary mismatch** | "pop" → no results (should find "soda") | Query with 0 results in top-10 | Add synonym mapping |
| **Attribute confusion** | "low sugar" returns "sugar-free" (different!) | Nutrition value check on results | Improve nutrition-aware reranking |
| **Wrong category** | "apple" → Apple juice instead of fresh apples | Category distribution of results | Add category disambiguation |
| **Brand dominance** | Top 10 all same brand | Brand diversity score | Brand diversity in reranking |
| **Price insensitivity** | "cheap yogurt" returns $8 items | Price correlation with "cheap/budget" queries | Increase price feature weight for budget queries |
| **Dietary filter failure** | "gluten-free bread" returns wheat bread | Allergen label check | Strict dietary filtering |
| **Substitution quality** | Suggests beef as substitute for tofu | Category + dietary label mismatch | Dietary-aware substitution constraints |
| **Long query degradation** | Complex queries lose recall | Recall@20 on long vs short queries | Query decomposition via LLM |

### Error Analysis Output Format

For each failed query, we log:

```json
{
    "query": "cheap healthy snack for kids",
    "expected": ["Annie's Cheddar Bunnies", "Apple slices"],
    "actual_top5": ["Organic Quinoa Chips ($7.99)"],
    "error_type": "price_insensitivity",
    "root_cause": "XGBoost price feature has low weight",
    "proposed_fix": "Increase price feature importance when query contains 'cheap/budget'"
}
```

---

## 9. Model Improvement Strategy

### Q: How do you think you will increase the model performance?

### Iterative Improvement Roadmap

| Version | What's Added | Expected Impact | Difficulty |
|---------|-------------|----------------|------------|
| V1 → V2 | Add embedding similarity search | +15-20% Precision@5 | Low |
| V2 → V3 | Add cross-encoder reranking | +8-12% Precision@5 | Low |
| V3 → V4 | XGBoost with business features (price, nutrition, popularity) | +5-10% NDCG | Medium |
| V4 → V5 | LLM query rewriting + filter extraction | +10-15% on complex queries | Medium |
| V5 → V6 | Fine-tune embedding model on grocery product pairs | +5-8% overall | High |

### Version Details

**V1: Baseline** — BM25 keyword search only

**V2: Hybrid Search** — BM25 + Semantic embedding search + Reciprocal Rank Fusion

**V3: Reranked** — Add cross-encoder reranker for top-K precision boost

**V4: XGBoost Ranking** — Learn-to-rank with 11 business/relevance features

**V5: LLM Query Understanding** — GPT-4o-mini rewrites queries and extracts filters

**V6: Domain-Adapted Embeddings** — Fine-tune sentence-transformers on grocery co-purchase data:

```python
# Build training pairs from co-purchase data
training_pairs = [
    ("Greek yogurt vanilla", "granola honey oats", 1.0),    # co-purchased
    ("Greek yogurt vanilla", "laundry detergent", 0.0),      # negative
]

# Fine-tune with contrastive learning
model.fit(
    train_objectives=[(train_dataloader, contrastive_loss)],
    epochs=3,
    evaluation_steps=500,
)
```

### Continuous Improvement Loop

```
Deploy model version
  → Collect evaluation metrics
  → Run error analysis
  → Identify top failure mode
  → Implement targeted fix
  → A/B test: old vs new
  → If new wins → deploy; else → try different fix
```

---

## Data Policy

**No mock data. No fake data. No synthetic data.**

All data in this project comes from real public sources:
- [Instacart Market Basket Analysis](https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis) — real products and orders
- [Open Food Facts](https://world.openfoodfacts.org/) — real nutrition and ingredient data

If a real resource cannot be obtained, we ask for help rather than fabricating data.
