# Grocery Intelligence — Project Plan

## Project Name

**Smart Grocery AI Assistant for Search and Substitute Recommendation**

## Goal

Build an applied AI system for online grocery marketplaces that adapts pre-trained language and embedding models to domain-specific tasks.

## One-Line Description

Built an applied AI system for online grocery marketplaces that adapts a base language/embedding model to domain-specific tasks such as semantic product search, substitute recommendation, and user-facing shopping assistance.

---

## 3 Core Features

| # | Feature | Description |
|---|---------|-------------|
| 1 | **Semantic Product Search** | Users search "low-sugar yogurt" → system returns ranked, relevant products with reasoning |
| 2 | **Substitute Recommendation** | Item out of stock → suggest alternatives (cheaper, healthier, same brand, dietary-specific) |
| 3 | **Shopping Q&A Assistant** | User asks "What can I use instead of heavy cream?" → LLM answers grounded in real product catalog |

### Starting Point

Feature 1 + 2 combined: **"Grocery product semantic search + substitute recommendation"**
- Best balance of business value, demo appeal, and technical depth

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.12 |
| Backend | FastAPI |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector Store | ChromaDB |
| LLM | OpenAI GPT-4o-mini |
| Reranking | cross-encoder (`ms-marco-MiniLM-L-6-v2`) |
| Learn-to-Rank | XGBoost |
| Keyword Search | BM25 (rank-bm25) |
| Data Processing | pandas, scikit-learn |
| Frontend | Streamlit |
| Testing | pytest |

---

## Data Sources (All Real)

| Source | What | URL |
|--------|------|-----|
| Instacart Market Basket | 50K real products, 3.4M orders | [Kaggle](https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis) |
| Open Food Facts | Nutrition, ingredients, allergens for 2M+ products | [openfoodfacts.org](https://world.openfoodfacts.org/) |

---

## 4-Week Development Plan

### Week 1: Data Preparation + Embedding Pipeline
- Download Instacart dataset from Kaggle
- Download Open Food Facts data
- Clean and merge datasets
- Build product text representations
- Generate embeddings with sentence-transformers
- Store in ChromaDB

### Week 2: Search System + Ranking Logic
- Implement BM25 keyword search (baseline)
- Implement semantic search with embeddings
- Build hybrid search with Reciprocal Rank Fusion
- Add cross-encoder reranking
- Compare baselines vs models

### Week 3: Substitute Recommendation + LLM Integration
- Build substitute recommendation engine
- Implement healthier/cheaper/similar alternatives
- Integrate GPT-4o-mini for query rewriting
- Add explanation generation
- Build XGBoost learn-to-rank

### Week 4: Frontend Demo + Evaluation + Error Analysis
- Build Streamlit frontend
- Run full evaluation (Precision@K, NDCG, MRR, Recall, Latency)
- Conduct error analysis
- Write documentation
- Polish demo for presentation

---

## Search Pipeline Architecture

```
1. User inputs natural language query
2. LLM rewrites query and extracts filters (optional)
3. BM25 keyword search → top-100 candidates
4. Semantic embedding search → top-100 candidates
5. Reciprocal Rank Fusion merges candidates
6. Cross-encoder reranks top-50
7. XGBoost final ranking with business features
8. LLM generates explanation
9. Return results + reasoning to user
```

## Substitute Pipeline Architecture

```
1. User selects a product (or product is out of stock)
2. System extracts product attributes (category, nutrition, price, dietary)
3. Embedding similarity finds nearest products
4. Filter by constraints (same category, price range, dietary compatibility)
5. Group by type: similar / healthier / cheaper / same brand
6. LLM generates substitution explanation
7. Return ranked substitutes with reasons
```

---

## Why This Project Demonstrates Interview-Ready Skills

| Skill | How This Project Shows It |
|-------|--------------------------|
| Base model adaptation | Generic embeddings → grocery-specific query normalization + reranking |
| Evaluation design | Precision@K, NDCG, MRR, Recall, Latency benchmarks |
| Error analysis | Categorized failure types with root causes and fixes |
| System design | Multi-stage pipeline: retrieval → reranking → LLM → API |
| Real data engineering | Loading, cleaning, merging real datasets |
| Business understanding | Price, nutrition, dietary features in ranking |
