# Grocery Intelligence

An applied AI system for online grocery marketplaces that adapts pre-trained language and embedding models to domain-specific tasks: **semantic product search**, **substitute recommendation**, and **shopping assistance**.

## Features

### 1. Semantic Product Search
Users search naturally — "low-sugar yogurt", "cheap high-protein breakfast" — and get ranked, relevant results with explanations.

### 2. Smart Substitute Recommendation
When products are out of stock, the system recommends alternatives: same brand, cheaper, healthier, or dietary-specific substitutes.

### 3. Personalized Recommendations
A **two-tower neural retrieval** model (trained on 1M real purchases) generates personalized candidates: the user vector is pooled online from purchase history and queried against precomputed item vectors with **FAISS**. Validated offline at **Recall@10 +56%** vs a popularity baseline (NDCG@10 +62%), with graceful popularity fallback for cold-start users.

### 4. Shopping Q&A Assistant
Users ask questions like "What can I use instead of heavy cream?" and get answers grounded in real product data.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.12 |
| Backend API | FastAPI |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector Store | ChromaDB |
| LLM | OpenAI GPT-4o-mini (via API) |
| Reranking | cross-encoder (`ms-marco-MiniLM-L-6-v2`) |
| Learn-to-Rank | XGBoost |
| Data Processing | pandas, scikit-learn |
| Frontend | Streamlit |
| Testing | pytest |

## Data Sources

Products, purchases, search/click behavior, and nutrition are **all real** —
no mock or synthetic data.

| Source | What | URL |
|--------|------|-----|
| Instacart Market Basket | 50K real products, 3.4M orders | [Kaggle](https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis) |
| Open Food Facts | Nutrition macros & ingredients (matched to ~15% of the catalog) | [openfoodfacts.org](https://world.openfoodfacts.org/) |

**One exception — prices are synthetic.** The Instacart catalog ships no price
data, so `src/pricing/synthetic_prices.py` derives a *deterministic, plausible*
price for each product from its department, name, nutrition grade, and order
popularity. Prices are clearly labelled as estimates in the UI; they exist only
to make the cart/checkout flow feel real and are **not** real market prices.

## Project Structure

```
grocery-intelligence/
├── src/
│   ├── data/           # Data loading, cleaning, enrichment
│   ├── models/         # Embedding, reranking, LLM integration
│   ├── search/         # Semantic search engine
│   ├── recommend/      # Substitute recommendation logic
│   ├── api/            # FastAPI endpoints
│   ├── evaluation/     # Metrics, eval pipelines
│   └── utils/          # Shared utilities
├── data/
│   ├── raw/            # Original datasets (gitignored)
│   ├── processed/      # Cleaned, enriched product data
│   ├── embeddings/     # Pre-computed vector embeddings
│   └── eval/           # Evaluation datasets
├── notebooks/          # Jupyter notebooks for exploration
├── tests/              # Unit and integration tests
├── scripts/            # Data download, processing scripts
├── frontend/           # Streamlit app
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Models Used

| Model | Type | Purpose | Training Required |
|-------|------|---------|-------------------|
| `all-MiniLM-L6-v2` | Encoder Transformer | Product & query embeddings | No (pre-trained) |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-Encoder Transformer | Rerank search results | No (pre-trained) |
| GPT-4o-mini | Decoder Transformer (LLM) | Query rewriting, explanations | No (API, prompt engineering) |
| XGBoost | Gradient Boosted Trees | Learn-to-rank with business features | Yes (light, seconds) |
| Two-Tower | Dual-encoder (frozen item tower + history user tower) | Personalized candidate generation | Yes (on real purchases) |
| BM25 | Statistical (TF-IDF variant) | Keyword baseline search | No |

## Quick Start

```bash
# Clone
git clone https://github.com/youseihuayu-wonderful/grocery-intelligence.git
cd grocery-intelligence

# Setup virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download data
python scripts/download_data.py

# (Optional) Train the two-tower recommender to enable the /recommend endpoint.
# Without it, /recommend gracefully falls back to popularity-based picks.
python scripts/train_two_tower.py --epochs 20   # GPU recommended (Kaggle T4)

# Run the API
uvicorn src.api.main:app --reload

# Run the frontend
streamlit run frontend/app.py
```

See **[DEMO.md](DEMO.md)** for a scripted end-to-end walkthrough (search →
nutrition filtering → substitutes → recommendations → checkout → metrics).

## License

MIT
