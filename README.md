# Grocery Intelligence

An applied AI system for online grocery marketplaces that adapts pre-trained language and embedding models to domain-specific tasks: **semantic product search**, **substitute recommendation**, and **shopping assistance**.

## Features

### 1. Semantic Product Search
Users search naturally — "low-sugar yogurt", "cheap high-protein breakfast" — and get ranked, relevant results with explanations.

### 2. Smart Substitute Recommendation
When products are out of stock, the system recommends alternatives: same brand, cheaper, healthier, or dietary-specific substitutes.

### 3. Shopping Q&A Assistant
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

All data is **real** — no mock or synthetic data.

| Source | What | URL |
|--------|------|-----|
| Instacart Market Basket | 50K real products, 3.4M orders | [Kaggle](https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis) |
| Open Food Facts | Nutrition, ingredients, allergens for 2M+ products | [openfoodfacts.org](https://world.openfoodfacts.org/) |

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

# Run the API
uvicorn src.api.main:app --reload

# Run the frontend
streamlit run frontend/app.py
```

## License

MIT
