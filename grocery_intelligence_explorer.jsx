import { useState, useEffect, useRef } from "react";

// ─── Data ──────────────────────────────────────────────────────────────────────

const PIPELINE_STAGES = [
  {
    id: "data",
    label: "Data Ingestion",
    icon: "🗄️",
    color: "#6366f1",
    bg: "#eef2ff",
    border: "#a5b4fc",
    file: "src/data/loader.py",
    subtitle: "Instacart + Open Food Facts",
    tagline: "Two real-world datasets merged into one enriched product catalog",
    overview: `Raw product data from two public datasets is loaded, merged, and saved as a Parquet file — the single source of truth for the entire pipeline.`,
    details: [
      { label: "Instacart catalog", value: "49,688 products with aisle & department labels" },
      { label: "Open Food Facts", value: "Nutrition, brand, ingredients for matched products" },
      { label: "Join strategy", value: "Exact name match after lowercasing & stripping punctuation" },
      { label: "Output", value: "data/processed/product_catalog.parquet" },
    ],
    code: `def build_product_catalog() -> pd.DataFrame:
    instacart_df = load_instacart_products()
    off_df = load_open_food_facts()
    catalog = enrich_products(instacart_df, off_df)

    # Rename to unified schema
    catalog = catalog.rename(columns={
        "aisle": "category",
        "brands": "brand",
        "ingredients_text": "ingredients",
        "energy-kcal_100g": "calories_100g",
        "proteins_100g": "protein_100g",
        "sugars_100g": "sugar_100g",
    })
    catalog.to_parquet("data/processed/product_catalog.parquet")
    return catalog`,
    whyItMatters: "Real datasets (not toy examples) signal production-readiness to interviewers. Merging heterogeneous sources demonstrates data engineering skill.",
    columns: ["product_id","product_name","category","department","brand","ingredients","calories_100g","protein_100g","sugar_100g","nutrition_grade"],
  },
  {
    id: "embeddings",
    label: "Embeddings",
    icon: "🧠",
    color: "#8b5cf6",
    bg: "#f5f3ff",
    border: "#c4b5fd",
    file: "src/models/embeddings.py",
    subtitle: "sentence-transformers · 384-dim vectors",
    tagline: "Every product is encoded into a dense vector so semantic similarity can be computed",
    overview: `The all-MiniLM-L6-v2 model (a distilled BERT) encodes product text into 384-dimensional L2-normalized vectors. These are pre-computed once and saved to disk, so search is fast at query time.`,
    details: [
      { label: "Model", value: "all-MiniLM-L6-v2 (sentence-transformers)" },
      { label: "Dimensions", value: "384 — compact but expressive" },
      { label: "Normalization", value: "L2 norm → cosine similarity becomes a dot product" },
      { label: "Batch size", value: "64 products per forward pass" },
      { label: "Storage", value: "data/embeddings/product_embeddings.npy" },
    ],
    code: `def build_product_text(row: dict) -> str:
    """Combine fields into a rich text representation."""
    parts = [str(row.get("product_name", ""))]
    for field in ["category", "department", "brand"]:
        if row.get(field):
            parts.append(str(row[field]))
    if row.get("ingredients"):
        parts.append(str(row["ingredients"])[:200])
    return " | ".join(parts)
    # e.g. "Chobani Greek Yogurt | dairy | yogurt | gluten-free low-fat"

class ProductEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode(query, normalize_embeddings=True)

    def build_product_embeddings(self, texts, ids, save=True):
        embeddings = self.model.encode(
            texts, batch_size=64,
            normalize_embeddings=True,  # cosine = dot product
        )
        if save:
            np.save("data/embeddings/product_embeddings.npy", embeddings)
        return embeddings`,
    whyItMatters: "Embedding-based retrieval catches intent even when the user's words don't exactly match a product name. 'Low-sugar snack for kids' → relevant yogurts & granola bars.",
  },
  {
    id: "bm25",
    label: "BM25 Index",
    icon: "🔑",
    color: "#0ea5e9",
    bg: "#f0f9ff",
    border: "#7dd3fc",
    file: "src/search/engine.py",
    subtitle: "Keyword retrieval · rank_bm25",
    tagline: "Classic IR: exact-term matching with TF-IDF weighting and document length normalization",
    overview: `BM25 (Best Match 25) scores each product by how often query terms appear, discounted by document length. It's fast and excellent at exact-keyword queries like brand names or product codes.`,
    details: [
      { label: "Library", value: "rank_bm25 (BM25Okapi variant)" },
      { label: "Input", value: "Tokenized product text (lowercase, split on whitespace)" },
      { label: "Output", value: "Top-100 candidates with BM25 relevance scores" },
      { label: "Strength", value: "Exact brand/product name matches, rare keywords" },
      { label: "Weakness", value: "Misses synonyms, paraphrases, multilingual queries" },
    ],
    code: `# Build index once at startup
tokenized = [text.lower().split() for text in self.product_texts]
self.bm25 = BM25Okapi(tokenized)

# At query time
bm25_scores = self.bm25.get_scores(query.lower().split())
bm25_top_idx = np.argsort(bm25_scores)[-100:][::-1]
# → indices of top-100 BM25 candidates`,
    whyItMatters: "Keyword search is still the fastest path to precision for specific product queries. It's the industry baseline that hybrid systems improve upon.",
    formula: "score(q,d) = Σ IDF(qᵢ) · f(qᵢ,d)·(k₁+1) / (f(qᵢ,d) + k₁·(1-b+b·|d|/avgdl))",
  },
  {
    id: "rrf",
    label: "Hybrid RRF",
    icon: "🔀",
    color: "#10b981",
    bg: "#ecfdf5",
    border: "#6ee7b7",
    file: "src/search/engine.py",
    subtitle: "Reciprocal Rank Fusion · merge BM25 + semantic",
    tagline: "The best of both worlds: exact-match precision meets semantic recall",
    overview: `Reciprocal Rank Fusion (RRF) merges the BM25 and semantic ranking lists without needing to know their raw scores. Each candidate's RRF score is the sum of 1/(k+rank) across all lists, where k=60 smooths rank differences.`,
    details: [
      { label: "BM25 pool", value: "Top-100 keyword candidates" },
      { label: "Semantic pool", value: "Top-100 embedding similarity candidates" },
      { label: "RRF constant k", value: "60 (reduces sensitivity to top ranks)" },
      { label: "Merged pool", value: "Union, re-sorted by fused score → top-50 for reranking" },
    ],
    code: `candidate_scores = {}
k = 60  # RRF constant

# Add BM25 ranks
for rank, idx in enumerate(bm25_top_idx):
    candidate_scores[idx] = candidate_scores.get(idx, 0) + 1 / (k + rank + 1)

# Add semantic ranks
for rank, idx in enumerate(semantic_top_idx):
    candidate_scores[idx] = candidate_scores.get(idx, 0) + 1 / (k + rank + 1)

# Sort by fused score → pass top-50 to reranker
fused = sorted(candidate_scores, key=lambda x: candidate_scores[x], reverse=True)
top_50 = fused[:50]`,
    whyItMatters: "RRF is score-agnostic, requiring no calibration between BM25 and cosine similarity scales. It's been shown to match or beat learned combination weights on many benchmarks.",
    formula: "RRF(d) = Σᵣ 1 / (k + rankᵣ(d))",
  },
  {
    id: "reranker",
    label: "Cross-Encoder Reranker",
    icon: "🎯",
    color: "#f59e0b",
    bg: "#fffbeb",
    border: "#fcd34d",
    file: "src/models/reranker.py",
    subtitle: "ms-marco-MiniLM-L-6-v2 · precision pass",
    tagline: "A second, slower model that jointly reads query + product text for higher accuracy",
    overview: `Unlike the bi-encoder (which embeds query and product separately), the cross-encoder reads the query and product text as a concatenated pair — achieving much higher precision at the cost of speed. Applied only to the top-50 candidates from RRF.`,
    details: [
      { label: "Model", value: "cross-encoder/ms-marco-MiniLM-L-6-v2" },
      { label: "Input", value: "Pairs: [query, product_text] × 50" },
      { label: "Output", value: "Single relevance score per pair → re-sorted top-K" },
      { label: "Trained on", value: "MS MARCO passage retrieval benchmark" },
      { label: "Trade-off", value: "2× slower than bi-encoder, but measurably more accurate" },
    ],
    code: `class ProductReranker:
    def __init__(self, model="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model)

    def rerank(self, query, product_texts, product_ids, top_k=10):
        # Build (query, doc) pairs — cross-attention over both
        pairs = [[query, text] for text in product_texts]
        scores = self.model.predict(pairs)  # one relevance score each

        results = [
            {"product_id": pid, "text": text, "relevance_score": float(score)}
            for pid, text, score in zip(product_ids, product_texts, scores)
        ]
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:top_k]`,
    whyItMatters: "The retrieval → reranking two-stage pattern is the industry standard (used by Google, Bing, DoorDash). The cross-encoder adds significant NDCG lift without making retrieval slow.",
  },
  {
    id: "llm",
    label: "LLM Layer",
    icon: "✨",
    color: "#ec4899",
    bg: "#fdf2f8",
    border: "#f9a8d4",
    file: "src/models/llm.py",
    subtitle: "GPT-4o-mini · query rewriting + explanations",
    tagline: "Language understanding at the edges: parse intent in, explain results out",
    overview: `GPT-4o-mini serves two roles: (1) query rewriting — parsing natural language like "cheap healthy snack for kids" into a structured query + filter object; (2) explanation generation — writing a 2-3 sentence narrative justifying the top results.`,
    details: [
      { label: "Model", value: "gpt-4o-mini (cost-efficient, fast)" },
      { label: "Query rewrite", value: "Extracts rewritten_query, filters (price, dietary), intent" },
      { label: "Response format", value: "JSON mode — no parsing errors" },
      { label: "Explanation", value: "References specific attributes (price, calories, dietary flags)" },
      { label: "Temperature", value: "0 for rewriting (deterministic), 0.3 for explanations" },
    ],
    code: `def rewrite_query(query: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content":
                "Extract intent, rewrite for semantic search, extract filters.\\n"
                "Return JSON: {rewritten_query, filters, intent}"},
            {"role": "user", "content": query},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(response.choices[0].message.content)

# "cheap healthy snack for kids under $5" →
# {
#   "rewritten_query": "healthy snack for children",
#   "filters": {"max_price": 5.0, "dietary": ["healthy"]},
#   "intent": "search"
# }`,
    whyItMatters: "Wrapping retrieval with LLM query understanding is the pattern behind Instacart's AI search, DoorDash's ordering assistant, and Amazon's Rufus chatbot.",
  },
  {
    id: "substitute",
    label: "Substitute Engine",
    icon: "🔄",
    color: "#14b8a6",
    bg: "#f0fdfa",
    border: "#99f6e4",
    file: "src/recommend/substitute.py",
    subtitle: "Out-of-stock recovery · similar / healthier / cheaper",
    tagline: "When a product is unavailable, intelligently surface the best alternatives",
    overview: `Uses the same product embeddings to find semantically similar substitutes, then applies business rules: same-category filter, price ratio cap, stock status check. Three modes: similar (embedding cosine), healthier (lower sugar + higher protein + fewer calories), cheaper (same category, lower price).`,
    details: [
      { label: "Similarity metric", value: "Cosine similarity via dot product (normalized vectors)" },
      { label: "Category filter", value: "Restricts candidates to same aisle/category" },
      { label: "Price cap", value: "max_price_ratio=1.5 (at most 50% more expensive)" },
      { label: "Stock check", value: "Excludes out-of-stock candidates" },
      { label: "Health scoring", value: "sugar↓×2 + protein↑ + calories↓×0.5" },
    ],
    code: `def find_substitutes(self, product_id, top_k=5,
                       same_category=True, max_price_ratio=1.5):
    original_embedding = self.embeddings[product_idx]
    similarities = np.dot(self.embeddings, original_embedding)

    # Business rule filters
    candidate_mask = np.ones(len(self.catalog), dtype=bool)
    candidate_mask[product_idx] = False  # exclude self
    if same_category:
        candidate_mask &= (catalog["category"] == original["category"]).values
    if "price" in catalog.columns:
        max_price = original["price"] * max_price_ratio
        candidate_mask &= (catalog["price"] <= max_price).values

    masked_similarities[~candidate_mask] = -1
    top_indices = np.argsort(masked_similarities)[-top_k:][::-1]
    return [catalog.iloc[i] for i in top_indices]`,
    whyItMatters: "Out-of-stock substitution directly impacts revenue — it's a core feature at every major grocery platform. Demonstrating it shows understanding of e-commerce business value.",
  },
  {
    id: "api",
    label: "FastAPI Service",
    icon: "⚡",
    color: "#f97316",
    bg: "#fff7ed",
    border: "#fed7aa",
    file: "src/api/main.py",
    subtitle: "REST API · /search · /substitute · /products",
    tagline: "Production-ready HTTP interface wrapping the entire ML pipeline",
    overview: `FastAPI exposes the search and substitute pipeline as a typed REST service. Pydantic models enforce request/response schemas. Three endpoints cover the core use cases.`,
    details: [
      { label: "POST /search", value: "Hybrid search with optional reranking" },
      { label: "POST /substitute", value: "Similar / healthier / cheaper alternatives" },
      { label: "GET /products", value: "Browse catalog, filter by category" },
      { label: "GET /health", value: "Health check for load balancer probes" },
      { label: "Docs", value: "Auto-generated OpenAPI spec at /docs" },
    ],
    code: `@app.post("/search", response_model=SearchResponse)
async def search_products(request: SearchRequest):
    results = search_engine.search(
        query=request.query,
        top_k=request.top_k,
        use_reranker=request.use_reranker,
    )
    return SearchResponse(
        query=request.query,
        results=[ProductResult(**r) for r in results],
        total_results=len(results),
    )

@app.post("/substitute", response_model=SubstituteResponse)
async def get_substitutes(request: SubstituteRequest):
    if request.substitute_type == "healthier":
        subs = recommender.find_healthier_alternatives(request.product_id)
    elif request.substitute_type == "cheaper":
        subs = recommender.find_cheaper_alternatives(request.product_id)
    else:
        subs = recommender.find_substitutes(request.product_id)
    return SubstituteResponse(substitutes=subs)`,
    whyItMatters: "Demonstrating the ability to ship a real API (not just a notebook) separates ML engineers who can deploy from those who can only prototype.",
  },
  {
    id: "metrics",
    label: "Evaluation",
    icon: "📊",
    color: "#64748b",
    bg: "#f8fafc",
    border: "#cbd5e1",
    file: "src/evaluation/metrics.py",
    subtitle: "Precision@K · NDCG · MRR · Recall · Latency",
    tagline: "Rigorous offline evaluation so you can prove the system actually works",
    overview: `The evaluation module implements four standard IR metrics plus latency percentiles. Given a set of queries with known relevant products, it measures how well the search engine ranks them.`,
    details: [
      { label: "Precision@K", value: "Fraction of top-K results that are relevant" },
      { label: "NDCG@K", value: "Position-aware ranking quality (penalizes relevant items ranked low)" },
      { label: "MRR", value: "1/rank of the first relevant result (best for single-answer queries)" },
      { label: "Recall@20", value: "Fraction of all relevant items found in top-20" },
      { label: "Latency P50/P99", value: "Median and worst-case response time in milliseconds" },
    ],
    code: `def ndcg_at_k(retrieved_ids, relevant_ids, k):
    """Normalized Discounted Cumulative Gain — penalizes low-ranked relevant items."""
    dcg = sum(
        1.0 / np.log2(i + 2)                   # discount by position
        for i, rid in enumerate(retrieved_ids[:k])
        if rid in relevant_ids
    )
    ideal_count = min(len(relevant_ids), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_count))
    return dcg / idcg if idcg > 0 else 0.0

def mean_reciprocal_rank(retrieved_ids, relevant_ids):
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant_ids:
            return 1.0 / (i + 1)   # rank is 1-indexed
    return 0.0`,
    whyItMatters: "Quoting concrete metric numbers (e.g. 'NDCG@10: 0.74, MRR: 0.81') in an interview is a strong signal. It shows you close the loop between model building and measurement.",
    metricsTable: [
      { metric: "Precision@5", formula: "# relevant in top-5 / 5", range: "0–1, higher is better" },
      { metric: "NDCG@10", formula: "DCG / Ideal DCG", range: "0–1, higher is better" },
      { metric: "MRR", formula: "mean of 1/rank₁", range: "0–1, higher is better" },
      { metric: "Recall@20", formula: "# found / # total relevant", range: "0–1, higher is better" },
    ],
  },
];

const QUERY_EXAMPLES = [
  { q: "low sugar yogurt", flow: ["data","embeddings","bm25","rrf","reranker","llm"] },
  { q: "cheap protein snack under $3", flow: ["data","embeddings","bm25","rrf","reranker","llm"] },
  { q: "gluten-free pasta", flow: ["data","embeddings","bm25","rrf","reranker","llm"] },
  { q: "organic baby food substitute", flow: ["data","embeddings","substitute","llm"] },
];

// ─── Helpers ───────────────────────────────────────────────────────────────────

function cn(...classes) { return classes.filter(Boolean).join(" "); }

function Badge({ children, color }) {
  return (
    <span style={{ background: color + "22", color, border: `1px solid ${color}55` }}
      className="inline-block text-xs font-semibold px-2 py-0.5 rounded-full mr-1 mb-1">
      {children}
    </span>
  );
}

function CodeBlock({ code }) {
  return (
    <pre className="text-xs font-mono bg-gray-950 text-green-300 rounded-xl p-4 overflow-x-auto leading-relaxed border border-gray-800 shadow-inner whitespace-pre-wrap">
      {code.trim()}
    </pre>
  );
}

// ─── Pipeline Node ─────────────────────────────────────────────────────────────

function PipelineNode({ stage, isActive, isAnimated, onClick, isLast }) {
  return (
    <div className="flex items-center">
      <button
        onClick={onClick}
        className="flex flex-col items-center group transition-transform hover:scale-105"
        style={{ minWidth: 80 }}
      >
        <div
          className="relative rounded-2xl p-3 transition-all duration-200 shadow-md"
          style={{
            background: isActive ? stage.color : stage.bg,
            border: `2px solid ${isActive ? stage.color : stage.border}`,
            boxShadow: isActive ? `0 0 18px ${stage.color}55` : undefined,
            transform: isAnimated ? "scale(1.15)" : undefined,
          }}
        >
          <span className="text-2xl">{stage.icon}</span>
          {isAnimated && (
            <span className="absolute -top-1 -right-1 w-3 h-3 rounded-full bg-yellow-400 animate-ping" />
          )}
        </div>
        <span
          className="mt-1.5 text-center font-semibold leading-tight"
          style={{
            fontSize: 10,
            color: isActive ? stage.color : "#64748b",
            maxWidth: 72,
          }}
        >
          {stage.label}
        </span>
      </button>
      {!isLast && (
        <div className="flex items-center mx-1 mt-[-14px]">
          <div className="w-5 h-0.5 bg-gray-300" />
          <svg width="8" height="10" viewBox="0 0 8 10" className="text-gray-300">
            <path d="M0 0 L8 5 L0 10 Z" fill="currentColor"/>
          </svg>
        </div>
      )}
    </div>
  );
}

// ─── Detail Panel ──────────────────────────────────────────────────────────────

function DetailPanel({ stage }) {
  const [tab, setTab] = useState("overview");
  const tabs = ["overview","code","why"];

  return (
    <div className="rounded-2xl overflow-hidden shadow-xl border"
      style={{ borderColor: stage.border, background: "#fff" }}>

      {/* Header */}
      <div className="p-5 pb-3" style={{ background: stage.bg }}>
        <div className="flex items-start gap-3">
          <div className="text-4xl">{stage.icon}</div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <h2 className="text-xl font-bold" style={{ color: stage.color }}>{stage.label}</h2>
              <Badge color={stage.color}>{stage.file}</Badge>
            </div>
            <p className="text-sm text-gray-500 mt-0.5">{stage.subtitle}</p>
            <p className="text-sm font-medium text-gray-700 mt-1 italic">"{stage.tagline}"</p>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex gap-1 mt-3">
          {tabs.map(t => (
            <button key={t} onClick={() => setTab(t)}
              className="px-3 py-1 rounded-lg text-xs font-semibold capitalize transition-all"
              style={tab===t
                ? { background: stage.color, color: "#fff" }
                : { background: "#fff", color: stage.color, border: `1px solid ${stage.border}` }
              }>
              {t === "why" ? "Why It Matters" : t.charAt(0).toUpperCase()+t.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div className="p-5">
        {tab === "overview" && (
          <div>
            <p className="text-sm text-gray-700 leading-relaxed mb-4">{stage.overview}</p>
            <div className="grid gap-2">
              {stage.details.map((d,i) => (
                <div key={i} className="flex gap-3 text-sm border-b border-gray-100 pb-2">
                  <span className="font-semibold text-gray-500 shrink-0 w-36">{d.label}</span>
                  <span className="text-gray-800">{d.value}</span>
                </div>
              ))}
            </div>
            {stage.formula && (
              <div className="mt-4 p-3 rounded-xl border font-mono text-xs text-center"
                style={{ background: stage.bg, borderColor: stage.border, color: stage.color }}>
                {stage.formula}
              </div>
            )}
            {stage.columns && (
              <div className="mt-4">
                <p className="text-xs font-semibold text-gray-400 uppercase mb-2">Output Schema</p>
                <div className="flex flex-wrap gap-1">
                  {stage.columns.map(c => <Badge key={c} color={stage.color}>{c}</Badge>)}
                </div>
              </div>
            )}
            {stage.metricsTable && (
              <div className="mt-4 overflow-x-auto">
                <table className="w-full text-xs border-collapse">
                  <thead>
                    <tr style={{ background: stage.bg }}>
                      {["Metric","Formula","Range"].map(h => (
                        <th key={h} className="text-left p-2 font-semibold text-gray-500">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {stage.metricsTable.map((row,i) => (
                      <tr key={i} className="border-t border-gray-100">
                        <td className="p-2 font-mono font-semibold" style={{ color: stage.color }}>{row.metric}</td>
                        <td className="p-2 font-mono text-gray-600">{row.formula}</td>
                        <td className="p-2 text-gray-500">{row.range}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
        {tab === "code" && (
          <div>
            <p className="text-xs text-gray-400 mb-3 font-mono">{stage.file}</p>
            <CodeBlock code={stage.code} />
          </div>
        )}
        {tab === "why" && (
          <div className="flex items-start gap-3 p-4 rounded-xl"
            style={{ background: stage.bg, border: `1px solid ${stage.border}` }}>
            <span className="text-2xl mt-0.5">💡</span>
            <p className="text-sm text-gray-700 leading-relaxed">{stage.whyItMatters}</p>
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Flow Simulator ────────────────────────────────────────────────────────────

function FlowSimulator({ onHighlight }) {
  const [selectedQuery, setSelectedQuery] = useState(null);
  const [step, setStep] = useState(-1);
  const [running, setRunning] = useState(false);
  const timerRef = useRef(null);

  const runFlow = (qi) => {
    if (running) return;
    const flow = QUERY_EXAMPLES[qi].flow;
    setSelectedQuery(qi);
    setStep(0);
    setRunning(true);
    onHighlight(flow[0]);
    let s = 0;
    timerRef.current = setInterval(() => {
      s++;
      if (s < flow.length) {
        setStep(s);
        onHighlight(flow[s]);
      } else {
        clearInterval(timerRef.current);
        setRunning(false);
        setStep(-1);
        onHighlight(null);
      }
    }, 800);
  };

  useEffect(() => () => clearInterval(timerRef.current), []);

  return (
    <div className="rounded-2xl border border-gray-200 bg-white shadow-md p-4">
      <h3 className="text-sm font-bold text-gray-700 mb-3 flex items-center gap-2">
        <span>▶</span> Query Flow Simulator
      </h3>
      <p className="text-xs text-gray-500 mb-3">Click a query to watch data travel through the pipeline</p>
      <div className="grid grid-cols-2 gap-2">
        {QUERY_EXAMPLES.map((ex, i) => (
          <button key={i} onClick={() => runFlow(i)}
            disabled={running}
            className={cn(
              "text-left text-xs p-2.5 rounded-xl border transition-all",
              selectedQuery===i && running
                ? "border-indigo-400 bg-indigo-50 font-semibold"
                : "border-gray-200 bg-gray-50 hover:border-indigo-300 hover:bg-indigo-50",
              running && selectedQuery!==i && "opacity-40 cursor-not-allowed"
            )}>
            <span className="text-base mr-1">🔍</span>
            <span className="text-gray-700">"{ex.q}"</span>
          </button>
        ))}
      </div>
      {running && (
        <div className="mt-3 text-xs text-indigo-600 font-medium flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-indigo-500 animate-pulse" />
          Processing: {QUERY_EXAMPLES[selectedQuery]?.flow[step] && PIPELINE_STAGES.find(s=>s.id===QUERY_EXAMPLES[selectedQuery].flow[step])?.label}...
        </div>
      )}
    </div>
  );
}

// ─── Architecture Overview Card ────────────────────────────────────────────────

function ArchCard() {
  return (
    <div className="rounded-2xl border border-gray-200 bg-gradient-to-br from-slate-900 to-slate-800 text-white p-5 shadow-xl">
      <h3 className="font-bold text-base mb-1">🏗️ System Architecture</h3>
      <p className="text-xs text-slate-400 mb-4">Two-stage retrieval → reranking → LLM, a production ML pattern</p>
      <div className="space-y-2 text-xs">
        {[
          { stage: "Stage 1: Retrieval", desc: "BM25 + Semantic → RRF merge", note: "Fast, high recall", color: "#6366f1" },
          { stage: "Stage 2: Reranking", desc: "Cross-encoder on top-50", note: "Slow, high precision", color: "#f59e0b" },
          { stage: "Stage 3: Generation", desc: "LLM query parsing + explanations", note: "Context-aware UX", color: "#ec4899" },
        ].map(r => (
          <div key={r.stage} className="flex items-center gap-3 p-2 rounded-lg bg-white/5">
            <div className="w-2 h-8 rounded-full shrink-0" style={{ background: r.color }} />
            <div>
              <p className="font-semibold text-white/90">{r.stage}</p>
              <p className="text-slate-400">{r.desc} <span className="text-slate-500">· {r.note}</span></p>
            </div>
          </div>
        ))}
      </div>
      <div className="mt-4 p-3 rounded-xl bg-white/5 border border-white/10">
        <p className="text-xs font-semibold text-slate-300 mb-2">🏢 Used at companies like</p>
        <div className="flex flex-wrap gap-1">
          {["Instacart","DoorDash","Amazon Fresh","Walmart","Uber Eats"].map(c => (
            <span key={c} className="text-xs px-2 py-0.5 rounded-full bg-white/10 text-slate-300">{c}</span>
          ))}
        </div>
      </div>
    </div>
  );
}

// ─── App ───────────────────────────────────────────────────────────────────────

export default function App() {
  const [activeStage, setActiveStage] = useState(PIPELINE_STAGES[0].id);
  const [animatedStage, setAnimatedStage] = useState(null);

  const active = PIPELINE_STAGES.find(s => s.id === activeStage);

  const handleHighlight = (id) => {
    if (id) {
      setAnimatedStage(id);
      setActiveStage(id);
    } else {
      setAnimatedStage(null);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-indigo-50 font-sans">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-xl font-extrabold text-gray-900 flex items-center gap-2">
              🛒 Grocery Intelligence
              <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-indigo-100 text-indigo-700 ml-1">
                Interactive Explorer
              </span>
            </h1>
            <p className="text-xs text-gray-500 mt-0.5">
              Production-grade ML pipeline · Retrieval → Reranking → LLM · Interview portfolio project
            </p>
          </div>
          <div className="hidden sm:flex items-center gap-2">
            <span className="text-xs text-gray-400">Target roles</span>
            <div className="flex gap-1">
              {["Applied ML","AI Engineer","Senior MLE"].map(r => (
                <span key={r} className="text-xs px-2 py-0.5 rounded-full bg-green-100 text-green-700 font-medium">{r}</span>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 py-6">

        {/* Pipeline strip */}
        <div className="bg-white rounded-2xl shadow-md border border-gray-100 p-4 mb-6">
          <p className="text-xs font-semibold text-gray-400 uppercase mb-3">Full Pipeline — click any stage to explore</p>
          <div className="flex items-center flex-wrap gap-y-3 overflow-x-auto pb-1">
            {PIPELINE_STAGES.map((s, i) => (
              <PipelineNode
                key={s.id}
                stage={s}
                isActive={activeStage === s.id}
                isAnimated={animatedStage === s.id}
                onClick={() => setActiveStage(s.id)}
                isLast={i === PIPELINE_STAGES.length - 1}
              />
            ))}
          </div>
        </div>

        {/* Main grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">

          {/* Left column */}
          <div className="space-y-5">
            <FlowSimulator onHighlight={handleHighlight} />
            <ArchCard />

            {/* Stage nav list */}
            <div className="rounded-2xl border border-gray-200 bg-white shadow-md p-4">
              <h3 className="text-sm font-bold text-gray-700 mb-3">All Components</h3>
              <div className="space-y-1">
                {PIPELINE_STAGES.map(s => (
                  <button key={s.id} onClick={() => setActiveStage(s.id)}
                    className="w-full text-left flex items-center gap-2 px-3 py-2 rounded-xl text-xs transition-all"
                    style={activeStage===s.id
                      ? { background: s.bg, color: s.color, fontWeight: 700, borderLeft: `3px solid ${s.color}` }
                      : { color: "#64748b", background: "transparent" }
                    }>
                    <span className="text-base">{s.icon}</span>
                    <span>{s.label}</span>
                    <span className="ml-auto text-gray-300">{s.file.split("/").pop()}</span>
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Main detail panel */}
          <div className="lg:col-span-2">
            {active && <DetailPanel key={active.id} stage={active} />}
          </div>
        </div>

        {/* Bottom bar */}
        <div className="mt-6 rounded-2xl bg-gradient-to-r from-indigo-600 to-purple-600 text-white p-5 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 shadow-lg">
          <div>
            <p className="font-bold text-sm">🎯 Interview Demo Script (20 min)</p>
            <p className="text-indigo-200 text-xs mt-1">
              1. Show live search via Streamlit · 2. Walk the pipeline (Data → Embed → BM25 → RRF → Rerank → LLM) · 3. Demo substitute engine · 4. Quote metrics
            </p>
          </div>
          <div className="flex gap-2 shrink-0">
            <div className="text-center px-3 py-2 bg-white/10 rounded-xl">
              <p className="text-xl font-bold">$150K</p>
              <p className="text-xs text-indigo-200">floor</p>
            </div>
            <div className="text-center px-3 py-2 bg-white/20 rounded-xl">
              <p className="text-xl font-bold">$350K</p>
              <p className="text-xs text-indigo-200">ceiling</p>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}
