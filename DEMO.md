# Grocery Intelligence — Demo Walkthrough

A scripted, end-to-end demo of the system: **search → nutrition filtering →
substitutes → personalized recommendations → basket → checkout → model metrics**.
Every query below is real and the expected results are what the system actually
returns on the bundled Instacart + Open Food Facts data.

---

## 0. Start the app

```bash
cd grocery-intelligence
source venv/bin/activate

# Terminal 1 — API (first start loads models; wait for "API ready")
uvicorn src.api.main:app --port 8000

# Terminal 2 — UI
streamlit run frontend/app.py        # opens http://localhost:8000 -> :8501
```

> The two-tower `/recommend` endpoint needs the trained model
> (`models/two_tower.pt` + `item_vectors.npy`). If it's absent, recommendations
> gracefully fall back to popularity. Generate it with
> `python scripts/train_two_tower.py --epochs 20` (GPU recommended), then
> `python scripts/evaluate_models.py` to refresh the Model Performance tab.

---

## 1. Personalized recommendations (two-tower)  — `🏠 Home`

1. In **"Sign in as a demo user"**, pick a user with many orders (e.g. the
   first option — a heavy buyer).
2. The **🎯 Recommended for You** section shows two-tower candidates with a
   relevance score per item.
3. Toggle **"Show only new products"** → switches from the reorder task
   (repurchases allowed) to novel-item discovery.

**Talking point:** the user vector is pooled *online* from the user's purchase
history and queried against precomputed item vectors with FAISS — so it
generalizes to unseen users and to ad-hoc baskets, with a popularity fallback
for cold-start.

API equivalent:

```bash
# Personalized (reorder task)
curl -s "http://localhost:8000/recommend?user_id=201268&top_k=5" | jq '.source, .products[].product_name'

# Novel-item discovery
curl -s "http://localhost:8000/recommend?user_id=201268&top_k=5&exclude_purchased=true" | jq '.products[].product_name'

# Cold-start user -> honest popularity fallback ("source":"popularity")
curl -s "http://localhost:8000/recommend?user_id=999999999&top_k=3" | jq '.source'
```

---

## 2. Honest health / nutrition search  — `🔍 Smart Search`

Search: **`high protein low sugar breakfast`**

Expected:
- A green banner: *"Auto-detected health filters: 💪 High Protein 🍯 Low Sugar"*
- Every result genuinely has **protein ≥ 15g and sugar ≤ 5g** (verified Open
  Food Facts data) — e.g. *Brownie Crunch High Protein Bar* (33g protein, 0g
  sugar), *Breakfast Burrito* (16g / 1g).
- A caveat noting nutrition data covers ~15% of the catalog.

**Talking point:** natural-language nutrition intent is parsed into attribute
filters, and search **filter-then-ranks the whole qualifying subset** by
semantic relevance. Products lacking verified nutrition data can never
masquerade as qualifying — no false health claims.

Try also: `low sugar yogurt`, `vegan snacks`, `organic whole wheat bread`.

API equivalent:

```bash
curl -s -X POST http://localhost:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"high protein low sugar breakfast","top_k":6}' \
  | jq '.applied_filters, [.results[] | {name:.product_name, protein:.protein_100g, sugar:.sugar_100g}]'
```

---

## 3. Substitute finder  — `🔄 Substitute Finder`

Pick an out-of-stock product and view substitutes. Each comes with
human-readable reasons (*"Same category"*, *"Lower sugar"*, *"$X cheaper"*).

```bash
# e.g. substitutes for product 69
curl -s -X POST http://localhost:8000/substitute \
  -H 'Content-Type: application/json' \
  -d '{"product_id":69,"top_k":5}' \
  | jq '[.substitutes[] | {name:.product_name, reasons:.substitution_reasons}]'
```

---

## 4. Basket recommendations  — `🤖 Shopping Assistant` / Cart

Add a few items (e.g. chicken breast, salad, yogurt) and view
"frequently bought together" / complementary suggestions.

```bash
curl -s "http://localhost:8000/related/13176?top_k=5" | jq '[.related[].product_name]'
```

---

## 5. Cart & checkout  — `🛒 Cart`

Add items, apply promotions, and run the demo checkout.

**Honesty note (call it out):** prices are clearly labelled **"est."** — the
Instacart catalog ships no price data, so prices are deterministic *synthetic
estimates* (department + nutrition + popularity). Everything else is real data.

---

## 6. Model performance  — `🧪 Model Performance`

Shows **measured, reproducible** offline metrics (no hard-coded numbers):

| Two-tower vs popularity (grocery reorder task) | Lift |
|---|---|
| Recall@10 | **+56%** |
| NDCG@10 | **+62%** |

Plus per-stage search latency (cross-encoder rerank dominates ≈ 90 ms of the
~122 ms p50), data scale (49,688 products · 1M purchases · 5K users), and the
honest ~15% nutrition-coverage disclosure.

Regenerate anytime:

```bash
python scripts/evaluate_models.py   # writes data/eval/model_eval.json
```

---

## Screenshots / recording

Screenshots are best captured live in the browser while running the demo above
(macOS: `Cmd-Shift-4`; full recording: QuickTime screen recording). Suggested
shots, one per section: **Home/recommendations**, **health-filtered search**,
**substitutes**, **cart with `est.` prices**, **Model Performance**.

To automate capture, install Playwright and drive the running UI:

```bash
pip install playwright && playwright install chromium
# then a short script can navigate http://localhost:8501 and screenshot each tab
```

> Automated capture isn't bundled because it pulls in a headless-browser
> dependency; the live-capture flow above is the recommended path for a demo
> reel.
