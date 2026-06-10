"""Cross-encoder reranker: int8 dynamic quantization — speed AND quality A/B.

The profiler showed the cross-encoder rerank is ~70% of search latency. Dynamic
int8 quantization (CPU, no calibration data) shrinks the Linear layers. This must
not wreck relevance, so we measure BOTH:
  - speed: median rerank latency, fp32 vs int8, on the real top-50 fused candidates
  - quality: top-10 overlap and score rank-correlation (Spearman) vs the fp32 ranking

No mock data: candidates come from the real BM25 + semantic + RRF pipeline over
the 49K-product catalog.

PLATFORM NOTE: PyTorch dynamic int8 quantization needs a CPU qengine and does not
run on Apple Silicon MPS (the quantize op is unimplemented for MPS, and this
machine's CPU inference path for the model is unreliable). This script is meant to
be run on a Linux/x86 (fbgemm) or CUDA host — e.g. Kaggle — where fp32-CPU vs
int8-CPU is a clean comparison. Kept here as the ready-to-run A/B harness.

Usage: ./venv/bin/python benchmark/quantize_reranker.py    # on Linux/Kaggle
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA_DIR = ROOT / "data"

from sentence_transformers import CrossEncoder  # noqa: E402
from src.search.engine import GrocerySearchEngine  # noqa: E402

QUERIES = [
    "low sugar yogurt", "organic whole milk", "gluten free bread",
    "high protein breakfast", "cheap pasta sauce", "almond butter",
    "sparkling water", "dark chocolate", "baby spinach", "ground coffee",
    "cheddar cheese", "frozen pizza", "olive oil", "greek yogurt",
    "chicken breast", "brown rice", "oat milk", "peanut free snacks",
    "low carb tortilla", "kombucha",
]
RERANK_K = 50  # matches engine.search's rerank_count cap


def fused_candidate_texts(engine: GrocerySearchEngine, query: str) -> list[str]:
    """Reproduce the engine's BM25 + semantic + RRF candidate selection."""
    bm25_scores = engine.bm25.get_scores(query.lower().split())
    bm25_top = np.argsort(bm25_scores)[-100:][::-1]
    q_emb = engine.embedder.embed_query(query)
    sims = np.dot(engine.embeddings, q_emb)
    sem_top = np.argsort(sims)[-100:][::-1]
    scores: dict[int, float] = {}
    k = 60
    for rank, idx in enumerate(bm25_top):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)
    for rank, idx in enumerate(sem_top):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)
    fused = sorted(scores, key=lambda x: scores[x], reverse=True)[:RERANK_K]
    return [engine.product_texts[i] for i in fused]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation (Pearson on ranks)."""
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / denom) if denom else 1.0


def bench_model(model: CrossEncoder, pairs_per_query: list[list], repeats: int = 3):
    """Return (median_ms_per_query, list_of_score_arrays)."""
    times: list[float] = []
    scores_out: list[np.ndarray] = []
    for pairs in pairs_per_query:
        sc = None
        for _ in range(repeats):
            t0 = time.perf_counter()
            sc = model.predict(pairs)
            times.append((time.perf_counter() - t0) * 1000)
        scores_out.append(np.asarray(sc, dtype=float))
    return statistics.median(times), scores_out


def main() -> None:
    print("Loading catalog + engine (for real candidates)...", file=sys.stderr)
    catalog = pd.read_parquet(DATA_DIR / "processed" / "product_catalog.parquet")
    embeddings = np.load(DATA_DIR / "embeddings" / "product_embeddings.npy")
    engine = GrocerySearchEngine(catalog=catalog, embeddings=embeddings)

    # Build the real rerank workload: [query, candidate_text] pairs per query.
    pairs_per_query = []
    for q in QUERIES:
        texts = fused_candidate_texts(engine, q)
        pairs_per_query.append([[q, t] for t in texts])

    # Dynamic int8 quantization is a CPU technique (MPS lacks the op), so compare
    # fp32-CPU vs int8-CPU — the realistic "quantize for CPU serving" claim.
    print("Loading fp32 + int8 cross-encoders (CPU)...", file=sys.stderr)
    fp32 = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    int8 = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    torch.backends.quantized.engine = "qnnpack"  # ARM (Apple Silicon) qengine
    int8.model = torch.quantization.quantize_dynamic(
        int8.model, {torch.nn.Linear}, dtype=torch.qint8
    )

    # Warm up both (not timed).
    for m in (fp32, int8):
        for _ in range(2):
            m.predict(pairs_per_query[0])

    print("Timing fp32...", file=sys.stderr)
    fp32_ms, fp32_scores = bench_model(fp32, pairs_per_query)
    print("Timing int8...", file=sys.stderr)
    int8_ms, int8_scores = bench_model(int8, pairs_per_query)

    # Quality: top-10 overlap and Spearman of the 50 scores, per query.
    overlaps, spearmans = [], []
    for s32, s8 in zip(fp32_scores, int8_scores):
        top32 = set(np.argsort(s32)[-10:])
        top8 = set(np.argsort(s8)[-10:])
        overlaps.append(len(top32 & top8) / 10.0)
        spearmans.append(spearman(s32, s8))

    speedup = fp32_ms / int8_ms if int8_ms else float("nan")
    print("\n" + "=" * 56)
    print(f"Cross-encoder rerank: fp32 vs int8  (50 real candidates x {len(QUERIES)} queries)")
    print("=" * 56)
    print(f"  fp32 median : {fp32_ms:7.2f} ms / query")
    print(f"  int8 median : {int8_ms:7.2f} ms / query")
    print(f"  speedup     : {speedup:7.2f}x")
    print("\nQuality vs fp32 ranking (higher = more faithful):")
    print(f"  top-10 overlap : {statistics.mean(overlaps):.3f}  (min {min(overlaps):.2f})")
    print(f"  Spearman corr  : {statistics.mean(spearmans):.4f}  (min {min(spearmans):.3f})")

    outdir = ROOT / "benchmark" / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "quantize_reranker.csv", "w") as f:
        f.write("metric,value\n")
        f.write(f"fp32_ms_per_query,{fp32_ms:.3f}\n")
        f.write(f"int8_ms_per_query,{int8_ms:.3f}\n")
        f.write(f"speedup,{speedup:.3f}\n")
        f.write(f"top10_overlap_mean,{statistics.mean(overlaps):.4f}\n")
        f.write(f"spearman_mean,{statistics.mean(spearmans):.4f}\n")
    print(f"\nWrote {outdir / 'quantize_reranker.csv'}", file=sys.stderr)


if __name__ == "__main__":
    main()
