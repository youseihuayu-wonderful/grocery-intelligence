"""Serving latency + throughput benchmark for the hybrid search pipeline.

Measures end-to-end latency (p50/p95/p99), throughput (QPS), and a per-stage
breakdown (BM25 / semantic / fusion / rerank / enrich) over a fixed set of real
queries against the real 49K-product catalog. No mock data.

Methodology:
  - warm up the engine (model load, JIT, OS cache) before timing
  - >= N timed samples per query, median + tail percentiles reported
  - per-stage timings read from GrocerySearchEngine.last_timings
  - single process (this machine has no GPU); numbers are CPU wall-clock

Usage:
  ./venv/bin/python benchmark/bench_search.py                 # default
  ./venv/bin/python benchmark/bench_search.py --no-rerank     # skip cross-encoder
  ./venv/bin/python benchmark/bench_search.py --repeats 5
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA_DIR = ROOT / "data"

from src.search.engine import GrocerySearchEngine  # noqa: E402

# Representative real grocery queries (head + torso + tail intents).
QUERIES = [
    "low sugar yogurt", "organic whole milk", "gluten free bread",
    "high protein breakfast", "cheap pasta sauce", "almond butter",
    "sparkling water", "dark chocolate", "baby spinach", "ground coffee",
    "cheddar cheese", "frozen pizza", "olive oil", "greek yogurt",
    "chicken breast", "brown rice", "oat milk", "peanut free snacks",
    "low carb tortilla", "kombucha",
]


def pctl(xs: list[float], p: float) -> float:
    """Linear-interpolated percentile of xs (p in [0,100])."""
    if not xs:
        return float("nan")
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    rank = (p / 100) * (len(s) - 1)
    lo = int(rank)
    frac = rank - lo
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * frac


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5, help="timed samples per query")
    ap.add_argument("--warmup", type=int, default=2, help="warmup passes per query")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--no-rerank", action="store_true")
    args = ap.parse_args()
    use_reranker = not args.no_rerank

    print("Loading real catalog + embeddings...", file=sys.stderr)
    catalog = pd.read_parquet(DATA_DIR / "processed" / "product_catalog.parquet")
    embeddings = np.load(DATA_DIR / "embeddings" / "product_embeddings.npy")
    engine = GrocerySearchEngine(catalog=catalog, embeddings=embeddings)
    print(f"  {len(catalog)} products, embeddings {embeddings.shape}", file=sys.stderr)

    # Warm up (model load, lazy reranker, OS cache) — not timed.
    for q in QUERIES[:5]:
        for _ in range(args.warmup):
            engine.search(q, top_k=args.top_k, use_reranker=use_reranker)

    e2e: list[float] = []
    stage_acc: dict[str, list[float]] = {}
    print(f"\nTiming {len(QUERIES)} queries x {args.repeats} samples "
          f"(rerank={'on' if use_reranker else 'off'})...", file=sys.stderr)
    for q in QUERIES:
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            engine.search(q, top_k=args.top_k, use_reranker=use_reranker)
            e2e.append((time.perf_counter() - t0) * 1000)
            for stage, ms in engine.last_timings.items():
                stage_acc.setdefault(stage, []).append(ms)

    n = len(e2e)
    p50, p95, p99 = pctl(e2e, 50), pctl(e2e, 95), pctl(e2e, 99)
    mean = statistics.mean(e2e)
    qps = 1000.0 / mean  # single-thread sequential throughput

    print("\n" + "=" * 56)
    print(f"End-to-end latency over {n} requests  (rerank={'on' if use_reranker else 'off'})")
    print("=" * 56)
    print(f"  p50   : {p50:8.1f} ms")
    print(f"  p95   : {p95:8.1f} ms")
    print(f"  p99   : {p99:8.1f} ms")
    print(f"  mean  : {mean:8.1f} ms")
    print(f"  min   : {min(e2e):8.1f} ms")
    print(f"  max   : {max(e2e):8.1f} ms")
    print(f"  QPS   : {qps:8.2f}  (single-thread sequential)")

    print("\nPer-stage median latency (share of mean e2e):")
    order = ["bm25", "semantic", "fusion", "rerank", "enrich"]
    for stage in order:
        xs = stage_acc.get(stage)
        if not xs:
            continue
        med = statistics.median(xs)
        share = 100 * statistics.mean(xs) / mean
        print(f"  {stage:9s}: {med:8.2f} ms   ({share:5.1f}%)")

    outdir = ROOT / "benchmark" / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    tag = "rerank" if use_reranker else "norerank"
    csv = outdir / f"search_latency_{tag}.csv"
    with open(csv, "w") as f:
        f.write("metric,value_ms\n")
        for name, val in [("p50", p50), ("p95", p95), ("p99", p99),
                          ("mean", mean), ("min", min(e2e)), ("max", max(e2e))]:
            f.write(f"{name},{val:.3f}\n")
        for stage in order:
            xs = stage_acc.get(stage)
            if xs:
                f.write(f"stage_{stage}_median,{statistics.median(xs):.3f}\n")
    print(f"\nWrote {csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
