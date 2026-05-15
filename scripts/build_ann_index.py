"""Build FAISS ANN indices (FLAT + IVF) from precomputed product embeddings.

Reads:
    data/embeddings/product_embeddings.npy
    data/processed/product_catalog.parquet   (for product_id mapping)

Writes:
    data/embeddings/ann_flat/
    data/embeddings/ann_ivf/

Usage:
    python3 scripts/build_ann_index.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make ``src`` importable when running directly.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.search.ann_index import ANNIndex, benchmark  # noqa: E402


EMBEDDINGS_PATH = ROOT / "data" / "embeddings" / "product_embeddings.npy"
CATALOG_PATH = ROOT / "data" / "processed" / "product_catalog.parquet"
FLAT_DIR = ROOT / "data" / "embeddings" / "ann_flat"
IVF_DIR = ROOT / "data" / "embeddings" / "ann_ivf"

N_QUERIES = 100
TOP_K = 50
IVF_NLIST = 100


def _dir_size_mb(path: Path) -> float:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total / (1024 * 1024)


def _print_table(results: dict) -> None:
    print()
    print(f"{'method':<14} {'p50 (ms)':>10} {'p95 (ms)':>10} {'p99 (ms)':>10} {'recall@K':>10}")
    print("-" * 58)
    for name in ("numpy", "faiss_flat", "faiss_ivf"):
        r = results[name]
        print(
            f"{name:<14} "
            f"{r['p50_ms']:>10.3f} "
            f"{r['p95_ms']:>10.3f} "
            f"{r['p99_ms']:>10.3f} "
            f"{r['recall']:>10.4f}"
        )
    print()


def main() -> int:
    if not EMBEDDINGS_PATH.exists():
        print(f"ERROR: embeddings file not found: {EMBEDDINGS_PATH}")
        return 1
    if not CATALOG_PATH.exists():
        print(f"ERROR: product catalog not found: {CATALOG_PATH}")
        return 1

    print(f"Loading embeddings from {EMBEDDINGS_PATH}")
    embeddings = np.load(EMBEDDINGS_PATH)
    print(f"  shape={embeddings.shape}, dtype={embeddings.dtype}")

    print(f"Loading catalog from {CATALOG_PATH}")
    catalog = pd.read_parquet(CATALOG_PATH)
    product_ids = catalog["product_id"].astype(int).tolist()
    print(f"  {len(product_ids)} products in catalog")

    if len(product_ids) != embeddings.shape[0]:
        print(
            f"ERROR: catalog rows ({len(product_ids)}) != "
            f"embedding rows ({embeddings.shape[0]})"
        )
        return 1

    # ------------------------------------------------------------------
    # Build FLAT
    # ------------------------------------------------------------------
    print("\nBuilding FAISS FLAT (exact inner product)...")
    flat = ANNIndex.build(embeddings, product_ids, index_type="flat")
    flat.save(FLAT_DIR)
    print(f"  saved to {FLAT_DIR} ({_dir_size_mb(FLAT_DIR):.2f} MB, {len(flat)} vectors)")

    # ------------------------------------------------------------------
    # Build IVF
    # ------------------------------------------------------------------
    print(f"\nBuilding FAISS IVF (nlist={IVF_NLIST})...")
    ivf = ANNIndex.build(embeddings, product_ids, index_type="ivf", nlist=IVF_NLIST)
    ivf.save(IVF_DIR)
    print(f"  saved to {IVF_DIR} ({_dir_size_mb(IVF_DIR):.2f} MB, {len(ivf)} vectors)")

    # ------------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------------
    print(f"\nBenchmarking with {N_QUERIES} random queries, top_k={TOP_K}...")
    results = benchmark(embeddings, product_ids, n_queries=N_QUERIES, top_k=TOP_K)
    _print_table(results)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
