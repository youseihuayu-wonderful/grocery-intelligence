"""Retrain the LTR model from the behavioral event log.

Loads:
  * ``data/processed/product_catalog.parquet`` -- product catalog
  * ``data/embeddings/product_embeddings.npy`` -- pre-computed embeddings
  * ``data/processed/behavior.db`` -- 1M+ purchase events

Builds a :class:`GrocerySearchEngine` against that data, then calls
:func:`retrain_ltr_from_behavior` with the production budget
(``max_users=500``, ``queries_per_user=3``, ``candidates_per_query=30``)
so the whole run finishes in well under 10 minutes.

The trained model is saved to ``models/ltr_behavioral.xgb`` and the
summary dict is printed.

macOS note
----------
sentence-transformers (via PyTorch) and XGBoost each bring their own
OpenMP runtime. When both are loaded in the same Python process on
macOS the second one to spin up a thread pool aborts the process
silently. Setting ``OMP_NUM_THREADS=1`` *before* any of these libraries
import their native extensions avoids the clash. We do that at the
very top of this script so users don't have to remember the magic
environment variables.
"""

from __future__ import annotations

import os

# Must be set *before* the first numpy/xgboost/torch import. See the
# module docstring for context.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Make ``src`` importable when running the script directly.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.embeddings import ProductEmbedder  # noqa: E402
from src.models.ltr_retrain import retrain_ltr_from_behavior  # noqa: E402
from src.recommend.behavior import BehaviorLogger  # noqa: E402
from src.search.engine import GrocerySearchEngine  # noqa: E402

CATALOG_PATH = ROOT / "data" / "processed" / "product_catalog.parquet"
EMBEDDINGS_PATH = ROOT / "data" / "embeddings" / "product_embeddings.npy"
BEHAVIOR_DB = ROOT / "data" / "processed" / "behavior.db"
OUTPUT_MODEL = ROOT / "models" / "ltr_behavioral.xgb"


def main() -> None:
    overall_t0 = time.time()
    print("=" * 70)
    print("LTR Retraining from Behavioral Events")
    print("=" * 70)

    # ---- Inputs --------------------------------------------------------
    for required in (CATALOG_PATH, EMBEDDINGS_PATH, BEHAVIOR_DB):
        if not required.exists():
            raise FileNotFoundError(required)

    print(f"\nLoading catalog from {CATALOG_PATH}")
    t0 = time.time()
    catalog = pd.read_parquet(CATALOG_PATH)
    print(f"  {len(catalog):,} products loaded in {time.time() - t0:.1f}s")

    print(f"\nLoading embeddings from {EMBEDDINGS_PATH}")
    t0 = time.time()
    embeddings = np.load(EMBEDDINGS_PATH)
    print(f"  shape={embeddings.shape} loaded in {time.time() - t0:.1f}s")

    print("\nInitializing ProductEmbedder (sentence-transformers)")
    t0 = time.time()
    embedder = ProductEmbedder()
    print(f"  ready in {time.time() - t0:.1f}s")

    print("\nBuilding GrocerySearchEngine")
    t0 = time.time()
    engine = GrocerySearchEngine(
        catalog=catalog, embeddings=embeddings, embedder=embedder
    )
    print(f"  ready in {time.time() - t0:.1f}s")

    print(f"\nOpening behavior log at {BEHAVIOR_DB}")
    with BehaviorLogger(BEHAVIOR_DB) as behavior:
        total = behavior.count_events()
        purchases = behavior.count_events("purchase")
        print(f"  {total:,} total events, {purchases:,} purchases")

        # ---- Retrain ----------------------------------------------------
        print("\nRetraining LTR model from behavioral signal...")
        print(
            "  budget: max_users=500, queries_per_user=3, "
            "candidates_per_query=30"
        )
        summary = retrain_ltr_from_behavior(
            behavior_logger=behavior,
            catalog=catalog,
            search_engine=engine,
            output_path=OUTPUT_MODEL,
            max_users=500,
            queries_per_user=3,
            candidates_per_query=30,
        )

    # ---- Report --------------------------------------------------------
    print()
    print("=" * 70)
    print("Retraining complete.")
    print("=" * 70)
    print(json.dumps(summary, indent=2, default=float))

    # Quick post-hoc sanity check: load the model and run a tiny predict.
    print("\nSanity check: loading saved model and predicting on 5 candidates")
    from src.models.ltr import LTRModel

    ltr = LTRModel()
    ltr.load(OUTPUT_MODEL)

    sample_query = "fresh produce"
    bm25_scores = engine.bm25.get_scores(sample_query.lower().split())
    sem_scores = np.dot(engine.embeddings, engine.embedder.embed_query(sample_query))
    sample_indices = list(range(5))
    scores = ltr.predict(
        query=sample_query,
        candidate_indices=sample_indices,
        bm25_scores=bm25_scores,
        semantic_scores=sem_scores,
        catalog=catalog,
    )
    print(f"  scores: {scores.tolist()}")

    print(f"\nTotal wall-clock: {time.time() - overall_t0:.1f}s")


if __name__ == "__main__":
    main()
