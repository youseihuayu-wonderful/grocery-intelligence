"""FAISS approximate-nearest-neighbor index for product embeddings.

Wraps a FAISS index so the rest of the codebase doesn't depend on faiss directly.
Supports both exact (IndexFlatIP) and approximate (IndexIVFFlat) search.

Embeddings are assumed L2-normalized so inner product == cosine similarity.

Quick example:
    idx = ANNIndex.build(embeddings, product_ids, index_type="ivf", nlist=100)
    results = idx.search(query_vec, top_k=50, nprobe=10)
    # results: [(product_id, similarity_score), ...]
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np


# ---------------------------------------------------------------------------
# ANNIndex
# ---------------------------------------------------------------------------


class ANNIndex:
    """FAISS-backed approximate nearest neighbor index for product embeddings."""

    INDEX_FILENAME = "index.faiss"
    META_FILENAME = "metadata.json"

    def __init__(self, dim: int = 384):
        self.dim = int(dim)
        self.index: faiss.Index | None = None
        self.product_ids: list[int] = []
        self.index_type: str = "flat"
        self.nlist: int = 0

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        embeddings: np.ndarray,
        product_ids: list[int] | Iterable[int],
        index_type: str = "flat",
        nlist: int = 100,
    ) -> "ANNIndex":
        """Build a FAISS index from precomputed L2-normalized embeddings."""
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError(
                f"embeddings must be 2-D, got shape {embeddings.shape}"
            )

        product_ids = list(product_ids)
        if len(product_ids) != embeddings.shape[0]:
            raise ValueError(
                f"product_ids length ({len(product_ids)}) != "
                f"embeddings rows ({embeddings.shape[0]})"
            )

        n, dim = embeddings.shape
        index_type = index_type.lower()

        if index_type == "flat":
            index = faiss.IndexFlatIP(dim)
            index.add(embeddings)
            effective_nlist = 0

        elif index_type == "ivf":
            # FAISS requires at least nlist * a few training points. Cap nlist
            # so small synthetic datasets (tests) still work.
            effective_nlist = max(1, min(int(nlist), max(1, n // 8)))
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(
                quantizer, dim, effective_nlist, faiss.METRIC_INNER_PRODUCT
            )
            index.train(embeddings)
            index.add(embeddings)
            # Default nprobe — caller can override at search() time.
            index.nprobe = min(10, effective_nlist)

        else:
            raise ValueError(
                f"Unknown index_type {index_type!r}; expected 'flat' or 'ivf'"
            )

        obj = cls(dim=dim)
        obj.index = index
        obj.product_ids = [int(pid) for pid in product_ids]
        obj.index_type = index_type
        obj.nlist = effective_nlist
        return obj

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 50,
        nprobe: int = 10,
    ) -> list[tuple[int, float]]:
        """Return up to top_k (product_id, similarity) pairs, sorted desc."""
        if self.index is None:
            raise RuntimeError("Index is empty; build or load first.")

        query = np.ascontiguousarray(query_embedding, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if query.shape[1] != self.dim:
            raise ValueError(
                f"query dim {query.shape[1]} != index dim {self.dim}"
            )

        if self.index_type == "ivf":
            # IVF: configure how many Voronoi cells to scan.
            self.index.nprobe = max(1, min(int(nprobe), self.nlist or int(nprobe)))

        k = min(int(top_k), len(self.product_ids))
        if k <= 0:
            return []

        scores, idxs = self.index.search(query, k)
        scores = scores[0]
        idxs = idxs[0]

        results: list[tuple[int, float]] = []
        for score, faiss_row in zip(scores, idxs):
            if faiss_row < 0:
                # FAISS returns -1 for empty slots (can happen with IVF when
                # too few candidates exist in the probed cells).
                continue
            results.append((self.product_ids[int(faiss_row)], float(score)))
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialize the FAISS index + product_id mapping to a directory."""
        if self.index is None:
            raise RuntimeError("Nothing to save — build or load an index first.")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path / self.INDEX_FILENAME))

        metadata = {
            "dim": self.dim,
            "index_type": self.index_type,
            "nlist": self.nlist,
            "product_ids": self.product_ids,
        }
        with open(path / self.META_FILENAME, "w") as f:
            json.dump(metadata, f)

    @classmethod
    def load(cls, path: str | Path) -> "ANNIndex":
        """Load a previously saved index."""
        path = Path(path)
        index_path = path / cls.INDEX_FILENAME
        meta_path = path / cls.META_FILENAME

        if not index_path.exists():
            raise FileNotFoundError(f"Missing FAISS index file: {index_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing metadata file: {meta_path}")

        with open(meta_path) as f:
            metadata = json.load(f)

        obj = cls(dim=int(metadata["dim"]))
        obj.index = faiss.read_index(str(index_path))
        obj.product_ids = [int(pid) for pid in metadata["product_ids"]]
        obj.index_type = str(metadata["index_type"])
        obj.nlist = int(metadata.get("nlist", 0))
        return obj

    # ------------------------------------------------------------------
    # Dunders
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        if self.index is None:
            return 0
        return int(self.index.ntotal)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def _percentile_ms(times_s: list[float], pct: float) -> float:
    """Percentile of latency samples (seconds), returned in milliseconds."""
    if not times_s:
        return 0.0
    return float(np.percentile(times_s, pct) * 1000.0)


def _numpy_topk(
    embeddings: np.ndarray, query: np.ndarray, top_k: int
) -> list[int]:
    """Exact top-k row indices by inner product (cosine for normalized vecs)."""
    sims = embeddings @ query
    if top_k >= sims.shape[0]:
        idx = np.argsort(-sims)
    else:
        # argpartition + sort within the top-k slice — same recipe as engine.py.
        idx = np.argpartition(-sims, top_k - 1)[:top_k]
        idx = idx[np.argsort(-sims[idx])]
    return idx.tolist()


def _recall_at_k(predicted: list[int], ground_truth: list[int]) -> float:
    if not ground_truth:
        return 1.0
    overlap = len(set(predicted) & set(ground_truth))
    return overlap / len(ground_truth)


def benchmark(
    embeddings: np.ndarray,
    product_ids: list[int],
    n_queries: int = 100,
    top_k: int = 50,
) -> dict:
    """Benchmark numpy linear scan vs FAISS flat vs FAISS IVF.

    Samples ``n_queries`` queries from the embedding matrix itself (so the
    correct neighbors are guaranteed to be in the catalog) and measures top-k
    retrieval latency for each method. Recall is computed against the numpy
    ground truth.
    """
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    n_total = embeddings.shape[0]
    n_queries = min(int(n_queries), n_total)

    rng = np.random.default_rng(seed=42)
    query_rows = rng.choice(n_total, size=n_queries, replace=False)
    queries = embeddings[query_rows]

    # Build both FAISS indices once.
    flat = ANNIndex.build(embeddings, product_ids, index_type="flat")
    ivf = ANNIndex.build(embeddings, product_ids, index_type="ivf", nlist=100)

    # Per-query latency samples.
    numpy_times: list[float] = []
    flat_times: list[float] = []
    ivf_times: list[float] = []

    # IDs returned per query, used for recall comparisons.
    numpy_ids: list[list[int]] = []
    flat_ids: list[list[int]] = []
    ivf_ids: list[list[int]] = []

    for q in queries:
        # numpy baseline
        t0 = time.perf_counter()
        idxs = _numpy_topk(embeddings, q, top_k)
        numpy_times.append(time.perf_counter() - t0)
        numpy_ids.append([int(product_ids[i]) for i in idxs])

        # FAISS flat
        t0 = time.perf_counter()
        flat_res = flat.search(q, top_k=top_k)
        flat_times.append(time.perf_counter() - t0)
        flat_ids.append([pid for pid, _ in flat_res])

        # FAISS IVF
        t0 = time.perf_counter()
        ivf_res = ivf.search(q, top_k=top_k, nprobe=10)
        ivf_times.append(time.perf_counter() - t0)
        ivf_ids.append([pid for pid, _ in ivf_res])

    recall_flat = float(
        np.mean([_recall_at_k(p, gt) for p, gt in zip(flat_ids, numpy_ids)])
    )
    recall_ivf = float(
        np.mean([_recall_at_k(p, gt) for p, gt in zip(ivf_ids, numpy_ids)])
    )

    return {
        "numpy": {
            "p50_ms": _percentile_ms(numpy_times, 50),
            "p95_ms": _percentile_ms(numpy_times, 95),
            "p99_ms": _percentile_ms(numpy_times, 99),
            "recall": 1.0,
        },
        "faiss_flat": {
            "p50_ms": _percentile_ms(flat_times, 50),
            "p95_ms": _percentile_ms(flat_times, 95),
            "p99_ms": _percentile_ms(flat_times, 99),
            "recall": recall_flat,
        },
        "faiss_ivf": {
            "p50_ms": _percentile_ms(ivf_times, 50),
            "p95_ms": _percentile_ms(ivf_times, 95),
            "p99_ms": _percentile_ms(ivf_times, 99),
            "recall": recall_ivf,
        },
    }
