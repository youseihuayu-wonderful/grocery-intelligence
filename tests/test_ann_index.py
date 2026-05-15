"""Tests for the FAISS-backed ANN index module."""

from __future__ import annotations

import numpy as np
import pytest

from src.search.ann_index import ANNIndex, benchmark


DIM = 32
N = 100
SEED = 7


@pytest.fixture(scope="module")
def synthetic_data() -> tuple[np.ndarray, list[int]]:
    """100 random L2-normalized vectors with sparse integer product_ids."""
    rng = np.random.default_rng(SEED)
    embeddings = rng.standard_normal((N, DIM)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Use non-contiguous product_ids so we know we're remapping, not just
    # returning FAISS's row indices.
    product_ids = [1000 + 7 * i for i in range(N)]
    return embeddings, product_ids


def _numpy_ground_truth(
    embeddings: np.ndarray,
    query: np.ndarray,
    product_ids: list[int],
    top_k: int,
) -> list[tuple[int, float]]:
    sims = embeddings @ query
    order = np.argsort(-sims)[:top_k]
    return [(product_ids[i], float(sims[i])) for i in order]


# ---------------------------------------------------------------------------
# Basic API
# ---------------------------------------------------------------------------


def test_build_flat_returns_correct_length(synthetic_data):
    embeddings, product_ids = synthetic_data
    idx = ANNIndex.build(embeddings, product_ids, index_type="flat")
    assert len(idx) == N
    assert idx.dim == DIM
    assert idx.index_type == "flat"


def test_search_returns_top_k_product_ids_sorted_desc(synthetic_data):
    embeddings, product_ids = synthetic_data
    idx = ANNIndex.build(embeddings, product_ids, index_type="flat")
    query = embeddings[42]

    results = idx.search(query, top_k=10)

    assert len(results) == 10
    # All returned product_ids must come from the catalog.
    for pid, _ in results:
        assert pid in product_ids
    # Scores must be non-ascending.
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)
    # The query is in the catalog so its own product_id should be the top hit.
    assert results[0][0] == product_ids[42]
    assert results[0][1] == pytest.approx(1.0, abs=1e-4)


def test_search_returns_fewer_results_when_top_k_exceeds_index(synthetic_data):
    embeddings, product_ids = synthetic_data
    idx = ANNIndex.build(embeddings, product_ids, index_type="flat")
    query = embeddings[0]

    results = idx.search(query, top_k=N + 50)
    assert len(results) == N


# ---------------------------------------------------------------------------
# Correctness vs numpy baseline
# ---------------------------------------------------------------------------


def test_flat_recall_matches_numpy_exactly(synthetic_data):
    embeddings, product_ids = synthetic_data
    idx = ANNIndex.build(embeddings, product_ids, index_type="flat")

    rng = np.random.default_rng(SEED + 1)
    query_rows = rng.choice(N, size=10, replace=False)

    for row in query_rows:
        q = embeddings[row]
        truth = _numpy_ground_truth(embeddings, q, product_ids, top_k=20)
        got = idx.search(q, top_k=20)

        # FLAT must match numpy exactly (same product_ids, same order).
        assert [pid for pid, _ in got] == [pid for pid, _ in truth]
        for (g_pid, g_score), (t_pid, t_score) in zip(got, truth):
            assert g_pid == t_pid
            assert g_score == pytest.approx(t_score, abs=1e-5)


def test_ivf_recall_at_default_nprobe_is_high(synthetic_data):
    embeddings, product_ids = synthetic_data
    idx = ANNIndex.build(embeddings, product_ids, index_type="ivf", nlist=8)

    rng = np.random.default_rng(SEED + 2)
    query_rows = rng.choice(N, size=20, replace=False)

    top_k = 10
    recalls = []
    for row in query_rows:
        q = embeddings[row]
        truth_ids = {
            pid
            for pid, _ in _numpy_ground_truth(embeddings, q, product_ids, top_k)
        }
        got_ids = {pid for pid, _ in idx.search(q, top_k=top_k, nprobe=4)}
        recalls.append(len(got_ids & truth_ids) / len(truth_ids))

    mean_recall = float(np.mean(recalls))
    assert mean_recall >= 0.85, f"IVF recall too low: {mean_recall:.3f}"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_save_and_load_roundtrip_flat(tmp_path, synthetic_data):
    embeddings, product_ids = synthetic_data
    idx = ANNIndex.build(embeddings, product_ids, index_type="flat")
    save_dir = tmp_path / "flat"
    idx.save(save_dir)

    loaded = ANNIndex.load(save_dir)
    assert len(loaded) == len(idx)
    assert loaded.dim == idx.dim
    assert loaded.product_ids == idx.product_ids
    assert loaded.index_type == "flat"

    query = embeddings[5]
    before = idx.search(query, top_k=10)
    after = loaded.search(query, top_k=10)

    assert [pid for pid, _ in before] == [pid for pid, _ in after]
    for (b_pid, b_score), (a_pid, a_score) in zip(before, after):
        assert b_pid == a_pid
        assert b_score == pytest.approx(a_score, abs=1e-6)


def test_save_and_load_roundtrip_ivf(tmp_path, synthetic_data):
    embeddings, product_ids = synthetic_data
    idx = ANNIndex.build(embeddings, product_ids, index_type="ivf", nlist=8)
    save_dir = tmp_path / "ivf"
    idx.save(save_dir)

    loaded = ANNIndex.load(save_dir)
    assert loaded.index_type == "ivf"
    assert loaded.nlist == idx.nlist

    query = embeddings[11]
    before = idx.search(query, top_k=10, nprobe=4)
    after = loaded.search(query, top_k=10, nprobe=4)
    assert [pid for pid, _ in before] == [pid for pid, _ in after]


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def test_benchmark_returns_expected_structure(synthetic_data):
    embeddings, product_ids = synthetic_data
    results = benchmark(embeddings, product_ids, n_queries=10, top_k=10)

    assert set(results.keys()) == {"numpy", "faiss_flat", "faiss_ivf"}
    for method, metrics in results.items():
        assert {"p50_ms", "p95_ms", "p99_ms", "recall"} <= set(metrics.keys())
        assert metrics["p50_ms"] >= 0
        assert metrics["p95_ms"] >= metrics["p50_ms"] - 1e-6
        assert metrics["p99_ms"] >= metrics["p95_ms"] - 1e-6
        assert 0.0 <= metrics["recall"] <= 1.0

    # numpy is the ground truth, by construction.
    assert results["numpy"]["recall"] == pytest.approx(1.0)
    # FAISS flat must match numpy exactly on the same data.
    assert results["faiss_flat"]["recall"] == pytest.approx(1.0)
