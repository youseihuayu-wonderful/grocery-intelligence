"""Search and recommendation quality metrics.

Metrics:
- Precision@K: fraction of top-K results that are relevant
- NDCG@K: normalized discounted cumulative gain (position-aware)
- MRR: mean reciprocal rank of first relevant result
- Recall@K: fraction of all relevant items found in top-K
- Latency: response time percentiles
"""

import time
from dataclasses import dataclass, field

import numpy as np
from loguru import logger


@dataclass
class SearchMetrics:
    """Container for search quality metrics."""

    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    ndcg_at_10: float = 0.0
    mrr: float = 0.0
    recall_at_20: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p99_ms: float = 0.0


def precision_at_k(
    retrieved_ids: list[str], relevant_ids: set[str], k: int
) -> float:
    """Compute Precision@K.

    What fraction of the top-K results are relevant?
    """
    if k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_count = sum(1 for rid in top_k if rid in relevant_ids)
    return relevant_count / k


def ndcg_at_k(
    retrieved_ids: list[str], relevant_ids: set[str], k: int
) -> float:
    """Compute NDCG@K (Normalized Discounted Cumulative Gain).

    Measures ranking quality — are the most relevant items at the top?
    """
    if not relevant_ids or k == 0:
        return 0.0

    # DCG: sum of relevance / log2(rank + 1)
    dcg = 0.0
    for i, rid in enumerate(retrieved_ids[:k]):
        if rid in relevant_ids:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because rank starts at 1

    # Ideal DCG: all relevant items at the top
    ideal_count = min(len(relevant_ids), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_count))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def mean_reciprocal_rank(
    retrieved_ids: list[str], relevant_ids: set[str]
) -> float:
    """Compute MRR (Mean Reciprocal Rank).

    1 / rank of the first relevant result.
    """
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(
    retrieved_ids: list[str], relevant_ids: set[str], k: int
) -> float:
    """Compute Recall@K.

    What fraction of all relevant items did we find in top-K?
    """
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    found = len(top_k & relevant_ids)
    return found / len(relevant_ids)


def evaluate_search_engine(
    search_fn,
    eval_queries: list[dict],
) -> SearchMetrics:
    """Run full evaluation on a search engine.

    Args:
        search_fn: Callable that takes a query string and returns list of product IDs
        eval_queries: List of dicts with 'query' and 'relevant_ids' keys

    Returns:
        SearchMetrics with averaged scores across all queries.
    """
    all_p5, all_p10, all_ndcg, all_mrr, all_recall = [], [], [], [], []
    latencies = []

    for item in eval_queries:
        query = item["query"]
        relevant = set(item["relevant_ids"])

        start = time.perf_counter()
        results = search_fn(query)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

        retrieved = [str(r.get("product_id", r)) for r in results]

        all_p5.append(precision_at_k(retrieved, relevant, 5))
        all_p10.append(precision_at_k(retrieved, relevant, 10))
        all_ndcg.append(ndcg_at_k(retrieved, relevant, 10))
        all_mrr.append(mean_reciprocal_rank(retrieved, relevant))
        all_recall.append(recall_at_k(retrieved, relevant, 20))

    metrics = SearchMetrics(
        precision_at_5=np.mean(all_p5),
        precision_at_10=np.mean(all_p10),
        ndcg_at_10=np.mean(all_ndcg),
        mrr=np.mean(all_mrr),
        recall_at_20=np.mean(all_recall),
        latency_p50_ms=np.percentile(latencies, 50),
        latency_p99_ms=np.percentile(latencies, 99),
    )

    logger.info(
        f"Evaluation complete on {len(eval_queries)} queries:\n"
        f"  Precision@5:  {metrics.precision_at_5:.3f}\n"
        f"  Precision@10: {metrics.precision_at_10:.3f}\n"
        f"  NDCG@10:      {metrics.ndcg_at_10:.3f}\n"
        f"  MRR:          {metrics.mrr:.3f}\n"
        f"  Recall@20:    {metrics.recall_at_20:.3f}\n"
        f"  Latency P50:  {metrics.latency_p50_ms:.1f}ms\n"
        f"  Latency P99:  {metrics.latency_p99_ms:.1f}ms"
    )

    return metrics
