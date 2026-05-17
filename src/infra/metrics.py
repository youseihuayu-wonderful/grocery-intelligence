"""Prometheus metrics for FastAPI.

Use the `prometheus-client` library (add to requirements.txt).
Exposes:
    - api_requests_total{endpoint, method, status}     Counter
    - api_request_seconds_bucket{endpoint, method}      Histogram (response time)
    - api_search_results_total{endpoint}                Counter (total products returned)
    - cache_hits_total                                  Counter
    - cache_misses_total                                Counter
    - active_users{period}                              Gauge (1h / 24h / 7d)

A FastAPI middleware to record request count/latency.
A `/metrics` endpoint that returns the Prometheus text format.
"""

from __future__ import annotations

import time

from fastapi import FastAPI, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)


REQUEST_COUNTER = Counter(
    "api_requests_total",
    "API requests",
    ["endpoint", "method", "status"],
)
REQUEST_LATENCY = Histogram(
    "api_request_seconds",
    "API request latency",
    ["endpoint", "method"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
SEARCH_RESULTS = Counter(
    "api_search_results_total",
    "Total products returned across search responses",
    ["endpoint"],
)
CACHE_HITS = Counter("cache_hits_total", "Cache hits")
CACHE_MISSES = Counter("cache_misses_total", "Cache misses")
ACTIVE_USERS = Gauge("active_users", "Active users", ["period"])


def install_metrics(app: FastAPI) -> None:
    """Mount /metrics endpoint and a request-tracking middleware on the FastAPI app."""

    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        endpoint = request.url.path
        # Avoid /metrics recording itself.
        if endpoint != "/metrics":
            method = request.method
            status = str(response.status_code)
            REQUEST_COUNTER.labels(
                endpoint=endpoint, method=method, status=status
            ).inc()
            REQUEST_LATENCY.labels(endpoint=endpoint, method=method).observe(elapsed)
        return response

    @app.get("/metrics")
    async def metrics_endpoint():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
