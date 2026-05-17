"""Tests for src.infra.metrics (Prometheus instrumentation).

These tests are deliberately isolated from the production FastAPI app —
they spin up a small dedicated app via TestClient so we don't accidentally
import the full main.py with all its model dependencies.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.infra.metrics import (
    REQUEST_COUNTER,
    REQUEST_LATENCY,
    install_metrics,
)


def _build_app() -> FastAPI:
    """Fresh FastAPI app with metrics installed and a couple of toy routes."""
    app = FastAPI()

    @app.get("/ping")
    def ping():
        return {"ok": True}

    @app.get("/items/{item_id}")
    def items(item_id: int):
        return {"item_id": item_id}

    install_metrics(app)
    return app


# ---------------------------------------------------------- generate_latest


def test_generate_latest_returns_valid_text_format():
    output = generate_latest()
    assert isinstance(output, (bytes, bytearray))
    text = output.decode("utf-8")
    # Standard prometheus text-format markers
    assert "# HELP" in text
    assert "# TYPE" in text
    # Our own metrics should appear at least once (declared at module import)
    assert "api_requests_total" in text or "api_request_seconds" in text or "cache_hits_total" in text


# ----------------------------------------------------------- /metrics endpoint


def test_metrics_endpoint_returns_200_and_text_plain():
    app = _build_app()
    with TestClient(app) as client:
        resp = client.get("/metrics")
    assert resp.status_code == 200
    # CONTENT_TYPE_LATEST is "text/plain; version=0.0.4; charset=utf-8" in prom-client
    assert "text/plain" in resp.headers["content-type"]
    # Should look like prom text format
    assert "# HELP" in resp.text or "# TYPE" in resp.text


def test_metrics_content_type_matches_prom_client():
    app = _build_app()
    with TestClient(app) as client:
        resp = client.get("/metrics")
    assert CONTENT_TYPE_LATEST.split(";")[0] in resp.headers["content-type"]


# ---------------------------------------------------------- middleware behaviour


def test_middleware_records_request_count_and_latency():
    app = _build_app()
    with TestClient(app) as client:
        # Take snapshot of counter for /ping GET 200 before requests.
        before = REQUEST_COUNTER.labels(endpoint="/ping", method="GET", status="200")._value.get()

        for _ in range(3):
            r = client.get("/ping")
            assert r.status_code == 200

        after = REQUEST_COUNTER.labels(endpoint="/ping", method="GET", status="200")._value.get()
        assert after - before == 3

        # Histogram observation count must also have grown by 3
        hist = REQUEST_LATENCY.labels(endpoint="/ping", method="GET")
        # _sum is a Value; total observations recorded in histogram's _count via samples.
        samples = list(hist.collect())[0].samples
        # find the _count sample for this label set
        count_samples = [s for s in samples if s.name.endswith("_count")]
        assert count_samples, "expected _count samples for histogram"
        assert count_samples[0].value >= 3

        # Now GET /metrics and ensure we see those values reflected
        resp = client.get("/metrics")
        body = resp.text
        assert "api_requests_total" in body
        assert 'endpoint="/ping"' in body
        assert "api_request_seconds_bucket" in body


def test_metrics_endpoint_is_not_self_recorded():
    """/metrics must not record requests against itself to avoid noise."""
    app = _build_app()
    with TestClient(app) as client:
        # Snapshot of /metrics counter
        try:
            before = REQUEST_COUNTER.labels(
                endpoint="/metrics", method="GET", status="200"
            )._value.get()
        except Exception:
            before = 0
        client.get("/metrics")
        client.get("/metrics")
        try:
            after = REQUEST_COUNTER.labels(
                endpoint="/metrics", method="GET", status="200"
            )._value.get()
        except Exception:
            after = 0
        # Should not have moved
        assert after == before


def test_middleware_records_different_status_codes():
    app = _build_app()
    with TestClient(app) as client:
        # missing route -> 404
        before = 0
        try:
            before = REQUEST_COUNTER.labels(
                endpoint="/does-not-exist", method="GET", status="404"
            )._value.get()
        except Exception:
            before = 0
        client.get("/does-not-exist")
        after = REQUEST_COUNTER.labels(
            endpoint="/does-not-exist", method="GET", status="404"
        )._value.get()
        assert after - before == 1
