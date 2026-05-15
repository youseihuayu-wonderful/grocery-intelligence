"""Online evaluation metrics for A/B experiments.

This module closes the experimentation loop: ``src.experiments.ab_testing``
deterministically assigns users to variants, ``src.recommend.behavior``
records every interaction (view / click / add_to_cart / purchase), and
this module turns that behavioral event log into the per-variant metrics
that decide whether a treatment wins.

The metric surface intentionally mirrors what real ranking teams compute
in their experiment dashboards:

* **CTR** -- clicks divided by views. The simplest engagement signal.
* **Conversion rate** -- distinct ``(user, product)`` pairs that ended
  in a purchase, over distinct pairs with any prior impression.
* **MRR@K** -- "did the user click something near the top?" computed
  over click events with a known rank position.
* **Average click position** -- a cheap stand-in for "did the ranker
  promote relevant items?". Lower is better.

Inputs are plain ``list[dict]`` rows as returned by
:meth:`BehaviorLogger.get_events` -- a deliberate choice so the metric
functions are pure, easy to unit-test with hand-crafted events, and
not coupled to SQLite.

Public surface
--------------
* ``compute_ctr``, ``compute_conversion_rate``, ``compute_mrr_at_k``,
  ``compute_avg_click_position`` -- the four single-shot metric
  functions.
* ``VariantMetrics`` -- a dataclass bundling every metric a variant
  needs to report.
* ``compute_variant_metrics`` -- aggregates events per variant by
  mapping ``user_id -> variant`` through a caller-supplied dict.
* ``compare_variants`` -- A vs B summary including lift percentages
  and a winner-by-CTR.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Single-metric functions
# ---------------------------------------------------------------------------
def compute_ctr(events: list[dict]) -> float:
    """Click-through rate: ``clicks / views``.

    Returns ``0.0`` if there are no views (avoids division by zero).
    """
    n_views = sum(1 for e in events if e.get("event_type") == "view")
    n_clicks = sum(1 for e in events if e.get("event_type") == "click")
    if n_views == 0:
        return 0.0
    return n_clicks / n_views


def compute_conversion_rate(events: list[dict]) -> float:
    """Conversion rate over distinct ``(user_id, product_id)`` pairs.

    Numerator: distinct pairs with a ``purchase`` event.
    Denominator: distinct pairs with any prior ``view`` or ``click``
    impression.

    Returns ``0.0`` if there are no impressions.
    """
    impressions: set[tuple[Any, Any]] = set()
    purchases: set[tuple[Any, Any]] = set()
    for ev in events:
        pair = (ev.get("user_id"), ev.get("product_id"))
        et = ev.get("event_type")
        if et in ("view", "click"):
            impressions.add(pair)
        elif et == "purchase":
            purchases.add(pair)

    if not impressions:
        return 0.0
    converted = impressions & purchases
    return len(converted) / len(impressions)


def compute_mrr_at_k(events: list[dict], k: int = 10) -> float:
    """Mean Reciprocal Rank at K, averaged across click events.

    For every click event whose ``position`` is non-null and within
    ``[1, k]``, contribute ``1 / position``. Average over the count
    of such clicks. Returns ``0.0`` if there are no qualifying clicks.

    Useful as an answer to "did the user click something near the top?".
    """
    reciprocals: list[float] = []
    for ev in events:
        if ev.get("event_type") != "click":
            continue
        pos = ev.get("position")
        if pos is None:
            continue
        # Positions are 1-indexed; ignore zero / negative defensively.
        if pos < 1 or pos > k:
            continue
        reciprocals.append(1.0 / pos)

    if not reciprocals:
        return 0.0
    return sum(reciprocals) / len(reciprocals)


def compute_avg_click_position(events: list[dict]) -> float:
    """Average rank position of click events.

    Clicks whose ``position`` is ``None`` are ignored. Returns ``0.0``
    if no click events carry a position.
    """
    positions = [
        ev["position"]
        for ev in events
        if ev.get("event_type") == "click" and ev.get("position") is not None
    ]
    if not positions:
        return 0.0
    return sum(positions) / len(positions)


# ---------------------------------------------------------------------------
# Per-variant aggregation
# ---------------------------------------------------------------------------
@dataclass
class VariantMetrics:
    """Metrics for one arm of an A/B experiment."""

    variant: str
    n_users: int
    n_views: int
    n_clicks: int
    n_purchases: int
    ctr: float
    conversion_rate: float
    mrr_at_10: float
    avg_click_position: float


def compute_variant_metrics(
    behavior_logger,
    user_variant_map: dict[int, str],
    since: float | None = None,
) -> dict[str, VariantMetrics]:
    """Pull events from the logger and aggregate metrics per variant.

    Parameters
    ----------
    behavior_logger:
        A ``BehaviorLogger`` instance (or anything else exposing the
        same ``get_events(since=..., limit=...)`` signature).
    user_variant_map:
        Mapping of ``user_id -> variant_name``. Events whose
        ``user_id`` is NOT in this map are silently dropped -- they
        don't belong to the experiment.
    since:
        Optional epoch-seconds floor. ``None`` means "all time".

    Returns
    -------
    ``dict[variant_name, VariantMetrics]`` with one entry per variant
    that appears as a value in ``user_variant_map`` (including variants
    that ended up with zero events, so the dashboard always has a row
    per arm).
    """
    # Pull every event since the cutoff. ``limit`` is set to a value
    # larger than any plausible event count for this MVP; in a real
    # system we'd paginate, but for a 1M-row SQLite DB a single fetch
    # is fine.
    raw_events = behavior_logger.get_events(since=since, limit=10_000_000)

    # Group by variant, dropping users outside the experiment.
    events_by_variant: dict[str, list[dict]] = defaultdict(list)
    users_by_variant: dict[str, set[int]] = defaultdict(set)
    for ev in raw_events:
        uid = ev.get("user_id")
        if uid is None:
            continue
        variant = user_variant_map.get(uid)
        if variant is None:
            continue
        events_by_variant[variant].append(ev)
        users_by_variant[variant].add(uid)

    # Make sure every variant from the user-map has an entry, even if
    # no events landed for it (so the comparison code never KeyErrors).
    all_variants: set[str] = set(user_variant_map.values())
    for v in all_variants:
        events_by_variant.setdefault(v, [])
        users_by_variant.setdefault(v, set())

    out: dict[str, VariantMetrics] = {}
    for variant, evs in events_by_variant.items():
        n_views = sum(1 for e in evs if e.get("event_type") == "view")
        n_clicks = sum(1 for e in evs if e.get("event_type") == "click")
        n_purchases = sum(1 for e in evs if e.get("event_type") == "purchase")
        out[variant] = VariantMetrics(
            variant=variant,
            n_users=len(users_by_variant[variant]),
            n_views=n_views,
            n_clicks=n_clicks,
            n_purchases=n_purchases,
            ctr=compute_ctr(evs),
            conversion_rate=compute_conversion_rate(evs),
            mrr_at_10=compute_mrr_at_k(evs, k=10),
            avg_click_position=compute_avg_click_position(evs),
        )
    return out


# ---------------------------------------------------------------------------
# A vs B comparison
# ---------------------------------------------------------------------------
def _lift_pct(baseline: float, treatment: float) -> float | None:
    """Return ``(treatment - baseline) / baseline * 100``, or ``None``
    if ``baseline`` is zero (lift is undefined)."""
    if baseline == 0:
        return None
    return (treatment - baseline) / baseline * 100.0


def compare_variants(
    metrics_a: VariantMetrics,
    metrics_b: VariantMetrics,
) -> dict:
    """Pairwise comparison: how does B (treatment) move relative to A (control)?

    Returns a dict with the lift on CTR, conversion, and MRR, plus a
    winner picked by CTR. Lift is ``None`` (not ``inf``/``NaN``) when
    the control value is zero, so dashboards can render "n/a".

    Tie-breaking on CTR equality returns ``"tie"``.
    """
    ctr_lift = _lift_pct(metrics_a.ctr, metrics_b.ctr)
    conversion_lift = _lift_pct(metrics_a.conversion_rate, metrics_b.conversion_rate)
    mrr_lift = _lift_pct(metrics_a.mrr_at_10, metrics_b.mrr_at_10)

    if metrics_a.ctr == metrics_b.ctr:
        winner = "tie"
    elif metrics_b.ctr > metrics_a.ctr:
        winner = metrics_b.variant
    else:
        winner = metrics_a.variant

    return {
        "control": metrics_a.variant,
        "treatment": metrics_b.variant,
        "ctr_lift_pct": ctr_lift,
        "conversion_lift_pct": conversion_lift,
        "mrr_lift_pct": mrr_lift,
        "winner": winner,
        "sample_size_a": metrics_a.n_users,
        "sample_size_b": metrics_b.n_users,
    }
