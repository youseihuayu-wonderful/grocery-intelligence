"""Search analytics dashboard backend.

Aggregate-level metrics over the ``BehaviorLogger`` event log, suitable
for an admin/operator dashboard. The metrics here are intentionally
read-only and side-effect-free: they take a logger, run SQL aggregation
queries against its underlying SQLite connection, and return plain
dataclasses / dicts / lists that JSON-serialize cleanly.

Performance note: the production log has ~1M events. Naive
``get_events(limit=10_000_000)`` works but materializes every row in
Python first. For the per-query / per-product / per-day aggregations
in this module we push the work down into SQL ``GROUP BY`` queries
running on the indexed ``events`` table; this turns minute-scale
Python loops into millisecond-scale SQLite scans.

Public surface
--------------
* ``TopQuery``, ``FunnelMetrics`` -- result dataclasses.
* ``top_queries`` -- most-frequent search queries (with click stats).
* ``funnel_metrics`` -- view -> click -> cart -> purchase rates.
* ``hot_products`` -- top products by event type.
* ``daily_event_counts`` -- per-day breakdown of the last N days.
* ``category_breakdown`` -- events grouped by department (needs catalog).
* ``user_activity_summary`` -- per-user event roll-up.
* ``search_quality_signals`` -- high-level search-quality summary.
"""

from __future__ import annotations

import datetime as _dt
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.evaluation.online_metrics import compute_ctr


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------
@dataclass
class TopQuery:
    """One row in the "top searched queries" report."""

    query: str
    count: int
    distinct_users: int
    avg_position_clicked: float | None  # None if no clicks for this query


@dataclass
class FunnelMetrics:
    """End-to-end search-funnel summary."""

    n_views: int
    n_clicks: int
    n_add_to_cart: int
    n_purchases: int
    view_to_click_rate: float       # n_clicks / n_views
    click_to_cart_rate: float       # n_add_to_cart / n_clicks
    cart_to_purchase_rate: float    # n_purchases / n_add_to_cart
    overall_conversion: float       # n_purchases / n_views


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _get_conn(behavior_logger):
    """Return the underlying SQLite connection of a BehaviorLogger.

    The logger keeps it on the ``_conn`` private attribute; we access
    it for read-only aggregation queries. Falls back to ``None`` if
    the logger doesn't expose one, in which case callers should use
    ``get_events`` instead.
    """
    return getattr(behavior_logger, "_conn", None)


def _safe_div(num: float, denom: float) -> float:
    """``num / denom`` with a 0.0 floor when ``denom`` is zero."""
    if denom == 0:
        return 0.0
    return num / denom


# ---------------------------------------------------------------------------
# Top queries
# ---------------------------------------------------------------------------
def top_queries(
    behavior_logger,
    limit: int = 20,
    since: float | None = None,
) -> list[TopQuery]:
    """Most-frequently-searched queries, ordered by total event count.

    Aggregates over every event whose ``query`` column is non-null,
    grouping by exact query text (case-sensitive). For each top query
    we also report:

    * ``distinct_users`` -- number of distinct ``user_id`` values
      (NULL user_ids are excluded from this count).
    * ``avg_position_clicked`` -- mean of the ``position`` column over
      click events for that query, or ``None`` if there are no clicks
      with a non-null position.

    Parameters
    ----------
    behavior_logger:
        A ``BehaviorLogger`` (or compatible object).
    limit:
        Maximum number of queries to return.
    since:
        Optional epoch-seconds floor. ``None`` -> all-time.
    """
    conn = _get_conn(behavior_logger)
    if conn is None:
        return []

    params: list[Any] = []
    since_clause = ""
    if since is not None:
        since_clause = " AND timestamp >= ?"
        params.append(float(since))

    # Top queries by total event count.
    sql = f"""
    SELECT
        query,
        COUNT(*) AS n_events,
        COUNT(DISTINCT user_id) AS n_users
    FROM events
    WHERE query IS NOT NULL{since_clause}
    GROUP BY query
    ORDER BY n_events DESC, query ASC
    LIMIT ?
    """
    rows = conn.execute(sql, params + [int(limit)]).fetchall()
    if not rows:
        return []

    query_texts = [r["query"] for r in rows]

    # Per-query avg click position. Single query with IN (?, ?, ?, ...)
    # — pulled separately so we can leave queries with no clicks at None.
    placeholders = ",".join("?" * len(query_texts))
    click_sql = f"""
    SELECT
        query,
        AVG(position) AS avg_pos
    FROM events
    WHERE event_type = 'click'
      AND query IN ({placeholders})
      AND position IS NOT NULL{since_clause}
    GROUP BY query
    """
    click_params: list[Any] = list(query_texts)
    if since is not None:
        click_params.append(float(since))
    click_rows = conn.execute(click_sql, click_params).fetchall()
    avg_pos_by_query = {r["query"]: float(r["avg_pos"]) for r in click_rows}

    return [
        TopQuery(
            query=r["query"],
            count=int(r["n_events"]),
            distinct_users=int(r["n_users"] or 0),
            avg_position_clicked=avg_pos_by_query.get(r["query"]),
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Funnel metrics
# ---------------------------------------------------------------------------
def funnel_metrics(
    behavior_logger,
    since: float | None = None,
) -> FunnelMetrics:
    """Compute the view -> click -> cart -> purchase funnel.

    All rates default to ``0.0`` when their denominator is zero, so
    the dashboard never sees a division-by-zero crash.
    """
    conn = _get_conn(behavior_logger)
    if conn is None:
        return FunnelMetrics(
            n_views=0,
            n_clicks=0,
            n_add_to_cart=0,
            n_purchases=0,
            view_to_click_rate=0.0,
            click_to_cart_rate=0.0,
            cart_to_purchase_rate=0.0,
            overall_conversion=0.0,
        )

    params: list[Any] = []
    since_clause = ""
    if since is not None:
        since_clause = "WHERE timestamp >= ?"
        params.append(float(since))

    sql = f"""
    SELECT
        SUM(CASE WHEN event_type = 'view'        THEN 1 ELSE 0 END) AS n_views,
        SUM(CASE WHEN event_type = 'click'       THEN 1 ELSE 0 END) AS n_clicks,
        SUM(CASE WHEN event_type = 'add_to_cart' THEN 1 ELSE 0 END) AS n_cart,
        SUM(CASE WHEN event_type = 'purchase'    THEN 1 ELSE 0 END) AS n_purchase
    FROM events {since_clause}
    """
    row = conn.execute(sql, params).fetchone()

    n_views = int(row["n_views"] or 0) if row else 0
    n_clicks = int(row["n_clicks"] or 0) if row else 0
    n_cart = int(row["n_cart"] or 0) if row else 0
    n_purchase = int(row["n_purchase"] or 0) if row else 0

    return FunnelMetrics(
        n_views=n_views,
        n_clicks=n_clicks,
        n_add_to_cart=n_cart,
        n_purchases=n_purchase,
        view_to_click_rate=_safe_div(n_clicks, n_views),
        click_to_cart_rate=_safe_div(n_cart, n_clicks),
        cart_to_purchase_rate=_safe_div(n_purchase, n_cart),
        overall_conversion=_safe_div(n_purchase, n_views),
    )


# ---------------------------------------------------------------------------
# Hot products
# ---------------------------------------------------------------------------
def hot_products(
    behavior_logger,
    event_type: str = "purchase",
    limit: int = 20,
    since: float | None = None,
) -> list[tuple[int, int]]:
    """Top ``product_id``s by event count of the given type.

    Returns a list of ``(product_id, count)`` tuples sorted by count
    descending. Empty list when the log has no matching events.
    """
    conn = _get_conn(behavior_logger)
    if conn is None:
        return []

    params: list[Any] = [event_type]
    since_clause = ""
    if since is not None:
        since_clause = " AND timestamp >= ?"
        params.append(float(since))

    sql = f"""
    SELECT product_id, COUNT(*) AS n
    FROM events
    WHERE event_type = ?{since_clause}
    GROUP BY product_id
    ORDER BY n DESC, product_id ASC
    LIMIT ?
    """
    params.append(int(limit))
    rows = conn.execute(sql, params).fetchall()
    return [(int(r["product_id"]), int(r["n"])) for r in rows]


# ---------------------------------------------------------------------------
# Daily event counts
# ---------------------------------------------------------------------------
def daily_event_counts(
    behavior_logger,
    days_back: int = 30,
) -> list[dict]:
    """Per-day event counts across the last ``days_back`` days.

    Each entry is a dict with:

    * ``date``: ``"YYYY-MM-DD"`` (UTC)
    * ``n_views``, ``n_clicks``, ``n_add_to_cart``, ``n_purchases``: ints

    The full window of ``days_back`` days is returned (zero-fill for
    days with no events), oldest first.
    """
    conn = _get_conn(behavior_logger)

    today = _dt.datetime.now(_dt.timezone.utc).date()
    # Inclusive window: days_back days ending today.
    start_date = today - _dt.timedelta(days=days_back - 1)
    start_ts = _dt.datetime.combine(
        start_date, _dt.time(0, 0, 0), tzinfo=_dt.timezone.utc,
    ).timestamp()

    # Pre-fill the window with zeros so missing days are still present.
    by_day: dict[str, dict[str, int]] = {}
    for i in range(days_back):
        d = start_date + _dt.timedelta(days=i)
        by_day[d.isoformat()] = {
            "n_views": 0,
            "n_clicks": 0,
            "n_add_to_cart": 0,
            "n_purchases": 0,
        }

    if conn is not None:
        sql = """
        SELECT
            DATE(timestamp, 'unixepoch') AS day,
            event_type,
            COUNT(*) AS n
        FROM events
        WHERE timestamp >= ?
        GROUP BY day, event_type
        """
        rows = conn.execute(sql, (start_ts,)).fetchall()
        col_for_type = {
            "view": "n_views",
            "click": "n_clicks",
            "add_to_cart": "n_add_to_cart",
            "purchase": "n_purchases",
        }
        for r in rows:
            day = r["day"]
            col = col_for_type.get(r["event_type"])
            if col is None or day not in by_day:
                continue
            by_day[day][col] += int(r["n"])

    return [
        {"date": day, **counts}
        for day, counts in sorted(by_day.items())  # ISO date sorts naturally
    ]


# ---------------------------------------------------------------------------
# Category breakdown
# ---------------------------------------------------------------------------
def category_breakdown(
    behavior_logger,
    catalog: pd.DataFrame,
    event_type: str = "purchase",
    since: float | None = None,
) -> list[dict]:
    """Count events of ``event_type`` grouped by product department.

    Joins the per-product counts in the event log with the catalog's
    ``product_id -> department`` map. Returns:

        [{"department": str, "count": int, "share": float}, ...]

    sorted by count descending. ``share`` is ``count / total`` so the
    full list sums to 1.0 (subject to float rounding). Products whose
    ``department`` is missing in the catalog are bucketed under
    ``"unknown"``.
    """
    conn = _get_conn(behavior_logger)
    if conn is None or catalog is None or catalog.empty:
        return []

    params: list[Any] = [event_type]
    since_clause = ""
    if since is not None:
        since_clause = " AND timestamp >= ?"
        params.append(float(since))

    sql = f"""
    SELECT product_id, COUNT(*) AS n
    FROM events
    WHERE event_type = ?{since_clause}
    GROUP BY product_id
    """
    rows = conn.execute(sql, params).fetchall()
    if not rows:
        return []

    # Catalog department lookup: product_id -> department.
    dept_map: dict[int, str] = dict(
        zip(
            catalog["product_id"].astype(int).tolist(),
            catalog["department"].fillna("unknown").astype(str).tolist(),
        )
    )

    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        pid = int(r["product_id"])
        dept = dept_map.get(pid, "unknown")
        counts[dept] += int(r["n"])

    total = sum(counts.values())
    if total == 0:
        return []

    return sorted(
        [
            {"department": d, "count": c, "share": c / total}
            for d, c in counts.items()
        ],
        key=lambda x: x["count"],
        reverse=True,
    )


# ---------------------------------------------------------------------------
# User activity summary
# ---------------------------------------------------------------------------
def user_activity_summary(
    behavior_logger,
    user_id: int,
) -> dict:
    """Per-user activity roll-up.

    Returns:
        {
            "user_id": int,
            "total_events": int,
            "by_type": dict[str, int],
            "distinct_products": int,
            "first_event_at": float | None,
            "last_event_at": float | None,
            "active_days": int,
        }

    ``active_days`` is the number of distinct calendar days
    (UTC) the user has at least one event on. When the user has no
    events at all, all numeric fields are zero and the timestamps are
    ``None``.
    """
    conn = _get_conn(behavior_logger)
    out: dict[str, Any] = {
        "user_id": int(user_id),
        "total_events": 0,
        "by_type": {"view": 0, "click": 0, "add_to_cart": 0, "purchase": 0},
        "distinct_products": 0,
        "first_event_at": None,
        "last_event_at": None,
        "active_days": 0,
    }
    if conn is None:
        return out

    uid = int(user_id)

    summary_row = conn.execute(
        """
        SELECT
            COUNT(*)              AS total,
            COUNT(DISTINCT product_id) AS distinct_products,
            MIN(timestamp)        AS first_ts,
            MAX(timestamp)        AS last_ts
        FROM events
        WHERE user_id = ?
        """,
        (uid,),
    ).fetchone()

    total = int(summary_row["total"] or 0) if summary_row else 0
    if total == 0:
        return out

    out["total_events"] = total
    out["distinct_products"] = int(summary_row["distinct_products"] or 0)
    out["first_event_at"] = (
        float(summary_row["first_ts"]) if summary_row["first_ts"] is not None else None
    )
    out["last_event_at"] = (
        float(summary_row["last_ts"]) if summary_row["last_ts"] is not None else None
    )

    type_rows = conn.execute(
        """
        SELECT event_type, COUNT(*) AS n
        FROM events
        WHERE user_id = ?
        GROUP BY event_type
        """,
        (uid,),
    ).fetchall()
    for r in type_rows:
        et = r["event_type"]
        if et in out["by_type"]:
            out["by_type"][et] = int(r["n"])

    day_row = conn.execute(
        """
        SELECT COUNT(DISTINCT DATE(timestamp, 'unixepoch')) AS n_days
        FROM events
        WHERE user_id = ?
        """,
        (uid,),
    ).fetchone()
    out["active_days"] = int(day_row["n_days"] or 0) if day_row else 0

    return out


# ---------------------------------------------------------------------------
# Search-quality signals
# ---------------------------------------------------------------------------
def search_quality_signals(
    behavior_logger,
    since: float | None = None,
) -> dict:
    """High-level signals operators glance at to spot search regressions.

    Returns:
        {
            "total_search_events": int,    # events where query IS NOT NULL
            "distinct_queries": int,
            "click_through_rate": float,   # clicks / searches (compute_ctr)
            "avg_click_position": float,   # mean position over click events
            "zero_result_rate": float | None,
        }

    ``zero_result_rate`` requires impression-level data we don't store
    (we don't log "search returned 0 results"); we return ``None`` so
    the UI can render "n/a".
    """
    conn = _get_conn(behavior_logger)
    out = {
        "total_search_events": 0,
        "distinct_queries": 0,
        "click_through_rate": 0.0,
        "avg_click_position": 0.0,
        "zero_result_rate": None,
    }
    if conn is None:
        return out

    params: list[Any] = []
    since_clause = ""
    if since is not None:
        since_clause = " AND timestamp >= ?"
        params.append(float(since))

    # Total events with a non-null query, plus distinct query count.
    row = conn.execute(
        f"""
        SELECT
            COUNT(*) AS n_events,
            COUNT(DISTINCT query) AS n_queries
        FROM events
        WHERE query IS NOT NULL{since_clause}
        """,
        params,
    ).fetchone()
    if row:
        out["total_search_events"] = int(row["n_events"] or 0)
        out["distinct_queries"] = int(row["n_queries"] or 0)

    # Click-through rate over query-bearing events. Reuse the existing
    # ``compute_ctr`` helper: it just walks a list of dicts and counts
    # 'view' vs 'click' rows, so we pass a synthetic mini-iterable. Get
    # the two counts once via SQL and feed them in.
    counts = conn.execute(
        f"""
        SELECT
            SUM(CASE WHEN event_type='view'  THEN 1 ELSE 0 END) AS n_v,
            SUM(CASE WHEN event_type='click' THEN 1 ELSE 0 END) AS n_c
        FROM events
        WHERE query IS NOT NULL{since_clause}
        """,
        params,
    ).fetchone()
    n_v = int(counts["n_v"] or 0) if counts else 0
    n_c = int(counts["n_c"] or 0) if counts else 0
    fake_events = (
        [{"event_type": "view"}] * n_v + [{"event_type": "click"}] * n_c
    )
    out["click_through_rate"] = compute_ctr(fake_events)

    # Avg click position over events that ARE clicks AND have a position.
    # No requirement here that ``query`` is non-null (positions are
    # populated for clicks on search-result lists).
    pos_params: list[Any] = []
    pos_clause = ""
    if since is not None:
        pos_clause = " AND timestamp >= ?"
        pos_params.append(float(since))

    pos_row = conn.execute(
        f"""
        SELECT AVG(position) AS avg_pos
        FROM events
        WHERE event_type = 'click'
          AND position IS NOT NULL{pos_clause}
        """,
        pos_params,
    ).fetchone()
    if pos_row and pos_row["avg_pos"] is not None:
        out["avg_click_position"] = float(pos_row["avg_pos"])

    return out
