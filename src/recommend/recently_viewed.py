"""Recently Viewed feature.

Computes a per-user "Recently Viewed" rail by reading the
:class:`src.recommend.behavior.BehaviorLogger` events table. Every
Amazon / TikTok-style ecommerce shows this rail at the top of the
homepage and search — it is one of the highest-CTR carousels in
production e-commerce because it surfaces things the shopper was
already considering.

Design notes
------------
* **Stdlib only.** All logic is one SQL query plus tiny in-memory
  passes; we never load pandas / numpy here. The module is hot-path:
  it is called on every homepage / search render.
* **Deduplication in Python, not SQL.** SQLite's window functions
  could do it with ``ROW_NUMBER()``, but the ``LIMIT`` interacts
  awkwardly with ``GROUP BY`` when we also want "ordered by latest
  event". Walking a small Python list (≤ ``limit*5`` rows) is
  simpler, easier to test, and trivially fast for the sizes we
  care about (≤ 100 rows per call).
* **Overfetch factor 5.** A user with 100 recent events of which 95
  are repeat views of the same 3 products would otherwise return
  3 dedup'd products instead of ``limit``. Pulling ``limit*5`` rows
  gives the dedup walk room to find ``limit`` distinct products in
  the vast majority of realistic histories. (If the user truly only
  has 3 distinct products in their last ``limit*5`` events, the
  rail returns 3 — that is the correct behavior.)
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.recommend.behavior import BehaviorLogger


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_recently_viewed(
    behavior_logger: "BehaviorLogger",
    user_id: int,
    limit: int = 20,
    event_types: tuple[str, ...] = ("view", "click"),
) -> list[int]:
    """Return product_ids the user has viewed/clicked, most recent first.

    The result is deduplicated: each product appears at most once,
    using its **latest** timestamp to determine ordering. This is the
    classic "Recently Viewed" semantics — if you look at product 42
    three times across the day, it appears once, ranked by the most
    recent visit.

    Parameters
    ----------
    behavior_logger:
        Source of events. Read-only access via raw SQL.
    user_id:
        The user whose history we are rendering.
    limit:
        Maximum number of distinct products to return.
    event_types:
        Which event types count as "viewed". Defaults to
        ``("view", "click")`` — purchase / add-to-cart are explicitly
        excluded so the rail is "things I was browsing", not "things
        I already bought".

    Returns
    -------
    list[int]
        Distinct product_ids, most-recent first. Empty list if the
        user has no matching events.

    Algorithm
    ---------
    1. Query events where ``user_id == given`` and
       ``event_type IN event_types``, ordered by ``timestamp DESC``,
       limited to ``limit * 5`` rows (overfetch so dedup has room).
    2. Walk in order, keeping each product_id's **first** (latest)
       occurrence.
    3. Return up to ``limit`` product_ids.
    """
    if limit <= 0:
        return []
    if not event_types:
        return []

    # Overfetch so deduplication has enough room to still return
    # ``limit`` distinct products in the common case where the user
    # has many repeat views of a few items.
    fetch_n = limit * 5

    placeholders = ",".join("?" for _ in event_types)
    sql = (
        f"SELECT product_id FROM events "
        f"WHERE user_id = ? AND event_type IN ({placeholders}) "
        f"ORDER BY timestamp DESC, id DESC "
        f"LIMIT ?"
    )
    params: list[object] = [int(user_id), *event_types, int(fetch_n)]
    cur = behavior_logger._conn.execute(sql, params)

    seen: set[int] = set()
    out: list[int] = []
    for row in cur.fetchall():
        pid = int(row["product_id"])
        if pid in seen:
            continue
        seen.add(pid)
        out.append(pid)
        if len(out) >= limit:
            break
    return out


def get_view_count(
    behavior_logger: "BehaviorLogger",
    user_id: int,
    product_id: int,
) -> int:
    """How many times has the user viewed/clicked this specific product?

    Counts both ``view`` and ``click`` events (not ``add_to_cart`` /
    ``purchase``). Returns ``0`` if the user has never touched the
    product.
    """
    cur = behavior_logger._conn.execute(
        "SELECT COUNT(*) AS n FROM events "
        "WHERE user_id = ? AND product_id = ? "
        "AND event_type IN ('view', 'click')",
        (int(user_id), int(product_id)),
    )
    row = cur.fetchone()
    return int(row["n"]) if row else 0


def get_co_viewed_in_session(
    behavior_logger: "BehaviorLogger",
    user_id: int,
    session_window_minutes: int = 60,
    min_co_occurrences: int = 2,
    top_k: int = 10,
) -> list[tuple[int, int, int]]:
    """Find product pairs the user has co-viewed in a recent session.

    Useful for a "Continue browsing" carousel: if I just looked at
    product A and product B together in this session, then next time
    I open the app, surfacing the pair (A, B) reminds me where I
    left off.

    Algorithm
    ---------
    1. Pull view/click events for ``user_id`` in the last
       ``session_window_minutes``.
    2. Count events per distinct product inside the window.
    3. For every unordered pair ``(A, B)`` of distinct products
       touched in the window, the co-occurrence count is defined as
       ``min(count_A, count_B)`` — the most "co-views" of the two
       that we can plausibly attribute to overlapping browsing.
       Pairs are canonicalised as ``(min, max)`` so ``(A,B)`` and
       ``(B,A)`` count as the same.
    4. Return ``[(product_a, product_b, count), ...]`` sorted by
       count desc, filtered to ``count >= min_co_occurrences``,
       limited to ``top_k`` pairs.

    Returns empty list if the user has fewer than 2 distinct events
    in the window (you need 2 events to form a pair).

    Parameters
    ----------
    behavior_logger:
        Source of events.
    user_id:
        Whose session to inspect.
    session_window_minutes:
        How far back to look. Default 60 minutes ≈ one shopping
        session.
    min_co_occurrences:
        Suppress noise — only return pairs that co-occur at least
        this many times. Default 2; a single accidental visit to two
        products in the same session is too weak a signal.
    top_k:
        Cap on the number of pairs returned.
    """
    if session_window_minutes <= 0 or top_k <= 0:
        return []

    # ``time.time()`` rather than ``datetime.now()`` so it lines up
    # exactly with how :class:`BehaviorLogger.log_event` stamps rows.
    import time as _time

    cutoff = _time.time() - (session_window_minutes * 60.0)

    cur = behavior_logger._conn.execute(
        "SELECT product_id FROM events "
        "WHERE user_id = ? AND event_type IN ('view', 'click') "
        "AND timestamp >= ? "
        "ORDER BY timestamp ASC, id ASC",
        (int(user_id), float(cutoff)),
    )
    rows = cur.fetchall()
    if len(rows) < 2:
        return []

    # Per-product event counts inside the window. For an unordered
    # pair (A, B) the co-occurrence count is ``min(count_A, count_B)``
    # — i.e. how many "co-views" of the two we can plausibly attribute
    # to the same browsing session. Logging A once and B once → pair
    # count 1; logging both twice → pair count 2. This is the model
    # described in the spec ("once we add a second view of same pair
    # → count=2").
    product_counts: dict[int, int] = defaultdict(int)
    for row in rows:
        product_counts[int(row["product_id"])] += 1

    # Pairs only exist when 2+ distinct products were touched.
    if len(product_counts) < 2:
        return []

    pair_counts: dict[tuple[int, int], int] = {}
    distinct_pids = sorted(product_counts.keys())
    for a, b in combinations(distinct_pids, 2):
        pair_counts[(a, b)] = min(product_counts[a], product_counts[b])

    # Filter + rank.
    filtered = [
        (a, b, c) for (a, b), c in pair_counts.items() if c >= min_co_occurrences
    ]
    filtered.sort(key=lambda t: (-t[2], t[0], t[1]))
    return filtered[:top_k]


def log_view(
    behavior_logger: "BehaviorLogger",
    user_id: int | None,
    product_id: int,
    query: str | None = None,
    position: int | None = None,
) -> int:
    """Log a ``view`` event. Convenience wrapper.

    Equivalent to::

        behavior_logger.log_event(
            product_id=product_id,
            event_type="view",
            user_id=user_id,
            query=query,
            position=position,
        )

    Returns the inserted event id.
    """
    return behavior_logger.log_event(
        product_id=int(product_id),
        event_type="view",
        user_id=int(user_id) if user_id is not None else None,
        query=query,
        position=position,
    )
