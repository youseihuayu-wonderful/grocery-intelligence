"""Behavioral signal tracking module.

SQLite-backed event log for user interactions with the grocery
intelligence app. Records views, clicks, add-to-cart, and purchase
events so future learning-to-rank (LTR) training has real behavioral
data to learn from.

The logger is intentionally minimal: a single ``events`` table with
indexes that match the typical query patterns (per-user history,
per-product popularity, per-type counts, per-(user, product) joins).
We use only the stdlib ``sqlite3`` module - no SQLAlchemy or any
extra dependency - so this module can run anywhere Python runs.

Public surface
--------------
* ``EVENT_TYPES`` -- tuple of allowed event-type strings.
* ``DEFAULT_DB_PATH`` -- on-disk location for the production log.
* ``BehaviorLogger`` -- context-manager wrapper over an SQLite
  connection with insert / query / summary / feature-matrix helpers.

The class supports use as a context manager, e.g.::

    with BehaviorLogger() as log:
        log.log_event(product_id=42, event_type="click", user_id=1)
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
EVENT_TYPES: tuple[str, ...] = ("view", "click", "add_to_cart", "purchase")

DEFAULT_DB_PATH: Path = Path("data/processed/behavior.db")


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp    REAL    NOT NULL,
    user_id      INTEGER,
    product_id   INTEGER NOT NULL,
    event_type   TEXT    NOT NULL,
    query        TEXT,
    position     INTEGER
);
CREATE INDEX IF NOT EXISTS idx_user ON events(user_id);
CREATE INDEX IF NOT EXISTS idx_product ON events(product_id);
CREATE INDEX IF NOT EXISTS idx_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_user_product ON events(user_id, product_id);
"""


# ---------------------------------------------------------------------------
# BehaviorLogger
# ---------------------------------------------------------------------------
class BehaviorLogger:
    """SQLite-backed event log for user interactions.

    Parameters
    ----------
    db_path:
        Filesystem location of the SQLite database. Defaults to
        :data:`DEFAULT_DB_PATH`. The parent directory is created
        on demand. Passing ``":memory:"`` produces an in-memory
        database (useful for tests).
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            db_path = DEFAULT_DB_PATH
        self.db_path: Path | str
        if isinstance(db_path, Path) or (
            isinstance(db_path, str) and db_path != ":memory:"
        ):
            p = Path(db_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self.db_path = p
            conn_target: str = str(p)
        else:
            # in-memory
            self.db_path = ":memory:"
            conn_target = ":memory:"

        # ``check_same_thread=False`` lets callers share the logger across
        # threads (e.g. an API server with worker threads). All access is
        # still serialized by SQLite's own locking.
        self._conn = sqlite3.connect(conn_target, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # Pragmatic perf tweaks: WAL is much faster for concurrent
        # readers/writers, NORMAL sync is safe enough for analytics.
        try:
            self._conn.execute("PRAGMA journal_mode = WAL;")
            self._conn.execute("PRAGMA synchronous = NORMAL;")
        except sqlite3.DatabaseError:
            # WAL is not supported for ``:memory:`` databases on some
            # platforms; the failure is harmless.
            pass
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Insertion helpers
    # ------------------------------------------------------------------
    def log_event(
        self,
        product_id: int,
        event_type: str,
        user_id: int | None = None,
        query: str | None = None,
        position: int | None = None,
        timestamp: float | None = None,
    ) -> int:
        """Insert one event. Returns the row id.

        Raises
        ------
        ValueError
            If ``event_type`` is not one of :data:`EVENT_TYPES`.
        """
        if event_type not in EVENT_TYPES:
            raise ValueError(
                f"Unknown event_type {event_type!r}. "
                f"Expected one of {EVENT_TYPES}."
            )
        if timestamp is None:
            timestamp = time.time()

        cur = self._conn.execute(
            """
            INSERT INTO events
                (timestamp, user_id, product_id, event_type, query, position)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (timestamp, user_id, int(product_id), event_type, query, position),
        )
        self._conn.commit()
        # ``lastrowid`` may be ``None`` per the typeshed signature, but
        # for an autoincrement INSERT it's always populated.
        return int(cur.lastrowid or 0)

    def log_events_bulk(self, events: list[dict[str, Any]]) -> int:
        """Bulk-insert events inside a single transaction.

        Each dict accepts the same keys as :meth:`log_event`. Returns
        the number of rows inserted. Validation (event_type membership)
        is applied per row.
        """
        if not events:
            return 0

        rows: list[tuple[Any, ...]] = []
        now = time.time()
        for ev in events:
            event_type = ev.get("event_type")
            if event_type not in EVENT_TYPES:
                raise ValueError(
                    f"Unknown event_type {event_type!r}. "
                    f"Expected one of {EVENT_TYPES}."
                )
            product_id = ev.get("product_id")
            if product_id is None:
                raise ValueError("product_id is required for every event.")
            rows.append(
                (
                    ev.get("timestamp", now),
                    ev.get("user_id"),
                    int(product_id),
                    event_type,
                    ev.get("query"),
                    ev.get("position"),
                )
            )

        # Explicit transaction: a single commit at the end instead of
        # one per row. With WAL journaling this is the difference
        # between O(1) fsyncs and O(N) fsyncs - i.e. seconds vs minutes
        # at 1M rows.
        try:
            self._conn.execute("BEGIN")
            self._conn.executemany(
                """
                INSERT INTO events
                    (timestamp, user_id, product_id, event_type, query, position)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        return len(rows)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------
    def get_events(
        self,
        user_id: int | None = None,
        product_id: int | None = None,
        event_type: str | None = None,
        since: float | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return events matching the given filters, newest first.

        ``limit`` caps the number of rows returned. All filter
        arguments are optional and are combined with ``AND``.
        """
        clauses: list[str] = []
        params: list[Any] = []
        if user_id is not None:
            clauses.append("user_id = ?")
            params.append(int(user_id))
        if product_id is not None:
            clauses.append("product_id = ?")
            params.append(int(product_id))
        if event_type is not None:
            if event_type not in EVENT_TYPES:
                raise ValueError(
                    f"Unknown event_type {event_type!r}. "
                    f"Expected one of {EVENT_TYPES}."
                )
            clauses.append("event_type = ?")
            params.append(event_type)
        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(float(since))

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = (
            "SELECT id, timestamp, user_id, product_id, event_type, "
            "query, position "
            f"FROM events {where} ORDER BY timestamp DESC, id DESC "
            "LIMIT ?"
        )
        params.append(int(limit))

        cur = self._conn.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]

    def count_events(self, event_type: str | None = None) -> int:
        """Total event count, optionally filtered by ``event_type``."""
        if event_type is None:
            cur = self._conn.execute("SELECT COUNT(*) AS n FROM events")
        else:
            if event_type not in EVENT_TYPES:
                raise ValueError(
                    f"Unknown event_type {event_type!r}. "
                    f"Expected one of {EVENT_TYPES}."
                )
            cur = self._conn.execute(
                "SELECT COUNT(*) AS n FROM events WHERE event_type = ?",
                (event_type,),
            )
        row = cur.fetchone()
        return int(row["n"]) if row else 0

    def user_event_summary(self, user_id: int) -> dict[str, Any]:
        """Return ``{'total', 'by_type', 'distinct_products'}`` for a user."""
        uid = int(user_id)
        total_row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM events WHERE user_id = ?",
            (uid,),
        ).fetchone()
        total = int(total_row["n"]) if total_row else 0

        by_type: dict[str, int] = {t: 0 for t in EVENT_TYPES}
        cur = self._conn.execute(
            "SELECT event_type, COUNT(*) AS n FROM events "
            "WHERE user_id = ? GROUP BY event_type",
            (uid,),
        )
        for row in cur.fetchall():
            by_type[row["event_type"]] = int(row["n"])

        dist_row = self._conn.execute(
            "SELECT COUNT(DISTINCT product_id) AS n "
            "FROM events WHERE user_id = ?",
            (uid,),
        ).fetchone()
        distinct_products = int(dist_row["n"]) if dist_row else 0

        return {
            "total": total,
            "by_type": by_type,
            "distinct_products": distinct_products,
        }

    def product_event_summary(self, product_id: int) -> dict[str, Any]:
        """Return ``{'total', 'by_type', 'distinct_users'}`` for a product."""
        pid = int(product_id)
        total_row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM events WHERE product_id = ?",
            (pid,),
        ).fetchone()
        total = int(total_row["n"]) if total_row else 0

        by_type: dict[str, int] = {t: 0 for t in EVENT_TYPES}
        cur = self._conn.execute(
            "SELECT event_type, COUNT(*) AS n FROM events "
            "WHERE product_id = ? GROUP BY event_type",
            (pid,),
        )
        for row in cur.fetchall():
            by_type[row["event_type"]] = int(row["n"])

        # Anonymous events (user_id IS NULL) are excluded from the
        # distinct-user count - they can't be attributed to a person.
        dist_row = self._conn.execute(
            "SELECT COUNT(DISTINCT user_id) AS n "
            "FROM events WHERE product_id = ? AND user_id IS NOT NULL",
            (pid,),
        ).fetchone()
        distinct_users = int(dist_row["n"]) if dist_row else 0

        return {
            "total": total,
            "by_type": by_type,
            "distinct_users": distinct_users,
        }

    # ------------------------------------------------------------------
    # Feature matrix for downstream LTR training
    # ------------------------------------------------------------------
    def get_feature_matrix(self) -> pd.DataFrame:
        """Build a per-(user, product) feature DataFrame for LTR training.

        Aggregates every event in the log into one row per
        ``(user_id, product_id)`` pair with columns:

        ``n_views, n_clicks, n_add_to_cart, n_purchases,
        last_event_ts, conversion``

        ``conversion`` is ``1`` if the pair has any purchase event,
        else ``0``. The returned DataFrame has ``user_id`` and
        ``product_id`` as plain columns (not the index), which is
        the shape most LightGBM/XGBoost LTR setups expect.
        """
        sql = """
        SELECT
            user_id,
            product_id,
            SUM(CASE WHEN event_type = 'view'        THEN 1 ELSE 0 END) AS n_views,
            SUM(CASE WHEN event_type = 'click'       THEN 1 ELSE 0 END) AS n_clicks,
            SUM(CASE WHEN event_type = 'add_to_cart' THEN 1 ELSE 0 END) AS n_add_to_cart,
            SUM(CASE WHEN event_type = 'purchase'    THEN 1 ELSE 0 END) AS n_purchases,
            MAX(timestamp) AS last_event_ts
        FROM events
        GROUP BY user_id, product_id
        """
        df = pd.read_sql_query(sql, self._conn)
        if df.empty:
            return pd.DataFrame(
                columns=[
                    "user_id",
                    "product_id",
                    "n_views",
                    "n_clicks",
                    "n_add_to_cart",
                    "n_purchases",
                    "last_event_ts",
                    "conversion",
                ]
            )
        # SQLite SUM returns NULL for empty groups, but our CASE
        # expressions always produce 0/1 - so the counts are integer.
        for col in ("n_views", "n_clicks", "n_add_to_cart", "n_purchases"):
            df[col] = df[col].fillna(0).astype("int64")
        df["conversion"] = (df["n_purchases"] > 0).astype("int8")
        return df

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Close the underlying SQLite connection. Idempotent."""
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                # Mark closed so repeat calls are safe.
                self._conn = None  # type: ignore[assignment]

    def __enter__(self) -> "BehaviorLogger":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
