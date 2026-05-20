"""SQLite-backed Subscribe & Save (auto-reorder) persistence.

A single SQLite database holds a ``subscriptions`` table keyed on an
auto-incrementing ``id``. Each row records a user's standing order for
a product at a fixed cadence (e.g. milk every week):

    subscriptions(
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id           INTEGER NOT NULL,
        product_id        INTEGER NOT NULL,
        qty               INTEGER NOT NULL DEFAULT 1,
        frequency         TEXT    NOT NULL,
        next_delivery_at  REAL    NOT NULL,
        active            INTEGER NOT NULL DEFAULT 1,
        created_at        REAL    NOT NULL
    )

with indices on ``(user_id, active)`` and ``(next_delivery_at)`` for the
two hot lookup paths: a user's active list and the cron's "what's due
right now?" sweep.

The store is safe to use as a context manager. Stdlib ``sqlite3`` only.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.shopping.cart import CartStore


# Default location, relative to the project root. Created on demand.
DEFAULT_SUBS_DB = Path("data/processed/subscriptions.db")


# Public allow-list of frequency names.
FREQUENCIES: tuple[str, ...] = (
    "weekly",
    "biweekly",
    "monthly",
    "every_2_months",
    "every_3_months",
)

# Mapping from a frequency name to the number of days between deliveries.
# Months are normalized to 30 days for simplicity (good enough for a
# revenue-estimate / next-delivery calendar; not a billing system).
FREQUENCY_DAYS: dict[str, int] = {
    "weekly": 7,
    "biweekly": 14,
    "monthly": 30,
    "every_2_months": 60,
    "every_3_months": 90,
}

_SECONDS_PER_DAY = 86400.0


class SubscriptionStore:
    """SQLite-backed Subscribe & Save persistence.

    Single DB, single table ``subscriptions``. The schema is documented
    at module level.
    """

    # --------------------------------------------------------------
    # Construction / lifecycle
    # --------------------------------------------------------------
    def __init__(self, db_path: str | Path | None = None):
        """Open or create the database.

        Parameters
        ----------
        db_path:
            Filesystem path to the SQLite database. ``None`` defaults
            to :data:`DEFAULT_SUBS_DB`. The parent directory is created
            if it does not exist. Pass ``":memory:"`` for tests.
        """
        if db_path is None:
            db_path = DEFAULT_SUBS_DB
        path_str = str(db_path)
        if path_str != ":memory:":
            parent = Path(path_str).expanduser().resolve().parent
            parent.mkdir(parents=True, exist_ok=True)

        # check_same_thread=False mirrors CartStore so the same store
        # can serve requests from FastAPI worker threads.
        self._conn: sqlite3.Connection = sqlite3.connect(
            path_str, check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON;")
        self._create_schema()

    def _create_schema(self) -> None:
        cur = self._conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS subscriptions (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id           INTEGER NOT NULL,
                product_id        INTEGER NOT NULL,
                qty               INTEGER NOT NULL DEFAULT 1,
                frequency         TEXT    NOT NULL,
                next_delivery_at  REAL    NOT NULL,
                active            INTEGER NOT NULL DEFAULT 1,
                created_at        REAL    NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_subs_user_active
                ON subscriptions(user_id, active);
            CREATE INDEX IF NOT EXISTS idx_subs_next_delivery
                ON subscriptions(next_delivery_at);
            """
        )
        self._conn.commit()

    # --------------------------------------------------------------
    # Mutations
    # --------------------------------------------------------------
    def subscribe(
        self,
        user_id: int,
        product_id: int,
        frequency: str,
        qty: int = 1,
        first_delivery_offset_days: int = 0,
    ) -> int:
        """Create a subscription. Returns the new ``subscription_id``.

        ``frequency`` must be one of :data:`FREQUENCIES`; anything else
        raises ``ValueError``. ``qty`` must be a positive integer.

        ``next_delivery_at`` is set to ``now + first_delivery_offset_days
        * 86400``; pass ``0`` for "already due" (useful for the cron
        demo) or a positive value to schedule into the future.
        """
        if frequency not in FREQUENCY_DAYS:
            raise ValueError(
                f"unknown frequency: {frequency!r}. "
                f"Must be one of {FREQUENCIES}"
            )
        if qty <= 0:
            raise ValueError("qty must be > 0")

        now = time.time()
        next_at = now + float(first_delivery_offset_days) * _SECONDS_PER_DAY
        cur = self._conn.execute(
            """
            INSERT INTO subscriptions
                (user_id, product_id, qty, frequency,
                 next_delivery_at, active, created_at)
            VALUES (?, ?, ?, ?, ?, 1, ?)
            """,
            (
                int(user_id),
                int(product_id),
                int(qty),
                frequency,
                next_at,
                now,
            ),
        )
        self._conn.commit()
        sub_id = cur.lastrowid
        assert sub_id is not None  # AUTOINCREMENT always returns one
        return int(sub_id)

    def cancel(self, subscription_id: int) -> None:
        """Soft-cancel by setting ``active=0``.

        Cancelled rows remain in history so the user (and the analytics
        side) can still see what they had. The cron skips them via the
        ``active=1`` predicate.
        """
        self._conn.execute(
            "UPDATE subscriptions SET active = 0 WHERE id = ?",
            (int(subscription_id),),
        )
        self._conn.commit()

    def update_frequency(self, subscription_id: int, frequency: str) -> None:
        """Change the cadence on an existing subscription.

        Does *not* shift ``next_delivery_at`` — the change applies from
        the next fulfillment onwards. ``frequency`` is validated.
        """
        if frequency not in FREQUENCY_DAYS:
            raise ValueError(
                f"unknown frequency: {frequency!r}. "
                f"Must be one of {FREQUENCIES}"
            )
        self._conn.execute(
            "UPDATE subscriptions SET frequency = ? WHERE id = ?",
            (frequency, int(subscription_id)),
        )
        self._conn.commit()

    def update_qty(self, subscription_id: int, qty: int) -> None:
        """Change the quantity on an existing subscription. ``qty`` > 0."""
        if qty <= 0:
            raise ValueError("qty must be > 0; use cancel() to stop a sub")
        self._conn.execute(
            "UPDATE subscriptions SET qty = ? WHERE id = ?",
            (int(qty), int(subscription_id)),
        )
        self._conn.commit()

    def advance_next_delivery(self, subscription_id: int) -> None:
        """Push ``next_delivery_at`` forward by one cadence interval.

        Called by the fulfillment cron after the subscription has been
        added to the user's cart. Looks up the row's frequency to find
        the correct delta, so callers don't have to remember.
        """
        cur = self._conn.execute(
            "SELECT frequency, next_delivery_at FROM subscriptions WHERE id = ?",
            (int(subscription_id),),
        )
        row = cur.fetchone()
        if row is None:
            # Silently no-op on unknown id — matches CartStore.remove_from_cart
            # behaviour on absent rows. Callers can use get_user_subscriptions
            # to verify existence.
            return
        days = FREQUENCY_DAYS[row["frequency"]]
        new_next = float(row["next_delivery_at"]) + days * _SECONDS_PER_DAY
        self._conn.execute(
            "UPDATE subscriptions SET next_delivery_at = ? WHERE id = ?",
            (new_next, int(subscription_id)),
        )
        self._conn.commit()

    # --------------------------------------------------------------
    # Queries
    # --------------------------------------------------------------
    def get_user_subscriptions(
        self,
        user_id: int,
        active_only: bool = True,
    ) -> list[dict]:
        """Return the user's subscriptions, soonest delivery first.

        Each dict has keys ``id``, ``product_id``, ``qty``, ``frequency``,
        ``next_delivery_at``, ``active``, ``created_at``, and a derived
        ``days_until_next`` (a float — negative when overdue).
        """
        if active_only:
            cur = self._conn.execute(
                """
                SELECT id, product_id, qty, frequency, next_delivery_at,
                       active, created_at
                FROM subscriptions
                WHERE user_id = ? AND active = 1
                ORDER BY next_delivery_at ASC
                """,
                (int(user_id),),
            )
        else:
            cur = self._conn.execute(
                """
                SELECT id, product_id, qty, frequency, next_delivery_at,
                       active, created_at
                FROM subscriptions
                WHERE user_id = ?
                ORDER BY next_delivery_at ASC
                """,
                (int(user_id),),
            )

        now = time.time()
        out: list[dict] = []
        for row in cur.fetchall():
            d = dict(row)
            d["days_until_next"] = (
                float(d["next_delivery_at"]) - now
            ) / _SECONDS_PER_DAY
            out.append(d)
        return out

    def get_due_subscriptions(
        self,
        as_of: float | None = None,
    ) -> list[dict]:
        """Return all *active* subscriptions whose ``next_delivery_at <=
        as_of`` (defaults to ``time.time()``).

        These are the rows the fulfillment cron should act on right now.
        Ordered by ``next_delivery_at ASC`` (oldest-due first), which
        keeps fulfillment deterministic.
        """
        if as_of is None:
            as_of = time.time()
        cur = self._conn.execute(
            """
            SELECT id, user_id, product_id, qty, frequency,
                   next_delivery_at, active, created_at
            FROM subscriptions
            WHERE active = 1 AND next_delivery_at <= ?
            ORDER BY next_delivery_at ASC
            """,
            (float(as_of),),
        )
        return [dict(row) for row in cur.fetchall()]

    def count_active(self, user_id: int) -> int:
        """Number of currently active subscriptions for a user."""
        cur = self._conn.execute(
            "SELECT COUNT(*) FROM subscriptions WHERE user_id = ? AND active = 1",
            (int(user_id),),
        )
        row = cur.fetchone()
        return int(row[0]) if row else 0

    def estimated_monthly_value(
        self,
        user_id: int,
        price_map: dict[int, float],
    ) -> float:
        """Estimated monthly recurring spend for the user's active subs.

        For each active subscription::

            monthly_freq = 30 / FREQUENCY_DAYS[frequency]
            value += monthly_freq * qty * price_map.get(product_id, 0)

        Products missing from ``price_map`` contribute zero (the catalog
        is large and the API can pass a sparse map). Result is rounded
        to two decimal places (currency-friendly).
        """
        cur = self._conn.execute(
            """
            SELECT product_id, qty, frequency
            FROM subscriptions
            WHERE user_id = ? AND active = 1
            """,
            (int(user_id),),
        )
        total = 0.0
        for row in cur.fetchall():
            days = FREQUENCY_DAYS[row["frequency"]]
            monthly_freq = 30.0 / days
            price = float(price_map.get(int(row["product_id"]), 0.0))
            total += monthly_freq * int(row["qty"]) * price
        return round(total, 2)

    # --------------------------------------------------------------
    # Lifecycle helpers
    # --------------------------------------------------------------
    def close(self) -> None:
        try:
            self._conn.close()
        except sqlite3.Error:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ----------------------------------------------------------------------
# Cron-style fulfillment helper
# ----------------------------------------------------------------------
def fulfill_due_subscriptions(
    store: "SubscriptionStore",
    cart_store: "CartStore",
    as_of: float | None = None,
) -> list[dict]:
    """Add every due subscription's product to its user's cart.

    This is the "Subscribe & Save delivery day" job in miniature. For
    each subscription returned by :meth:`SubscriptionStore.get_due_subscriptions`,
    we:

    1. Call ``cart_store.add_to_cart(user_id, product_id, qty)``.
    2. Advance the row's ``next_delivery_at`` by one cadence interval.

    Returns one summary dict per row fired, in the same order that
    :meth:`get_due_subscriptions` produced them (oldest-due first).
    The summary keys ``subscription_id``, ``user_id``, ``product_id``,
    ``qty`` are what the API needs to render "we just placed an
    auto-order for you" in the UI.
    """
    due = store.get_due_subscriptions(as_of=as_of)
    fired: list[dict] = []
    for sub in due:
        # Step 1: add the item to the user's cart. If it's already in
        # their cart, this just increments — which is the right
        # behaviour for repeat fulfillment.
        cart_store.add_to_cart(
            user_id=int(sub["user_id"]),
            product_id=int(sub["product_id"]),
            qty=int(sub["qty"]),
        )
        # Step 2: roll the next-delivery clock forward.
        store.advance_next_delivery(int(sub["id"]))

        fired.append(
            {
                "subscription_id": int(sub["id"]),
                "user_id": int(sub["user_id"]),
                "product_id": int(sub["product_id"]),
                "qty": int(sub["qty"]),
            }
        )
    return fired
