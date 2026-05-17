"""SQLite-backed shopping cart and wishlist persistence.

A single SQLite database holds two tables (``cart_items`` and
``wishlist_items``) keyed by ``(user_id, product_id)``. Both tables
share the schema::

    (user_id INTEGER, product_id INTEGER, qty INTEGER DEFAULT 1,
     added_at REAL NOT NULL)

with an index on ``user_id`` for fast per-user lookups.

The class is safe to use as a context manager — see ``__enter__`` /
``__exit__`` — and uses ``stdlib sqlite3`` only (no external deps).
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Iterable

# Default location, relative to the project root. Created on demand.
DEFAULT_CART_DB = Path("data/processed/shopping.db")


class CartStore:
    """Cart and wishlist persistence on top of SQLite.

    Single DB, two tables: ``cart_items`` and ``wishlist_items``.

    Schema for both::

        (user_id INTEGER, product_id INTEGER, qty INTEGER DEFAULT 1,
         added_at REAL NOT NULL)

    Indexed on ``user_id``. The pair ``(user_id, product_id)`` is the
    primary key in both tables, so a re-add of the same product is an
    increment (cart) or a no-op (wishlist).
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
            to :data:`DEFAULT_CART_DB`. The parent directory is created
            if it does not exist. Pass ``":memory:"`` for tests.
        """
        if db_path is None:
            db_path = DEFAULT_CART_DB
        # ``:memory:`` is a magic SQLite path; do not touch the FS for it.
        path_str = str(db_path)
        if path_str != ":memory:":
            parent = Path(path_str).expanduser().resolve().parent
            parent.mkdir(parents=True, exist_ok=True)

        # ``check_same_thread=False`` keeps things flexible if the caller
        # accesses the store from a worker thread (e.g. FastAPI).
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
            CREATE TABLE IF NOT EXISTS cart_items (
                user_id     INTEGER NOT NULL,
                product_id  INTEGER NOT NULL,
                qty         INTEGER NOT NULL DEFAULT 1,
                added_at    REAL    NOT NULL,
                PRIMARY KEY (user_id, product_id)
            );
            CREATE INDEX IF NOT EXISTS idx_cart_user
                ON cart_items(user_id);

            CREATE TABLE IF NOT EXISTS wishlist_items (
                user_id     INTEGER NOT NULL,
                product_id  INTEGER NOT NULL,
                qty         INTEGER NOT NULL DEFAULT 1,
                added_at    REAL    NOT NULL,
                PRIMARY KEY (user_id, product_id)
            );
            CREATE INDEX IF NOT EXISTS idx_wishlist_user
                ON wishlist_items(user_id);
            """
        )
        self._conn.commit()

    # --------------------------------------------------------------
    # Cart
    # --------------------------------------------------------------
    def add_to_cart(
        self, user_id: int, product_id: int, qty: int = 1
    ) -> None:
        """Add a product to the user's cart.

        If the product is already in the cart, increment ``qty`` by the
        given amount. ``qty`` must be a positive integer.
        """
        if qty <= 0:
            raise ValueError("qty must be > 0; use update_cart_qty/remove_from_cart for removal")
        now = time.time()
        # UPSERT: on conflict, increment qty by the new value (do NOT
        # overwrite). Leave added_at as the original first-added time
        # so "most recent added first" surfaces genuinely new additions.
        self._conn.execute(
            """
            INSERT INTO cart_items (user_id, product_id, qty, added_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id, product_id)
            DO UPDATE SET qty = qty + excluded.qty
            """,
            (int(user_id), int(product_id), int(qty), now),
        )
        self._conn.commit()

    def remove_from_cart(self, user_id: int, product_id: int) -> None:
        """Remove the row entirely (regardless of qty)."""
        self._conn.execute(
            "DELETE FROM cart_items WHERE user_id = ? AND product_id = ?",
            (int(user_id), int(product_id)),
        )
        self._conn.commit()

    def update_cart_qty(
        self, user_id: int, product_id: int, qty: int
    ) -> None:
        """Set ``qty`` to the given absolute value.

        If ``qty <= 0`` the row is removed entirely.
        """
        if qty <= 0:
            self.remove_from_cart(user_id, product_id)
            return
        now = time.time()
        # Upsert: if the row doesn't exist yet, create it; if it does,
        # overwrite qty (NOT increment — that's add_to_cart's job).
        self._conn.execute(
            """
            INSERT INTO cart_items (user_id, product_id, qty, added_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id, product_id)
            DO UPDATE SET qty = excluded.qty
            """,
            (int(user_id), int(product_id), int(qty), now),
        )
        self._conn.commit()

    def get_cart(self, user_id: int) -> list[dict]:
        """Return cart items as ``[{product_id, qty, added_at}, ...]``.

        Sorted by ``added_at`` descending (most recent first).
        """
        cur = self._conn.execute(
            """
            SELECT product_id, qty, added_at
            FROM cart_items
            WHERE user_id = ?
            ORDER BY added_at DESC
            """,
            (int(user_id),),
        )
        return [dict(row) for row in cur.fetchall()]

    def clear_cart(self, user_id: int) -> None:
        """Empty the user's cart."""
        self._conn.execute(
            "DELETE FROM cart_items WHERE user_id = ?", (int(user_id),)
        )
        self._conn.commit()

    def cart_count(self, user_id: int) -> int:
        """Total number of *distinct* products in the user's cart."""
        cur = self._conn.execute(
            "SELECT COUNT(*) FROM cart_items WHERE user_id = ?",
            (int(user_id),),
        )
        row = cur.fetchone()
        return int(row[0]) if row else 0

    # --------------------------------------------------------------
    # Wishlist
    # --------------------------------------------------------------
    def add_to_wishlist(self, user_id: int, product_id: int) -> None:
        """Add a product to the user's wishlist. No-op if already there."""
        now = time.time()
        self._conn.execute(
            """
            INSERT INTO wishlist_items (user_id, product_id, qty, added_at)
            VALUES (?, ?, 1, ?)
            ON CONFLICT(user_id, product_id) DO NOTHING
            """,
            (int(user_id), int(product_id), now),
        )
        self._conn.commit()

    def remove_from_wishlist(self, user_id: int, product_id: int) -> None:
        """Remove the product from the wishlist."""
        self._conn.execute(
            "DELETE FROM wishlist_items WHERE user_id = ? AND product_id = ?",
            (int(user_id), int(product_id)),
        )
        self._conn.commit()

    def get_wishlist(self, user_id: int) -> list[dict]:
        """Return wishlist items as ``[{product_id, added_at}, ...]``.

        Sorted by ``added_at`` descending (most recent first).
        """
        cur = self._conn.execute(
            """
            SELECT product_id, added_at
            FROM wishlist_items
            WHERE user_id = ?
            ORDER BY added_at DESC
            """,
            (int(user_id),),
        )
        return [dict(row) for row in cur.fetchall()]

    def is_in_wishlist(self, user_id: int, product_id: int) -> bool:
        cur = self._conn.execute(
            "SELECT 1 FROM wishlist_items WHERE user_id = ? AND product_id = ? LIMIT 1",
            (int(user_id), int(product_id)),
        )
        return cur.fetchone() is not None

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
