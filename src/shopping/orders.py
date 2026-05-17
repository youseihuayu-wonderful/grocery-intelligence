"""Order-history store backed by the real Instacart CSVs.

The Instacart dataset ships two large CSVs:

* ``orders.csv``                  ~3.4M rows (one row per order)
* ``order_products__prior.csv``   ~32M rows  (one row per item-in-order)

A naive load of both is roughly **3 GB** of RAM with default
``pandas`` dtypes. To keep the in-memory footprint manageable we

1. cast everything to the smallest integer dtype that fits
   (``uint32`` / ``uint8`` / ``float16``) and
2. optionally cap to the top-N most-active users (configurable via
   ``user_cap``; ``None`` loads everyone).

With ``user_cap=50_000`` the store keeps under ~1 GB resident; with
``user_cap=None`` you should expect 2–3 GB.

Trade-off documented: we pre-load the per-user index at startup so
that all subsequent ``get_user_orders``/``buy_again_top`` calls are
O(1) dict lookups. The alternative — lazy scan on each call — would
make the first call per user slow (whole-file scan) and require
keeping CSV file handles or doing per-call ``pyarrow`` filtering.

The store keeps only the columns the public API needs:

* From ``orders.csv``: ``order_id``, ``user_id``, ``order_number``,
  ``order_dow``, ``order_hour_of_day``, ``days_since_prior_order``,
  ``eval_set``.
* From ``order_products__prior.csv``: ``order_id``, ``product_id``,
  ``add_to_cart_order``, ``reordered``.
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# Default cap. ``None`` = load all users (will use ~2-3 GB RAM).
# Adjust depending on your machine. The real-data sanity script uses
# ``None`` so that high-volume users from the existing user profiles
# (e.g. ``user_id=206105``) are guaranteed to be present.
DEFAULT_USER_CAP: int | None = None


class OrderHistoryStore:
    """In-memory order history for the Instacart dataset.

    All queries are O(1) per user once the index is built. Construction
    is expensive (reads the CSVs once); subsequent queries are dict
    lookups.

    Footprint depends on ``user_cap``:

    +-----------------+---------+----------+
    | ``user_cap``    | Orders  | Approx.  |
    +=================+=========+==========+
    | ``5_000``       |   100K  |   ~50 MB |
    +-----------------+---------+----------+
    | ``50_000``      |  ~1.0M  |  ~500 MB |
    +-----------------+---------+----------+
    | ``None`` (all)  |   3.4M  |    ~2 GB |
    +-----------------+---------+----------+
    """

    def __init__(
        self,
        orders_csv: str | Path,
        order_products_csv: str | Path,
        user_cap: int | None = DEFAULT_USER_CAP,
        eval_sets: Iterable[str] = ("prior",),
        verbose: bool = False,
    ):
        """Build the in-memory index.

        Parameters
        ----------
        orders_csv:
            Path to ``orders.csv``. Required columns: ``order_id``,
            ``user_id``, ``order_number``, ``order_dow``,
            ``order_hour_of_day``, ``days_since_prior_order``,
            ``eval_set``.
        order_products_csv:
            Path to ``order_products__prior.csv``. Required columns:
            ``order_id``, ``product_id``, ``add_to_cart_order``,
            ``reordered``.
        user_cap:
            If set, only the top-N users (by order count) are kept.
            ``None`` loads everyone.
        eval_sets:
            Which eval-set partitions to retain (Instacart uses
            ``prior``, ``train``, ``test``). The companion
            ``order_products__prior.csv`` only has items for
            ``prior`` orders, so the default keeps just those.
        verbose:
            Print progress messages to stderr.
        """
        self._orders_csv = Path(orders_csv)
        self._order_products_csv = Path(order_products_csv)
        self._user_cap = user_cap
        self._verbose = verbose

        # Will be populated below.
        # user_id -> list of order_ids (newest first)
        self._user_orders: dict[int, list[int]] = {}
        # order_id -> dict(order_number, order_dow, order_hour_of_day,
        #                  days_since_prior_order, user_id)
        self._orders_meta: dict[int, dict] = {}
        # order_id -> list of (product_id, reordered, add_to_cart_order)
        self._order_items: dict[int, list[tuple[int, int, int]]] = {}
        # user_id -> Counter of product_id -> count
        self._user_product_counts: dict[int, Counter] = {}

        self._build_index(eval_sets=set(eval_sets))

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------
    def _log(self, msg: str) -> None:
        if self._verbose:
            print(msg, file=sys.stderr, flush=True)

    def _build_index(self, eval_sets: set[str]) -> None:
        self._log(f"[orders] reading {self._orders_csv}")
        # Compact dtypes — keeps RAM low. days_since_prior_order has
        # NaNs (first order of every user) so it stays float32.
        orders = pd.read_csv(
            self._orders_csv,
            dtype={
                "order_id": "uint32",
                "user_id": "uint32",
                "eval_set": "category",
                "order_number": "uint16",
                "order_dow": "uint8",
                "order_hour_of_day": "uint8",
                "days_since_prior_order": "float32",
            },
        )
        # Keep only requested eval sets (default: ``prior``).
        if eval_sets:
            orders = orders[orders["eval_set"].isin(eval_sets)]
        self._log(f"[orders] {len(orders):,} order rows after eval-set filter")

        # Optional cap to the top-N most-active users.
        if self._user_cap is not None:
            top_users = (
                orders["user_id"]
                .value_counts()
                .head(self._user_cap)
                .index
            )
            keep = orders["user_id"].isin(top_users)
            orders = orders[keep]
            self._log(
                f"[orders] capped to {self._user_cap:,} most-active "
                f"users; {len(orders):,} order rows remain"
            )

        # Build orders metadata and the per-user order list.
        # Sort by user_id then order_number descending so that the
        # per-user lists end up newest-first naturally.
        orders = orders.sort_values(
            ["user_id", "order_number"], ascending=[True, False]
        )

        order_ids_np = orders["order_id"].to_numpy(dtype=np.uint32)
        user_ids_np = orders["user_id"].to_numpy(dtype=np.uint32)
        order_numbers_np = orders["order_number"].to_numpy(dtype=np.uint16)
        order_dow_np = orders["order_dow"].to_numpy(dtype=np.uint8)
        order_hour_np = orders["order_hour_of_day"].to_numpy(dtype=np.uint8)
        days_since_np = orders["days_since_prior_order"].to_numpy(
            dtype=np.float32
        )

        user_orders: dict[int, list[int]] = defaultdict(list)
        for i in range(len(orders)):
            oid = int(order_ids_np[i])
            uid = int(user_ids_np[i])
            days = float(days_since_np[i])
            self._orders_meta[oid] = {
                "user_id": uid,
                "order_number": int(order_numbers_np[i]),
                "order_dow": int(order_dow_np[i]),
                "order_hour_of_day": int(order_hour_np[i]),
                # Instacart marks the FIRST order of each user with NaN.
                "days_since_prior_order": None if np.isnan(days) else days,
            }
            user_orders[uid].append(oid)
        self._user_orders = dict(user_orders)
        self._log(f"[orders] indexed {len(self._user_orders):,} users")

        # Free the orders DataFrame before reading the (much larger)
        # order_products CSV.
        kept_order_ids = set(self._orders_meta.keys())
        del orders
        del order_ids_np, user_ids_np, order_numbers_np
        del order_dow_np, order_hour_np, days_since_np

        # ---------- order_products__prior.csv ----------
        self._log(f"[orders] reading {self._order_products_csv}")
        op = pd.read_csv(
            self._order_products_csv,
            dtype={
                "order_id": "uint32",
                "product_id": "uint32",
                "add_to_cart_order": "uint16",
                "reordered": "uint8",
            },
        )
        self._log(f"[orders] {len(op):,} order-product rows read")

        if self._user_cap is not None:
            op = op[op["order_id"].isin(kept_order_ids)]
            self._log(
                f"[orders] filtered order-products to "
                f"{len(op):,} rows for kept orders"
            )

        # Build order_id -> list of (product_id, reordered, add_to_cart_order)
        # We want each list sorted by add_to_cart_order ascending so the
        # reorder_items convenience method returns items "in original
        # cart order".
        op = op.sort_values(["order_id", "add_to_cart_order"])
        order_ids_np = op["order_id"].to_numpy(dtype=np.uint32)
        product_ids_np = op["product_id"].to_numpy(dtype=np.uint32)
        reordered_np = op["reordered"].to_numpy(dtype=np.uint8)
        add_to_cart_np = op["add_to_cart_order"].to_numpy(dtype=np.uint16)

        order_items: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
        for i in range(len(op)):
            oid = int(order_ids_np[i])
            order_items[oid].append(
                (
                    int(product_ids_np[i]),
                    int(reordered_np[i]),
                    int(add_to_cart_np[i]),
                )
            )
        self._order_items = dict(order_items)
        self._log(f"[orders] indexed items for {len(self._order_items):,} orders")

        # Build user -> product counter for buy_again_top.
        user_product_counts: dict[int, Counter] = defaultdict(Counter)
        for uid, oids in self._user_orders.items():
            counter = user_product_counts[uid]
            for oid in oids:
                for pid, _reordered, _atc in self._order_items.get(oid, ()):
                    counter[pid] += 1
        self._user_product_counts = dict(user_product_counts)
        self._log(
            f"[orders] computed buy-again counters for "
            f"{len(self._user_product_counts):,} users"
        )

        del op, order_ids_np, product_ids_np, reordered_np, add_to_cart_np

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_user_orders(self, user_id: int) -> list[dict]:
        """Return the user's prior orders, newest first.

        See module docstring for the dict shape.
        """
        uid = int(user_id)
        order_ids = self._user_orders.get(uid)
        if not order_ids:
            return []
        return [self._build_order_dict(oid) for oid in order_ids]

    def get_order(self, order_id: int) -> dict | None:
        """Return a single order's dict (same shape as get_user_orders entries)."""
        oid = int(order_id)
        if oid not in self._orders_meta:
            return None
        return self._build_order_dict(oid)

    def reorder_items(self, order_id: int) -> list[int]:
        """Return just the product_ids from an order, in original cart order."""
        items = self._order_items.get(int(order_id))
        if not items:
            return []
        # Items are already sorted by add_to_cart_order during build.
        return [pid for pid, _r, _atc in items]

    def buy_again_top(self, user_id: int, top_k: int = 20) -> list[int]:
        """Most-frequently-purchased product_ids for the user, desc by count."""
        counter = self._user_product_counts.get(int(user_id))
        if not counter:
            return []
        return [pid for pid, _ in counter.most_common(top_k)]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_order_dict(self, order_id: int) -> dict:
        meta = self._orders_meta[order_id]
        items = self._order_items.get(order_id, [])
        return {
            "order_id": int(order_id),
            "order_number": meta["order_number"],
            "order_dow": meta["order_dow"],
            "order_hour_of_day": meta["order_hour_of_day"],
            "days_since_prior_order": meta["days_since_prior_order"],
            "items": [
                {"product_id": pid, "reordered": bool(reordered)}
                for pid, reordered, _atc in items
            ],
            "n_items": len(items),
        }

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def n_users(self) -> int:
        return len(self._user_orders)

    @property
    def n_orders(self) -> int:
        return len(self._orders_meta)

    @property
    def n_items_total(self) -> int:
        return sum(len(v) for v in self._order_items.values())
