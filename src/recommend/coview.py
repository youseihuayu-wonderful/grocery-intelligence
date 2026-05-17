"""Customers Also Viewed recommendation engine.

User-level item-item co-occurrence model. Conceptually different from
:class:`src.recommend.fbt.FrequentlyBoughtTogether`:

* **FBT**  groups items by ``order_id`` -- "purchased in the same basket".
* **Coview** groups items by ``user_id`` -- "touched by the same user,
  across any of their sessions or orders".

This captures "people like this also looked at" carousels: even if the
user bought the items on different days, if they keep showing up in the
same user's history, the items have an affinity worth surfacing.

Lift scoring
------------
For two products ``A`` and ``B`` with ``count(A, B)`` users touching
both and ``count(A)``, ``count(B)`` users touching each individually::

    lift(A, B) = count(A, B) * total_users / (count(A) * count(B))

A lift > 1 means the pair co-occurs in user histories more often than
random chance would predict. The model keeps the top-K partners per
item by lift, after filtering pairs below ``min_support``.

The behavior log is read through whatever event types are configured
(``view``, ``click``, ``purchase``). The seeded log currently has only
purchase events, but once real view-event traffic accumulates the same
model trains on those without any code change.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from src.recommend.behavior import BehaviorLogger


class CustomersAlsoViewed:
    """User-level item-item co-occurrence with lift scoring.

    Differs from FBT in that it pairs items co-occurring in the SAME USER's
    history, across different sessions/orders, not just same-basket.
    Suitable for 'Customers also viewed' carousels.
    """

    def __init__(
        self,
        related_map: dict[int, list[tuple[int, float]]] | None = None,
    ):
        """Initialize. Use :meth:`fit` or :meth:`load` to populate."""
        self.related_map: dict[int, list[tuple[int, float]]] = related_map or {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(
        self,
        behavior_logger: "BehaviorLogger",
        event_types: tuple[str, ...] = ("view", "click", "purchase"),
        min_support: int = 30,
        top_k: int = 20,
    ) -> "CustomersAlsoViewed":
        """Build item-item co-occurrence at the USER level.

        Args:
            behavior_logger: A :class:`BehaviorLogger` instance whose
                events table will be scanned for the configured
                ``event_types``.
            event_types: Tuple of event-type strings to include.
                Events of other types are ignored. Anonymous events
                (``user_id IS NULL``) are always ignored, since they
                can't be attributed to a user.
            min_support: Minimum number of distinct users co-touching
                a pair for it to survive into the model. Filters noise.
            top_k: Keep at most this many partners per product, ranked
                by descending lift.

        Returns:
            self, for chaining.

        Algorithm
        ---------
        1. Stream the events table once, restricted to the chosen
           event types and to rows with a non-null ``user_id``,
           sorted by ``user_id``.
        2. For each user, build the set of distinct product_ids they
           touched. Generate all unordered, distinct pairs.
        3. Count co-occurrences across users; count per-item users.
        4. Filter pairs below ``min_support``.
        5. Score with the lift formula
           ``count(A,B) * total_users / (count(A) * count(B))``.
        6. Keep the top-K partners per item by lift, sorted desc.
        """
        # Pull only the columns we need, filtered server-side. Streaming
        # via the underlying connection keeps memory bounded even for
        # multi-million-row logs. We rely on the documented private
        # ``_conn`` attribute of BehaviorLogger -- same pattern that
        # ``BehaviorLogger.get_feature_matrix`` uses internally.
        if not event_types:
            raise ValueError("event_types must contain at least one event type")

        placeholders = ",".join("?" for _ in event_types)
        sql = (
            "SELECT user_id, product_id FROM events "
            f"WHERE user_id IS NOT NULL AND event_type IN ({placeholders}) "
            "ORDER BY user_id"
        )

        # ---- 1. Group rows by user_id while streaming -----------------
        pair_counts: defaultdict[tuple[int, int], int] = defaultdict(int)
        item_user_counts: Counter[int] = Counter()
        total_users = 0

        cur = behavior_logger._conn.execute(sql, tuple(event_types))
        current_user: int | None = None
        current_items: set[int] = set()

        def _flush(items: set[int]) -> None:
            nonlocal total_users
            if not items:
                return
            total_users += 1
            for pid in items:
                item_user_counts[pid] += 1
            if len(items) >= 2:
                # combinations() over a sorted list gives canonical (a < b)
                # pairs so (a, b) and (b, a) are not double-counted.
                ordered = sorted(items)
                for a, b in combinations(ordered, 2):
                    pair_counts[(a, b)] += 1

        for user_id, product_id in cur:
            if current_user is None:
                current_user = int(user_id)
                current_items = {int(product_id)}
                continue
            if int(user_id) != current_user:
                _flush(current_items)
                current_user = int(user_id)
                current_items = {int(product_id)}
            else:
                current_items.add(int(product_id))
        # Flush the final group.
        if current_user is not None:
            _flush(current_items)

        # ---- 2. Filter by min_support, compute lift -------------------
        raw: defaultdict[int, list[tuple[int, float]]] = defaultdict(list)
        for (a, b), c in pair_counts.items():
            if c < min_support:
                continue
            ca = item_user_counts[a]
            cb = item_user_counts[b]
            if ca == 0 or cb == 0:
                continue
            lift = (c * total_users) / (ca * cb)
            raw[a].append((b, lift))
            raw[b].append((a, lift))

        # ---- 3. Top-K per item by lift --------------------------------
        related_map: dict[int, list[tuple[int, float]]] = {}
        for pid, partners in raw.items():
            partners.sort(key=lambda x: x[1], reverse=True)
            related_map[pid] = partners[:top_k]

        self.related_map = related_map
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def get_related(
        self, product_id: int, top_k: int = 10
    ) -> list[tuple[int, float]]:
        """Return up to ``top_k`` ``(partner_id, lift_score)`` pairs.

        Pairs are sorted by descending lift. Returns an empty list for
        an unknown product id or one with no partners that survived
        the support filter.
        """
        partners = self.related_map.get(int(product_id), [])
        return partners[:top_k]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Serialize the related_map to disk.

        Uses parquet for ``.parquet`` suffix (long-form, inspectable in
        pandas), pickle otherwise.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix == ".parquet":
            rows: list[tuple[int, int, float, int]] = []
            for pid, partners in self.related_map.items():
                for rank, (partner, lift) in enumerate(partners):
                    rows.append((int(pid), int(partner), float(lift), int(rank)))
            df = pd.DataFrame(
                rows,
                columns=["product_id", "partner_id", "lift", "rank"],
            )
            df.to_parquet(path, index=False)
        else:
            import pickle

            with open(path, "wb") as fh:
                pickle.dump(self.related_map, fh)

    @classmethod
    def load(cls, path: str | Path) -> "CustomersAlsoViewed":
        """Load a previously-saved model from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
            df = df.sort_values(["product_id", "rank"])
            related_map: dict[int, list[tuple[int, float]]] = defaultdict(list)
            for pid, partner, lift in zip(
                df["product_id"].astype(int).tolist(),
                df["partner_id"].astype(int).tolist(),
                df["lift"].astype(float).tolist(),
            ):
                related_map[pid].append((partner, lift))
            return cls(related_map=dict(related_map))
        else:
            import pickle

            with open(path, "rb") as fh:
                data = pickle.load(fh)
            return cls(related_map=data)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.related_map)

    def __contains__(self, pid: int) -> bool:
        return int(pid) in self.related_map
