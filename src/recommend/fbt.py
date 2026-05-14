"""Frequently Bought Together (FBT) recommendation engine.

Item-item co-purchase recommendations built from order history using
market basket analysis. Scores partner items with lift on a sparse
co-occurrence matrix derived from real Instacart 'prior' baskets.

Lift interpretation:
    lift(A, B) = P(B | A) / P(B)
               = (count(A, B) * total_baskets) / (count(A) * count(B))

A lift > 1 means A and B co-occur more often than independent chance
would predict; the higher the lift, the stronger the association.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Iterable

import pandas as pd


class FrequentlyBoughtTogether:
    """Item-item co-purchase recommendations from order history.

    Uses lift-based scoring on a co-occurrence matrix built from
    real Instacart 'prior' order data (3.4M baskets).
    """

    def __init__(
        self,
        related_map: dict[int, list[tuple[int, float]]] | None = None,
    ):
        """Initialize. Use .fit() or .load() to populate."""
        self.related_map: dict[int, list[tuple[int, float]]] = related_map or {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(
        self,
        order_products: pd.DataFrame,
        min_support: int = 50,
        top_k: int = 20,
        candidate_ids: Iterable[int] | None = None,
        progress_every: int = 100_000,
    ) -> "FrequentlyBoughtTogether":
        """Compute item-item co-occurrence with lift scoring.

        Args:
            order_products: DataFrame with columns 'order_id', 'product_id'.
                One row per item-in-basket. Items in the same order are
                "bought together".
            min_support: Minimum number of co-occurrences for a pair to
                be considered. Filters out noise.
            top_k: Keep only the top-K related items per product.
            candidate_ids: Optional iterable of product_ids to restrict
                the model to. Massively reduces work for large catalogs.
            progress_every: Print progress every N processed orders.

        Returns:
            self, for chaining.
        """
        if "order_id" not in order_products.columns or "product_id" not in order_products.columns:
            raise ValueError("order_products must have columns 'order_id' and 'product_id'")

        # ---- 1. Optional candidate filter --------------------------------
        if candidate_ids is not None:
            candidate_set = set(int(pid) for pid in candidate_ids)
            filtered = order_products[order_products["product_id"].isin(candidate_set)]
        else:
            filtered = order_products

        # Ensure we only iterate orders that still have >= 2 items
        # after candidate filtering (otherwise no pairs can be formed).
        order_sizes = filtered.groupby("order_id")["product_id"].size()
        good_orders = order_sizes[order_sizes >= 2].index
        filtered = filtered[filtered["order_id"].isin(good_orders)]

        # ---- 2. Per-item counts (for the lift denominator) --------------
        item_counts: Counter[int] = Counter(filtered["product_id"].astype(int).tolist())
        total_baskets = int(filtered["order_id"].nunique())

        # ---- 3. Pair counts via itertools.combinations ------------------
        pair_counts: defaultdict[tuple[int, int], int] = defaultdict(int)
        processed = 0

        # Sort by order_id once so groupby can stream linearly.
        # filtered is small after the candidate filter so .sort_values is fine.
        sorted_df = filtered.sort_values("order_id")
        for _, group in sorted_df.groupby("order_id", sort=False):
            # Deduplicate items within the basket so reorder rows
            # don't double-count a pair from a single order.
            items = sorted({int(pid) for pid in group["product_id"]})
            if len(items) < 2:
                processed += 1
                continue
            for a, b in combinations(items, 2):
                pair_counts[(a, b)] += 1
            processed += 1
            if progress_every and processed % progress_every == 0:
                print(
                    f"  processed {processed:,} orders, "
                    f"{len(pair_counts):,} unique pairs so far"
                )

        # ---- 4. Filter by min_support, compute lift ---------------------
        # Build raw partners dict mapping product_id -> list[(partner, lift)]
        raw: defaultdict[int, list[tuple[int, float]]] = defaultdict(list)
        for (a, b), c in pair_counts.items():
            if c < min_support:
                continue
            ca = item_counts[a]
            cb = item_counts[b]
            if ca == 0 or cb == 0:
                continue
            lift = (c * total_baskets) / (ca * cb)
            raw[a].append((b, lift))
            raw[b].append((a, lift))

        # ---- 5. Top-K per item by lift ---------------------------------
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
        """Return up to top_k (product_id, lift_score) pairs for the input.

        Returns empty list if product has no entries in the model
        (low-volume product or unknown id).
        """
        partners = self.related_map.get(int(product_id), [])
        return partners[:top_k]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Serialize the related_map to disk.

        Uses parquet for the .parquet suffix (long-form, inspectable
        in pandas), pickle otherwise.
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

    def load(self, path: str | Path) -> "FrequentlyBoughtTogether":
        """Load a previously-saved model. Returns self for chaining."""
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
            self.related_map = dict(related_map)
        else:
            import pickle

            with open(path, "rb") as fh:
                self.related_map = pickle.load(fh)
        return self

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.related_map)

    def __contains__(self, pid: int) -> bool:
        return int(pid) in self.related_map
