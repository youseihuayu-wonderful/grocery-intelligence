"""Tests for the Frequently Bought Together (FBT) recommender."""

from __future__ import annotations

import pandas as pd
import pytest

from src.recommend.fbt import FrequentlyBoughtTogether


# Product IDs used in the synthetic dataset.
PID_A = 1
PID_B = 2
PID_C = 3
PID_D = 4
PID_E = 5

# Hand-crafted: A and B always appear together. C is independent
# (appears in some baskets that contain A but never enough to dominate),
# D and E are noise that should drop below min_support.
SYNTHETIC_ORDERS: list[tuple[int, int]] = []
# 10 orders containing both A and B
for order_id in range(1, 11):
    SYNTHETIC_ORDERS.append((order_id, PID_A))
    SYNTHETIC_ORDERS.append((order_id, PID_B))
# Order 11 only has C and D (so C has support outside A-baskets)
SYNTHETIC_ORDERS.append((11, PID_C))
SYNTHETIC_ORDERS.append((11, PID_D))
# Order 12 only has C and E
SYNTHETIC_ORDERS.append((12, PID_C))
SYNTHETIC_ORDERS.append((12, PID_E))
# Order 13 has A, B, and C (one weak co-occurrence with C)
SYNTHETIC_ORDERS.append((13, PID_A))
SYNTHETIC_ORDERS.append((13, PID_B))
SYNTHETIC_ORDERS.append((13, PID_C))


@pytest.fixture
def order_products() -> pd.DataFrame:
    return pd.DataFrame(SYNTHETIC_ORDERS, columns=["order_id", "product_id"])


def test_a_and_b_are_related(order_products: pd.DataFrame) -> None:
    fbt = FrequentlyBoughtTogether().fit(
        order_products, min_support=2, top_k=10, progress_every=0
    )
    related_for_a = fbt.get_related(PID_A)
    assert related_for_a, "A should have related items"
    partners = {pid for pid, _ in related_for_a}
    assert PID_B in partners
    # Lift for (A, B) should be high — they always co-occur
    lift_b = dict(related_for_a)[PID_B]
    assert lift_b > 1.0


def test_lift_is_symmetric(order_products: pd.DataFrame) -> None:
    fbt = FrequentlyBoughtTogether().fit(
        order_products, min_support=2, top_k=10, progress_every=0
    )
    a_partners = dict(fbt.get_related(PID_A))
    b_partners = dict(fbt.get_related(PID_B))
    assert PID_B in a_partners
    assert PID_A in b_partners
    # Exactly the same lift score (since the formula is symmetric)
    assert a_partners[PID_B] == pytest.approx(b_partners[PID_A])


def test_unknown_product_returns_empty(order_products: pd.DataFrame) -> None:
    fbt = FrequentlyBoughtTogether().fit(
        order_products, min_support=2, top_k=10, progress_every=0
    )
    assert fbt.get_related(999_999) == []


def test_save_load_roundtrip_parquet(
    order_products: pd.DataFrame, tmp_path
) -> None:
    fbt = FrequentlyBoughtTogether().fit(
        order_products, min_support=2, top_k=10, progress_every=0
    )
    out = tmp_path / "fbt.parquet"
    fbt.save(out)
    assert out.exists()

    loaded = FrequentlyBoughtTogether().load(out)
    assert dict(loaded.get_related(PID_A)) == dict(fbt.get_related(PID_A))
    assert dict(loaded.get_related(PID_B)) == dict(fbt.get_related(PID_B))


def test_save_load_roundtrip_pickle(
    order_products: pd.DataFrame, tmp_path
) -> None:
    fbt = FrequentlyBoughtTogether().fit(
        order_products, min_support=2, top_k=10, progress_every=0
    )
    out = tmp_path / "fbt.pkl"
    fbt.save(out)
    assert out.exists()

    loaded = FrequentlyBoughtTogether().load(out)
    assert loaded.related_map == fbt.related_map


def test_min_support_filters_low_count_pairs(
    order_products: pd.DataFrame,
) -> None:
    """With min_support=5, (C, D) and (C, E) each have count 1 — filtered out.
    Only (A, B) has enough support to survive."""
    fbt = FrequentlyBoughtTogether().fit(
        order_products, min_support=5, top_k=10, progress_every=0
    )
    # D and E should not appear anywhere (they had only single-pair support).
    assert fbt.get_related(PID_D) == []
    assert fbt.get_related(PID_E) == []

    # A's related should at most contain B — not the weak (A, C) pair.
    a_partners = {pid for pid, _ in fbt.get_related(PID_A)}
    assert PID_B in a_partners
    assert PID_C not in a_partners
