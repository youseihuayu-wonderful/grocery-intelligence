"""Tests for the algorithmic badge computation module.

These tests use the real product catalog so they exercise the same data
distribution the production module will see.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.recommend.badges import BADGE_LABELS, compute_badges

DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture(scope="module")
def catalog() -> pd.DataFrame:
    """Load the real product catalog."""
    path = DATA_DIR / "processed" / "product_catalog.parquet"
    if not path.exists():
        pytest.skip("Product catalog not found. Run data pipeline first.")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def badges(catalog: pd.DataFrame) -> dict[int, list[str]]:
    """Compute badges once and reuse across assertions."""
    return compute_badges(catalog)


def test_returns_entry_for_nearly_every_product(catalog, badges):
    """Every product (give or take a tiny margin) should appear in the result."""
    assert isinstance(badges, dict)
    assert len(badges) >= 49_000
    # And the keys should be a subset of the real product_ids.
    catalog_ids = set(catalog["product_id"].tolist())
    assert set(badges.keys()).issubset(catalog_ids)


def test_at_least_100_bestsellers(badges):
    """Top 5% of ~50k products should yield well over 100 bestsellers."""
    bestseller_count = sum(1 for tags in badges.values() if "bestseller" in tags)
    assert bestseller_count >= 100, (
        f"Expected at least 100 bestsellers, got {bestseller_count}"
    )


def test_bestseller_threshold_is_at_least_90th_percentile(catalog, badges):
    """The minimum order_count among bestsellers must sit at or above the
    90th percentile of the catalog — guarding against the threshold drifting
    too low if quantile logic regresses."""
    bestseller_ids = {pid for pid, tags in badges.items() if "bestseller" in tags}
    bestseller_rows = catalog[catalog["product_id"].isin(bestseller_ids)]
    min_bestseller_orders = bestseller_rows["order_count"].min()
    p90 = catalog["order_count"].quantile(0.90)
    assert min_bestseller_orders >= p90, (
        f"Bestseller floor ({min_bestseller_orders}) is below the 90th "
        f"percentile of order_count ({p90})"
    )


def test_badge_labels_cover_all_badge_ids(badges):
    """Every badge string emitted by compute_badges must have a label."""
    emitted = {tag for tags in badges.values() for tag in tags}
    missing = emitted - set(BADGE_LABELS.keys())
    assert not missing, f"Badge ids without labels: {missing}"


def test_low_sugar_never_assigned_when_sugar_is_missing(catalog, badges):
    """A product missing sugar_100g must NOT receive low-sugar."""
    no_sugar_data = catalog[catalog["sugar_100g"].isna()]["product_id"].tolist()
    offenders = [pid for pid in no_sugar_data if "low-sugar" in badges.get(pid, [])]
    assert not offenders, (
        f"{len(offenders)} products without sugar data were tagged low-sugar"
    )


def test_bestseller_and_popular_are_mutually_exclusive(badges):
    """A product is either bestseller or popular, never both."""
    conflicts = [
        pid
        for pid, tags in badges.items()
        if "bestseller" in tags and "popular" in tags
    ]
    assert not conflicts, f"{len(conflicts)} products tagged both bestseller and popular"
