"""Tests for the Customers Also Viewed recommender.

Uses an in-memory BehaviorLogger seeded with a small synthetic event
log so every test runs in milliseconds and never touches the real
production database.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.recommend.behavior import BehaviorLogger
from src.recommend.coview import CustomersAlsoViewed


# Product IDs used in the synthetic event log.
PID_A = 1
PID_B = 2
PID_C = 3
PID_D = 4
PID_E = 5


# Hand-crafted: users 1..5 each "touch" products A and B (so the pair
# (A, B) has the maximum possible co-occurrence). User 6 touches C+D
# alone and user 7 touches C+E alone (so {C,D} and {C,E} each have
# support 1 -- below the test's min_support threshold).
#
# Events are split across event types and across multiple timestamps
# per user, to verify that:
#   1) We pair items at the USER level (not basket/order),
#   2) The model handles mixed event_types correctly,
#   3) Duplicate events for the same (user, product) are deduplicated
#      into a single "touch" per user.
SYNTHETIC_EVENTS: list[dict] = []


def _seed_pair_events(events: list[dict]) -> None:
    """Populate the SYNTHETIC_EVENTS module list."""
    # Users 1..5: each user purchases A and views B (different events,
    # same user -> they count as a co-occurring pair at user-level).
    for user_id in range(1, 6):
        events.append({
            "product_id": PID_A, "event_type": "purchase",
            "user_id": user_id, "timestamp": 1_700_000_000.0 + user_id,
        })
        events.append({
            "product_id": PID_B, "event_type": "view",
            "user_id": user_id, "timestamp": 1_700_000_100.0 + user_id,
        })
    # Duplicate event for user 1 -- should NOT inflate the pair count.
    events.append({
        "product_id": PID_A, "event_type": "click",
        "user_id": 1, "timestamp": 1_700_000_200.0,
    })

    # User 6 touches only C and D.
    events.append({
        "product_id": PID_C, "event_type": "purchase",
        "user_id": 6, "timestamp": 1_700_000_300.0,
    })
    events.append({
        "product_id": PID_D, "event_type": "purchase",
        "user_id": 6, "timestamp": 1_700_000_301.0,
    })

    # User 7 touches only C and E.
    events.append({
        "product_id": PID_C, "event_type": "purchase",
        "user_id": 7, "timestamp": 1_700_000_400.0,
    })
    events.append({
        "product_id": PID_E, "event_type": "purchase",
        "user_id": 7, "timestamp": 1_700_000_401.0,
    })


_seed_pair_events(SYNTHETIC_EVENTS)


@pytest.fixture
def logger() -> BehaviorLogger:
    """Fresh in-memory logger seeded with the synthetic events."""
    log = BehaviorLogger(":memory:")
    log.log_events_bulk(SYNTHETIC_EVENTS)
    yield log
    log.close()


def test_a_and_b_are_related(logger: BehaviorLogger) -> None:
    """A and B are touched by the same 5 users -> top related pair."""
    coview = CustomersAlsoViewed().fit(
        logger,
        event_types=("view", "click", "purchase"),
        min_support=2,
        top_k=10,
    )
    related_for_a = coview.get_related(PID_A)
    assert related_for_a, "A should have related items"
    partners = {pid for pid, _ in related_for_a}
    assert PID_B in partners
    lift_b = dict(related_for_a)[PID_B]
    assert lift_b > 1.0


def test_lift_is_symmetric(logger: BehaviorLogger) -> None:
    """lift(A, B) == lift(B, A) -- model writes the same score both ways."""
    coview = CustomersAlsoViewed().fit(
        logger,
        event_types=("view", "click", "purchase"),
        min_support=2,
        top_k=10,
    )
    a_partners = dict(coview.get_related(PID_A))
    b_partners = dict(coview.get_related(PID_B))
    assert PID_B in a_partners
    assert PID_A in b_partners
    assert a_partners[PID_B] == pytest.approx(b_partners[PID_A])


def test_min_support_filters_pairs(logger: BehaviorLogger) -> None:
    """With min_support=3, the (C, D) and (C, E) pairs (count 1 each) drop out.

    Only the (A, B) pair (count 5) survives.
    """
    coview = CustomersAlsoViewed().fit(
        logger,
        event_types=("view", "click", "purchase"),
        min_support=3,
        top_k=10,
    )
    assert coview.get_related(PID_D) == []
    assert coview.get_related(PID_E) == []
    a_partners = {pid for pid, _ in coview.get_related(PID_A)}
    assert PID_B in a_partners
    assert PID_C not in a_partners


def test_unknown_product_returns_empty(logger: BehaviorLogger) -> None:
    coview = CustomersAlsoViewed().fit(
        logger,
        event_types=("view", "click", "purchase"),
        min_support=2,
        top_k=10,
    )
    assert coview.get_related(999_999) == []


def test_save_load_roundtrip_parquet(
    logger: BehaviorLogger, tmp_path: Path,
) -> None:
    coview = CustomersAlsoViewed().fit(
        logger,
        event_types=("view", "click", "purchase"),
        min_support=2,
        top_k=10,
    )
    out = tmp_path / "coview.parquet"
    coview.save(out)
    assert out.exists()

    loaded = CustomersAlsoViewed.load(out)
    assert dict(loaded.get_related(PID_A)) == dict(coview.get_related(PID_A))
    assert dict(loaded.get_related(PID_B)) == dict(coview.get_related(PID_B))


def test_save_load_roundtrip_pickle(
    logger: BehaviorLogger, tmp_path: Path,
) -> None:
    coview = CustomersAlsoViewed().fit(
        logger,
        event_types=("view", "click", "purchase"),
        min_support=2,
        top_k=10,
    )
    out = tmp_path / "coview.pkl"
    coview.save(out)
    assert out.exists()

    loaded = CustomersAlsoViewed.load(out)
    assert loaded.related_map == coview.related_map


def test_user_level_pairing_across_events(logger: BehaviorLogger) -> None:
    """The same user_id touching A (purchase) and B (view) should count
    as a single user-level co-occurrence, even though they are different
    event types and different timestamps."""
    coview = CustomersAlsoViewed().fit(
        logger,
        event_types=("view", "click", "purchase"),
        min_support=2,
        top_k=10,
    )
    # A and B must be partners despite never sharing an event_type.
    assert PID_B in dict(coview.get_related(PID_A))


def test_event_type_filter_excludes_other_types() -> None:
    """Restricting to event_types=('view',) should ignore purchases.

    With our synthetic log, only B is touched as a 'view' (and only by
    users 1..5). C/D/E events are all purchases -> they should be
    invisible to the model.
    """
    log = BehaviorLogger(":memory:")
    try:
        log.log_events_bulk(SYNTHETIC_EVENTS)
        coview = CustomersAlsoViewed().fit(
            log,
            event_types=("view",),
            min_support=1,
            top_k=10,
        )
        # No pairs possible since views are all of product B alone.
        assert coview.get_related(PID_C) == []
        assert coview.get_related(PID_D) == []
        # A has no view events -> A is not a key in the model.
        assert PID_A not in coview
    finally:
        log.close()


def test_anonymous_events_ignored() -> None:
    """Events with user_id=None must NOT contribute to pair counts."""
    log = BehaviorLogger(":memory:")
    try:
        # Two anonymous purchases of A and B -- should be ignored.
        log.log_events_bulk([
            {"product_id": PID_A, "event_type": "purchase",
             "user_id": None, "timestamp": 1_700_000_000.0},
            {"product_id": PID_B, "event_type": "purchase",
             "user_id": None, "timestamp": 1_700_000_001.0},
        ])
        coview = CustomersAlsoViewed().fit(
            log,
            event_types=("purchase",),
            min_support=1,
            top_k=10,
        )
        # Nothing survives because we have zero attributable users.
        assert PID_A not in coview
        assert PID_B not in coview
    finally:
        log.close()


def test_len_and_contains(logger: BehaviorLogger) -> None:
    coview = CustomersAlsoViewed().fit(
        logger,
        event_types=("view", "click", "purchase"),
        min_support=2,
        top_k=10,
    )
    assert len(coview) > 0
    assert PID_A in coview
    assert 999_999 not in coview
