"""End-to-end A/B experimentation demo.

Wires together every piece of the experimentation stack:

    YAML registry  ->  variant assignment  ->  behavior log  ->  metrics

The script samples 1,000 random users from the seeded behavior database,
assigns each one to a variant of the ``ranking_v1`` experiment, then
runs the events through ``compute_variant_metrics`` and
``compare_variants`` to produce a per-variant report.

Run::

    cd /Users/shihuayu/grocery-intelligence
    source venv/bin/activate
    python3 scripts/simulate_experiment.py

About the seeded data
---------------------
The 1M seeded events are *all* purchase events (no views/clicks logged
yet by upstream agents). That means CTR and conversion rate will both
come back as 0.0 in this demo -- the metric pipeline is correct, the
data just doesn't include impressions yet. ``n_purchases`` will be
nonzero per variant, which proves the per-variant aggregation works.
"""

from __future__ import annotations

import random
import sqlite3
import sys
import time
from pathlib import Path

# Make ``src`` importable when running this script directly.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.online_metrics import (  # noqa: E402
    compare_variants,
    compute_variant_metrics,
)
from src.experiments.ab_testing import ExperimentRegistry, assign_variant  # noqa: E402
from src.recommend.behavior import DEFAULT_DB_PATH, BehaviorLogger  # noqa: E402


EXPERIMENTS_YAML = REPO_ROOT / "experiments" / "ranking_v1.yaml"
BEHAVIOR_DB = REPO_ROOT / DEFAULT_DB_PATH

N_USERS = 1000
SEED = 42  # reproducible demo


def _sample_distinct_user_ids(db_path: Path, n: int, seed: int) -> list[int]:
    """Pull ``n`` distinct user_ids from the events table.

    We go straight to SQLite for this because ``BehaviorLogger`` doesn't
    expose a "distinct user ids" helper, and adding one to ``behavior.py``
    is out of scope (other agents own that file). One query against the
    ``idx_user`` index is the cheap path.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute(
            "SELECT DISTINCT user_id FROM events WHERE user_id IS NOT NULL"
        )
        all_uids = [row[0] for row in cur.fetchall()]
    finally:
        conn.close()

    rng = random.Random(seed)
    rng.shuffle(all_uids)
    return all_uids[:n]


def _print_variant_table(metrics: dict) -> None:
    """Render a small fixed-width table of per-variant metrics."""
    cols = (
        ("variant", 20),
        ("n_users", 9),
        ("n_views", 9),
        ("n_clicks", 9),
        ("n_purchases", 12),
        ("ctr", 8),
        ("conversion", 11),
        ("mrr@10", 8),
        ("avg_click_pos", 14),
    )
    header = "  ".join(name.ljust(w) for name, w in cols)
    print(header)
    print("-" * len(header))
    for v in metrics.values():
        row = (
            v.variant.ljust(20),
            f"{v.n_users}".ljust(9),
            f"{v.n_views}".ljust(9),
            f"{v.n_clicks}".ljust(9),
            f"{v.n_purchases}".ljust(12),
            f"{v.ctr:.4f}".ljust(8),
            f"{v.conversion_rate:.4f}".ljust(11),
            f"{v.mrr_at_10:.4f}".ljust(8),
            f"{v.avg_click_position:.4f}".ljust(14),
        )
        print("  ".join(row))


def main() -> None:
    wall_start = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Load the experiment registry from YAML
    # ------------------------------------------------------------------
    print(f"Loading experiment registry from {EXPERIMENTS_YAML} ...")
    registry = ExperimentRegistry.load(EXPERIMENTS_YAML)
    experiment = registry.get("ranking_v1")
    if experiment is None:
        raise SystemExit("Experiment 'ranking_v1' not found in registry.")
    print(
        f"  -> experiment={experiment.name}, "
        f"enabled={experiment.enabled}, "
        f"variants={[v.name for v in experiment.variants]}"
    )

    # ------------------------------------------------------------------
    # 2. Sample users that actually have events in the log
    # ------------------------------------------------------------------
    print(f"\nSampling {N_USERS} random user_ids from {BEHAVIOR_DB} ...")
    user_ids = _sample_distinct_user_ids(BEHAVIOR_DB, N_USERS, SEED)
    print(f"  -> {len(user_ids)} distinct users selected")

    # ------------------------------------------------------------------
    # 3. Assign each user to a variant
    # ------------------------------------------------------------------
    print("\nAssigning users to variants ...")
    user_variant_map: dict[int, str] = {
        uid: assign_variant(experiment, uid).name for uid in user_ids
    }
    # Quick sanity check on the split.
    split: dict[str, int] = {}
    for v in user_variant_map.values():
        split[v] = split.get(v, 0) + 1
    for name, n in sorted(split.items()):
        print(f"  -> {name}: {n} users")

    # ------------------------------------------------------------------
    # 4. Aggregate metrics per variant from the behavior log
    # ------------------------------------------------------------------
    print("\nComputing per-variant metrics from the behavior log ...")
    metrics_start = time.perf_counter()
    with BehaviorLogger(BEHAVIOR_DB) as logger:
        variant_metrics = compute_variant_metrics(logger, user_variant_map)
    metrics_secs = time.perf_counter() - metrics_start
    print(f"  -> finished in {metrics_secs:.2f}s")

    # ------------------------------------------------------------------
    # 5. Print the per-variant report
    # ------------------------------------------------------------------
    print("\nPer-variant metrics:")
    _print_variant_table(variant_metrics)

    # ------------------------------------------------------------------
    # 6. Compare control vs treatment
    # ------------------------------------------------------------------
    print("\nComparison (control vs treatment):")
    control_name = experiment.variants[0].name
    treatment_name = experiment.variants[1].name
    control_metrics = variant_metrics[control_name]
    treatment_metrics = variant_metrics[treatment_name]
    cmp = compare_variants(control_metrics, treatment_metrics)
    for k, v in cmp.items():
        print(f"  {k}: {v}")

    # ------------------------------------------------------------------
    # 7. Note: seeded data is purchase-only, so CTR/conversion are 0
    # ------------------------------------------------------------------
    print(
        "\nNOTE: the seeded behavior log contains purchase events only "
        "(no views/clicks),\n"
        "      so ctr and conversion_rate read as 0.0 here. The point\n"
        "      of this demo is to show the metrics pipeline works "
        "end-to-end and\n"
        "      that the per-variant aggregation is correct -- which is "
        "visible from\n"
        "      the nonzero n_purchases column."
    )

    wall_secs = time.perf_counter() - wall_start
    print(f"\nTotal wall-clock time: {wall_secs:.2f}s")


if __name__ == "__main__":
    main()
