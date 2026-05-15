"""Tests for the A/B testing framework."""

from collections import Counter
from pathlib import Path

import pytest

from src.experiments.ab_testing import (
    Experiment,
    ExperimentRegistry,
    Variant,
    assign_variant,
    get_variant_config,
)


SAMPLE_YAML = (
    Path(__file__).resolve().parent.parent / "experiments" / "ranking_v1.yaml"
)


def _make_experiment(name: str = "test_exp", enabled: bool = True) -> Experiment:
    return Experiment(
        name=name,
        description="50/50 split test",
        enabled=enabled,
        variants=[
            Variant(name="control", traffic_weight=0.5, config={"x": 1}),
            Variant(name="treatment", traffic_weight=0.5, config={"x": 2}),
        ],
    )


class TestAssignVariant:
    def test_none_user_id_returns_first_variant(self):
        exp = _make_experiment()
        variant = assign_variant(exp, None)
        assert variant.name == "control"

    def test_deterministic_for_same_user(self):
        exp = _make_experiment()
        # Same (exp, user_id) should map to the same variant across many calls.
        for user_id in [1, 42, 9999, 12345]:
            first = assign_variant(exp, user_id)
            for _ in range(20):
                assert assign_variant(exp, user_id).name == first.name

    def test_traffic_split_is_roughly_correct(self):
        exp = _make_experiment()
        counts: Counter[str] = Counter()
        for user_id in range(1, 10_001):
            v = assign_variant(exp, user_id)
            counts[v.name] += 1

        assert abs(counts["control"] - 5000) < 200, counts
        assert abs(counts["treatment"] - 5000) < 200, counts
        assert counts["control"] + counts["treatment"] == 10_000

    def test_different_experiments_do_not_correlate(self):
        """If experiment_name is part of the hash, two experiments should produce
        independent assignments. Same user can be control in A and treatment in B.
        For two 50/50 experiments, the overlap of users-in-control should be
        ~50% (not 100% which would mean perfect correlation).
        """
        exp_a = _make_experiment(name="experiment_a")
        exp_b = _make_experiment(name="experiment_b")

        control_in_a: set[int] = set()
        control_in_b: set[int] = set()
        for user_id in range(1, 1001):
            if assign_variant(exp_a, user_id).name == "control":
                control_in_a.add(user_id)
            if assign_variant(exp_b, user_id).name == "control":
                control_in_b.add(user_id)

        # Expect ~50% of A's controls to also be controls in B.
        overlap = len(control_in_a & control_in_b)
        # |control_in_a| ~ 500, so overlap should be ~250.
        # Tolerance band: 150..350 (well away from 0 or 500).
        assert 150 <= overlap <= 350, (
            f"overlap={overlap}, |A|={len(control_in_a)}, |B|={len(control_in_b)}"
        )

        # And it must NOT be ~100% of A's controls (that would mean perfect correlation).
        assert overlap < len(control_in_a) * 0.9


class TestExperimentRegistry:
    def test_load_parses_sample_yaml(self):
        registry = ExperimentRegistry.load(SAMPLE_YAML)

        ranking = registry.get("ranking_v1")
        assert ranking is not None
        assert ranking.enabled is True
        assert len(ranking.variants) == 2
        assert ranking.variants[0].name == "control"
        assert ranking.variants[0].config["popularity_weight"] == 0.15
        assert ranking.variants[1].name == "popularity-boost"
        assert ranking.variants[1].config["popularity_weight"] == 0.35

        rewrite = registry.get("query_rewrite_v1")
        assert rewrite is not None
        assert rewrite.enabled is False

    def test_load_lists_active(self):
        registry = ExperimentRegistry.load(SAMPLE_YAML)
        active = registry.list_active()
        assert len(active) == 1
        assert active[0].name == "ranking_v1"

    def test_load_raises_if_weights_do_not_sum_to_one(self, tmp_path):
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text(
            """
experiments:
  - name: bad_exp
    description: weights don't sum to 1
    enabled: true
    variants:
      - name: a
        traffic_weight: 0.4
        config: {}
      - name: b
        traffic_weight: 0.4
        config: {}
"""
        )
        with pytest.raises(ValueError, match="traffic_weights summing to"):
            ExperimentRegistry.load(bad_yaml)

    def test_get_unknown_returns_none(self):
        registry = ExperimentRegistry.load(SAMPLE_YAML)
        assert registry.get("does_not_exist") is None


class TestGetVariantConfig:
    def test_unknown_experiment_returns_control_default(self):
        registry = ExperimentRegistry.load(SAMPLE_YAML)
        default = {"alpha": 0.5}
        name, cfg = get_variant_config(registry, "nonexistent", user_id=1, default_config=default)
        assert name == "control"
        assert cfg == {"alpha": 0.5}

    def test_unknown_experiment_with_no_default(self):
        registry = ExperimentRegistry.load(SAMPLE_YAML)
        name, cfg = get_variant_config(registry, "nonexistent", user_id=1)
        assert name == "control"
        assert cfg == {}

    def test_disabled_experiment_returns_control_default(self):
        registry = ExperimentRegistry.load(SAMPLE_YAML)
        default = {"use_llm_rewrite": False}
        name, cfg = get_variant_config(
            registry, "query_rewrite_v1", user_id=1, default_config=default
        )
        assert name == "control"
        assert cfg == {"use_llm_rewrite": False}

    def test_enabled_experiment_returns_variant_config(self):
        registry = ExperimentRegistry.load(SAMPLE_YAML)
        name, cfg = get_variant_config(registry, "ranking_v1", user_id=42)
        # Must be one of the variant names defined in the YAML.
        assert name in {"control", "popularity-boost"}
        assert "alpha" in cfg
        assert "popularity_weight" in cfg
        # And it must match the variant's actual config.
        exp = registry.get("ranking_v1")
        assert exp is not None
        matching = [v for v in exp.variants if v.name == name][0]
        assert cfg == matching.config

    def test_returned_config_is_a_copy(self):
        """Mutating the returned config should not mutate the registry's variant."""
        registry = ExperimentRegistry.load(SAMPLE_YAML)
        _, cfg = get_variant_config(registry, "ranking_v1", user_id=42)
        cfg["alpha"] = 999
        # Re-read and ensure it's unchanged.
        _, cfg2 = get_variant_config(registry, "ranking_v1", user_id=42)
        assert cfg2["alpha"] != 999
