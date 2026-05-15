"""A/B testing framework for ranking and recommendation experiments.

This module provides a minimal but real experimentation framework that supports
comparing ranking variants. Every real e-commerce system (Amazon, TikTok,
Instacart) decides ranking changes via A/B experiments, not gut feeling.

Key properties:
    - Deterministic, consistent hashing: the SAME user always gets the SAME
      variant for a given experiment, even across runs and restarts.
    - Experiment isolation: the same user can be in variant A for one
      experiment and variant B for another, because the hash includes the
      experiment name.
    - YAML-driven config so experiments can be edited without code changes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Variant:
    """One arm of an A/B experiment."""

    name: str  # e.g. "control" or "popularity-boost"
    traffic_weight: float  # in [0, 1]; weights across variants in an experiment must sum to 1
    config: dict  # variant-specific knobs (e.g. {"alpha": 0.3, "popularity_weight": 0.25})


@dataclass(frozen=True)
class Experiment:
    """An A/B experiment definition."""

    name: str
    description: str
    enabled: bool
    variants: list[Variant]


class ExperimentRegistry:
    """In-memory registry of running experiments, loaded from YAML."""

    def __init__(self, experiments: list[Experiment]):
        self.experiments = {exp.name: exp for exp in experiments}

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentRegistry":
        """Load experiments from a YAML file.

        Validates traffic_weight sums to 1.0 (+/- 1e-6) for each experiment.
        Returns a registry instance.
        """
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        if not raw or "experiments" not in raw:
            return cls([])

        experiments: list[Experiment] = []
        for exp_dict in raw["experiments"]:
            variants = [
                Variant(
                    name=v["name"],
                    traffic_weight=float(v["traffic_weight"]),
                    config=dict(v.get("config", {})),
                )
                for v in exp_dict["variants"]
            ]

            total_weight = sum(v.traffic_weight for v in variants)
            if abs(total_weight - 1.0) > 1e-6:
                raise ValueError(
                    f"Experiment '{exp_dict['name']}' has traffic_weights summing to "
                    f"{total_weight}, expected 1.0 (+/- 1e-6)"
                )

            experiments.append(
                Experiment(
                    name=exp_dict["name"],
                    description=exp_dict.get("description", ""),
                    enabled=bool(exp_dict.get("enabled", False)),
                    variants=variants,
                )
            )

        return cls(experiments)

    def get(self, name: str) -> Experiment | None:
        """Return the experiment with the given name, or None."""
        return self.experiments.get(name)

    def list_active(self) -> list[Experiment]:
        """Return only experiments with enabled=True."""
        return [exp for exp in self.experiments.values() if exp.enabled]


def assign_variant(experiment: Experiment, user_id: int | None) -> Variant:
    """Deterministically assign a user to a variant using consistent hashing.

    Algorithm:
        - If user_id is None, return the first variant (control behavior)
        - Otherwise hash (experiment.name + str(user_id)) with hashlib.md5
        - Take hash modulo 1_000_000 to get a number in [0, 1_000_000)
        - Walk variants in order, accumulating traffic_weight * 1_000_000
        - Return the first variant whose cumulative range covers the hash

    This means the SAME user always gets the SAME variant for an experiment,
    even across runs. That's the foundation of A/B testing -- consistency.
    """
    if user_id is None:
        return experiment.variants[0]

    key = f"{experiment.name}{user_id}".encode("utf-8")
    digest = hashlib.md5(key).hexdigest()
    bucket = int(digest, 16) % 1_000_000

    cumulative = 0
    for variant in experiment.variants:
        cumulative += int(variant.traffic_weight * 1_000_000)
        if bucket < cumulative:
            return variant

    # Fallback for edge case where cumulative rounding leaves a gap
    # (only happens if all variants exhausted). Return last variant.
    return experiment.variants[-1]


def get_variant_config(
    registry: ExperimentRegistry,
    experiment_name: str,
    user_id: int | None,
    default_config: dict | None = None,
) -> tuple[str, dict]:
    """High-level helper: return (variant_name, merged_config) for an experiment.

    - If the experiment doesn't exist or isn't enabled, returns
      ("control", default_config or {})
    - Otherwise assigns the user to a variant and returns
      (variant.name, variant.config)
    """
    experiment = registry.get(experiment_name)
    if experiment is None or not experiment.enabled:
        return ("control", dict(default_config) if default_config else {})

    variant = assign_variant(experiment, user_id)
    return (variant.name, dict(variant.config))
