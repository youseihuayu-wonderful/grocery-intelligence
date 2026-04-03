"""Tests for evaluation metrics."""

from src.evaluation.metrics import (
    precision_at_k,
    ndcg_at_k,
    mean_reciprocal_rank,
    recall_at_k,
)


class TestPrecisionAtK:
    def test_perfect_precision(self):
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = {"a", "b", "c", "d", "e"}
        assert precision_at_k(retrieved, relevant, 5) == 1.0

    def test_zero_precision(self):
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b", "c"}
        assert precision_at_k(retrieved, relevant, 3) == 0.0

    def test_partial_precision(self):
        retrieved = ["a", "x", "b", "y", "c"]
        relevant = {"a", "b", "c"}
        assert precision_at_k(retrieved, relevant, 5) == 0.6

    def test_k_greater_than_retrieved(self):
        retrieved = ["a", "b"]
        relevant = {"a", "b", "c"}
        assert precision_at_k(retrieved, relevant, 5) == 0.4

    def test_k_zero(self):
        assert precision_at_k(["a"], {"a"}, 0) == 0.0


class TestNDCG:
    def test_perfect_ndcg(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert ndcg_at_k(retrieved, relevant, 3) == 1.0

    def test_zero_ndcg(self):
        retrieved = ["x", "y", "z"]
        relevant = {"a", "b"}
        assert ndcg_at_k(retrieved, relevant, 3) == 0.0

    def test_empty_relevant(self):
        assert ndcg_at_k(["a", "b"], set(), 5) == 0.0


class TestMRR:
    def test_first_position(self):
        retrieved = ["a", "b", "c"]
        relevant = {"a"}
        assert mean_reciprocal_rank(retrieved, relevant) == 1.0

    def test_second_position(self):
        retrieved = ["x", "a", "b"]
        relevant = {"a"}
        assert mean_reciprocal_rank(retrieved, relevant) == 0.5

    def test_not_found(self):
        retrieved = ["x", "y", "z"]
        relevant = {"a"}
        assert mean_reciprocal_rank(retrieved, relevant) == 0.0


class TestRecall:
    def test_full_recall(self):
        retrieved = ["a", "b", "c", "d"]
        relevant = {"a", "b"}
        assert recall_at_k(retrieved, relevant, 4) == 1.0

    def test_partial_recall(self):
        retrieved = ["a", "x", "y", "z"]
        relevant = {"a", "b"}
        assert recall_at_k(retrieved, relevant, 4) == 0.5

    def test_zero_recall(self):
        retrieved = ["x", "y"]
        relevant = {"a", "b"}
        assert recall_at_k(retrieved, relevant, 2) == 0.0

    def test_empty_relevant(self):
        assert recall_at_k(["a"], set(), 1) == 0.0
