"""Unit tests for src.evaluation metric functions."""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.evaluation import (
    extract_doc_uuids,
    f1_at_k,
    hit_rate,
    mrr,
    precision_at_k,
    recall_at_k,
    summarize_results,
)


def make_result(source_uuid: str) -> MagicMock:
    """Build a minimal mock that mimics a NodeWithScore returned by a retriever."""
    m = MagicMock()
    m.node.source_node.node_id = source_uuid
    return m


# ── recall_at_k ────────────────────────────────────────────────────────────────

class TestRecallAtK:
    def test_perfect_recall(self):
        assert recall_at_k(["a", "b"], ["a", "b"]) == 1.0

    def test_no_overlap(self):
        assert recall_at_k(["c"], ["a", "b"]) == 0.0

    def test_partial_recall(self):
        assert recall_at_k(["a", "c"], ["a", "b"]) == 0.5

    def test_empty_relevant(self):
        assert recall_at_k(["a"], []) == 0.0

    def test_superset_predicted(self):
        # All relevant retrieved; extra predicted docs don't affect recall.
        assert recall_at_k(["a", "b", "c"], ["a", "b"]) == 1.0


# ── precision_at_k ─────────────────────────────────────────────────────────────

class TestPrecisionAtK:
    def test_perfect_precision(self):
        assert precision_at_k(["a", "b"], ["a", "b"]) == 1.0

    def test_no_overlap(self):
        assert precision_at_k(["c"], ["a"]) == 0.0

    def test_partial_precision(self):
        assert precision_at_k(["a", "c"], ["a"]) == 0.5

    def test_empty_predicted(self):
        assert precision_at_k([], ["a"]) == 0.0


# ── f1_at_k ────────────────────────────────────────────────────────────────────

class TestF1AtK:
    def test_perfect_f1(self):
        assert f1_at_k(["a", "b"], ["a", "b"]) == 1.0

    def test_zero_f1_no_overlap(self):
        assert f1_at_k(["c"], ["a"]) == 0.0

    def test_zero_f1_both_empty(self):
        assert f1_at_k([], []) == 0.0

    def test_harmonic_mean(self):
        # recall=0.5, precision=0.5 → F1=0.5
        assert f1_at_k(["a", "c"], ["a", "b"]) == pytest.approx(0.5)


# ── hit_rate ───────────────────────────────────────────────────────────────────

class TestHitRate:
    def test_hit(self):
        assert hit_rate(["a", "b"], ["b"]) == 1.0

    def test_miss(self):
        assert hit_rate(["c"], ["a"]) == 0.0

    def test_empty_relevant(self):
        assert hit_rate(["a"], []) == 0.0


# ── mrr ────────────────────────────────────────────────────────────────────────

class TestMRR:
    def test_first_rank(self):
        assert mrr(["a", "b"], ["a"]) == pytest.approx(1.0)

    def test_second_rank(self):
        assert mrr(["x", "a"], ["a"]) == pytest.approx(0.5)

    def test_third_rank(self):
        assert mrr(["x", "y", "a"], ["a"]) == pytest.approx(1 / 3)

    def test_not_found(self):
        assert mrr(["x", "y"], ["a"]) == 0.0

    def test_multiple_relevant_uses_first_hit(self):
        # "b" is at rank 2, "a" is at rank 1 → MRR = 1.0
        assert mrr(["a", "b"], ["a", "b"]) == pytest.approx(1.0)


# ── extract_doc_uuids ──────────────────────────────────────────────────────────

class TestExtractDocUuids:
    def test_deduplicates_same_source(self):
        results = [make_result("doc1"), make_result("doc1"), make_result("doc2")]
        assert extract_doc_uuids(results) == ["doc1", "doc2"]

    def test_preserves_rank_order(self):
        results = [make_result("doc2"), make_result("doc1")]
        assert extract_doc_uuids(results) == ["doc2", "doc1"]

    def test_empty_input(self):
        assert extract_doc_uuids([]) == []

    def test_all_unique(self):
        results = [make_result("a"), make_result("b"), make_result("c")]
        assert extract_doc_uuids(results) == ["a", "b", "c"]


# ── summarize_results ──────────────────────────────────────────────────────────

class TestSummarizeResults:
    def test_returns_all_metric_keys(self):
        df = pd.DataFrame({
            "recall": [0.5, 1.0],
            "precision": [0.5, 1.0],
            "f1": [0.5, 1.0],
            "hit_rate": [1.0, 1.0],
            "mrr": [0.5, 1.0],
        })
        summary = summarize_results(df)
        assert set(summary.keys()) == {"recall", "precision", "f1", "hit_rate", "mrr"}

    def test_computes_mean(self):
        df = pd.DataFrame({
            "recall": [0.5, 1.0],
            "precision": [0.0, 1.0],
            "f1": [0.5, 0.5],
            "hit_rate": [1.0, 1.0],
            "mrr": [0.5, 1.0],
        })
        summary = summarize_results(df)
        assert summary["recall"] == pytest.approx(0.75)
        assert summary["precision"] == pytest.approx(0.5)
        assert summary["f1"] == pytest.approx(0.5)
        assert summary["hit_rate"] == pytest.approx(1.0)
