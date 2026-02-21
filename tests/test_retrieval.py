"""Unit tests for src.retrieval (RRF and hybrid retrieval factory)."""

from unittest.mock import MagicMock

import pytest
from llama_index.core.schema import NodeWithScore, TextNode

from src.retrieval import make_hybrid_retrieval_fn, reciprocal_rank_fusion


def make_node(node_id: str, score: float = 1.0) -> NodeWithScore:
    """Build a real NodeWithScore with a minimal TextNode."""
    return NodeWithScore(node=TextNode(id_=node_id, text="sample"), score=score)


# ── reciprocal_rank_fusion ─────────────────────────────────────────────────────

class TestReciprocalRankFusion:
    def test_single_list_preserves_order(self):
        nodes = [make_node("a"), make_node("b"), make_node("c")]
        result = reciprocal_rank_fusion([nodes])
        assert [n.node.node_id for n in result] == ["a", "b", "c"]

    def test_node_appearing_in_both_lists_ranks_first(self):
        list1 = [make_node("a"), make_node("b")]
        list2 = [make_node("b"), make_node("c")]
        result = reciprocal_rank_fusion([list1, list2])
        # "b" gets RRF score from both lists → highest total score
        assert result[0].node.node_id == "b"

    def test_deduplicates_across_lists(self):
        list1 = [make_node("a"), make_node("b")]
        list2 = [make_node("a"), make_node("c")]
        result = reciprocal_rank_fusion([list1, list2])
        ids = [n.node.node_id for n in result]
        assert ids.count("a") == 1

    def test_output_contains_all_unique_nodes(self):
        list1 = [make_node("a"), make_node("b")]
        list2 = [make_node("c"), make_node("d")]
        result = reciprocal_rank_fusion([list1, list2])
        assert set(n.node.node_id for n in result) == {"a", "b", "c", "d"}

    def test_scores_are_positive(self):
        nodes = [make_node("a"), make_node("b")]
        result = reciprocal_rank_fusion([nodes])
        assert all(n.score > 0 for n in result)

    def test_empty_lists_return_empty(self):
        assert reciprocal_rank_fusion([[], []]) == []

    def test_rrf_k_parameter_affects_scores(self):
        nodes = [make_node("a")]
        # k=0 gives score 1/1=1.0; k=60 gives score 1/61
        result_k0 = reciprocal_rank_fusion([nodes], k=0)
        result_k60 = reciprocal_rank_fusion([nodes], k=60)
        assert result_k0[0].score > result_k60[0].score


# ── make_hybrid_retrieval_fn ───────────────────────────────────────────────────

class TestMakeHybridRetrievalFn:
    def _make_retriever(self, results: list[NodeWithScore]) -> MagicMock:
        retriever = MagicMock()
        retriever.retrieve.return_value = results
        return retriever

    def test_returns_callable(self):
        fn = make_hybrid_retrieval_fn(
            self._make_retriever([]),
            self._make_retriever([]),
        )
        assert callable(fn)

    def test_combines_results_from_both_retrievers(self):
        vector_results = [make_node("v1"), make_node("v2")]
        bm25_results   = [make_node("b1"), make_node("b2")]
        fn = make_hybrid_retrieval_fn(
            self._make_retriever(vector_results),
            self._make_retriever(bm25_results),
        )
        result_ids = {n.node.node_id for n in fn("test query")}
        assert result_ids == {"v1", "v2", "b1", "b2"}

    def test_bm25_query_preprocessor_is_applied(self):
        bm25_retriever = self._make_retriever([])
        fn = make_hybrid_retrieval_fn(
            self._make_retriever([]),
            bm25_retriever,
            bm25_query_preprocessor=str.lower,
        )
        fn("UPPER CASE QUERY")
        bm25_retriever.retrieve.assert_called_once_with("upper case query")

    def test_no_preprocessor_passes_query_unchanged(self):
        bm25_retriever = self._make_retriever([])
        fn = make_hybrid_retrieval_fn(
            self._make_retriever([]),
            bm25_retriever,
        )
        fn("Original Query")
        bm25_retriever.retrieve.assert_called_once_with("Original Query")
