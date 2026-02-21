from collections import defaultdict
from typing import Callable

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.schema import NodeWithScore


def reciprocal_rank_fusion(
    ranked_lists: list[list[NodeWithScore]], k: int = 60
) -> list[NodeWithScore]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion (RRF).

    For each document, RRF score = sum over all lists of 1 / (k + rank).
    The constant k (default 60) dampens the effect of high rankings.
    Reference: Cormack et al., 2009.

    Args:
        ranked_lists: List of ranked result lists to merge. Each inner list is
            a sequence of ``NodeWithScore`` objects ordered by relevance.
        k: Smoothing constant that controls sensitivity to high rankings.
            Larger values reduce the advantage of top-ranked results.

    Returns:
        Single merged list of ``NodeWithScore`` objects sorted by descending
        RRF score. Each node appears exactly once; its score is the sum of its
        RRF contributions across all input lists.
    """
    rrf_scores: dict[str, float] = defaultdict(float)
    node_map: dict[str, NodeWithScore] = {}

    for ranked_list in ranked_lists:
        for rank, node_with_score in enumerate(ranked_list, start=1):
            node_id = node_with_score.node.node_id
            rrf_scores[node_id] += 1.0 / (k + rank)
            if node_id not in node_map:
                node_map[node_id] = node_with_score

    sorted_ids = sorted(rrf_scores, key=lambda nid: rrf_scores[nid], reverse=True)
    return [
        NodeWithScore(node=node_map[nid].node, score=rrf_scores[nid])
        for nid in sorted_ids
    ]


def make_hybrid_retrieval_fn(
    vector_retriever: BaseRetriever,
    bm25_retriever: BaseRetriever,
    bm25_query_preprocessor: Callable[[str], str] | None = None,
) -> Callable[[str], list[NodeWithScore]]:
    """Return a retrieval function that fuses vector + BM25 results with RRF.

    Args:
        vector_retriever: Dense vector retriever (e.g. bge-large backed by
            ChromaDB).
        bm25_retriever: Sparse BM25 retriever.
        bm25_query_preprocessor: Optional callable applied to the query string
            before BM25 retrieval only. Pass ``str.lower`` when the corpus was
            indexed with aggressive lowercasing (e.g. ``clean_text()``) to
            avoid a silent query–corpus case mismatch in term matching.

    Returns:
        Callable that accepts a query string and returns a merged, RRF-scored
        list of ``NodeWithScore`` objects.
    """

    def retrieve(query: str) -> list[NodeWithScore]:
        vector_results = vector_retriever.retrieve(query)
        bm25_query = bm25_query_preprocessor(query) if bm25_query_preprocessor else query
        bm25_results = bm25_retriever.retrieve(bm25_query)
        return reciprocal_rank_fusion([vector_results, bm25_results])

    return retrieve


def make_rerank_retrieval_fn(
    retriever: BaseRetriever,
    reranker: SentenceTransformerRerank,
) -> Callable[[str], list[NodeWithScore]]:
    """Return a retrieval function that retrieves then re-ranks.

    Args:
        retriever: Initial retriever used to fetch candidate nodes.
        reranker: Cross-encoder reranker that reorders the candidates.

    Returns:
        Callable that accepts a query string and returns the candidate nodes
        reordered by the cross-encoder score.
    """

    def retrieve(query: str) -> list[NodeWithScore]:
        candidates = retriever.retrieve(query)
        return reranker.postprocess_nodes(candidates, query_str=query)

    return retrieve
