from collections import defaultdict

from llama_index.core.schema import NodeWithScore


def reciprocal_rank_fusion(
    ranked_lists: list[list[NodeWithScore]], k: int = 60
) -> list[NodeWithScore]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion (RRF).

    For each document, RRF score = sum over all lists of 1 / (k + rank).
    The constant k (default 60) dampens the effect of high rankings.
    Reference: Cormack et al., 2009.
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
