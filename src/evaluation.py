"""Retrieval evaluation metrics and evaluation runner.

This module is intentionally limited to *evaluation* concerns only.
Index construction lives in ``src.indexing``.
"""

import ast
from typing import Callable

import pandas as pd
from llama_index.core.schema import NodeWithScore
from tqdm import tqdm


# ── Per-query metrics ──────────────────────────────────────────────────────────


def recall_at_k(predicted: list[str], relevant: list[str]) -> float:
    """Fraction of relevant documents that appear in the predicted set.

    Args:
        predicted: Ranked list of retrieved document IDs.
        relevant: Ground-truth list of relevant document IDs.

    Returns:
        Recall score in [0, 1]. Returns 0.0 if ``relevant`` is empty.
    """
    if not relevant:
        return 0.0
    return len(set(predicted) & set(relevant)) / len(relevant)


def precision_at_k(predicted: list[str], relevant: list[str]) -> float:
    """Fraction of predicted documents that are relevant.

    Args:
        predicted: Ranked list of retrieved document IDs.
        relevant: Ground-truth list of relevant document IDs.

    Returns:
        Precision score in [0, 1]. Returns 0.0 if ``predicted`` is empty.
    """
    if not predicted:
        return 0.0
    return len(set(predicted) & set(relevant)) / len(predicted)


def f1_at_k(predicted: list[str], relevant: list[str]) -> float:
    """Harmonic mean of precision and recall.

    Args:
        predicted: Ranked list of retrieved document IDs.
        relevant: Ground-truth list of relevant document IDs.

    Returns:
        F1 score in [0, 1]. Returns 0.0 if both precision and recall are 0.
    """
    p = precision_at_k(predicted, relevant)
    r = recall_at_k(predicted, relevant)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def hit_rate(predicted: list[str], relevant: list[str]) -> float:
    """1.0 if at least one relevant document is retrieved, else 0.0.

    Args:
        predicted: Ranked list of retrieved document IDs.
        relevant: Ground-truth list of relevant document IDs.

    Returns:
        1.0 on any hit, 0.0 on miss. Returns 0.0 if ``relevant`` is empty.
    """
    if not relevant:
        return 0.0
    return 1.0 if set(predicted) & set(relevant) else 0.0


def mrr(predicted: list[str], relevant: list[str]) -> float:
    """Reciprocal rank of the first relevant result (0 if none found).

    Args:
        predicted: Ranked list of retrieved document IDs.
        relevant: Ground-truth list of relevant document IDs.

    Returns:
        1 / rank of the first relevant document, or 0.0 if no relevant
        document appears in ``predicted``.
    """
    relevant_set = set(relevant)
    for rank, doc_id in enumerate(predicted, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


_METRICS: dict[str, Callable[[list[str], list[str]], float]] = {
    "recall": recall_at_k,
    "precision": precision_at_k,
    "f1": f1_at_k,
    "hit_rate": hit_rate,
    "mrr": mrr,
}


# ── Helpers ────────────────────────────────────────────────────────────────────


def extract_doc_uuids(results: list[NodeWithScore]) -> list[str]:
    """Deduplicate chunk-level results to unique source-document UUIDs.

    Preserves rank order: the first chunk from each document determines its
    position in the returned list.

    Args:
        results: Ranked list of ``NodeWithScore`` objects returned by a retriever.

    Returns:
        Ordered list of unique source-document UUIDs, deduplicated by first
        occurrence.
    """
    seen: set[str] = set()
    uuids: list[str] = []
    for res in results:
        doc_id = res.node.source_node.node_id
        if doc_id not in seen:
            seen.add(doc_id)
            uuids.append(doc_id)
    return uuids


# ── Evaluation runner ──────────────────────────────────────────────────────────


def evaluate_retrieval_fn(
    retrieval_fn: Callable[[str], list[NodeWithScore]],
    queries_df: pd.DataFrame,
    sample_size: int = 500,
    random_state: int = 42,
    desc: str = "Evaluating",
) -> pd.DataFrame:
    """Evaluate any retrieval function on a sample of labelled queries.

    Args:
        retrieval_fn: Callable that maps a query string to a ranked list of
            ``NodeWithScore`` objects (vector retriever, hybrid pipeline, etc.).
        queries_df: DataFrame with ``query`` and ``result`` columns. ``result``
            must be a stringified list of ground-truth document UUIDs.
        sample_size: Number of queries to evaluate (sampled without replacement).
        random_state: Random seed for reproducibility.
        desc: Progress-bar label shown by tqdm.

    Returns:
        DataFrame with one row per sampled query and one column per metric
        (recall, precision, f1, hit_rate, mrr).
    """
    sample = (
        queries_df.sample(min(sample_size, len(queries_df)), random_state=random_state)
        .copy()
        .reset_index(drop=True)
    )

    records: list[dict] = []
    for idx in tqdm(range(len(sample)), desc=desc):
        query_text = sample.loc[idx, "query"]
        relevant = ast.literal_eval(sample.loc[idx, "result"])
        predicted = extract_doc_uuids(retrieval_fn(query_text))

        record = {"query": query_text}
        for name, fn in _METRICS.items():
            record[name] = fn(predicted, relevant)
        records.append(record)

    return pd.DataFrame(records)


def summarize_results(results_df: pd.DataFrame) -> dict[str, float]:
    """Return mean of each metric across all evaluated queries.

    Args:
        results_df: DataFrame produced by ``evaluate_retrieval_fn``.

    Returns:
        Dict mapping metric name to its macro-average, rounded to 4 decimal
        places.
    """
    return {m: round(results_df[m].mean(), 4) for m in _METRICS}
