import ast
from typing import Callable

import chromadb
import pandas as pd
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Metrics (per-query)
# ---------------------------------------------------------------------------


def recall_at_k(predicted: list[str], relevant: list[str]) -> float:
    """Fraction of relevant documents that appear in predictions."""
    if not relevant:
        return 0.0
    return len(set(predicted) & set(relevant)) / len(relevant)


def precision_at_k(predicted: list[str], relevant: list[str]) -> float:
    """Fraction of predictions that are relevant."""
    if not predicted:
        return 0.0
    return len(set(predicted) & set(relevant)) / len(predicted)


def f1_at_k(predicted: list[str], relevant: list[str]) -> float:
    """Harmonic mean of precision and recall."""
    p = precision_at_k(predicted, relevant)
    r = recall_at_k(predicted, relevant)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def hit_rate(predicted: list[str], relevant: list[str]) -> float:
    """1.0 if at least one relevant document is in predictions, else 0.0."""
    if not relevant:
        return 0.0
    return 1.0 if set(predicted) & set(relevant) else 0.0


def mrr(predicted: list[str], relevant: list[str]) -> float:
    """Mean Reciprocal Rank — 1/rank of the first relevant result."""
    relevant_set = set(relevant)
    for rank, doc_id in enumerate(predicted, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def extract_doc_uuids(results: list[NodeWithScore]) -> list[str]:
    """Deduplicate NodeWithScore results by source document UUID, preserving rank order."""
    seen: set[str] = set()
    uuids: list[str] = []
    for res in results:
        doc_id = res.node.source_node.node_id
        if doc_id not in seen:
            seen.add(doc_id)
            uuids.append(doc_id)
    return uuids


METRICS = {
    "recall": recall_at_k,
    "precision": precision_at_k,
    "f1": f1_at_k,
    "hit_rate": hit_rate,
    "mrr": mrr,
}


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------


def evaluate_retrieval_fn(
    retrieval_fn: Callable[[str], list[NodeWithScore]],
    queries_df: pd.DataFrame,
    sample_size: int = 500,
    random_state: int = 42,
    desc: str = "Evaluating",
) -> pd.DataFrame:
    """Evaluate any retrieval function that maps query → list[NodeWithScore].

    This is the core evaluation function. It works with plain retrievers,
    hybrid pipelines, re-ranking pipelines, or any custom logic.

    Parameters
    ----------
    retrieval_fn : Callable[[str], list[NodeWithScore]]
        A function that takes a query string and returns ranked results.
    queries_df : pd.DataFrame
        Must have columns ``query`` and ``result`` (stringified list of UUIDs).
    sample_size : int
        Number of queries to evaluate on.
    random_state : int
        Seed for reproducibility.
    desc : str
        Progress bar description.

    Returns
    -------
    pd.DataFrame with per-query metrics.
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
        for name, fn in METRICS.items():
            record[name] = fn(predicted, relevant)
        records.append(record)

    return pd.DataFrame(records)


def summarize_results(results_df: pd.DataFrame) -> dict[str, float]:
    """Return mean of each metric."""
    return {m: round(results_df[m].mean(), 4) for m in METRICS}


# ---------------------------------------------------------------------------
# Index builder helper
# ---------------------------------------------------------------------------


def build_index(
    corpus_df: pd.DataFrame,
    collection_name: str,
    chunk_size: int = 256,
    chunk_overlap: int = 20,
    embed_model_name: str = "BAAI/bge-small-en-v1.5",
    chroma_path: str = "./chroma_db",
) -> VectorStoreIndex:
    """Build (or load) a ChromaDB-backed vector index.

    If the collection already has documents, the existing index is loaded
    instead of re-indexing.
    """
    documents = [
        Document(text=row["text"], doc_id=row["uuid"])
        for _, row in corpus_df.iterrows()
    ]

    db = chromadb.PersistentClient(path=chroma_path)
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    embed_model = HuggingFaceEmbedding(model_name=embed_model_name)

    # If collection already populated, just load
    if chroma_collection.count() > 0:
        return VectorStoreIndex.from_vector_store(
            vector_store=vector_store, embed_model=embed_model
        )

    # Otherwise build from scratch
    node_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = node_parser.get_nodes_from_documents(documents)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex(
        nodes,
        embed_model=embed_model,
        storage_context=storage_context,
        show_progress=True,
    )
