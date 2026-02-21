"""Vector index construction and loading.

Wraps LlamaIndex + ChromaDB to provide a single ``build_index`` function used
by both the experiment runner and the API server.
"""

import chromadb
import pandas as pd
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.config import (
    CHROMA_COLLECTION,
    CHROMA_DB_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBED_MODEL_NAME,
    SIMILARITY_TOP_K,
)


def build_index(
    corpus_df: pd.DataFrame,
    collection_name: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    embed_model_name: str = EMBED_MODEL_NAME,
    chroma_path: str | None = None,
) -> VectorStoreIndex:
    """Build (or load) a ChromaDB-backed vector index from a corpus DataFrame.

    If the named collection already contains documents the existing index is
    returned immediately — no re-embedding is performed.

    Parameters
    ----------
    corpus_df:
        DataFrame with at least ``text`` and ``uuid`` columns.
    collection_name:
        ChromaDB collection name. Use distinct names for different
        corpus/model combinations to avoid cross-contamination.
    chunk_size:
        Token budget per chunk (SentenceSplitter).
    chunk_overlap:
        Overlap in tokens between adjacent chunks.
    embed_model_name:
        HuggingFace model identifier for the embedding model.
    chroma_path:
        Override the default ChromaDB persistence directory.
    """
    path = str(chroma_path or CHROMA_DB_PATH)
    documents = [
        Document(text=row["text"], doc_id=row["uuid"])
        for _, row in corpus_df.iterrows()
    ]

    db = chromadb.PersistentClient(path=path)
    collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    embed_model = HuggingFaceEmbedding(model_name=embed_model_name)

    if collection.count() > 0:
        return VectorStoreIndex.from_vector_store(
            vector_store=vector_store, embed_model=embed_model
        )

    node_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = node_parser.get_nodes_from_documents(documents)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex(
        nodes,
        embed_model=embed_model,
        storage_context=storage_context,
        show_progress=True,
    )


def load_index(
    collection_name: str = CHROMA_COLLECTION,
    embed_model_name: str = EMBED_MODEL_NAME,
    chroma_path: str | None = None,
) -> VectorStoreIndex:
    """Load a pre-built index from ChromaDB (no re-embedding).

    Used by the API server at startup. Raises ``ValueError`` if the collection
    is empty (index has not been built yet — run ``run-experiments`` first).
    """
    path = str(chroma_path or CHROMA_DB_PATH)
    db = chromadb.PersistentClient(path=path)
    collection = db.get_or_create_collection(collection_name)

    if collection.count() == 0:
        raise ValueError(
            f"ChromaDB collection '{collection_name}' is empty. "
            "Run 'run-experiments' to build the index before starting the API."
        )

    vector_store = ChromaVectorStore(chroma_collection=collection)
    embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store, embed_model=embed_model
    )


def retrieve_doc_uuids(
    query: str,
    index: VectorStoreIndex,
    top_k: int = SIMILARITY_TOP_K,
) -> list[str]:
    """Return deduplicated source-document UUIDs for a query, ranked by relevance."""
    retriever = index.as_retriever(similarity_top_k=top_k)
    results = retriever.retrieve(query)

    seen: set[str] = set()
    uuids: list[str] = []
    for node in results:
        doc_id = node.node.source_node.node_id
        if doc_id not in seen:
            seen.add(doc_id)
            uuids.append(doc_id)
    return uuids
