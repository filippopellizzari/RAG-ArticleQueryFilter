import uuid

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    TextClassificationPipeline,
)


def predict_toxic_query(query: str) -> str:
    """Toxic classifier inference."""
    # load toxic classifier
    model_path = "JungleLee/bert-toxic-comment-classification"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)
    pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer)
    # inference
    result = pipeline(query)[0]["label"]
    return result


def is_query_valid(query: str) -> bool:
    """Return True if the content of the query is non-toxic."""
    result = predict_toxic_query(query)
    if result == "non-toxic":
        return True
    return False


def load_chroma_index() -> VectorStoreIndex:
    """Load Chroma DB index from disk."""
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("quickstart")
    chroma_vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    chroma_index = VectorStoreIndex.from_vector_store(
        vector_store=chroma_vector_store, embed_model=embed_model
    )
    return chroma_index


def retrieve_doc_uuids(query: str, chroma_index: VectorStoreIndex) -> list[str]:
    retriever = chroma_index.as_retriever()
    # retrieve documents uuids
    uuids = [res.node.source_node.node_id for res in retriever.retrieve(query)]
    return uuids
