import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    TextClassificationPipeline,
)

# Retrieval configuration — update these after running 05_retrieval_experiments.ipynb
SIMILARITY_TOP_K = 10
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
CHROMA_COLLECTION = "quickstart"
CHROMA_DB_PATH = "./chroma_db"

# Load toxic classifier once at module level to avoid reloading on every request
_TOXIC_MODEL_PATH = "JungleLee/bert-toxic-comment-classification"
_toxic_tokenizer = BertTokenizer.from_pretrained(_TOXIC_MODEL_PATH)
_toxic_model = BertForSequenceClassification.from_pretrained(
    _TOXIC_MODEL_PATH, num_labels=2
)
_toxic_pipeline = TextClassificationPipeline(
    model=_toxic_model, tokenizer=_toxic_tokenizer
)


def predict_toxic_query(query: str) -> str:
    """Toxic classifier inference."""
    return _toxic_pipeline(query)[0]["label"]


def is_query_valid(query: str) -> bool:
    """Return True if the content of the query is non-toxic."""
    return predict_toxic_query(query) == "non-toxic"


def load_chroma_index() -> VectorStoreIndex:
    """Load Chroma DB index from disk."""
    db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    chroma_collection = db.get_or_create_collection(CHROMA_COLLECTION)
    chroma_vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    return VectorStoreIndex.from_vector_store(
        vector_store=chroma_vector_store, embed_model=embed_model
    )


def retrieve_doc_uuids(query: str, chroma_index: VectorStoreIndex) -> list[str]:
    """Retrieve relevant document UUIDs for a query."""
    retriever = chroma_index.as_retriever(similarity_top_k=SIMILARITY_TOP_K)
    results = retriever.retrieve(query)
    # Deduplicate by source document UUID while preserving rank order
    seen: set[str] = set()
    uuids: list[str] = []
    for res in results:
        doc_id = res.node.source_node.node_id
        if doc_id not in seen:
            seen.add(doc_id)
            uuids.append(doc_id)
    return uuids
