"""Centralised configuration for the RAG Article Query Filter project.

All runtime constants live here. Import from this module everywhere else —
never hard-code paths, model names, or tuning parameters in other modules.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────

CORPUS_RAW_PATH = Path("dataset/corpus.csv")
CORPUS_PATH     = Path("dataset/corpus_minimal_clean.csv")
QUERIES_PATH    = Path("dataset/queries.csv")
CHROMA_DB_PATH      = Path("chroma_db")

# ── Retrieval (best config from experiments — see results/retrieval_experiments.md) ──

EMBED_MODEL_NAME  = "BAAI/bge-large-en-v1.5"
CHROMA_COLLECTION = "v2_bge_large"
SIMILARITY_TOP_K  = 5
CHUNK_SIZE        = 256
CHUNK_OVERLAP     = CHUNK_SIZE // 10   # 26

# ── Evaluation ─────────────────────────────────────────────────────────────────

SAMPLE_SIZE  = 500
RANDOM_STATE = 42

# ── Query filtering ────────────────────────────────────────────────────────────

TOXIC_MODEL_PATH = "JungleLee/bert-toxic-comment-classification"
