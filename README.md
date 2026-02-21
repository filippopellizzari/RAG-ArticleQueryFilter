# RAG Article Query Filter

A hybrid retrieval system for news articles that combines dense vector search and BM25 keyword matching, with a BERT-based toxicity gate on incoming queries.

Built as a self-contained ML engineering project to demonstrate systematic retrieval experimentation, modular Python packaging, and production-ready API design.

---

## Architecture

```
Raw corpus (CSV)
      │
      ▼
┌──────────────────────┐   SentenceSplitter (256 tok)  ┌────────────────┐
│      src/data        │ ────────────────────────────► │  src/indexing  │
│  minimal_clean_text  │                               │  ChromaDB      │
└─────────────┘                                     │  bge-large-v1.5│
                                                    └───────┬────────┘
                                                            │
                        ┌───────────────────────────────────┤
                        │                                   │
               ┌────────▼────────┐                 ┌────────▼────────┐
               │  Vector Search  │                 │  BM25 Retriever │
               │  bge-large-v1.5 │                 │  word tokeniser │
               └────────┬────────┘                 └────────┬────────┘
                        │                                   │
                        └──────────────┬────────────────────┘
                                       │ Reciprocal Rank Fusion
                                       ▼
                              Ranked document UUIDs
                                       │
                              ┌────────▼────────┐
                              │ src/query_filter │
                              │  BERT toxicity  │
                              └────────┬────────┘
                                       │
                              ┌────────▼────────┐
                              │    src/api       │
                              │    FastAPI       │
                              └─────────────────┘
```

---

## Results

Five retrieval strategies evaluated on 500 queries (seed 42) from a labelled set of 2 330.
Full methodology and findings: [`results/retrieval_experiments.md`](results/retrieval_experiments.md)

| Experiment | F1 | Recall | Precision |
|-----------|---:|-------:|----------:|
| Baseline — bge-small, top_k=2 | 0.323 | 0.271 | 0.440 |
| BM25 word-tokenised, top_k=5 | 0.396 | 0.460 | 0.373 |
| bge-base vector, top_k=5 | 0.349 | 0.396 | 0.337 |
| bge-base + BM25 hybrid (RRF) | 0.403 | 0.440 | 0.401 |
| **bge-large + BM25 hybrid (RRF)** | **0.405** | **0.441** | **0.408** |

**+25.5% F1** improvement over baseline. Key insight: on this entity-heavy news corpus,
BM25 dominates standalone vector search; a fair RRF hybrid closes the precision gap.

---

## Setup

```bash
# Install dependencies (uv recommended)
uv sync

# Or with pip
pip install -e .
```

**Requirements:** Python 3.11–3.12, ~5 GB disk (models + vector DB).

---

## Usage

### 1. Prepare the corpus

```bash
prepare-data
```

Produces `dataset/corpus_minimal_clean.csv` (whitespace-normalised, casing preserved).

### 2. Run retrieval experiments

```bash
run-experiments
```

Builds ChromaDB indices for each embedding model (cached on re-runs) and prints the
full results table. See [`results/retrieval_experiments.md`](results/retrieval_experiments.md)
for the pre-computed results.

### 3. Run the tests

```bash
# Install dev extras first (pytest + httpx)
uv sync --extra dev

pytest tests/
```

All tests run without a pre-built index or GPU — heavy dependencies (ChromaDB, BERT,
embedding models) are mocked at the test boundary.

### 4. Start the API

```bash
uvicorn src.api:app --reload
```

Interactive docs at <http://127.0.0.1:8000/docs>.

**Note:** The API loads the `v2_bge_large` index at startup. Run `run-experiments` at
least once before serving to ensure the index is built.

#### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check |
| `POST` | `/query?query=…` | Return ranked article UUIDs |

The toxicity filter rejects harmful queries with HTTP 422 before retrieval runs.

---

## Dataset

| File | Description |
|------|-------------|
| `dataset/corpus.csv` | 609 news articles (title, body, uuid, category, …) |
| `dataset/corpus_minimal_clean.csv` | Whitespace-normalised — produced by `prepare-data` |
| `dataset/queries.csv` | 2 330 labelled queries with ground-truth article UUIDs |

---

## Tech stack

| Component | Library |
|-----------|---------|
| Vector indexing | [LlamaIndex](https://www.llamaindex.ai/) + [ChromaDB](https://docs.trychroma.com/) |
| Dense embeddings | [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) via HuggingFace |
| Keyword retrieval | BM25 (rank_bm25) with word-level tokeniser |
| Rank fusion | Reciprocal Rank Fusion (Cormack et al., 2009) |
| Toxicity filter | [JungleLee/bert-toxic-comment-classification](https://huggingface.co/JungleLee/bert-toxic-comment-classification) |
| API | [FastAPI](https://fastapi.tiangolo.com/) |
| Linting | [Ruff](https://docs.astral.sh/ruff/) |

---

## Project layout

```
src/
├── config.py        centralised constants (paths, model names, tuning params)
├── data.py          corpus text cleaning + prepare-data entry point
├── indexing.py      ChromaDB index construction and loading
├── retrieval.py     RRF, hybrid retrieval, reranking utilities
├── evaluation.py    retrieval metrics (recall, precision, F1, hit rate, MRR)
├── query_filter.py  lazy-loaded BERT toxicity classifier
├── experiments.py   5-experiment runner + run-experiments entry point
└── api.py           FastAPI application

results/
└── retrieval_experiments.md   experiment report (methodology + results + findings)

tests/                         tests
```
