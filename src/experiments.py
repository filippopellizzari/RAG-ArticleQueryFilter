"""Retrieval quality experiment runner.

Runs five retrieval experiments on the minimal-clean corpus and prints a
comparison table. Results are documented in ``results/retrieval_experiments.md``.

Usage
-----
    run-experiments          # via pyproject.toml [project.scripts]
    python -m src.experiments
"""

import logging
import re

import pandas as pd
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import NodeWithScore
from llama_index.retrievers.bm25 import BM25Retriever

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CORPUS_PATH,
    QUERIES_PATH,
    RANDOM_STATE,
    SAMPLE_SIZE,
    SIMILARITY_TOP_K,
)
from src.evaluation import evaluate_retrieval_fn, summarize_results
from src.indexing import build_index
from src.retrieval import reciprocal_rank_fusion

log = logging.getLogger(__name__)

TOP_K = SIMILARITY_TOP_K  # local alias used throughout this module


# ── Helpers ────────────────────────────────────────────────────────────────────

def word_tokenize(text: str) -> list[str]:
    """Extract alphanumeric tokens, preserving case.

    Improvement over rank_bm25's default whitespace split:
    - "City."  → ["City"]     not ["City."]
    - "2-0"    → ["2", "0"]   numeric components separated correctly
    - Preserves casing for entity matching on the minimal-clean corpus.
    """
    return re.findall(r"[A-Za-z0-9]+", text)


def make_hybrid_fn(
    vector_retriever,
    bm25_retriever,
    top_k: int = TOP_K,
):
    """Return a fair RRF hybrid retriever truncated to *top_k* chunks.

    Without truncation a hybrid returns up to ``2×top_k`` results, giving it an
    unfair recall advantage over single-retriever baselines at the same top_k.
    """
    def retrieve(query: str) -> list[NodeWithScore]:
        merged = reciprocal_rank_fusion([
            vector_retriever.retrieve(query),
            bm25_retriever.retrieve(query),
        ])
        return merged[:top_k]
    return retrieve


# ── Experiment runner ──────────────────────────────────────────────────────────

def run() -> dict[str, dict[str, float]]:
    """Execute all five experiments and return a dict of {label: metrics}."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    log.info("Loading corpus and queries …")
    corpus_df  = pd.read_csv(CORPUS_PATH)
    queries_df = pd.read_csv(QUERIES_PATH)
    log.info("Corpus: %d articles | Queries: %d", len(corpus_df), len(queries_df))

    documents   = [Document(text=r["text"], doc_id=r["uuid"]) for _, r in corpus_df.iterrows()]
    node_parser = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes       = node_parser.get_nodes_from_documents(documents)
    log.info("Chunks: %d (size=%d, overlap=%d)", len(nodes), CHUNK_SIZE, CHUNK_OVERLAP)

    results: dict[str, dict] = {}

    # ── 1. Baseline ────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Exp 1 / Baseline — bge-small-en-v1.5 | top_k=2 | chunk=%d", CHUNK_SIZE)
    baseline_index     = build_index(corpus_df, collection_name="v2_baseline")
    baseline_retriever = baseline_index.as_retriever(similarity_top_k=2)
    results["1. Baseline (bge-small, k=2)"] = summarize_results(
        evaluate_retrieval_fn(
            baseline_retriever.retrieve, queries_df, SAMPLE_SIZE, RANDOM_STATE,
            desc="Baseline",
        )
    )

    # ── 2. BM25 ───────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Exp 2 / BM25 — word tokeniser | top_k=%d", TOP_K)
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=TOP_K,
        tokenizer=word_tokenize,
    )
    results["2. BM25 word-tokenised (k=5)"] = summarize_results(
        evaluate_retrieval_fn(
            bm25_retriever.retrieve, queries_df, SAMPLE_SIZE, RANDOM_STATE,
            desc="BM25",
        )
    )

    # ── 3. bge-base vector ────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Exp 3 / bge-base-en-v1.5 — vector only | top_k=%d", TOP_K)
    bge_base_index = build_index(
        corpus_df,
        collection_name="v2_bge_base",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        embed_model_name="BAAI/bge-base-en-v1.5",
    )
    bge_base_retriever = bge_base_index.as_retriever(similarity_top_k=TOP_K)
    results["3. bge-base Vector (k=5)"] = summarize_results(
        evaluate_retrieval_fn(
            bge_base_retriever.retrieve, queries_df, SAMPLE_SIZE, RANDOM_STATE,
            desc="bge-base",
        )
    )

    # ── 4. bge-base + BM25 hybrid ─────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Exp 4 / bge-base + BM25 Hybrid (fair RRF) | top_k=%d", TOP_K)
    hybrid_base_fn = make_hybrid_fn(bge_base_retriever, bm25_retriever, top_k=TOP_K)
    results["4. bge-base + BM25 Hybrid (k=5)"] = summarize_results(
        evaluate_retrieval_fn(
            hybrid_base_fn, queries_df, SAMPLE_SIZE, RANDOM_STATE,
            desc="bge-base Hybrid",
        )
    )

    # ── 5. bge-large + BM25 hybrid ────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Exp 5 / bge-large-en-v1.5 + BM25 Hybrid | top_k=%d", TOP_K)
    log.info("(First run downloads ~335 M-param model and indexes the corpus)")
    bge_large_index = build_index(
        corpus_df,
        collection_name="v2_bge_large",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        embed_model_name="BAAI/bge-large-en-v1.5",
    )
    bge_large_retriever = bge_large_index.as_retriever(similarity_top_k=TOP_K)
    hybrid_large_fn     = make_hybrid_fn(bge_large_retriever, bm25_retriever, top_k=TOP_K)
    results["5. bge-large + BM25 Hybrid (k=5)"] = summarize_results(
        evaluate_retrieval_fn(
            hybrid_large_fn, queries_df, SAMPLE_SIZE, RANDOM_STATE,
            desc="bge-large Hybrid",
        )
    )

    return results


def main() -> None:
    """CLI entry point: run experiments and print the results table."""
    results    = run()
    summary_df = pd.DataFrame(results).T
    summary_df.index.name = "experiment"

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(summary_df.to_string(float_format=lambda x: f"{x:.4f}"))

    baseline_f1 = results["1. Baseline (bge-small, k=2)"]["f1"]
    best_name   = summary_df["f1"].idxmax()
    best_f1     = summary_df.loc[best_name, "f1"]

    print(f"\nBest configuration : {best_name}")
    print(
        f"F1 improvement     : {baseline_f1:.4f} → {best_f1:.4f} "
        f"(+{(best_f1 - baseline_f1) / baseline_f1 * 100:.1f}%)"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
