# Retrieval Quality Experiments

## Setup

**Corpus**: 609 news articles (`corpus_minimal_clean.csv`).
Text preprocessing: whitespace normalisation only — casing and unicode are preserved.

**Evaluation**: 500 queries sampled at random (seed 42) from a labelled query set of 2 330.
Each query has a ground-truth list of relevant article UUIDs.
Retrieval operates at the **chunk level** but is evaluated at the **document level**:
`extract_doc_uuids` deduplicates chunks by their source document before computing metrics.

**Metrics**

| Metric | Definition |
|--------|-----------|
| Recall | \|predicted ∩ relevant\| / \|relevant\| |
| Precision | \|predicted ∩ relevant\| / \|predicted\| |
| F1 | Harmonic mean of recall and precision |
| Hit rate | 1 if at least one relevant document is retrieved, else 0 |
| MRR | 1 / rank of the first relevant document |

All scores are macro-averaged over the 500 sampled queries.

**Fairness note on hybrid retrieval**: in experiments 4–5, the RRF-merged chunk list is
truncated to `top_k` before document-level deduplication. Without this truncation a hybrid
retriever would return up to `2×top_k` results, giving it an unfair recall advantage over
single-retriever baselines at the same `top_k`.

---

## Results

| # | Experiment | Recall | Precision | F1 | Hit Rate | MRR |
|---|-----------|-------:|----------:|---:|---------:|----:|
| 1 | Baseline — bge-small-en-v1.5, top_k=2, chunk=256 | 0.2705 | 0.4400 | 0.3225 | 0.5740 | 0.5220 |
| 2 | BM25 — word tokeniser, top_k=5 | 0.4597 | 0.3729 | 0.3961 | 0.7240 | 0.6110 |
| 3 | bge-base-en-v1.5 vector, top_k=5 | 0.3958 | 0.3372 | 0.3488 | 0.6860 | 0.5765 |
| 4 | bge-base-en-v1.5 + BM25 hybrid (fair RRF), top_k=5 | 0.4395 | 0.4009 | **0.4031** | 0.7240 | 0.5991 |
| 5 | bge-large-en-v1.5 + BM25 hybrid (fair RRF), top_k=5 | 0.4412 | 0.4081 | **0.4048** | 0.7260 | 0.5884 |

**Best configuration**: Experiment 5 — bge-large + BM25 hybrid
**F1 improvement over baseline**: 0.3225 → 0.4048 (+25.5%)

---

## Key Findings

### 1. BM25 with a proper tokeniser dominates standalone vector search

BM25 (Exp 2) achieves F1=0.396, outperforming even bge-base (Exp 3, F1=0.349).
The word-level tokeniser (`re.findall(r'[A-Za-z0-9]+', text)`) matters: the default
whitespace split leaves punctuation attached to tokens ("City." ≠ "City"), silently
penalising entity matches in a news corpus where proper nouns are the primary query signal.

### 2. The fair hybrid finally beats BM25

Without fair truncation the hybrid did *not* beat BM25 on F1.
The cause: the merged RRF list was not truncated, returning up to 10 results instead of 5
and deflating precision. With consistent `top_k` truncation, the hybrid (Exp 4) achieves
F1=0.403, beating BM25 by +0.007. The gain comes entirely from precision (0.401 vs 0.373):
semantic re-ranking within the fused list promotes the most relevant results.

### 3. bge-large adds marginal value over bge-base

Scaling from bge-base (109M params) to bge-large (335M params) yields +0.002 absolute F1
(Exp 5 vs Exp 4). On a 609-article corpus, the embedding quality is not the binding
constraint: BM25 provides the dominant ranking signal, and the semantic component saturates
quickly as model size grows.

### 4. BM25 is the strongest single retriever for MRR

BM25 alone achieves MRR=0.611 — higher than any hybrid configuration (0.599, 0.588).
This means BM25 is best at placing the *first* relevant document at rank 1. RRF fusion
redistributes scores across both retrieval signals, which occasionally demotes a high-
confidence BM25 result. For latency-sensitive single-answer use cases, BM25 alone is the
better choice.

### 5. The corpus characteristics drive the architecture

Entity-heavy news queries (team names, player names, event dates) are best served by
exact keyword matching. Semantic embeddings add recall for paraphrase queries but cannot
replace BM25 for precise entity matching. The winning stack treats both as first-class
retrievers and combines them via RRF.
