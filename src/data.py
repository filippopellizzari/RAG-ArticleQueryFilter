"""Text preprocessing for the news article corpus.

``minimal_clean_text`` is the production strategy: whitespace normalisation only,
casing and unicode preserved. Confirmed best for retrieval by ablation —
see ``results/retrieval_experiments.md``.

``clean_text`` (aggressive: lowercase + ASCII-only) is kept for reference and
notebook experiments but is not used in the production pipeline.
"""

import logging
import re

import pandas as pd

from src.config import (
    CORPUS_PATH,
    CORPUS_RAW_PATH,
)

log = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Aggressive normalisation: lowercase and strip non-ASCII characters.

    Non-ASCII characters are replaced with a space (not removed outright) to
    avoid silently merging adjacent tokens (e.g. "word1\u2014word2" must not
    become "word1word2").
    """
    text = re.sub(r"\s+", " ", text).strip()
    text = text.lower()
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text.strip()


def minimal_clean_text(text: str) -> str:
    """Conservative normalisation: collapse whitespace only.

    Preserves casing, punctuation, and unicode — critical for BM25 entity
    matching and for embedding models trained on mixed-case text.
    """
    return re.sub(r"\s+", " ", text).strip()


def _build_corpus(raw_path: str | None = None) -> pd.DataFrame:
    path = raw_path or CORPUS_RAW_PATH
    corpus = pd.read_csv(path)
    corpus["text"] = corpus["title"].fillna("") + " " + corpus["body"].fillna("")
    return corpus


def main() -> None:
    """CLI entry point: produce the cleaned corpus from the raw CSV."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    log.info("Loading raw corpus from %s", CORPUS_RAW_PATH)
    corpus = _build_corpus()
    log.info("Loaded %d articles", len(corpus))

    corpus["text"] = corpus["text"].apply(minimal_clean_text)
    corpus.to_csv(CORPUS_PATH, index=False)
    log.info("Saved corpus → %s", CORPUS_PATH)


if __name__ == "__main__":
    main()
