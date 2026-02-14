import logging
import re

import pandas as pd

INPUT_CORPUS_FILENAME = "dataset/corpus.csv"
OUTPUT_CORPUS_FILENAME = "dataset/corpus_clean.csv"
OUTPUT_CORPUS_MINIMAL_FILENAME = "dataset/corpus_minimal_clean.csv"


def clean_text(text: str) -> str:
    """Aggressive normalization: lowercase, strip non-ASCII."""
    text = text.replace("\n", " ")
    text = text.lower().strip()
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    return text


def minimal_clean_text(text: str) -> str:
    """Minimal normalization: preserve casing and unicode, only fix whitespace."""
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.info("Starting data cleaning")
    corpus = pd.read_csv(INPUT_CORPUS_FILENAME)

    # Create a text column by concatenating title and body
    corpus["text"] = corpus["title"] + " " + corpus["body"]

    # Aggressive clean (original)
    corpus_clean = corpus.copy()
    corpus_clean["text"] = corpus_clean["text"].apply(clean_text)
    corpus_clean.to_csv(OUTPUT_CORPUS_FILENAME, index=False)
    logging.info("Saved %s", OUTPUT_CORPUS_FILENAME)

    # Minimal clean (preserves casing + unicode)
    corpus_minimal = corpus.copy()
    corpus_minimal["text"] = corpus_minimal["text"].apply(minimal_clean_text)
    corpus_minimal.to_csv(OUTPUT_CORPUS_MINIMAL_FILENAME, index=False)
    logging.info("Saved %s", OUTPUT_CORPUS_MINIMAL_FILENAME)

    logging.info("End")


if __name__ == "__main__":
    main()
