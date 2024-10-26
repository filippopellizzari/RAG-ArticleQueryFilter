import logging
import re

import pandas as pd

INPUT_CORPUS_FILENAME = "dataset/corpus.csv"
OUTPUT_CORPUS_FILENAME = "dataset/corpus_clean.csv"


def clean_text(text: str) -> str:
    """Normalize text"""
    # remove newline
    text = text.replace("\n", " ")
    # lowercase and remove extra spaces
    text = text.lower().strip()
    # remove unicode
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    return text


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.info("Starting data cleaning")
    # Read corpus dataset
    corpus = pd.read_csv(INPUT_CORPUS_FILENAME)

    # Create a text column by concatenating title and body
    corpus["text"] = corpus["title"] + " " + corpus["body"]
    # Clean text
    corpus["text"] = corpus["text"].apply(clean_text)

    logging.info("Saving output")
    corpus.to_csv(OUTPUT_CORPUS_FILENAME)

    logging.info("End")


if __name__ == "__main__":
    main()
