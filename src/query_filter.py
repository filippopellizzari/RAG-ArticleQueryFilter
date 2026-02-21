"""Toxic-query filtering using a fine-tuned BERT classifier.

The model is lazy-loaded on first use via ``@lru_cache`` so that importing this
module does not trigger a heavyweight download or GPU allocation. The API and
CLI tools that import ``is_query_valid`` only pay the loading cost when the
function is actually called for the first time.

Model: JungleLee/bert-toxic-comment-classification
A bert-base-uncased variant fine-tuned on the Jigsaw Toxic Comment dataset.
Reference: https://huggingface.co/JungleLee/bert-toxic-comment-classification
"""

from functools import lru_cache

from transformers import pipeline as hf_pipeline

from src.config import TOXIC_MODEL_PATH


@lru_cache(maxsize=1)
def _pipeline():
    """Load and cache the toxicity classifier (first call only)."""
    return hf_pipeline("text-classification", model=TOXIC_MODEL_PATH)


def is_query_valid(query: str) -> bool:
    """Return True if the query does not contain harmful content."""
    return _pipeline()(query)[0]["label"] == "non-toxic"
