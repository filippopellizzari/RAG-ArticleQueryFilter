"""Unit tests for src.data text-cleaning functions."""

import pytest

from src.data import clean_text, minimal_clean_text


class TestCleanText:
    def test_lowercases(self):
        assert clean_text("Hello World") == "hello world"

    def test_collapses_whitespace(self):
        assert clean_text("a  b\t\nc") == "a b c"

    def test_replaces_non_ascii_with_space(self):
        # Regression: non-ASCII must become a space, not be deleted,
        # to avoid silently merging adjacent tokens ("word1—word2" → "word1word2").
        assert clean_text("word1\u2014word2") == "word1 word2"

    def test_strips_leading_trailing_whitespace(self):
        assert clean_text("  hello  ") == "hello"

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_pure_non_ascii_collapses_to_empty(self):
        assert clean_text("\u00e9\u00e0") == ""


class TestMinimalCleanText:
    def test_preserves_casing(self):
        assert minimal_clean_text("Hello World") == "Hello World"

    def test_collapses_whitespace(self):
        assert minimal_clean_text("a  b\t\nc") == "a b c"

    def test_preserves_unicode(self):
        assert minimal_clean_text("caf\u00e9 \u2014 na\u00efve") == "caf\u00e9 \u2014 na\u00efve"

    def test_strips_leading_trailing_whitespace(self):
        assert minimal_clean_text("  hello  ") == "hello"

    def test_empty_string(self):
        assert minimal_clean_text("") == ""

    def test_does_not_lowercase(self):
        assert minimal_clean_text("Premier League") == "Premier League"
