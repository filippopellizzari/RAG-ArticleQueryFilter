"""Unit tests for src.data text-cleaning functions."""

from src.data import minimal_clean_text


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
