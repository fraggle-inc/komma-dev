# coding: utf8
"""Tests for string tokenization."""
# pylint: disable=protected-access,too-many-public-methods,no-self-use,too-few-public-methods,C0103

from komma_dev.tokenization import Tokenizer, _split_string


class TestTokenization(object):
    """Test class for string tokenization."""

    def test_substrings(self):
        """Test substrings."""
        dirty = "Der  var 100 kr. i pungen."
        expected = ["Der", "  ", "var", " ", "100", " ", "kr.", " ", "i", " ", "pungen."]
        actual = _split_string(dirty)
        assert expected == actual

    def test_tokenization(self):
        dirty = "Der  var 100 kr. i pungen."
        expected = ["Der", "  ", "var", " ", "100", " ", "kr", ".", " ", "i", " ", "pungen", "."]
        tokenizer = Tokenizer()
        actual = [token.name for token in tokenizer.tokenize(dirty)]
        print("expected:", expected[0])
        print("actual:", actual[0])
        assert expected == actual
