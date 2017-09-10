# coding: utf8
"""Tests for string tokenization."""
# pylint: disable=protected-access,too-many-public-methods,no-self-use,too-few-public-methods,C0103

from komma_dev.parsing import StringParser, _split_string


class TestTokenization(object):
    """Test class for string tokenization."""

    def test_substrings(self):
        """Test that a simple string can be split correctly."""
        dirty = "Der  var 100 kr. i pungen"
        expected = ["Der", "  ", "var", " ", "100", " ", "kr.", " ", "i", " ", "pungen"]
        actual = _split_string(dirty)
        assert expected == actual

    def test_tokenization(self):
        """Test that a simple string can be tokenized correctly."""
        dirty = "Der  var 100 kr. i pungen"
        expected = ["Der", "  ", "var", " ", "100", " ", "kr.", " ", "i", " ", "pungen"]
        parser = StringParser()
        actual = [token.name for token in parser._tokenize(dirty)]
        print("expected:", expected[0])
        print("actual:", actual[0])
        assert expected == actual

    def test_parse_simple(self):
        """Test that a simple string can be parsed correctly."""
        dirty = "Der  var 100 kr. i pungen"
        expected_chunk_names = ["Der", "var", "100", "kr.", "i", "pungen"]
        parser = StringParser()
        actual_sentence = parser.parse(dirty)
        actual_chunks_names = [token.name for token in actual_sentence.chunks]
        assert expected_chunk_names == actual_chunks_names
        assert dirty == actual_sentence.text
        expected_features = ["Der", "var", "NUMBER", "kr.", "i", "pungen"]
        assert expected_features == actual_sentence.features

    def test_parse_complicated(self):
        """Test that a string including multiple whitespace as well as commas and number groups is parsed correctly."""
        dirty = "Du  kan ringe, til 'mig' på +45 89674523, eller 7856"
        expected_chunk_names = ["Du", "kan", "ringe,", "til", "'", "mig", "'", "på", "+45 89674523,", "eller", "7856"]
        parser = StringParser()
        actual_sentence = parser.parse(dirty)
        actual_chunks_names = [token.name for token in actual_sentence.chunks]
        assert expected_chunk_names == actual_chunks_names
        assert dirty == actual_sentence.text
        expected_features = ["Du", "kan", "ringe", "til", "'", "mig", "'", "på", "NUMBER", "eller", "NUMBER"]
        assert expected_features == actual_sentence.features
        expected_commas = [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0]
        assert expected_commas == actual_sentence.commas
