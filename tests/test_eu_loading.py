# coding: utf8
"""Tests for loading of EU data."""
# pylint: disable=protected-access,too-many-public-methods,no-self-use,too-few-public-methods,C0103

import komma_dev.eu
import py.test


class TestLoading(object):
    """Test class for loading of EU data."""

    @py.test.mark.slow
    def test_substrings(self):
        """Test that EU data can be loaded correctly."""
        eu_data = komma_dev.eu.load('da')
        expected_length = 1968800
        actual_length = len(eu_data)
        assert expected_length == actual_length

        expected_first = "Genoptagelse af sessionen"
        actual_first = eu_data[0]
        assert expected_first == actual_first

        expected_last = "(Mødet hævet kl. 10.50)"
        actual_last = eu_data[-1]
        assert expected_last == actual_last
