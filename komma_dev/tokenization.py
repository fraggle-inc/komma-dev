# coding: utf8
"""Functions for building vocabulary."""
import regex


class Token():
    """Class representing a single token."""

    def __init__(self, name, kind, start_index, end_index):
        """Initialize a new token."""
        self.name = name
        self.kind = kind
        self.start_index = start_index
        self.end_index = end_index

    def __repr__(self):
        """Return a printable string representing the token."""
        # return "%s (%s)" % (self.name.replace(" ", "_"), self.kind)
        # return "%s" % (self.name.replace(" ", "_"))
        return self.name


WHITESPACE_TYPE = "WHITESPACE"
TEXT_TYPE = "TEXT"


def _get_type(char):
    if char == " ":
        return WHITESPACE_TYPE
    return TEXT_TYPE


def _split_string(text):
    substrings = []
    last_type = None
    start_index = None
    for index, char in enumerate(text):
        current_type = _get_type(char)
        if current_type == last_type:
            continue

        if last_type:
            temp = text[start_index:index]
            substrings += [temp]

        start_index = index
        last_type = current_type

    if last_type:
        substrings += [text[start_index:]]
    return substrings


class Tokenizer(object):
    """String tokenizer class."""

    def __init__(self):
        """Initialize a new tokenizer."""
        self.prefix_search = regex.compile(r'''^([\[\("'\.\-\?])''').search
        self.suffix_search = regex.compile(r'''([\]\)"'\.\-\?])$''').search

    def tokenize(self, text):
        """Tokenize specified string."""
        tokens = []
        for substring in _split_string(text):
            suffixes = []
            while substring:
                if self._find_prefix(substring):
                    split = self._find_prefix(substring)
                    token = Token(substring[:split], "prefix", 0, 0)
                    tokens.append(token)
                    substring = substring[split:]
                    continue

                if self._find_suffix(substring):
                    split = self._find_suffix(substring)
                    token = Token(substring[split:], "suffix", 0, 0)
                    suffixes.append(token)
                    substring = substring[:split]
                    continue

                token = Token(substring, "text", 0, 0)
                tokens.append(token)
                substring = ''
            tokens.extend(reversed(suffixes))
        return tokens

    def _find_prefix(self, string):
        if self.prefix_search is None:
            return 0
        match = self.prefix_search(string)
        return (match.end() - match.start()) if match is not None else 0

    def _find_suffix(self, string):
        if self.suffix_search is None:
            return 0
        match = self.suffix_search(string)
        return len(string) - (match.end() - match.start()) if match is not None else 0
