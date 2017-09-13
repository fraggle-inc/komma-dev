# coding: utf8
# pylint: disable=too-few-public-methods

"""Functions for parsing and tokenizing strings as sentences."""
import regex

WHITESPACE_TYPE = "WHITESPACE"
TEXT_TYPE = "TEXT"
NUMBER_TYPE = "NUMBER"
PREFIX_TYPE = "PREFIX"
SUFFIX_TYPE = "SUFFIX"
NUMBER_REGEX = regex.compile(r"^([\+-])?([0-9]+([\+0-9,\.-]*[0-9\.])?),?$")
NUMBER_PLACEHOLDER = "NUMBER"


class StringParser(object):
    """Class for parsing strings as sentences."""

    def __init__(self):
        """Initialize a new parser."""
        self.prefix_search = regex.compile(r'''^([\[\("'\-\?])''').search
        self.suffix_search = regex.compile(r'''([\]\)"'\-\?])$''').search

    def parse(self, text):
        """Parse specified string and return a Sentence instace."""
        tokens = self._tokenize(text)
        parsed = []
        stack = []
        for index, token in enumerate(tokens):
            if index == 0:
                stack += [token]
                continue

            if token.kind == WHITESPACE_TYPE:
                stack += [token]
                continue

            if (stack[0].kind != NUMBER_TYPE or token.kind != NUMBER_TYPE) and token.kind != WHITESPACE_TYPE:
                if len(stack) == 1:
                    parsed += [Chunk(stack[0].name, "", stack[0].kind)]
                    stack = [token]
                    continue

                temp_text = ''.join(t.name for t in stack[:-1])
                if stack[-1].kind == WHITESPACE_TYPE:
                    temp = Chunk(temp_text, stack[-1].name, stack[0].kind)
                else:
                    temp = Chunk(temp_text + stack[-1].name, "", stack[0].kind)
                parsed += [temp]
                stack = [token]
                continue

            stack += [token]
            continue

        if stack:
            temp_text = ''.join(t.name for t in stack)
            parsed += [Chunk(temp_text, "", stack[0].kind)]

        sentence = Sentence(parsed)
        assert text == sentence.text
        return sentence

    def _tokenize(self, text):
        """Tokenize specified string."""
        tokens = []
        for substring in _split_string(text):
            suffixes = []
            while substring:
                if self._find_prefix(substring):
                    split = self._find_prefix(substring)
                    token = Token(substring[:split], PREFIX_TYPE)
                    tokens.append(token)
                    substring = substring[split:]
                    continue

                if self._find_suffix(substring):
                    split = self._find_suffix(substring)
                    token = Token(substring[split:], SUFFIX_TYPE)
                    suffixes.append(token)
                    substring = substring[:split]
                    continue

                if substring.strip() == "":
                    token_type = WHITESPACE_TYPE
                elif NUMBER_REGEX.match(substring):
                    token_type = NUMBER_TYPE
                else:
                    token_type = TEXT_TYPE

                token = Token(substring, token_type)
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


class Sentence():
    """Class representing af single sentence."""

    def __init__(self, chunks):
        """Initialize a new sentence with specified chunks."""
        self.chunks = chunks

    def __str__(self):
        """Return a printable string representing the sentence."""
        return self.text

    @property
    def text(self):
        """Return the text of the sentence."""
        return ''.join([chunk.name + chunk.trailing_whitespace for chunk in self.chunks])

    @property
    def commas(self):
        """Return a list of length equal to number of chunks where 1 indicates a comma and zero indicates no comma."""
        # TODO: create a backing variable. Maybe return an NP array.
        return [1 if chunk.comma else 0 for chunk in self.chunks]

    @property
    def features(self):
        """Return array of features that can be used to pass to classifier."""
        return [chunk.feature for chunk in self.chunks]


class Chunk(object):
    """Class representing a chunk of a sentence."""

    def __init__(self, name, trailing_whitespace, kind):
        """Initialize a new chunk."""
        self.name = name
        self.trailing_whitespace = trailing_whitespace
        self.kind = kind

    def __str__(self):
        """Return a printable string representing the chunk."""
        return self.name + self.trailing_whitespace

    @property
    def comma(self):
        """Return a value indicating whether the chunk ends with a comma."""
        return self.name[-1] == ","

    @property
    def feature(self):
        """
        Return a representation of the chunk that can be used as a feature for the classifier.

        For number chunks, the string 'NUMBER' will be returned. For chunks containing a comma, that comma will be
        removed.
        """
        return NUMBER_PLACEHOLDER if self.kind == NUMBER_TYPE else self.clean_name

    @property
    def clean_name(self):
        """Return a clean version of the chunk contents. This means stripping trailing comma."""
        # TODO: Create a backing variable.
        temp = self.name
        # TODO: Why does this crash if the left conjunct is not there? That means we have empty tokens or comma only?
        while temp and temp[-1] == ",":
            temp = temp[:-1]
        temp = temp.replace(".", "")
        return temp


class Token(object):
    """Class representing a single token."""

    def __init__(self, name, kind):
        """Initialize a new token."""
        self.name = name
        self.kind = kind

    def __str__(self):
        """Return a printable string representing the token."""
        return self.name


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


def _get_type(char):
    if char == " ":
        return WHITESPACE_TYPE
    return TEXT_TYPE
