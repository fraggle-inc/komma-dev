# coding: utf8
"""Functions for building vocabulary."""
from collections import Counter


def build_vocabulary(sentences, n_words):
    """
    Build the vocabulary.

    Return a tuple of dictionaries. The first item is the vocabulary. The second item is a reverse vocabulary which
    can be used to look up a word given the ID.
    """
    chunks = [chunk.clean_name for sentence in sentences for chunk in sentence.chunks]
    counter = Counter(chunks)
    most_common_words = counter.most_common(n_words)
    vocabulary = {}
    reverse_vocabulary = {}
    for index, item in enumerate(most_common_words):
        word = item[0]
        vocabulary[word] = index + 1
        reverse_vocabulary[index + 1] = word
    return vocabulary, reverse_vocabulary
