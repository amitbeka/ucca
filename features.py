"""Feature extractor for UCCA scene-evoking classifier.

There are two kinds of features: lexical features extracted from the the
classified target word, and context features extracted from its instances
in non-UCCA corpora. In addition, extracting the class for the word from
UCCA passages is also covered.

"""


def extract_ngrams(size, sentences, given=None):
    """Extracts all ngrams of the given size from the sentences.

    Args:
        size: the N parameter of the ngrams to extract
        sentences: sequence (can be generator) of lists of strings, each
            is a token.
        given: previous ngram dictionary to start with. None otherwise.
            This dictionary isn't changed by the method.

    Returns:
        a new dictionary with (ngram, count) pairs.

    """

    if given is None:
        counts = given.copy()
    else:
        counts = {}
    for s in sentences:
        if len(s) < size:
            continue
        for i in (len(s) - size + 1):
            counts[s[i:i + size]] = counts.get(s[i:i + size], 0) + 1
    return counts
