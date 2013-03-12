"""Wiktionary-related classes and functionality."""

import re
from ucca.core import UCCAError


class MultipleLemmasError(UCCAError):
    pass


class LemmaNotFound(UCCAError):
    pass


class WiktLemmatizer:
    """Represents a lemmatizer based on wiktionary's inflected entries.

    On English wiktionary (and others) all inflected/transformed or alternative
    forms of a word in general receive their own entry, whose definition is
    a template of the kind {{X of|Y}}, e.g. {{abbreviation of|et cetera}}.

    These templates are parsed and used to create a static lemmatizer, where
    each form maps either to itself or to the main form extracted from the
    template (assuming one for each entry).

    """

    # The basic pattern form is {{X of|Y}}.
    # As this template can appear with other templates, we restrict X and Y
    # to exclude '}' symbols so we won't match two templates together.
    # Some templates have options, given as a third argument, e.g. {{X of|Y|Z}}
    # We therefore exclude '|' from X and Y matching and add a possible
    # match for this addition.
    _pattern = r'{{([^|}]+) of\|([^|}]+)\|?.*?}}'

    def __init__(self, raw):
        """Initializes a wiktionary lemmatizer object from raw entries.

        Args:
            raw: a list of wiktionary entries, given as tab-separated strings
                of language (ignored), phrase, Part-of-speech and definition.

        """
        self._mapping = {}
        for line in raw:
            _, phrase, pos, defn = line.split('\t')
            match = re.search(WiktLemmatizer._pattern, defn)
            if match is None:
                if self._mapping.get(phrase):
                    self._mapping[phrase].add((pos, phrase))
                else:
                    self._mapping[phrase] = {(pos, phrase)}
            else:
                orig = match.group(2)
                if self._mapping.get(phrase):
                    self._mapping[phrase].add((pos, orig))
                else:
                    self._mapping[phrase] = {(pos, orig)}

    def lemmatize(self, phrase, pos=None):
        """Lemmatizes the phrase and returns the lemmatized string.

        Args:
            phrase: the phrase to lemmatize
            pos: the part-of-speech of the phrase, according to which trying to
                lemmatize. If not given, picks (one of) the shortest lemma
                exists for that phrase

        Returns:
            the lemma of the phrase

        Raises:
            LemmaNotFound if the phrase doesn't exist in the lemmatizer.
            MultipleLemmasError if the phrase have more than one lemma for
            the POS given (isn't raised when pos is None).

        """
        if phrase not in self._mapping:
            raise LemmaNotFound()
        if pos is None:
            return sorted(self._mapping[phrase], key=lambda x: len(x[1]))[0][1]
        else:
            lemmas = [lemma for lpos, lemma in self._mapping[phrase]
                      if lpos == pos]
            if len(lemmas) == 0:
                raise LemmaNotFound
            elif len(lemmas) == 1:
                return lemmas[0]
            else:
                raise MultipleLemmasError

    def lemmas(self, phrase):
        """Returns a set of possible lemmas for the given phrase.

        Args:
            phrase: the phrase to lemmatize

        Returns:
            a set of (part-of-speech, lemma) tuples for all possible
            lemmatizations of the phrase. Can be empty.

        """
        return self._mapping.get(phrase, set())
