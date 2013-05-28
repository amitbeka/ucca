"""Wiktionary-related classes and functionality."""

import re
from ucca.core import UCCAError
from ucca.postags import POSTags


class MultipleLemmasError(UCCAError):
    pass


class LemmaNotFound(UCCAError):
    pass


class WiktEntry:
    """Represents an entry of a phrase, with its POS, definition and lemma."""

    _tags = {'Noun': POSTags.Noun,
             'Proper noun': POSTags.Noun,
             'Verb': POSTags.Verb}

    def __init__(self, phrase, pos, defn, lemma):
        self.phrase = phrase
        self.defn = defn
        self.lemma = lemma
        if pos in self._tags:
            self.pos = self._tags[pos]
        else:
            self.pos = POSTags.Other

    def __str__(self):
        return "{} ({})\t{}\t{}".format(self.phrase, self.lemma, self.pos,
                                        self.defn)


class Wiktionary:
    """Wiktionary object which provides lemmas and definitions.

    For lemmatization, we use the structure of wiktionary definitions.
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
        """Initializes a wiktionary object from raw entries.

        Args:
            raw: a list of wiktionary entries, given as tab-separated strings
                of language (ignored), phrase, Part-of-speech and definition.

        """
        self._entries = {}
        for line in raw:
            _, phrase, pos, defn = line.split('\t')
            match = re.search(Wiktionary._pattern, defn)
            if match is None:
                lemma = phrase
            else:
                lemma = match.group(2)
                if lemma.startswith('[[') and lemma.endswith(']]'):
                    lemma = lemma[2:-2]  # removing wikilink markup
            self._entries[phrase] = self._entries.get(phrase, []) + [WiktEntry(
                phrase, pos, defn, lemma)]

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
        if phrase not in self._entries:
            raise LemmaNotFound()
        # Phrase ==> possible lemmas ==> entries for these lemmas
        entries = []
        for orig_entry in self._entries[phrase]:
            entries.extend(self._entries.get(orig_entry.lemma, []))
        if pos is None:
            return sorted(entries, key=lambda x: len(x.phrase))[0].lemma
        else:
            lemmas = {e.lemma for e in entries if e.pos == pos}
            if len(lemmas) == 0:
                raise LemmaNotFound
            elif len(lemmas) == 1:
                return lemmas.pop()
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
        return {e.lemma for e in self._entries[phrase]}

    def by_form(self, phrase):
        """Returns a list of entries which have the phrase as their form.

        Args:
            phrase: the phrase to search for

        Returns:
            a list of all WiktEntry objects whose lemma is the given phrase,
            or that appear as a lemma of this phrase WiktEntry.

        """
        entries = []
        for e in self._entries.get(phrase, []):
            if e.phrase == e.lemma:
                entries.append(e)
            else:
                entries.extend(x for x in self._entries.get(e.lemma, [])
                               if x.pos == e.pos)
        return entries
