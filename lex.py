"""Contains lexical-related methods and classes."""

import nltk
import pickle
import xml.etree.ElementTree as ETree

from ucca import collins, wikt


class DixonVerbs:
    """Holds the different verb categories according to Dixon's theory (BTL).

    The verb categories are hierarichal, and contain a group, type and
    usually a subtype. Each verb can be part of numerous groups, with or
    without phrasal verbs (e.g. put NP on) and required prepositions
    (e.g. refer to), passivisation, and other transformations.

    """

    def __init__(self, root):
        """Creates a new instance using an XML root element of the categories.

        Args:
            root: an :class:ElementTree.Element object of the root element

        """

        SEP = ':'
        self._verbs = {}

        def _add(word, group, type_, subtype='MAIN'):
            cat = SEP.join([group, type_, subtype])
            if word in self._verbs:
                self._verbs[word].append(cat)
            else:
                self._verbs[word] = [cat]

        for group_elem in root.iterfind('group'):
            group = group_elem.get('id')
            for type_elem in group_elem.iterfind('type'):
                type_ = type_elem.get('id')
                for subtype_elem in type_elem.iterfind('subtype'):
                    subtype = subtype_elem.get('id')
                    for word in [x.text for x in
                                 subtype_elem.iterfind('member')]:
                        _add(word, group, type_, subtype)
                # For members directly under the type, w/o subtype
                for word in [x.text for x in type_elem.iterfind('member')]:
                    _add(word, group, type_)

    def by_phrase(self, phrase):
        """Returns a list of categories for the exact phrase given."""
        return self._verbs.get(phrase, [])

    def by_verb(self, verb):
        """Returns a dictionary of verb:categories according to verb given.

        Given a verb, returns all entries where the verb appears as one of
        the words (contrast to the whole phrase as exact match).

        """
        return {k: v for k, v in self._verbs.items() if verb in k.split()}

    def by_stem(self, stem):
        """Returns a dictionary of verb:categories if the stem matches.

        The stem needs to match the stemmed entry of the verb.

        """
        st = nltk.stem.snowball.EnglishStemmer()
        return {k: v for k, v in self._verbs.items()
                if stem in st.stem(k).split()}


class DixonIdentifier:

    def __init__(self, dixon_path, collins_path, wikt_path):
        with open(dixon_path) as f:
            self.dixon = DixonVerbs(ETree.ElementTree().parse(f))
        with open(collins_path, 'rb') as f:
            self.collins = collins.CollinsDictionary(pickle.load(f))
        with open(wikt_path) as f:
            raw_defs = f.read().split('\n')[:-1]  # last line is empty
            self.wikt = wikt.Wiktionary(raw_defs)
        self.stemmer = nltk.stem.snowball.EnglishStemmer()

    def get_categories(self, scene, head):
        if head.attrib.get('implicit'):
            return 'implicit'
        text = head.to_text()
        base_form = self.collins.by_form(text)
        # For both lemmatizers, we try to lowercase the first letter if it's
        # the only upper letter, because this can be a start-of-sentence
        # capitalization
        if not base_form and text.istitle():
            base_form = self.collins.by_form(text.lower())
        if not base_form:
            try:
                base_form = self.wikt.lemmatize(text)
            except wikt.LemmaNotFound:
                if text.istitle():
                    try:
                        base_form = self.wikt.lemmatize(text.lower())
                    except wikt.LemmaNotFound:
                        return 'no base form'
                else:
                    return 'no base form'
        else:
            base_form = base_form[0].key
        stem = self.stemmer.stem(base_form)
        return (base_form, stem, self.dixon.by_stem(stem))


class FormIdentifier:
    """Identifies which base forms are possible for a given phrase."""

    def __init__(self, collins_path, wikt_path):
        with open(collins_path, 'rb') as f:
            self.collins = collins.CollinsDictionary(pickle.load(f))
        with open(wikt_path) as f:
            raw_defs = f.read().split('\n')[:-1]  # last line is empty
            self.wikt = wikt.Wiktionary(raw_defs)

    def get_forms(self, phrase):
        coll_entries = self.collins.by_form(phrase)
        wikt_entries = self.wikt.by_form(phrase)
        if phrase.istitle():
            coll_entries += self.collins.by_form(phrase.lower())
            wikt_entries += self.wikt.by_form(phrase.lower())
        forms = {}
        for entry in coll_entries:
            forms[entry.key] = (forms.get(entry.key, set()) |
                                {s.pos for s in entry.senses})
        for entry in wikt_entries:
            forms[entry.lemma] = forms.get(entry.lemma, set()) | {entry.pos}
        return forms
