"""Contains lexical-related methods and classes."""

import nltk
import xml.etree.ElementTree as ETree


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
