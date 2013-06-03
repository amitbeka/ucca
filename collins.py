"""Data structures for Collins cobuild dictionary.

Parsing is in the legacy code, but it produces a dictionary with
some self-explained keys and values (and some errors too).

"""

import nltk
import random

from ucca.postags import POSTags

KEY_CONTEXT_SEP = '#'


class Entry:
    """Collins dictionary entry structure.

    Each entry has a unique key+context pair.

    Attributes:
        key: the key name (word or phrase) (read-only)
        context: the context of the key, if it exists. Contexts are available
            for entries with the same key, but different senses in general,
            e.g. 'apart' in 'stand apart' (position context) VS.
            'apart' in 'apart from her' (exceptions context).
            If no context exists, it is None. (read only)
        derived_forms: list of possible forms for the key. Cannot be empty.
        senses: list of possible senses for this key.

    """

    class Sense:
        """Represents one sense of a dictionary entry.

        A sense is composed of a meaning with examples, part-of-speech tag and
        other relevant data, but currently only the part-of-speech tag and
        usage line are extracted, and the examples + description are saved as
        is.

        Attributes:
            pos: part-of-speech tag
            usage: usage line, with its members separated by tabs
            desc: description and examples

        """

        _tags = {'N-': POSTags.Noun,
                 'VERB': POSTags.Verb,
                 'PHRASAL': POSTags.Verb,
                 'V-': POSTags.Verb,
                 'MODAL': POSTags.Modal}

        def __init__(self, sense_dic):
            self.usage = '\t'.join(sense_dic['POS'])
            self.pos = POSTags.Other
            self.desc = "\n".join([sense_dic['def']] + sense_dic['examples'])
            if not sense_dic['POS']:
                return
            for pattern, tag in self._tags.items():
                if sense_dic['POS'][0].startswith(pattern):
                    self.pos = tag
                    break

        def __str__(self):
            return "Part-of-speech: {}\nUsage: {}\n{}".format(self.pos,
                                                              self.usage,
                                                              self.desc)

    def __init__(self, key, forms, senses, context=None):
        """Initializes the class instance.

        Args:
            key: the key of the Entry
            forms: the derived forms of the Entry. If None, the key will be
                copied as the only form available.
            senses: list of sense entries, each is a dict
            context: context of the key, if exists (None if not)

        """
        self._key = key
        self._context = context
        self.derived_forms = forms if forms is not None else [key]
        self.senses = [self.Sense(x) for x in senses]

    @property
    def key(self):
        return self._key

    @property
    def context(self):
        return self._context

    def __hash__(self):
        return hash(self._key + KEY_CONTEXT_SEP + str(self._context))

    def __str__(self):
        new_key = self.key if self._context is None else '{} ({})'.format(
            self.key, self._context)
        return "Key: {}\nDerived Forms: {}\nSenses:\n{}".format(
            new_key, ', '.join(self.derived_forms),
            '\n===\n'.join(str(x) for x in self.senses))


class CollinsDictionary:
    """Collins dictionary structure."""

    def __init__(self, raw):
        """Initializes the class instance from a raw dictionary instance.

        Args:
            raw: a Python dict with each key mapped to another dict, with
                some pre-defined names. Keys with the form s1#s2 are split
                to key and context.

        """
        self._entries = {}
        for k, v in raw.items():
            if KEY_CONTEXT_SEP in k:
                key, context = k.split(KEY_CONTEXT_SEP)
            else:
                key, context = k, None
            if v['forms']:
                forms = v['forms']
            else:
                forms = [key]
            self._entries[k] = Entry(key, forms, v['senses'], context)

    def by_key(self, key):
        """Returns a list of entries whose key matches.

        Args:
            key: the key to match

        Returns:
            a list of matching :class:Entry objects, can be empty.

        """
        return [v for v in self._entries.values() if v.key == key]

    def by_form(self, form):
        """Returns a list of entries who have this derived forms.

        Args:
            form: the form to look for.

        Returns:
            a list of matching :class:Entry objects, can be empty.

        """
        return [v for v in self._entries.values() if form in v.derived_forms]

    def by_stem(self, stem):
        """Return a list of entries whose stemmed keys is as given.

        Uses nltk.stem.snowball.EnglishStemmer (Porter's algorithm).

        Args:
            stem: the stem to match.

        Returns:
            a list of matching :class:Entry objects, can eb empty.

        """
        st = nltk.stem.snowball.EnglishStemmer()
        return [v for v in self._entries.values() if st.stem(v.key) == stem]

    def random_entry(self, postag):
        """Returns a random entry from the dictionary.

        Args:
            poastag: the POSTag that the random entry should match.
            At least one of the senses must have it.

        Returns:
            an Entry object

        """
        keys = list(self._entries.keys())
        while True:
            candidate = self._entries[random.choice(keys)]
            if any(s.pos == postag for s in candidate.senses):
                break
        return candidate
