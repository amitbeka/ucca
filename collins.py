"""Data structures for Collins cobuild dictionary.

Parsing is in the legacy code, but it produces a dictionary with
some self-explained keys and values (and some errors too).

"""

import nltk

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

    """

    def __init__(self, key, forms, context=None):
        """Initializes the class instance.

        Args:
            key: the key of the Entry
            forms: the derived forms of the Entry. If None, the key will be
                copied as the only form available.
            context: context of the key, if exists (None if not)

        """
        self._key = key
        self._context = context
        self.derived_forms = forms if forms is not None else [key]

    @property
    def key(self):
        return self._key

    def context(self):
        return self._context

    def __hash__(self):
        return hash(self._key + KEY_CONTEXT_SEP + str(self._context))


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
            self._entries[k] = Entry(key, forms, context)

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
