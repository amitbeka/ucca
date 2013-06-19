"""Encapsulates all word and punctuation symbols layer.

Layer 0 is the basic layer for all the UCCA annotation, as it includes the
actual words and punctuation marks found in the :class:core.Passage.

Layer 0 has only one type of node, :class:Terminal. This is a subtype of
:class:core.Node, and can have one of two tags: Word or Punctuation.

"""

from ucca import core

# Layer ID
LAYER_ID = '0'


class NodeTags:
    Punct = 'Punctuation'
    Word = 'Word'
    __init__ = None


ATTRIB_KEYS = ('text', 'paragraph', 'paragraph_position')


class Terminal(core.Node):
    """Layer 0 Node type, represents a word or a punctuation mark.

    Terminals are :class:core.Node objects which represent a word or
    a punctuation mark in the :class:core.Passage object. They are immutable,
    as they shouldn't be changed throughout their use and have no children.
    Hence, they can be compared and hashed, unlike other core.Node subclasses.

    Attributes:
        ID: the unique ID of each Terminal is its global position in the
        Passage, e.g. ID=0.4 is the 4th Terminal in the :class:Passage.
        tag: from NodeTags
        layer: '0' (LAYER_ID)
        attrib: returns a copy of the attribute dictionary, so changing it
            will not affect the Terminal object
        text: text of the Terminal, whether punctuation or a word
        position: global position of the Terminal in the passage, starting at 1
        paragraph: which paragraph the Terminal belongs to, starting at 1
        para_pos: the position of the Terminal in the paragraph,
            starting at 1 (per paragraph).
        punct: whether the Terminal is a punctuation mark (boolean)

    """

    @property
    def text(self):
        return self.attrib['text']

    @property
    def position(self):
        # the format of ID is LAYER_ID + ID separator + position
        return int(self.ID[len(LAYER_ID) + len(core.Node.ID_SEPARATOR):])

    @property
    def para_pos(self):
        return self.attrib['paragraph_position']

    @property
    def paragraph(self):
        return self.attrib['paragraph']

    @property
    def attrib(self):
        return self._attrib.copy()

    @property
    def punct(self):
        return self.tag == NodeTags.Punct

    def equals(self, other, *, ordered=False):
        """Equals if the Terminals are of the same Layer, tag, position & text.

        Args:
            other: another Terminal to equal to
            ordered: unused, here for API conformness.

        Returns:
            True iff the two Terminals are equal.

        """
        return (self.layer.ID == other.layer.ID and self.text == other.text
                and self.position == other.position and self.tag == other.tag
                and self.paragraph == other.paragraph
                and self.para_pos == other.para_pos)

    def __eq__(self, other):
        """Equals if both of the same Passage, Layer, position, tag & text."""
        if other.layer.ID != LAYER_ID:
            return False
        return (self.root == other.root and self.layer.ID == other.layer.ID
                and self.position == other.position
                and self.text == other.text and self.tag == other.tag
                and self.paragraph == other.paragraph
                and self.para_pos == other.para_pos)

    def __hash__(self):
        """Hashes the Terminals aacording to its ID and text."""
        return hash(self.ID + str(self.text))

    def __str__(self):
        return self.text

    # Terminal are immutable (except the extra dictionary which is
    # just a temporary playground) and have no children, so enforce it
    add = None
    remove = None


class Layer0(core.Layer):
    """Represents the :class:Terminal objects layer.

    Attributes:
        words: a tuple of only the words (not punctuation) Terminals, ordered
        pairs: a tuple of (position, terminal) tuples of all Terminals, ordered

    """

    def __init__(self, root, attrib=None):
        return super().__init__(ID=LAYER_ID, root=root, attrib=attrib)

    @property
    def words(self):
        return tuple(x for x in self.all if not x.punct)

    @property
    def pairs(self):
        return tuple(zip(range(1, len(self.all) + 1), self._all))

    def by_position(self, pos):
        """Returns the Terminals at the position given.

        Args:
            pos: the position of the Terminal object

        Returns:
            the Terminal in this position

        Raises:
            IndexError if the position is out of bounds

        """
        return self._all[pos - 1]  # positions start at 1, not 0

    def add_terminal(self, text, punct, paragraph=1):
        """Adds the next Terminal at the next available position.

        Creates a :class:Terminal object with the next position, assuming that
        all positions are filled (no holes).

        Args:
            text: the text of the Terminal
            punct: boolen, whether it's a punctuation mark
            paragraph: paragraph number, defualts to 1

        Returns:
            the created Terminal

        Raises:
            DuplicateIdError: if trying to add an already existing Terminal,
                caused by un-ordered Terminal positions in the layer

        """
        position = len(self.all) + 1  # we want positions to start with 1
        if position > 1 and paragraph == self.all[-1].paragraph:
            para_pos = self.all[-1].para_pos + 1
        else:
            para_pos = 1

        tag = NodeTags.Punct if punct else NodeTags.Word
        return Terminal(ID="{}{}{}".format(LAYER_ID, core.Node.ID_SEPARATOR,
                                           position),
                        root=self.root, tag=tag,
                        attrib={'text': text,
                                'paragraph': paragraph,
                                'paragraph_position': para_pos})

    def copy(self, other_passage):
        """Creates a copied Layer0 object and Terminals in other_passage.

        Args:
            other_passage: the Passage to copy self to

        """
        other = Layer0(root=other_passage, attrib=self.attrib.copy())
        other.extra = self.extra.copy()
        for t in self._all:
            copied = other.add_terminal(t.text, t.punct, t.paragraph)
            copied.extra = t.extra.copy()


def is_punct(node):
    """Returns whether the unit is a layer0 punctuation (for all Units)."""
    return (node.punct if node.layer == LAYER_ID else False)
