"""This modules encapsulates all scene-related functions.

Scene-related functions are used to extract, identify and label scenes
in the foundational layer of UCCA.

"""

import re
from ucca import layer0, layer1


def extract_possible_scenes(passage):
    """Extracts all possible scenes from a Passage.

    Possible scenes are currently units which follow these restrictions:
        1. It is a Participant of a main relation (Process/State)
        2. It is not a remote participant
        If so, add the participant of it if has only one center and at least
        one elaborator, or add all centers otherwise.
        In addition, add all elaborators which have more than one child (less
        likely to have one-terminal elaborators as elaborating scenes).

    Args:
        passage: the core.Passage object to extract optional scenes from

    Returns:
        a list of core.Node objects which are candidates for scenes

    """
    l1 = passage.layer(layer1.LAYER_ID)
    ret = []
    for scene in (x for x in l1.all if x.tag == layer1.NodeTags.Foundational
                  and x.is_scene()):
        for p in (e.child for e in scene
                  if e.tag == layer1.EdgeTags.Participant and
                  not e.attrib.get('remote')):
            if p.is_scene() or (len(p.centers) == 1 and p.elaborators):
                ret.append(p)
            else:  # if there are more than one center, add all of them
                ret.extend(c for c in p.centers)
            ret.extend(e for e in p.elaborators if len(e) > 1)
    return ret


def extract_non_scenes_AGCE(passage):
    """Extracts all non-scene participant/center/elaborator/ground objects.

    Args:
        passage: core.Passage object to extract FNodes from

    Returns:
        a list of layer1.FundationalNodes objects, or an empty list

    """
    l1 = passage.layer(layer1.LAYER_ID)
    requested_tags = (layer1.EdgeTags.Participant, layer1.EdgeTags.Center,
                      layer1.EdgeTags.Elaborator, layer1.EdgeTags.Ground)
    return [fnode for fnode in l1.all
            if fnode.tag == layer1.NodeTags.Foundational and
            fnode.ftag in requested_tags]


def extract_head(fnode):
    """Extracts the head of the FoundationalNode given.

    The head is defined differently and recursively for scenes and non-scenes:
        1. For non-scenes, the head is defined as the head of the only Center
            (recursively). If there are multiple Centers, there is no head.
        2. For scenes, the head is the head of the Process/State (recursively).
        3. For Centers w/o further FoundationalNodes, the head is the Center
            itself.

    Returns:
        A core.FoundationalNode object of the head, or None if there is no head

    """
    if fnode.is_scene():
        return extract_head(fnode.process or fnode.state)
    elif len(fnode.centers) == 1:
        return extract_head(fnode.centers[0])
    elif all(e.tag in (layer1.EdgeTags.Terminal, layer1.EdgeTags.Punctuation)
             for e in fnode):
        return fnode
    else:
        return None


def is_noun(fnode):
    """Returns whether the fnode is a noun.

    Args:
        fnode: layer1.FoundationalNode object who have at least one Terminal
        as a child

    Returns:
        Whether the first Terminal child has NN* (PTB noun POS tag) in its
        extra[postag] attribute.

    """
    return fnode.terminals and re.match(r'NN.+',
                                        fnode.terminals[0].extra['postag'])


def filter_noun_heads(heads):
    """Extract only the possible noun heads from the given FNodes.

    Args:
        heads: list of fnodes, each have at least one Terminal as a child,
        and the Terminal have the attribute 'postag' in extra.

    Returns:
        a set of all tokens (Terminal text) which at least one of the postag
        attributes for them where a noun (PTB POSTags).

    """
    out = set()
    for head in heads:
        try:  # handling implicit processes/centers
            term = head.terminals[0]
            if re.match(r'NN.+', term.extra['postag']):
                out.add(term.text)
        except IndexError:
            pass
    return out


def extract_all_nouns(passages):
    """Extract all nouns in passages and their location.

    Extracted nouns are either single words with PTB POS tag of a noun,
    or phrases where all words are tagged as nouns.

    Args:
        passages: sequence of core.Passage object to extract nouns from.
        All Terminals should have the attribute 'postag' in extra data.

    Returns:
        dictionary of noun: [Terminal1, Terminal2 ...] where they appear.

    """
    nouns = {}
    for passage in passages:
        for term in passage.layer(layer0.LAYER_ID).all:
            if all(re.match(r'NN.+', tag)
                   for tag in term.extra['postag'].split()):
                nouns[term.text] = nouns.get(term.text, []) + [term]
    return nouns


def is_scene_evoking(terminal):
    """Returns whether the Terminal object is a scene-evoker.

    Scene-evoking Terminals are ones which are the head of a Process/State,
    where a head is the (recursive) only Center of an FNode (non-remote).

    Args:
        terminal: a layer0.Terminal object to check

    Returns:
        True iff the terminal given is a head of a scene (scene-evoker).

    """
    if not terminal.parents:
        return False
    fnode = terminal.parents[0]
    if fnode.tag != layer1.NodeTags.Foundational:
        return False
    while fnode:
        if len(fnode.centers) > 1:  # multiple centers, can't have a head
            break
        elif fnode.ftag in (layer1.EdgeTags.Process, layer1.EdgeTags.State):
            return True
        elif fnode.ftag == layer1.EdgeTags.Center:
            fnode = fnode.fparent
        else:
            break
    return False
