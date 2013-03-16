"""This modules encapsulates all scene-related functions.

Scene-related functions are used to extract, identify and label scenes
in the foundational layer of UCCA.

"""

from . import layer1


def extract_possible_scenes(passage):
    """Extracts all possible scenes from a Passage.

    Possible scenes are currently units which follow these restrictions:
        1. It is a Participant of a main relation (Process/State)
        2. It is a scene, or it has one Center (exactly) and at least one
            Elaborator
        3. It is not a remote participant

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
    return ret


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
