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

    Args:
        passage: the core.Passage object to extract optional scenes from

    Returns:
        a list of core.Node objects which are candidates for scenes

    """
    l1 = passage.layer(layer1.LAYER_ID)
    ret = []
    for scene in (x for x in l1.all if x.tag == layer1.NodeTags.Foundational
                  and x.is_scene()):
        for p in scene.participants:
            if p.is_scene() or (len(p.centers) == 1 and p.elaborators):
                ret.append(p)
    return ret
