"""Converter module between different UCCA annotation formats.

This module contains utilities to convert between UCCA annotation in different
forms, to/from the :class:core.Passage form, acts as a pivot for all
conversions.

The possible other formats are:
    site XML form
    standard XML form

"""

import operator
import re
import string
import xml.sax.saxutils
import xml.etree.ElementTree as ET

from ucca import core, layer0, layer1, util


class SiteXMLUnknownElement(core.UCCAError):
    pass


class SiteCfg:
    """Contains static configuration for conversion to/from the site XML.

    Attributes:
        Tags:
            XML Elements' tags in the site XML format of different annotation
            components - FNodes (Unit), Terminals, remote and implicit Units
            and linkages.

        Paths:
            Paths (from the XML root) to different parts of the annotation -
            the main units part, the discontiguous units, the paragraph
            elements and the annoation units.

        Types:
            Possible types for the Type attribute, which is roughly equivalent
            to Edge/Node tag. Only specially-handled types are here, which is
            the punctuation type.

        Attr:
            Attribute names in the XML elements (not all exist in all elements)
             - passage and site ID, discontiguous unit ID, UCCA tag, uncertain
             flag, user remarks and linkage arguments. NodeID is special
             because we set it for every unit that was already converted, and
             it's not present in the original XML.

        TagConversion: mapping of site XML tag attribute to layer1 edge tags.

    """

    class _Tags:
        Unit = 'unit'
        Terminal = 'word'
        Remote = 'remoteUnit'
        Implicit = 'implicitUnit'
        Linkage = 'linkage'

    class _Paths:
        Main = 'units'
        Paragraphs = 'units/unit/*'
        Annotation = 'units/unit/*/*'
        Discontiguous = 'unitGroups'

    class _Types:
        Punct = 'Punctuation'

    class _Attr:
        PassageID = 'passageID'
        SiteID = 'id'
        NodeID = 'internal_id'
        ElemTag = 'type'
        Uncertain = 'uncertain'
        Remarks = 'remarks'
        GroupID = 'unitGroupID'
        LinkageArgs = 'args'

    __init__ = None
    Tags = _Tags
    Paths = _Paths
    Types = _Types
    Attr = _Attr
    TagConversion = {'Linked U': layer1.EdgeTags.ParallelScene,
                     'Parallel Scene': layer1.EdgeTags.ParallelScene,
                     'Function': layer1.EdgeTags.Function,
                     'Participant': layer1.EdgeTags.Participant,
                     'Process': layer1.EdgeTags.Process,
                     'State': layer1.EdgeTags.State,
                     'aDverbial': layer1.EdgeTags.Adverbial,
                     'Center': layer1.EdgeTags.Center,
                     'Elaborator': layer1.EdgeTags.Elaborator,
                     'Linker': layer1.EdgeTags.Linker,
                     'Ground': layer1.EdgeTags.Ground,
                     'Connector': layer1.EdgeTags.Connector,
                     'Role Marker': layer1.EdgeTags.Relator,
                     'Relator': layer1.EdgeTags.Relator}


class SiteUtil:
    """Contains utility functions for converting to/from the site XML.

    Functions:
        unescape: converts escaped characters to their original form.
        set_id: sets the Node ID (internal) attribute in the XML element.
        get_node: gets the node corresponding to the element given from
            the mapping. If not found, returns None
        set_node: writes the element site ID + node pair to the mapping

    """
    __init__ = None
    unescape = lambda x: xml.sax.saxutils.unescape(x, {'&quot;': '"'})
    set_id = lambda e, ID: e.set(SiteCfg.Attr.NodeID, ID)
    get_node = lambda e, mapp: mapp.get(e.get(SiteCfg.Attr.SiteID))
    set_node = lambda e, n, mapp: mapp.update({e.get(SiteCfg.Attr.SiteID): n})


def _from_site_terminals(elem, passage, elem2node):
    """Extract the Terminals from the site XML format.

    Some of the terminals metadata (remarks, type) is saved in a wrapper unit
    which excapsulates each terminal, so we use both for creating our
    :class:layer0.Terminal objects.

    Args:
        elem: root element of the XML heirarchy
        passage: passage to add the Terminals to, already with Layer0 object
        elem2node: dictionary whose keys are site IDs and values are the
            created UCCA Nodes which are equivalent. This function updates the
            dictionary by mapping each word wrapper to a UCCA Terminal.

    """
    l0 = layer0.Layer0(passage)
    for para_num, paragraph in enumerate(elem.iterfind(
            SiteCfg.Paths.Paragraphs)):
        words = list(paragraph.iter(SiteCfg.Tags.Terminal))
        wrappers = []
        for word in words:
            # the list added has only one element, because XML is hierarichal
            wrappers.extend([x for x in paragraph.iter(SiteCfg.Tags.Unit)
                             if word in list(x)])
        for word, wrapper in zip(words, wrappers):
            punct = (wrapper.get(SiteCfg.Attr.ElemTag) == SiteCfg.Types.Punct)
            text = SiteUtil.unescape(word.text)
            # Paragraphs start at 1 and enumeration at 0, so add +1 to para_num
            t = passage.layer(layer0.LAYER_ID).add_terminal(text, punct,
                                                            para_num + 1)
            SiteUtil.set_id(word, t.ID)
            SiteUtil.set_node(wrapper, t, elem2node)


def _parse_site_units(elem, parent, passage, groups, elem2node):
    """Parses the given element in the site annotation.

    The parser works recursively by determining how to parse the current XML
    element, then adding it with a core.Edge onject to the parent given.
    After creating (or retrieving) the current node, which corresponds to the
    XML element given, we iterate its subelements and parse them recuresively.

    Args:
        elem: the XML element to parse
        parent: layer1.FouncdationalNode parent of the current XML element
        passage: the core.Passage we are converting to
        groups: the main XML element of the discontiguous units (unitGroups)
        elem2node: mapping between site IDs and Nodes, updated here

    Returns:
        a list of (parent, elem) pairs which weren't process, as they should
        be process last (usually because they contain references to not-yet
        created Nodes).

    """

    def _get_node(elem):
        """Given an XML element, returns its node if it was already created.

        If not created, returns None. If the element is a part of discontiguous
        unit, returns the discontiguous unit corresponding Node (if exists).

        """
        gid = elem.get(SiteCfg.Attr.GroupID)
        if gid is not None:
            return elem2node.get(gid)
        else:
            return SiteUtil.get_node(elem, elem2node)

    def _get_work_elem(elem):
        """Given XML element, return either itself or its discontiguos unit."""
        gid = elem.get(SiteCfg.Attr.GroupID)
        return (elem if gid is None
                else [elem for elem in groups
                      if elem.get(SiteCfg.Attr.SiteID) == gid][0])

    def _fill_attributes(elem, node):
        """Fills in node the remarks and uncertain attributes from XML elem."""
        if elem.get(SiteCfg.Attr.Uncertain) == 'true':
            node.attrib['uncertain'] = True
        if elem.get(SiteCfg.Attr.Remarks) is not None:
            node.extra['remarks'] = SiteUtil.unescape(
                elem.get(SiteCfg.Attr.Remarks))

    l1 = passage.layer(layer1.LAYER_ID)
    tbd = []

    # Unit tag means its a regular, heirarichally built unit
    if elem.tag == SiteCfg.Tags.Unit:
        node = _get_node(elem)
        # Only nodes created by now are the terminals, or discontiguous units
        if node is not None:
            if node.tag == layer0.NodeTags.Word:
                parent.add(layer1.EdgeTags.Terminal, node)
            elif node.tag == layer0.NodeTags.Punct:
                SiteUtil.set_node(elem, l1.add_punct(parent, node), elem2node)
            else:
                # if we got here, we are the second (or later) chunk of a
                # discontiguous unit, whose node was already created.
                # So, we don't need to create the node, just keep processing
                # our subelements (as subelements of the discontiguous unit)
                for subelem in elem:
                    tbd.extend(_parse_site_units(subelem, node, passage,
                                                 groups, elem2node))
        else:
            # Creating a new node, either regular or discontiguous.
            # Note that for discontiguous units we have a different work_elem,
            # because all the data on them are stored outside the heirarchy
            work_elem = _get_work_elem(elem)
            edge_tag = SiteCfg.TagConversion[work_elem.get(
                SiteCfg.Attr.ElemTag)]
            node = l1.add_fnode(parent, edge_tag)
            SiteUtil.set_node(work_elem, node, elem2node)
            _fill_attributes(work_elem, node)
            # For iterating the subelements, we don't use work_elem, as it may
            # out of the current XML heirarchy we are processing (discont..)
            for subelem in elem:
                tbd.extend(_parse_site_units(subelem, node, passage,
                                             groups, elem2node))
    # Implicit units have their own tag, and aren't recursive, but nonetheless
    # are treated the same as regular units
    elif elem.tag == SiteCfg.Tags.Implicit:
        edge_tag = SiteCfg.TagConversion[elem.get(SiteCfg.Attr.ElemTag)]
        node = l1.add_fnode(parent, edge_tag, implicit=True)
        SiteUtil.set_node(elem, node, elem2node)
        _fill_attributes(elem, node)
    # non-unit, probably remote or linkage, which should be created in the end
    else:
        tbd.append((parent, elem))

    return tbd


def _from_site_annotation(elem, passage, elem2node):
    """Parses site XML annotation.

    Parses the whole annotation, given that the terminals are already processed
    and converted and appear in elem2node.

    Args:
        elem: root XML element
        passage: the passage to create, with layer0, w/o layer1
        elem2node: mapping from site ID to Nodes, should contain the Terminals

    Raises:
        SiteXMLUnknownElement: if an unknown, unhandled element is found

    """
    tbd = []
    l1 = layer1.Layer1(passage)
    l1head = l1.heads[0]
    groups_root = elem.find(SiteCfg.Paths.Discontiguous)

    # this takes care of the heirarichal annotation
    for subelem in elem.iterfind(SiteCfg.Paths.Annotation):
        tbd.extend(_parse_site_units(subelem, l1head, passage, groups_root,
                                     elem2node))

    # Hadnling remotes and linkages, which usually contain IDs from all over
    # the annotation, hence must be taken care of after all elements are
    # converted
    for parent, elem in tbd:
        if elem.tag == SiteCfg.Tags.Remote:
            edge_tag = SiteCfg.TagConversion[elem.get(SiteCfg.Attr.ElemTag)]
            child = SiteUtil.get_node(elem, elem2node)
            l1.add_remote(parent, edge_tag, child)
        elif elem.tag == SiteCfg.Tags.Linkage:
            args = [elem2node[x] for x in
                    elem.get(SiteCfg.Attr.LinkageArgs).split(',')]
            l1.add_linkage(parent, *args)
        else:
            raise SiteXMLUnknownElement


def from_site(elem):
    """Converts site XML structure to :class:core.Passage object.

    Args:
        elem: root element of the XML structure

    Returns:
        The converted core.Passage object

    """
    pid = elem.find(SiteCfg.Paths.Main).get(SiteCfg.Attr.PassageID)
    passage = core.Passage(pid)
    elem2node = {}
    _from_site_terminals(elem, passage, elem2node)
    _from_site_annotation(elem, passage, elem2node)
    return passage


def to_standard(passage):
    """Converts a Passage object to a standard XML root element.

    The standard XML specification is not contained here, but it uses a very
    shallow structure with attributes to create hierarchy.

    Args:
        passage: the passage to convert

    Returns:
        the root element of the standard XML structure

    """

    # This utility stringifies the Unit's attributes for proper XML
    # we don't need to escape the character - the serializer of the XML element
    # will do it (e.g. tostring())
    stringify = lambda dic: {str(k): str(v) for k, v in dic.items()}

    # Utility to add an extra element if exists in the object
    add_extra = lambda obj, elem: obj.extra and ET.SubElement(
        elem, 'extra', stringify(obj.extra))

    # Adds attributes element (even if empty)
    add_attrib = lambda obj, elem: ET.SubElement(elem, 'attributes',
                                                 stringify(obj.attrib))

    root = ET.Element('root', passageID=str(passage.ID), annotationID='0')
    add_attrib(passage, root)
    add_extra(passage, root)

    for layer in sorted(passage.layers, key=operator.attrgetter('ID')):
        layer_elem = ET.SubElement(root, 'layer', layerID=layer.ID)
        add_attrib(layer, layer_elem)
        add_extra(layer, layer_elem)
        for node in layer.all:
            node_elem = ET.SubElement(layer_elem, 'node',
                                      ID=node.ID, type=node.tag)
            add_attrib(node, node_elem)
            add_extra(node, node_elem)
            for edge in node:
                edge_elem = ET.SubElement(node_elem, 'edge',
                                          toID=edge.child.ID, type=edge.tag)
                add_attrib(edge, edge_elem)
                add_extra(edge, edge_elem)
    return root


def from_standard(root, extra_funcs={}):

    attribute_converters = {
        'paragraph': (lambda x: int(x)),
        'paragraph_position': (lambda x: int(x)),
        'remote': (lambda x: True if x == 'True' else False),
        'implicit': (lambda x: True if x == 'True' else False),
        'uncertain': (lambda x: True if x == 'True' else False),
        'suggest': (lambda x: True if x == 'True' else False),
    }

    layer_objs = {layer0.LAYER_ID: layer0.Layer0,
                  layer1.LAYER_ID: layer1.Layer1}

    node_objs = {layer0.NodeTags.Word: layer0.Terminal,
                 layer0.NodeTags.Punct: layer0.Terminal,
                 layer1.NodeTags.Foundational: layer1.FoundationalNode,
                 layer1.NodeTags.Linkage: layer1.Linkage,
                 layer1.NodeTags.Punctuation: layer1.PunctNode}

    def get_attrib(elem):
        return {k: attribute_converters.get(k, str)(v)
                for k, v in elem.find('attributes').items()}

    def add_extra(obj, elem):
        if elem.find('extra') is not None:
            for k, v in elem.find('extra').items():
                obj.extra[k] = extra_funcs.get(k, str)(v)

    passage = core.Passage(root.get('passageID'), attrib=get_attrib(root))
    add_extra(passage, root)
    edge_elems = []
    for layer_elem in root.findall('layer'):
        layerID = layer_elem.get('layerID')
        layer = layer_objs[layerID](passage, attrib=get_attrib(layer_elem))
        add_extra(layer, layer_elem)
        # some nodes are created automatically, skip creating them when found
        # in the XML (they should have 'constant' IDs) but take their edges
        # and attributes/extra from the XML (may have changed from the defualt)
        created_nodes = {x.ID: x for x in layer.all}
        for node_elem in layer_elem.findall('node'):
            nodeID = node_elem.get('ID')
            tag = node_elem.get('type')
            node = (created_nodes[nodeID] if nodeID in created_nodes else
                    node_objs[tag](root=passage, ID=nodeID, tag=tag,
                                   attrib=get_attrib(node_elem)))
            add_extra(node, node_elem)
            edge_elems.extend((node, x) for x in node_elem.findall('edge'))

    # Adding edges (must have all nodes before doing so)
    for from_node, edge_elem in edge_elems:
        to_node = passage.nodes[edge_elem.get('toID')]
        tag = edge_elem.get('type')
        edge = from_node.add(tag, to_node, edge_attrib=get_attrib(edge_elem))
        add_extra(edge, edge_elem)

    return passage


def from_text(text, passage_id='1'):
    """Converts from tokenized strings to a Passage object.

    Args:
        text: a sequence of strings, where each one will be a new paragraph.

    Returns:
        a Passage object with only Terminals units.

    """
    p = core.Passage(passage_id)
    l0 = layer0.Layer0(p)
    punct = re.compile('^[{}]+$'.format(string.punctuation))

    for i, par in enumerate(text):
        for token in par.split():
            # i is paragraph index, but it starts with 0, so we need to add +1
            l0.add_terminal(text=token, punct=punct.match(token),
                            paragraph=(i + 1))
    return p


def to_text(passage, sentences=True):
    """Converts from a Passage object to tokenized strings.

    Args:
        passage: the Passage object to convert
        sentences: whether to break the Passage to sentences (one for string)
        or leave as one string. Defaults to True

    Returns:
        a list of strings - 1 if sentences=False, # of sentences otherwise

    """
    tokens = [x.text for x in sorted(passage.layer(layer0.LAYER_ID).all,
                                     key=lambda x: x.position)]
    # break2sentences return the positions of the end tokens, which is
    # always the index into tokens incremented by ones (tokens index starts
    # with 0, positions with 1). So in essence, it returns the index to start
    # the next sentence from, and we should add index 0 for the first sentence
    if sentences:
        starts = [0] + util.break2sentences(passage)
    else:
        starts = [0, len(tokens)]
    return [' '.join(tokens[starts[i]:starts[i + 1]])
            for i in range(len(starts) - 1)]
