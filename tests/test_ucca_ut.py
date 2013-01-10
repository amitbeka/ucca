"""Testing code for the ucca package, unit-testing only."""

import unittest
import operator

from ucca import core, layer0

# modifying: creating + add edges/nodes + ordering (node, layer) + frozen errs
# removing: creating + destroy nodes/edges + frozen errs


class CoreTests(unittest.TestCase):

    @staticmethod
    def _create_basic_passage():
        """Creates a basic :class:Passage to tinker with.

        Passage structure is as follows:
            Layer1: order by ID, heads = [1.2], all = [1.1, 1.2, 1.3]
            Layer2: order by node unique ID descending,
                    heads = all = [2.2, 2.1], attrib={'test': True}
            Nodes (tag):
                1.1 (1)
                1.3 (3), attrib={'node': True}
                1.2 (x), order by edge tag
                    children: 1.3 Edge: tag=test1, attrib={'Edge': True}
                              1.1 Edge: tag=test2
                2.1 (2), children [1.1, 1.2] with edge tags [test, test2]
                2.2 (2), children [1.1, 1.2, 1.3] with tags [test, test1, test]

        """
        p = core.Passage(ID='1')
        l1 = core.Layer(ID='1', root=p)
        l1 = core.Layer(ID='2', root=p, attrib={'test': True},
                        orderkey=lambda x: -1 * int(x.ID.split('.')[1]))

        # Order is explicitly different in order to break the alignment between
        # the ID/Edge ordering and the order of creation/addition
        node11 = core.Node(ID='1.1', root=p, tag='1')
        node13 = core.Node(ID='1.3', root=p, tag='3', attrib={'node': True})
        node12 = core.Node(ID='1.2', root=p, tag='x',
                           orderkey=operator.attrgetter('tag'))
        node21 = core.Node(ID='2.1', root=p, tag='2')
        node22 = core.Node(ID='2.2', root=p, tag='2')
        node12.add('test2', node11)
        node12.add('test1', node13, edge_attrib={'edge': True})
        node21.add('test2', node12)
        node21.add('test', node11)
        node22.add('test1', node12)
        node22.add('test', node13)
        node22.add('test', node11)
        return p

    def test_creation(self):

        p = self._create_basic_passage()

        self.assertEqual(p.ID, '1')
        self.assertEqual(p.root, p)
        self.assertDictEqual(p.attrib.copy(), {})
        self.assertEqual(p.layer('1').ID, '1')
        self.assertEqual(p.layer('2').ID, '2')
        self.assertRaises(KeyError, p.layer, '3')

        l1 = p.layer('1')
        l2 = p.layer('2')
        self.assertEqual(l1.root, p)
        self.assertEqual(l2.attrib['test'], True)
        self.assertNotEqual(l1.orderkey, l2.orderkey)
        self.assertSequenceEqual([x.ID for x in l1.all], ['1.1', '1.2', '1.3'])
        self.assertSequenceEqual([x.ID for x in l1.heads], ['1.2'])
        self.assertSequenceEqual([x.ID for x in l2.all], ['2.2', '2.1'])
        self.assertSequenceEqual([x.ID for x in l2.heads], ['2.2', '2.1'])

        node11, node12, node13 = l1.all
        node22, node21 = l2.all
        self.assertEqual(node11.ID, '1.1')
        self.assertEqual(node11.root, p)
        self.assertEqual(node11.layer.ID, '1')
        self.assertEqual(node11.tag, '1')
        self.assertEqual(len(node11), 0)
        self.assertSequenceEqual(node11.parents, [node12, node21, node22])
        self.assertSequenceEqual(node13.parents, [node12, node22])
        self.assertDictEqual(node13.attrib.copy(), {'node': True})
        self.assertEqual(len(node12), 2)
        self.assertSequenceEqual([x.child for x in node12], [node13, node11])
        self.assertDictEqual(node12[0].attrib.copy(), {'edge': True})
        self.assertSequenceEqual(node12.parents, [node22, node21])
        self.assertEqual(node21[0].ID, '2.1->1.1')
        self.assertEqual(node21[1].ID, '2.1->1.2')
        self.assertEqual(node22[0].ID, '2.2->1.1')
        self.assertEqual(node22[1].ID, '2.2->1.2')
        self.assertEqual(node22[2].ID, '2.2->1.3')

    def test_modifying(self):

        p = self._create_basic_passage()
        l1, l2 = p.layer('1'), p.layer('2')
        node11, node12, node13 = l1.all
        node22, node21 = l2.all

        # Testing attribute changes
        p.attrib['passage'] = 1
        self.assertDictEqual(p.attrib.copy(), {'passage': 1})
        del l2.attrib['test']
        self.assertDictEqual(l2.attrib.copy(), {})
        node13.attrib[1] = 1
        self.assertDictEqual(node13.attrib.copy(), {'node': True, 1: 1})
        self.assertEqual(len(node13.attrib), 2)
        self.assertEqual(node13.attrib.get('node'), True)
        self.assertEqual(node13.attrib.get('missing'), None)

        # Testing Node changes
        node14 = core.Node(ID='1.4', root=p, tag='4')
        node15 = core.Node(ID='1.5', root=p, tag='5')
        self.assertSequenceEqual(l1.all, [node11, node12, node13, node14,
                                          node15])
        self.assertSequenceEqual(l1.heads, [node12, node14, node15])
        node15.add('test', node11)
        self.assertSequenceEqual(node11.parents, [node12, node15, node21,
                                                 node22])
        node21.remove(node12)
        node21.remove(node21[0])
        self.assertEqual(len(node21), 0)
        self.assertSequenceEqual(node12.parents, [node22])
        self.assertSequenceEqual(node11.parents, [node12, node15, node22])
        node14.add('test', node15)
        self.assertSequenceEqual(l1.heads, [node12, node14])
        node12.destroy()
        self.assertSequenceEqual(l1.heads, [node13, node14])
        self.assertSequenceEqual([x.child for x in node22], [node11, node13])


class Layer0Tests(unittest.TestCase):
    """Tests module layer0 functionality."""

    def test_terminals(self):
        """Tests :class:layer0.Terminal new and inherited functionality."""
        p = core.Passage('1')
        l0 = layer0.Layer0(p)
        terms = []
        terms.append(layer0.Terminal(ID='0.1', root=p, tag=layer0.WORD_TAG,
                                     attrib={layer0._TEXT_ATTRIB: '1',
                                             layer0._PARAGRAPH_ATTRIB: 1,
                                             layer0._PARA_POS_ATTRIB: 1}))
        terms.append(layer0.Terminal(ID='0.2', root=p, tag=layer0.WORD_TAG,
                                     attrib={layer0._TEXT_ATTRIB: '2',
                                             layer0._PARAGRAPH_ATTRIB: 2,
                                             layer0._PARA_POS_ATTRIB: 1}))
        terms.append(layer0.Terminal(ID='0.3', root=p, tag=layer0.PUNCT_TAG,
                                     attrib={layer0._TEXT_ATTRIB: '.',
                                             layer0._PARAGRAPH_ATTRIB: 2,
                                             layer0._PARA_POS_ATTRIB: 2}))

        self.assertSequenceEqual([t.punct for t in terms],
                                 [False, False, True])
        self.assertSequenceEqual([t.text for t in terms], ['1', '2', '.'])
        self.assertSequenceEqual([t.position for t in terms], [1, 2, 3])
        self.assertSequenceEqual([t.paragraph for t in terms], [1, 2, 2])
        self.assertSequenceEqual([t.para_pos for t in terms], [1, 1, 2])
        self.assertFalse(terms[0] == terms[1])
        self.assertFalse(terms[0] == terms[2])
        self.assertFalse(terms[1] == terms[2])
        self.assertTrue(terms[0] == terms[0])

    def test_layer0(self):
        p = core.Passage('1')
        l0 = layer0.Layer0(p)
        t1 = l0.add_terminal(text='1', punct=False)
        t2 = l0.add_terminal(text='2', punct=True, paragraph=2)
        t3 = l0.add_terminal(text='3', punct=False, paragraph=2)
        self.assertSequenceEqual([x[0] for x in l0.pairs], [1, 2, 3])
        self.assertSequenceEqual([t.para_pos for t in l0.all], [1, 1, 2])
        self.assertSequenceEqual(l0.words, (t1, t3))
