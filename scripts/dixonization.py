"""Identifies and categorizes words according to Dixon's verb list."""

import sys
import xml.etree.ElementTree as ETree

from ucca import lex, convert, scenes


def main():
    """Runs DixonIdentifier and gathers statistics."""
    eng = lex.DixonIdentifier(
        '/home/beka/thesis/resources/dixon-verbs.xml',
        '/home/beka/thesis/resources/collins/collins.pickle')
    for path in sys.argv[1:]:
        run_file(path, eng)


def run_file(path, eng):
    """Site XML file ==> prints list of sceneness results"""
    with open(path) as f:
        root = ETree.ElementTree().parse(f)
    passage = convert.from_site(root)

    sc = scenes.extract_possible_scenes(passage)
    heads = [scenes.extract_head(x) for x in sc]

    for s, h in zip(sc, heads):
        print("Scene: " + s.to_text())
        if h is None:
            print("Head: None")
            continue
        print("Head: " + h.to_text())
        print("Categories: " + str(eng.get_categories(s, h)))


if __name__ == '__main__':
    main()
