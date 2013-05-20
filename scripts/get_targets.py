"""Identifies and categorizes words according to Dixon's verb list."""

import pickle
import sys
from ucca import scenes


def main():

    dbfile = sys.argv[1]
    with open(dbfile, 'rb') as f:
        passages = pickle.load(f)

    nouns = set()
    for p in passages:
        sc = scenes.extract_possible_scenes(p)
        heads = [scenes.extract_head(x) for x in sc]
        heads = [x for x in heads if x is not None]
        nouns.update(scenes.filter_noun_heads(heads))

    print('\n'.join(nouns))

if __name__ == '__main__':
    main()
