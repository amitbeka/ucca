"""Identifies and categorizes words according to Dixon's verb list."""

import argparse
import xml.etree.ElementTree as ETree

from ucca import lex, convert, scenes


class Result:
    def __init__(self, main_unit, head=None, lemma=None, stem=None, cat=None):
        self.main_unit = main_unit
        self.is_scene = main_unit.is_scene()
        self.head = head
        self.lemma = lemma
        self.stem = stem
        self.categories = {} if cat is None else cat
        self.main_cat = [(k, v) for k, v in self.categories.items()
                         if len(k) == min(len(k) for k in self.categories)]

    def __str__(self):
        return ("Unit: ({}) {}\nHead: {} ==> "
                "{} ==> {}\nCategories: {} {}".format(
                    ("SC" if self.is_scene else "NS"), self.main_unit,
                    self.head, self.lemma, self.stem, self.main_cat,
                    self.categories))


class Stats:
    def __init__(self):
        self.heads = []
        self.lemmas = []
        self.no_cats = []
        self.fulls = []
        self.lemma_count = {}
        self.cat_count = {}

    def update_counts(self):
        self.lemma_counts = {}
        self.cat_counts = {}
        for result in self.no_cats:
            self.lemma_count[result.lemma] = self.lemma_count.get(result.lemma,
                                                                  0) + 1
        for result in self.fulls:
            self.cat_count[str(result.main_cat[0][1][0])] = self.cat_count.get(
                str(result.main_cat[0][1][0]), 0) + 1


def main():
    """Runs DixonIdentifier and gathers statistics."""
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', nargs='*',
                        help="Site XML files to operate on")
    parser.add_argument('-v', '--verbs',
                        default='/home/beka/thesis/resources/dixon-verbs.xml',
                        help="Dixon's verb categories in XML file")
    parser.add_argument(
        '-c', '--collins',
        default='/home/beka/thesis/resources/collins/collins.pickle',
        help="Collins parsed dictionary in pickle file")
    parser.add_argument(
        '-w', '--wiktionary',
        default='/home/beka/thesis/resources/enwikt-defs-latest-en.tsv',
        help="Wiktionary definitions only in tab-separated format")

    args = parser.parse_args()
    eng = lex.DixonIdentifier(args.verbs, args.collins, args.wiktionary)
    stats = Stats()
    for path in args.filename:
        run_file(path, eng, stats)
    stats.heads.sort(key=lambda x: str(x.main_unit))
    stats.lemmas.sort(key=lambda x: str(x.head))
    stats.no_cats.sort(key=lambda x: str(x.head))
    stats.fulls.sort(key=lambda x: str(x.main_cat))
    stats.update_counts()
    for name, results in [('HEADS', stats.heads), ('LEMMAS', stats.lemmas),
                          ('EMPTY', stats.no_cats), ('FULLS', stats.fulls)]:
        print('=== {} ({}) ==='.format(name, len(results)))
        for result in results:
            print(str(result))
    print('=== COUNTS ===')
    for name, count in (sorted((y, x) for x, y in stats.lemma_count.items()) +
                        sorted((y, x) for x, y in stats.cat_count.items())):
        print("{}\t{}".format(name, count))


def run_file(path, eng, stats):
    """Site XML file ==> prints list of sceneness results"""
    with open(path) as f:
        root = ETree.ElementTree().parse(f)
    passage = convert.from_site(root)

    sc = scenes.extract_possible_scenes(passage)
    heads = [scenes.extract_head(x) for x in sc]

    for s, h in zip(sc, heads):
        if h is None:
            stats.heads.append(Result(s))
            continue
        out = eng.get_categories(s, h)
        if out == 'implicit':
            stats.heads.append(Result(s))
        elif out == 'no base form':
            stats.lemmas.append(Result(s, h))
        elif out[2]:
            stats.fulls.append(Result(s, h, *out))
        else:
            stats.no_cats.append(Result(s, h, *out))


if __name__ == '__main__':
    main()
