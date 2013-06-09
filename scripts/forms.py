import argparse
import xml.etree.ElementTree as ETree

from ucca import lex, convert
from ucca.postags import POSTags


def main():
    """Runs FormsIdentifier and gathers statistics."""
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', nargs='*',
                        help="Site XML files to operate on")
    parser.add_argument(
        '-c', '--collins',
        default='/home/beka/thesis/resources/collins/collins.pickle',
        help="Collins parsed dictionary in pickle file")
    parser.add_argument(
        '-w', '--wiktionary',
        default='/home/beka/thesis/resources/enwikt-defs-latest-en.tsv',
        help="Wiktionary definitions only in tab-separated format")

    args = parser.parse_args()
    eng = lex.FormIdentifier(args.collins, args.wiktionary)
    for path in args.filename:
        run_file(path, eng)
    #stats.heads.sort(key=lambda x: str(x.main_unit))
    #stats.lemmas.sort(key=lambda x: str(x.head))
    #stats.no_cats.sort(key=lambda x: str(x.head))
    #stats.fulls.sort(key=lambda x: str(x.main_cat))
    #stats.update_counts()
    #for name, results in [('HEADS', stats.heads), ('LEMMAS', stats.lemmas),
                          #('EMPTY', stats.no_cats), ('FULLS', stats.fulls)]:
        #print('=== {} ({}) ==='.format(name, len(results)))
        #for result in results:
            #print(str(result))
    #print('=== COUNTS ===')
    #for name, count in (sorted((y, x) for x, y in stats.lemma_count.items()) +
                        #sorted((y, x) for x, y in stats.cat_count.items())):
        #print("{}\t{}".format(name, count))


def run_file(path, eng):
    """Site XML file ==> prints list of sceneness results"""
    with open(path) as f:
        root = ETree.ElementTree().parse(f)
    passage = convert.from_site(root)
    words = [x.text for x in passage.layer('0').words]
    print(' '.join(words))
    for word in words:
        all_tagsets = eng.get_forms(word)
        all_postags = set()
        for tagset in all_tagsets.values():
            all_postags.update(tagset)
        print('{}\t{}'.format(word, all_postags))
        if eng.is_dual_vn(word):
            print(all_tagsets)
            print('========')


if __name__ == '__main__':
    main()
