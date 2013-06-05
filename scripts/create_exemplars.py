"""Creates new examplers from Collins entries."""
import pickle
import sys
import random
from ucca import collins
from ucca.postags import POSTags


def main():
    with open("/home/beka/thesis/resources/collins/collins.pickle", "rb") as f:
        col = collins.CollinsDictionary(pickle.load(f))
    with open(sys.argv[1]) as f:
        data = [line.strip().split('\t') for line in f if line.strip()]
        nouns = {k: float(v) for k, v in data}
    with open(sys.argv[1] + '.new', 'wt') as f:
        skipped = set()
        while True:
            entry = col.random_entry(POSTags.Noun)
            while entry.key in nouns or entry.key in skipped:
                entry = col.random_entry(POSTags.Noun)
            print("\n\n\n{}#{}".format(entry.key, entry.context))
            print("\n===\n".join(str(s) for s in entry.senses))
            user = input("select 0-5, Q or anything else to skip: ")
            if user == 'Q':
                break
            try:
                score = int(user)
                assert 0 <= score <= 5
                nouns[entry.key] = score
            except:
                skipped.add(entry.key)
                continue
            f.write("{}\t{}\n".format(entry.key, score / 5))
            f.flush()


if __name__ == '__main__':
    main()
