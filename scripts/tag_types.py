"""Tags words (as types, not tokens) as scene-evoking nouns."""
import pickle
import sys
from ucca import collins
from ucca.postags import POSTags


def main():
    with open("/home/beka/thesis/resources/collins/collins.pickle", "rb") as f:
        col = collins.CollinsDictionary(pickle.load(f))
    with open(sys.argv[1]) as f:
        words = [line.strip() for line in f if line.strip()]
        words = words[int(sys.argv[2]):]
    output = []
    for word in words:
        while True:
            entries = col.by_form(word)
            if not entries:
                output.append((word, 'NOT FOUND'))
                break
            if all(s.pos != POSTags.Noun
                    for e in entries for s in e.senses):
                output.append((word, 'NOT NOUN'))
                break
            for entry in entries:
                print("\n\n\n{}#{}".format(entry.key, entry.context))
                print("\n===\n".join(str(s) for s in entry.senses))
            user = input("select 0-5, Q or anything else to skip: ")
            if user == 'Q':
                return output
            try:
                score = int(user)
                assert 0 <= score <= 5
                output.append((word, user))
                break
            except:
                break
    return output


if __name__ == '__main__':
    out = main()
    with open(sys.argv[3], 'at') as f:
        f.write('\n'.join('\t'.join(x) for x in out) + '\n')
