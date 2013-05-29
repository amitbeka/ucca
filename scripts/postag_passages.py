from ucca import convert
import pickle
import os
import sys


def main():
    dbpath = sys.argv[1]
    textdir = sys.argv[2]
    with open(dbpath, 'rb') as f:
        passages = pickle.load(f)
    for p in passages:
        with open(os.path.join(textdir, p.ID + '.tagged')) as f:
            tokens = []
            for line in f:
                tokens.extend(line.split())
            for terminal in p.layer('0').all:
                num_tokens = len(terminal.text.split())
                curr, tokens = tokens[:num_tokens], tokens[num_tokens:]
                tags = [x.split('_')[1] for x in curr]
                terminal.extra['postag'] = " ".join(tags)
    with open(dbpath + '.tags', 'wb') as f:
        pickle.dump(passages, f)


if __name__ == '__main__':
    main()
