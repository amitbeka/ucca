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
            for token, terminal in zip(tokens, p.layer('0').all):
                print(token, terminal.text)
                word, tag = token.split('_')
                terminal.extra['postag'] = tag
    with open(dbpath + '.tags', 'wb') as f:
        pickle.dump(passages, f)


if __name__ == '__main__':
    main()
