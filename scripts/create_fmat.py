"""Converts a feature matrix as output by features.py morph extract
(one line per feature, space-separated values, each line with len(targets))
to a numpy feature matrix which can be used to classification."""
import pickle
import sys
import numpy as np


def main():
    with open(sys.argv[1]) as f:
        feature_lines = [x.strip().split() for x in f]
    feature_data = [[float(x) for x in line] for line in feature_lines]
    targets_data = list(zip(*feature_data))
    fmat = np.array(targets_data)
    with open(sys.argv[2], 'wb') as f:
        pickle.dump(fmat, f)


if __name__ == '__main__':
    main()
