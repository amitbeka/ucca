import pickle
import argparse
import numpy as np


def extend_labels(old_labels, new_scores, threshold):
    """Returns a new np.array of labels.

    Args:
        old_labels: numpy.array of int32, 0 or 1
        new_scores: list of floats, scores of the new targets
        threshold: a number which score greater than it will be labeled as 1

    Returns:
        a new numpy array of int32 with old_labels + len(new_scores) entries

    """

    added_labels = np.array([1 if x > threshold else 0 for x in new_scores],
                            dtype=np.int32)
    return np.append(old_labels, added_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('orig', type=argparse.FileType('rb'),
                        help='Original targets and labels pickle file')
    parser.add_argument('new', type=argparse.FileType('r'),
                        help='New targets and scores, tab-separated')
    parser.add_argument('out', type=argparse.FileType('wb'),
                        help='New output targets and labels pickle file')
    parser.add_argument('--threshold', type=float, default=0,
                        help='Threshold above it a target is labeled as 1')
    args = parser.parse_args()

    old_targets, old_labels = pickle.load(args.orig)
    args.orig.close()

    new_data = [line.strip().split('\t') for line in args.new]
    added_targets, added_scores = zip(*new_data)  # un-tupling to 2 lists
    added_scores = [float(x) for x in added_scores]

    new_targets = old_targets + list(added_targets)
    new_labels = extend_labels(old_labels, added_scores, args.threshold)
    pickle.dump((new_targets, new_labels), args.out)
    args.out.close()


if __name__ == '__main__':
    main()
