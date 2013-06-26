import pickle
import argparse
import random
import numpy as np

from ucca import classify

# run through evaluate and baseline


def get_data_objects(labels_fd, fmat_fd):
    """Gets file obj of 2 pickle files and returns (targets, labels, fmat)."""
    targets, labels = pickle.load(labels_fd)
    fmat = pickle.load(fmat_fd)
    labels_fd.close()
    fmat_fd.close()
    return targets, labels, fmat


def filter_data(targets, labels, fmat, ratio):
    """Filters the negative examples to be in ratio with the positive ones.

    Takes the negative examples randomly, and all positive examples.

    Args:
        targets: list of target phrases
        labels: numpy.array of 0/1, for negative/positive examples
        fmat: feature matrix (numpy.array of scores) of the targets
        ratio: desired ratio between positive and negative examples

    Returns:
        a tuple of filtered (targets, labels, fmat)

    """

    pos_targets = [t for t, la in zip(targets, labels) if la == 1]
    neg_targets = [t for t, la in zip(targets, labels) if la == 0]
    pos_fmat = fmat[[i for i, la in enumerate(labels) if la == 1]]
    neg_fmat = fmat[[i for i, la in enumerate(labels) if la == 0]]

    # Creates a random permutation of indices of negative examples with the
    # right ratio to positive ones
    neg_indices = list(range(len(neg_targets)))
    random.shuffle(neg_indices)
    neg_indices = neg_indices[:len(pos_targets) * ratio]
    new_targets = [neg_targets[i] for i in neg_indices]
    new_targets += pos_targets
    new_fmat = np.append(neg_fmat[np.array(neg_indices)], pos_fmat).reshape(
        len(neg_indices) + len(pos_fmat), len(neg_fmat[0]))
    new_labels = np.append(np.zeros(len(neg_indices), dtype=np.int32),
                           np.ones(len(pos_targets), dtype=np.int32))

    return new_targets, new_labels, new_fmat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('labels', type=argparse.FileType('rb'))
    parser.add_argument('fmat', type=argparse.FileType('rb'))
    parser.add_argument('collins', help='path to collins dict in pickle')
    parser.add_argument('wiktionary', help='path to wiktionary defs')
    parser.add_argument('method', choices=('c_svc', 'nu_svc_linear',
                                           'nu_svc_sigmoid', 'lr', 'gboost'),
                        help='classification method')
    parser.add_argument('--ratio', type=float, default=2)
    args = parser.parse_args()

    orig_targets, orig_labels, orig_fmat = get_data_objects(args.labels,
                                                            args.fmat)
    targets, labels, fmat = filter_data(orig_targets, orig_labels, orig_fmat,
                                        args.ratio)

    stats, detailed = classify.evaluate(fmat, labels, targets, args.method)
    # We use zip(*stats) because stats are [(prec1, rec1, acc1), ((prec2 ...))]
    # and this turns them into [(prec1, prec2 ..), (rec1, rec2 ..)] which is
    # what we want to use mean() on
    results = [np.mean([x for x in stat if x is not None])
               for stat in zip(*stats)]
    print("Evaluation result:")
    print("Precision: {} Recall: {} Accuracy: {}".format(*results))
    print("Baseline:")
    print("Precision: {} Recall: {} Accuracy: {}".format(
        *classify.evaluate_bl(labels, classify.baseline(targets, args.collins,
                                                        args.wiktionary))))
    print("Detailed Results:")
    for true_label in [0, 1]:
        for pred_label in [0, 1]:
            print("\n\nTrue: {} Pred: {} Targets:\n".format(true_label,
                                                            pred_label),
                  *detailed[true_label][pred_label], sep='\t')


if __name__ == '__main__':
    main()
