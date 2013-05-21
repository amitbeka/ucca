import mlpy
import numpy as np


def create_targets_array(targets_fd):
    """target+label line ==> labels ndarray, string tuple"""
    targets = []
    labels = []
    for line in targets_fd:
        if not line.strip():
            continue
        target, label = line.strip().split('\t')
        targets.append(target)
        labels.append(label)
    return np.array(labels, dtype=np.int32), tuple(targets)


def create_feature_matrix(scores_fd, targets, features):
    """scores is target\tfeature_name\tscore, targets and features are tuples
    of strings (without 'before'/'after' like feature_name) ==> feature matrix
    which should have been sparse, but it's not."""
    findex = lambda x: (features.index(x.split('_')[0])
                        if x.endswith('_before')
                        else features.index(x.split('_')[0]) + 1)

    mat = np.zeros((len(targets), len(features) * 2))
    for line in scores_fd:
        t, f, s = line.strip().split('\t')  # target, feature name, score
        mat[targets.index(t), findex(f)] = float(s)
    return mat


def evaluate(fmat, labels, method='svm', k=10):
    """Evaluates linear kernel SVM/logistic regression with k-fold cross
    validation."""
    cls = mlpy.LibSvm() if method == 'svm' else mlpy.LibLinear()
    out = []
    for tr, ts in mlpy.cv_kfold(len(labels), k, strat=labels):
        cls.learn(fmat[tr], labels[tr])
        pred = cls.pred(fmat[ts])
        import pdb; pdb.set_trace()
        acc = sum(x == int(y) for x, y in zip(labels[ts], pred)) / len(ts)
        print(acc)
        out.append(acc)
    return out
