import mlpy
import numpy as np
from ucca import lex


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


def evaluate(fmat, labels, targets, method='svm', k=10):
    """Evaluates linear kernel SVM/logistic regression with k-fold cross
    validation."""
    cls = mlpy.LibSvm() if method == 'svm' else mlpy.LibLinear()
    nptargets = np.array(targets)
    out = []
    detailed = [[[], []], [[], []]]
    for tr, ts in mlpy.cv_kfold(len(labels), k, strat=labels):
        cls.learn(fmat[tr], labels[tr])
        pred = cls.pred(fmat[ts])
        for target, x, y in zip(nptargets[ts], labels[ts], pred):
            detailed[x][int(y)].append(target)
        tp = [x == int(y) == 1
              for x, y in zip(labels[ts], pred)].count(True)
        tn = [x == int(y) == 0
              for x, y in zip(labels[ts], pred)].count(True)
        fp = [x == 0 and int(y) == 1
              for x, y in zip(labels[ts], pred)].count(True)
        fn = [x == 1 and int(y) == 0
              for x, y in zip(labels[ts], pred)].count(True)
        try:
            precision = tp / (tp + fp)
        except:
            precision = None
        try:
            recall = tp / (tp + fn)
        except:
            recall = None
        try:
            accuracy = (tp + tn) / (tp + tn + fp + fn)
        except:
            accuracy = None
        out.append((precision, recall, accuracy))
    return out, detailed


def baseline(targets, collins_path, wikt_path):
    """Classifies baseline by checking for zero or -ing derivations."""
    form_ident = lex.FormIdentifier(collins_path, wikt_path)
    labels = np.zeros(len(targets), dtype=np.int32)
    for i, target in enumerate(targets):
        if form_ident.is_dual_vn(target) or target.endswith('ing'):
            labels[i] = 1
    return labels


def evaluate_bl(labels_known, labels_guessed):
    """Evaluates the baseline, returns precision, recall and accuracy"""
    # True/False positive/negatives
    tp = [x == y == 1
          for x, y in zip(labels_known, labels_guessed)].count(True)
    tn = [x == y == 0
          for x, y in zip(labels_known, labels_guessed)].count(True)
    fp = [x == 0 and y == 1
          for x, y in zip(labels_known, labels_guessed)].count(True)
    fn = [x == 1 and y == 0
          for x, y in zip(labels_known, labels_guessed)].count(True)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return (precision, recall, accuracy)
