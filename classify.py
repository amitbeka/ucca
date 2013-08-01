import mlpy
import random
import numpy as np
from ucca import lex

from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier


# number which means the label is undecisive
UNDECISIVE_LABEL = 2


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


def train_classifier(fmat, labels, method):
    classifiers = {
        'c_svc': mlpy.LibSvm(),
        'nu_svc_linear': mlpy.LibSvm('nu_svc', 'linear'),
        'nu_svc_sigmoid': mlpy.LibSvm('nu_svc', 'sigmoid'),
        'c_svc_prob': mlpy.LibSvm(probability=True),
        'nu_svc_linear_prob': mlpy.LibSvm('nu_svc', 'linear',
                                          probability=True),
        'nu_svc_sigmoid_prob': mlpy.LibSvm('nu_svc', 'sigmoid',
                                           probability=True),
        'lr': mlpy.LibLinear(),
        'gboost': GradientBoostingClassifier()
    }
    clas = classifiers[method]
    if hasattr(clas, 'learn'):
        clas.learn(fmat, labels)
    else:
        clas.fit(fmat, labels)
    return clas


def predict_labels(clas, fmat):
    if hasattr(clas, 'pred'):
        labels = clas.pred(fmat)
    else:
        labels = clas.predict(fmat)
    return labels


def evaluate(fmat, labels, targets, method='c_svc', k=10):
    nptargets = np.array(targets)
    out = []
    detailed = [[[], []], [[], []]]
    for tr, ts in mlpy.cv_kfold(len(labels), k, strat=labels):
        clas = train_classifier(fmat[tr], labels[tr], method)
        try:
            pred = clas.pred(fmat[ts])
        except AttributeError:
            pred = clas.predict(fmat[ts])
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


def get_probabilities_prediction(fmat, clas):
    if hasattr(clas, 'pred_probability'):
        probs = clas.pred_probality(fmat)
        if clas.labels()[0] != 0:  # labels are reversed, [1, 0]
            probs = np.array([[x[1], x[0]] for x in probs])
    else:
        probs = clas.pred_proba(fmat)  # always in the right order
    return probs


def return_new_labels(probs, confidence_0, confidence_1):
    labels = np.zeros(len(probs), dtype=np.int32)
    for i, prob in enumerate(probs):
        if confidence_0 < prob < confidence_1:
            labels[i] = UNDECISIVE_LABEL
        elif prob <= confidence_0:
            labels[i] = 0
        elif prob >= confidence_1:
            labels[i] = 1
    return labels


def split_arrays(arr, values, *additional_arrays):
    indexers = [arr == value for value in values]
    split_arr = np.array([arr[indexer] for indexer in indexers])
    if not additional_arrays:
        return split_arr
    split_all = [split_arr]
    for array in additional_arrays:
        split_all.append(np.array([array[indexer] for indexer in indexers]))
    return split_all


PARAM_PRE_LABELS, PARAM_CONF0, PARAM_CONF1 = range(3)
PRE_LABELS_THRESH = 3


def self_train_classifier(fmat_all, pre_scores, targets_all, params_list,
                          method='c_svc_prob'):

    split_point = np.array([len(pre_scores)])
    pre_fmat, unlabeled_fmat = np.split(fmat_all, split_point)
    pre_targets, unlabeled_targets = np.split(targets_all, split_point)
    unlabeled_labels = np.zeros(len(unlabeled_targets), dtype=np.int32)
    unlabeled_labels.fill(UNDECISIVE_LABEL)
    pre_labels, pre_targets, pre_fmat = split_arrays(pre_scores, range(6),
                                                     pre_targets, pre_fmat)
    for pre_value in range(6):
        if pre_value < PRE_LABELS_THRESH:
            pre_labels[pre_value].fill(0)
        else:
            pre_labels[pre_value].fill(1)

    for params in params_list:
        # taking the relevant pre-labeled data
        pre_labels_to_take = list(params[PARAM_PRE_LABELS])
        labels = pre_labels[pre_labels_to_take][0]
        targets = pre_targets[pre_labels_to_take][0]
        fmat = pre_fmat[pre_labels_to_take][0]
        for score in pre_labels_to_take[1:]:
            labels = np.append(labels, pre_labels[score])
            targets = np.append(targets, pre_targets[score])
            fmat = np.append(fmat, pre_fmat[score], 0)

        # taking confident unlabeled data
        unlabel_to_take = (unlabeled_labels != UNDECISIVE_LABEL)
        labels = np.append(labels, unlabeled_labels[unlabel_to_take])
        targets = np.append(targets, unlabeled_targets[unlabel_to_take])
        fmat = np.append(fmat, unlabeled_fmat[unlabel_to_take], 0)

        # splitting to 0/1 and creating similar-size training sets
        labels, targets, fmat = split_arrays(labels, [0, 1], targets, fmat)
        if len(labels[0]) > 2 * len(labels[1]):
            to_take = list(range(len(labels[0])))
            random.shuffle(to_take)
            to_take = to_take[:2 * len(labels[1])]
            labels[0] = labels[0][to_take]
            targets[0] = targets[0][to_take]
            fmat[0] = fmat[0][to_take]
        labels = np.append(labels[0], labels[1])
        targets = np.append(targets[0], targets[1])
        fmat = np.append(fmat[0], fmat[1], 0)

        # training a classifier and predicting
        clas = train_classifier(fmat, labels, method)
        if hasattr(clas, 'pred_probability'):
            _, probs = zip(*clas.pred_probability(unlabeled_fmat))
        else:
            _, probs = zip(*clas.pred_proba(unlabeled_fmat))
        unlabeled_labels = return_new_labels(probs, params[PARAM_CONF0],
                                             params[PARAM_CONF1])

    return clas, unlabeled_targets, unlabeled_labels
