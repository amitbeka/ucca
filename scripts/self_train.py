import pickle
import random
import numpy as np
from itertools import combinations_with_replacement as comb_repeat

from ucca import classify, tokeneval

TARGETS_PATH = "/home/beka/thesis/db/nouns/targets-scores.nouns.pickle"
FMAT_PATH = "/home/beka/thesis/db/nouns/fmat_morph_dict.nouns"
PASSAGES_PATH = "/home/beka/thesis/db/db_18_6/huca_18_6_with_pos.pickle"


NUM_ITERATIONS_OPTS = tuple(range(2, 9))
PRE_LABELS_OPTS = ((0, 4, 5), (0, 1, 4, 5), (0, 1, 3, 4, 5),
                   (0, 1, 2, 3, 4, 5))
CONFIDENCE0_OPTS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
CONFIDENCE1_OPTS = (0.5, 0.6, 0.7, 0.8, 0.9)
SAMPLING_RATIO = 0.001


def params_generator():
    for num_iter in NUM_ITERATIONS_OPTS:
        for pre_labels in comb_repeat(PRE_LABELS_OPTS, num_iter):
            for conf0 in comb_repeat(CONFIDENCE0_OPTS, num_iter):
                for conf1 in comb_repeat(CONFIDENCE1_OPTS, num_iter):
                    if conf0 > conf1:
                        continue
                    if random.random() < SAMPLING_RATIO:
                        yield num_iter, pre_labels, conf0, conf1


def main():
    # Getting required data
    with open(TARGETS_PATH, "rb") as f:
        target_array, scores = pickle.load(f)
        target_list = target_array.tolist()
    with open(FMAT_PATH, "rb") as f:
        fmat = pickle.load(f)
    with open(PASSAGES_PATH, "rb") as f:
        passages = pickle.load(f)
    passages = passages[:22]
    del passages[18]
    terminals, token_labels = tokeneval.get_terminals_labels(passages)
    tokens = [x.text for x in terminals]

    # Running through random parameters settings
    for params in params_generator():
        params2 = list(zip(*params[1:]))
        print(params2)
        clas, _, _ = classify.self_train_classifier(fmat, scores,
                                                    target_array, params2)
        target_labels = [int(x >= classify.PRE_LABELS_THRESH) for x in scores]
        target_labels += list(classify.predict_labels(clas,
                                                      fmat[len(scores):]))
        stats = tokeneval.evaluate_with_type(tokens, token_labels,
                                             target_list, target_labels)
        print([len(x) for x in stats])


if __name__ == '__main__':
    main()
