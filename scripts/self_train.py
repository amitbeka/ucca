import pickle
import random
import numpy as np
from itertools import combinations_with_replacement as comb_repeat

from ucca import classify, tokeneval

DB_PATH = "/home/beka/thesis/db/"
TARGETS_PATH = DB_PATH + "nouns/targets-scores.nouns.pickle"
FMAT_PATH = DB_PATH + "nouns/fmat_morph_dict.nouns"
PASSAGES_PATH = DB_PATH + "db_18_6/huca_18_6_pos_random_filtered.pickle"


NUM_ITERATIONS_OPTS = tuple(range(2, 9))
PRE_LABELS_OPTS = ((0, 4, 5), (0, 4, 5), (0, 4, 5), (0, 4, 5), (0, 4, 5),
                   (0, 1, 4, 5), (0, 1, 4, 5), (0, 4, 5),
                   (0, 1, 3, 4, 5), (0, 1, 3, 4, 5),
                   (0, 1, 2, 3, 4, 5))
CONFIDENCE0_OPTS = (0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3,
                    0.4, 0.4, 0.4, 0.4, 0.5, 0.5, 0.6, 0.6, 0.7)
CONFIDENCE1_OPTS = (0.95, 0.95, 0.9, 0.9, 0.9, 0.9, 0.8, 0.8, 0.8, 0.8,
                    0.7, 0.7, 0.7, 0.6, 0.6, 0.5)


def params_generator(max_opts):
    prev_opts = []
    while len(prev_opts) < max_opts:
        num_iter = random.choice(NUM_ITERATIONS_OPTS)
        seq = range(num_iter - 1)
        while True:
            scores = random.sample(PRE_LABELS_OPTS, num_iter)
            scores.sort(key=len)
            if all(len(scores[i]) <= len(scores[i + 1]) for i in seq):
                break
        while True:
            conf0 = random.sample(CONFIDENCE0_OPTS, num_iter)
            conf0.sort()
            if all(conf0[i] <= conf0[i + 1] for i in seq):
                break
        while True:
            conf1 = random.sample(CONFIDENCE1_OPTS, num_iter)
            conf1.sort(reverse=True)
            if all(conf1[i] >= conf1[i + 1] for i in seq):
                if all(c0 <= c1 for c0, c1 in zip(conf0, conf1)):
                    break
        opt = tuple(x for x in zip(scores, conf0, conf1))
        if opt in prev_opts:
            continue
        prev_opts.append(opt)
        num_iter = random.choice(NUM_ITERATIONS_OPTS)
        yield opt


def main():
    # Getting required data
    with open(TARGETS_PATH, "rb") as f:
        target_array, scores = pickle.load(f)
        target_list = target_array.tolist()
    with open(FMAT_PATH, "rb") as f:
        fmat = pickle.load(f)
    with open(PASSAGES_PATH, "rb") as f:
        passages = pickle.load(f)
    passages = passages[:34]
    terminals, token_labels = tokeneval.get_terminals_labels(passages)
    tokens = [x.text for x in terminals]

    # Running through random parameters settings
    for params in params_generator(50000):
        clas, _, _ = classify.self_train_classifier(fmat, scores,
                                                    target_array, params)
        target_labels = [int(x >= classify.PRE_LABELS_THRESH) for x in scores]
        target_labels += list(classify.predict_labels(clas,
                                                      fmat[len(scores):]))
        stats = tokeneval.evaluate_with_type(tokens, token_labels,
                                             target_list, target_labels)
        print("\t".join([str(x)
                         for x in params] + [str(len(x)) for x in stats]))


if __name__ == '__main__':
    main()
