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
PRE_LABELS_OPTS = ((0, 4, 5), (0, 1, 4, 5), (0, 1, 3, 4, 5),
                   (0, 1, 2, 3, 4, 5))
CONFIDENCE0_OPTS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
CONFIDENCE1_OPTS = (0.5, 0.6, 0.7, 0.8, 0.9)


def params_generator(max_opts):
    prev_opts = []
    num_iter = random.choice(NUM_ITERATIONS_OPTS)
    while len(prev_opts) < max_opts:
        opt = tuple(tuple(random.choice(x) for x in
                          (PRE_LABELS_OPTS, CONFIDENCE0_OPTS,
                           CONFIDENCE1_OPTS))
                    for _ in range(num_iter))
        # check for legality, params change in the correct direction
        if opt in prev_opts:
            continue
        if any(x[1] > x[2] for x in opt):
            continue
        if any(len(opt[i][0]) > len(opt[i + 1][0])
               for i in range(num_iter - 1)):
            continue
        if any(opt[i + 1][1] < opt[i][1]
               for i in range(num_iter - 1)):
            continue
        if any(opt[i + 1][2] > opt[i][2]
               for i in range(num_iter - 1)):
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
    for params in params_generator(1000):
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
