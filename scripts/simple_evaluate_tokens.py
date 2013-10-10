import sys
import pickle
from ucca import classify, tokeneval

DB_PATH = "/home/beka/thesis/db/"
_, LABELS = pickle.load(open(DB_PATH + "nouns2/targets-labels.nouns2.pickle",
                             "rb"))
TARGETS = pickle.load(open(DB_PATH + "nouns2/targets-scores.nouns2.pickle",
                              "rb"))[0].tolist()
METHOD = sys.argv[1].split(',')
METHOD, PARAM = METHOD[0], float(METHOD[1])
FMAT = pickle.load(open(sys.argv[2], "rb"))
TOKENS_FMAT = None if len(sys.argv) < 4 else pickle.load(open(sys.argv[3],
                                                              "rb"))
NUM_PASSAGES = 50
PASSAGES_PATH = DB_PATH + "db_18_6/huca_18_6_pos_random_filtered.pickle"


def main():
    with open(PASSAGES_PATH, "rb") as f:
        passages = pickle.load(f)
    passages = passages[:NUM_PASSAGES]
    terminals, token_labels = tokeneval.get_terminals_labels(passages)
    tokens = [x.text for x in terminals]

    clas = classify.train_classifier(FMAT[:len(LABELS)], LABELS, METHOD,
                                     c_param=PARAM, nu_param=PARAM,
                                     learn_rate=PARAM, n_estimators=500)
    if TOKENS_FMAT is not None:  # use token evaluation, not type
        stats = tokeneval.evaluate_with_classifier( tokens, token_labels,
                                                   TARGETS, TOKENS_FMAT, clas)
    else:
        target_labels = LABELS.tolist()
        target_labels += classify.predict_labels(clas, FMAT[len(LABELS):]).tolist()
        stats = tokeneval.evaluate_with_type(tokens, token_labels, TARGETS,
                                            target_labels)

    print("\t".join(str(len(x)) for x in stats))


if __name__ == '__main__':
    main()
