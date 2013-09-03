import numpy as np
import pickle
from ucca import lex, tokeneval, features


DB_PATH = "/home/beka/thesis/db/"
PREFIXES = [x.strip() for x in open(DB_PATH + 'prefixes')]
SUFFIXES = [x.strip() for x in open(DB_PATH + 'suffixes')]
HFW = [x.strip() for x in open(DB_PATH + 'dict_features')]
FUNCWORDS = [x.strip().split() for x in open(DB_PATH + 'function-words')]
LIGHTVERBS = [x.strip().split() for x in open(DB_PATH + 'light-verbs')]
TARGETS = [x.strip() for x in open(DB_PATH + 'nouns2/targets.filtered')]
PASSAGES_PATH = DB_PATH + "db_18_6/huca_18_6_pos_random_filtered.pickle"
FMAT_PATH = DB_PATH + "nouns2/fmat_tokens_mdc.nouns2.pikle"
WIKT_PATH = '/home/beka/thesis/resources/enwikt-defs-latest-en.tsv'
COLLINS_PATH = '/home/beka/thesis/resources/collins/collins.pickle'
NUM_PASSAGES = 50

USE_MORPH_DICT = True
USE_HFW = False
USE_FUNCWORDS = True
USE_LIGHTVERBS = True


def main():
    with open(PASSAGES_PATH, "rb") as f:
        passages = pickle.load(f)
    passages = passages[:NUM_PASSAGES]
    terminals, token_labels = tokeneval.get_terminals_labels(passages)
    tokens_context = tokeneval.get_context(terminals, context=2)
    tokens = [x[0] for x in tokens_context]
    lemmas = [tokeneval.lemmatize(token, TARGETS) for token in tokens]
    lemmas_tuples = [(lemma,) for lemma in lemmas]
    form_ident = lex.FormIdentifier(COLLINS_PATH, WIKT_PATH)

    # First calculate all features which are computed together
    if USE_MORPH_DICT:
        res = features.extract_dict_features(lemmas_tuples, COLLINS_PATH)
        res = [x.split(' ') for x in res]
        res = [[int(x) for x in y] for y in res]
        dict_features = list(zip(*res))
    if USE_HFW:
        res = features.extract_hfw_dict_features(lemmas_tuples, COLLINS_PATH,
                                                 HFW)
        res = [x.split(' ') for x in res]
        res = [[int(x) for x in y] for y in res]
        hfw_features = list(zip(*res))

    # Creating a list of features for each token
    all_res = []
    print("finished init")
    for i, (token, pre_context, post_context) in enumerate(tokens_context):
        if i % 100 == 0: print(i)
        lemma = lemmas[i]
        res = []
        if USE_MORPH_DICT:
            res += [int(lemma.endswith(suffix)) for suffix in SUFFIXES]
            res += [int(lemma.startswith(prefix)) for prefix in PREFIXES]
            res.append(int(form_ident.is_dual_vn(lemma)))
            res.extend(dict_features[i])
        if USE_HFW:
            res.extend(hfw_features[i])
        if USE_FUNCWORDS:
            for funcwords in FUNCWORDS:
                if pre_context and pre_context[0].lower() in funcwords:
                    res.append(1)
                else:
                    res.append(0)
                if post_context and post_context[0].lower() in funcwords:
                    res.append(1)
                else:
                    res.append(0)
        if USE_LIGHTVERBS:
            for lightverbs in LIGHTVERBS:
                if ((pre_context and pre_context[0].lower() in lightverbs) or
                        (len(pre_context) > 1 and
                         pre_context[1].lower() in lightverbs)):
                    res.append(1)
                else:
                    res.append(0)
                if ((post_context and post_context[0].lower() in lightverbs) or
                        (len(post_context) > 1 and
                         post_context[1].lower() in lightverbs)):
                    res.append(1)
                else:
                    res.append(0)

        all_res.append(res)

    # Converting to numpy matrix
    fmat = np.array(all_res)
    with open(FMAT_PATH, 'wb') as f:
        pickle.dump(fmat, f)


if __name__ == '__main__':
    main()
