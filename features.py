"""Feature extractor for UCCA scene-evoking classifier.

There are two kinds of features: lexical features extracted from the the
classified target word, and context features extracted from its instances
in non-UCCA corpora. In addition, extracting the class for the word from
UCCA passages is also covered.

"""

import argparse
import sys
import re
import nltk
import pickle
from ucca import lex, collins
from ucca.postags import POSTags


DEFAULT_CHUNK_SIZE = 2 ** 20


def chunks(f, size=DEFAULT_CHUNK_SIZE):
    """Yields chunks of lines from a file until exhausted.

    Args:
        f: file to yield lines from
        size: hint to readlines() of how many bytes each yield should contain.

    Yields:
        list of strings (lines).

    """
    while True:
        x = f.readlines(size)
        if x:
            yield x
        else:
            raise StopIteration


def tokenize(sentences):
    """list of string sentences ==> list of tuples of stripped tokens"""
    return [tuple(x.strip().split()) for x in sentences if x.strip()]


def parse_ngram_line(line):
    """tab-separated ngram line ==> count (int), ngram (tuple)"""
    count, ngram = line.strip().split('\t')
    return int(count), tuple(ngram.split())


def extract_ngrams(size, sentences, counts, *, lemmatize=False):
    """Extracts all ngrams of the given size from the sentences.

    Args:
        size: the N parameter of the ngrams to extract
        sentences: sequence (can be generator) of tuples of strings, each
            is a token.
        counts: previous ngram dictionary to start with.
            This dictionary is changed by the method.
        lemmatize: whether to lemmatize the ngram tokens. Lemmatization is
            of noun forms, so plurals will be removed, but basically nothing
            else will change (verb derivations, adverbs etc.).

    Returns:
        counts dictionary (the same as the argument).

    """
    wn = nltk.stem.wordnet.WordNetLemmatizer()
    for sentence in sentences:
        if len(sentence) < size:
            continue
        s = tuple(wn.lemmatize(x) for x in sentence) if lemmatize else sentence
        for i in range((len(s) - size + 1)):
            counts[s[i:i + size]] = counts.get(s[i:i + size], 0) + 1
    return counts


def filter_ngrams(lines, *, threshold=1, exclude=frozenset(), match_all=False,
                  startswith=None, endswith=None):
    """Filters the given ngrams according to the keyword parameters.

    Args:
        lines: list of tab-seprated strings of count-ngram pairs,
            will be modified.
        threshold: what is the minimum count to keep the ngram in
        exclude: set of tokens which make the ngram being omitted.
            If one of the tokens appear in any position in the ngram, it
            is being excluded from the results.
        match_all: whether whole the tokens in the ngram should match
            exclude in order to be removed.
        startswith: a token which all remaining ngrams must start with.
            all ngrams go lemmatization in the first position in order
            to match. None if not active.
        endswith: same as startswith, but at the last ngram position.

    Returns:
        lines argument (after modification)

    """
    def _discard(line):
        count, ngram = parse_ngram_line(line)
        if ((startswith and wn.lemmatize(ngram[0]) != startswith) or
                (endswith and wn.lemmatize(ngram[-1]) != endswith)):
            return True
        if match_all:
            return (count < threshold or set(ngram) < exclude)
        else:
            return (count < threshold or exclude & set(ngram))

    wn = nltk.stem.wordnet.WordNetLemmatizer()
    indices = [i for i, line in enumerate(lines) if _discard(line)]
    for i in reversed(indices):
        del lines[i]
    return lines


def merge_ngrams(lines):
    """Merge adjacent ngram counts if they are refering to the same ngram.

    Assumes lines are sorted alphabetically by ngram, and yields the lines
    after merging the counts.

    Args:
        lines: a generator of tab-separated strings of count-ngram pairs,

    Yields:
        merged-count lines of ngrams

    """
    first = next(lines)  # first line with the ngram, will be the one yielded
    if not first:
        return
    for curr in lines:
        count1, ngram1 = parse_ngram_line(first)
        count2, ngram2 = parse_ngram_line(curr)
        if ngram1 == ngram2:
            first = '{}\t{}\n'.format(count1 + count2, ' '.join(ngram1))
        else:
            yield first
            first = curr
    yield first


def create_feature_counts(lines, features, pos=0):
    """Yields all ngram counts who has a feature phrase in the given position.

    Each ngram count in lines is tested for the existence of the feature phrase
    in ngram[pos:pos+len(feature)], and if it matches the line is yielded.

    Args:
        lines: a generator of tab-separated strings of count-ngram pairs,
        features: a sequence of feature tuples (of strings)
        pos: position in thee ngram to match the feature, default 0

    Yields:
        matching lines of ngrams

    """
    for line in lines:
        count, ngram = parse_ngram_line(line)
        if any(ngram[pos:pos + len(x)] == x for x in features):
            yield line


def calculate_ngram_features(lines, features, targets, divider=10 ** 5):
    """Returns a dictionary of feature scores for ngrams before/after.

    Args:
        lines: count-ngram lines with both the target and feature words in it
        features: a sequence of feature tuples (of strings)
        targets: a sequence of target words (strings)
        divider: an integer to divide all counts by, for normalizing features

    Returns:
        a dictionary: {target1: {feat1_before: score, feat1_after: score ...}}

    """
    for line in lines:
        count, ngram = parse_ngram_line(line)
        if ngram[:-1] in features and ngram[-1] in targets:
            print('{}\t{}_before\t{}'.format(ngram[-1], '_'.join(ngram[:-1]),
                                             count / divider))
        if ngram[1:] in features and ngram[0] in targets:
            print('{}\t{}_after\t{:12f}'.format(ngram[0], '_'.join(ngram[1:]),
                                                count / divider))


def has_suffix(targets, suffix):
    """0/1 tuple if suffix string applies to the end of the targets."""
    # targets are tuples, but we deal only with 1-word targets
    return tuple(int(x[0].endswith(suffix)) for x in targets)


def has_prefix(targets, prefix):
    """0/1 tuple if prefix string applies to the start of the targets."""
    # targets are tuples, but we deal only with 1-word targets
    return tuple(int(x[0].startswith(prefix)) for x in targets)


def extract_dict_features(targets, collins_path):
    targets = [' '.join(target) for target in targets]  # tuples to strings
    with open(collins_path, 'rb') as f:
        raw_dict = pickle.load(f)
    coll = collins.CollinsDictionary(raw_dict)
    feats = []
    descriptions = []
    for target in targets:
        entries = coll.by_key(target)
        if len(entries) != 1:  # we don't handle context-dependent entries
            descriptions.append([])
            continue
        descriptions.append([s.desc for s in entries[0].senses
                             if s.pos == POSTags.Noun])
    for mark in ('activity', 'process', 'act'):
        feats.append(' '.join(
            str(int(any(mark in d.split()[1:6] for d in tdesc)))
            for tdesc in descriptions))
    return feats


def extract_hfw_dict_features(targets, collins_path, hfw):
    targets = [' '.join(target) for target in targets]  # tuples to strings
    with open(collins_path, 'rb') as f:
        raw_dict = pickle.load(f)
    coll = collins.CollinsDictionary(raw_dict)
    feats = []
    descriptions = []
    for target in targets:
        entries = coll.by_key(target)
        if len(entries) != 1:  # we don't handle context-dependent entries
            descriptions.append([])
            continue
        descriptions.append([s.desc for s in entries[0].senses
                             if s.pos == POSTags.Noun])
    for word in hfw:
        rx = re.compile('\W{}\W'.format(word))
        feats.append(' '.join(
            str(int(any(rx.search(d) for d in tdesc)))
            for tdesc in descriptions))
    return feats


def print_progress(current, updated=[0]):
    """Prints progress to stderr by adding current to updated[0] each time."""
    updated[0] = updated[0] + current
    print("Processed: " + str(updated[0]), file=sys.stderr)


def parse_cmd():
    """Parses and validates cmd line arguments, then return them."""
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('ngrams', 'counts', 'morph'))
    parser.add_argument('action', choices=('extract', 'filter', 'merge',
                                           'score'))
    parser.add_argument('--ngram_size', type=int, default=1)
    parser.add_argument('--sort', action='store_true')
    parser.add_argument('--exclude')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--lemmatize', action='store_true')
    parser.add_argument('--threshold', type=int, default=1)
    parser.add_argument('--startswith')
    parser.add_argument('--endswith')
    parser.add_argument('--targets')
    parser.add_argument('--featurewords')
    parser.add_argument('--position', type=int, default=0)
    parser.add_argument('--suffixes')
    parser.add_argument('--prefixes')
    parser.add_argument('--collins')
    parser.add_argument('--wiktionary')
    parser.add_argument('--hfw')

    args = parser.parse_args()
    if args.exclude:
        with open(args.exclude) as f:
            tokens = f.readlines()
        args.exclude = {x.strip() for x in tokens}
    else:
        args.exclude = set()
    if args.targets:
        with open(args.targets) as f:
            args.targets = [tuple(x.strip().split(' ')) for x in f.readlines()]
    if args.featurewords:
        with open(args.featurewords) as f:
            args.featurewords = [tuple(x.strip().split(' '))
                                 for x in f.readlines()]
    if args.suffixes:
        with open(args.suffixes) as f:
            args.suffixes = tuple(x.strip() for x in f)

    if args.prefixes:
        with open(args.prefixes) as f:
            args.prefixes = tuple(x.strip() for x in f)
    if args.hfw:
        with open(args.hfw) as f:
            args.hfw = [x.strip() for x in f.readlines()]

    return args


def main():
    args = parse_cmd()

    if args.command == 'ngrams' and args.action == 'extract':
        counts = {}
        for data in chunks(sys.stdin):
            sentences = tokenize(data)
            counts = extract_ngrams(args.ngram_size, sentences, counts,
                                    lemmatize=args.lemmatize)
            print_progress(len(sentences))
        print("Finished processing", file=sys.stderr)
        if args.sort:
            sorted_counts = sorted(counts.items(), key=lambda x: x[1],
                                   reverse=True)
            print('\n'.join('\t'.join([str(value), ' '.join(ngram)])
                            for ngram, value in sorted_counts))
        else:
            for ngram, value in counts.items():
                print('\t'.join([str(value), ' '.join(ngram)]))

    if args.command == 'ngrams' and args.action == 'filter':
        for data in chunks(sys.stdin):
            print_progress(len(data))
            filtered = filter_ngrams(data, threshold=args.threshold,
                                     exclude=args.exclude,
                                     match_all=args.all,
                                     startswith=args.startswith,
                                     endswith=args.endswith)
            print(*filtered, sep='', end='')

    if args.command == 'ngrams' and args.action == 'merge':
        for new_line in merge_ngrams(sys.stdin):
            print(new_line, end='')

    if args.command == 'ngrams' and args.action == 'score':
        targets = [x[0] for x in args.targets]  # converting from tuples to str
        scores = calculate_ngram_features(sys.stdin, args.featurewords,
                                          targets)

    if args.command == 'counts' and args.action == 'extract':
        for res in create_feature_counts(sys.stdin, args.targets,
                                         args.position):
            print(res, end='')

    if args.command == 'morph' and args.action == 'extract':
        res = [" ".join(str(x) for x in has_suffix(args.targets, suffix))
               for suffix in args.suffixes]
        res += [" ".join(str(x) for x in has_prefix(args.targets, prefix))
                for prefix in args.prefixes]
        form_ident = lex.FormIdentifier(args.collins, args.wiktionary)
        dual_vn = []
        for target in args.targets:
            if len(target) == 1 and form_ident.is_dual_vn(target[0]):
                dual_vn.append('1')
            else:
                dual_vn.append('0')
        res.append(" ".join(dual_vn))
        res += extract_dict_features(args.targets, args.collins)
        res += extract_hfw_dict_features(args.targets, args.collins, args.hfw)
        print("\n".join(res))


if __name__ == '__main__':
    main()
