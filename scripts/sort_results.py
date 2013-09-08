import sys


PARAMS, FO, NF, TP, TN, FP, FN = range(7)


def precision(x):
    return x[TP] / (x[TP] + x[FP])


def recall(x):
    return x[TP] / (x[TP] + x[FN])


def accuracy(x):
    return (x[TP] + x[TN]) / x[FO]


def f1score(x):
    return (2 * precision(x) * recall(x)) / (precision(x) + recall(x))

KEY = f1score


def main():
    for filename in sys.argv[1:]:
        with open(filename) as f:
            lines = [x.strip().split('\t') for x in f.readlines()]
            lines = [[' '.join(x[:-6])] + [int(y) for y in x[-6:]]
                     for x in lines]
            lines.sort(key=KEY, reverse=True)
        with open(filename + '.sorted', 'wt') as f:
            f.write('\n'.join('{}\t{}\t{}'.format(
                x[PARAMS], '\t'.join(str(y) for y in x[FO:]), KEY(x))
                for x in lines))


if __name__ == '__main__':
    main()

