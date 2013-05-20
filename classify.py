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