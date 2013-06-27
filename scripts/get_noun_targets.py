"""call with DBFILE THRESH1 THRES2.

DBFILE - pickle file of list of passages, whose terminals are pos-tagged
THRESH1 - int, how many appearances a noun needs in order to be count as
a valid target (less than this threshold is ignored).
THRESH2 - float, what is the ratio between scene-evoking instances to not
which will label this target as scene evoker

Output: both prints the targets with num_instances and ratio to stdout,
and saves the targets and labels in a pickle file.

"""
from ucca import scenes
import sys
import pickle
import numpy as np

dbfile = sys.argv[1]
appear_thresh = int(sys.argv[2])
ratio_thresh = float(sys.argv[3])
with open(dbfile, 'rb') as f:
    passages = pickle.load(f)
nouns = scenes.extract_all_nouns(passages)
targets = []
labels = []
for noun, terminals in nouns.items():
    evokers = [scenes.is_scene_evoking(x) for x in terminals].count(True)
    print("{}\t{}\t{}".format(noun, evokers / len(terminals),
                              len(terminals)))
    targets.append(noun)
    if (len(terminals) >= appear_thresh and
            evokers / len(terminals) >= ratio_thresh):
        labels.append(1)
    else:
        labels.append(0)
np_labels = np.array(labels, dtype=np.int32)
with open(dbfile + '.tl.pickle', 'wb') as f:
    pickle.dump((targets, labels), f)
