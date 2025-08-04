import numpy as np
import json
import os
from collections import Counter

this_dir = os.path.dirname(__file__)
tsvd_path = os.path.join(this_dir, "tsvd_components.npy")
rf_path = os.path.join(this_dir, "rf_forest.json")

tsvd_components = np.load(tsvd_path)
n_classes = 5

with open(rf_path, "r") as f:
    forest = json.load(f)

def transform_tsvd(x, components):
    return np.dot(x, components.T)

def predict_tree(tree, x):
    node = 0
    while tree["children_left"][node] != -1:
        feature = tree["feature"][node]
        threshold = tree["threshold"][node]
        if x[feature] <= threshold:
            node = tree["children_left"][node]
        else:
            node = tree["children_right"][node]
    return np.argmax(tree["value"][node])

def predict_forest(forest, x):
    votes = [predict_tree(tree, x) for tree in forest]
    return max(set(votes), key=votes.count)

def predict_bitstream(features):
    reduced = transform_tsvd(features, tsvd_components)
    pred = predict_forest(forest, reduced)
    return pred
