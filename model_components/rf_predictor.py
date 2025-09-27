import numpy as np
import json
import os
from collections import Counter

tsvd_components = np.load(os.path.join(os.path.dirname(__file__), "tsvd_components.npy"))
n_classes = 5

# Loads a pre-trained random forest model from JSON
with open(os.path.join(os.path.dirname(__file__), "rf_forest.json"), "r") as f:  
    forest = json.load(f)

# Applies dimensionality reduction using Truncated SVD components
def transform_tsvd(x, components):
    return np.dot(x, components.T)

def predict_tree(tree, x):
    node = 0
    while tree["children_left"][node] != -1:  # Traverse the model tree
        feature = tree["feature"][node]
        threshold = tree["threshold"][node]
        if x[feature] <= threshold:
            node = tree["children_left"][node]
        else:
            node = tree["children_right"][node]
    return np.argmax(tree["value"][node])  # Pick the class with the highest count

# Aggregates predictions from all trees by majority vote
def predict_forest(forest, x):
    votes = [predict_tree(tree, x) for tree in forest]
    return max(set(votes), key=votes.count)

# Reduce feature dimensions using TSVD, then predict with the model
def predict_bitstream(features):
    reduced = transform_tsvd(features, tsvd_components)
    pred = predict_forest(forest, reduced)
    return pred
