import numpy as np
import json
import os

HERE = os.path.dirname(__file__)

with open(os.path.join(HERE, "meta.json"), "r") as f:
    META = json.load(f)

N_CLASSES = int(META.get("n_classes", 5))
FEATURE_LEN = int(META.get("feature_len", 266))

with open(os.path.join(HERE, "rf_forest.json"), "r") as f:
    forest = json.load(f)

def _as_feature_vec(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).ravel()
    if x.size != FEATURE_LEN:
        raise ValueError(
            f"Feature length mismatch: got {x.size}, expected {FEATURE_LEN}. "
            "Deploy-time feature extraction must match training-time features."
        )
    return x

def _leaf_value(tree, x: np.ndarray) -> np.ndarray:
    """
    Traverse tree, return leaf class-count vector (length N_CLASSES).
    Stored values are float16 in JSON; we accumulate as float32.
    """
    node = 0
    while tree["children_left"][node] != -1:
        feature = int(tree["feature"][node])
        threshold = float(tree["threshold"][node])
        if x[feature] <= threshold:
            node = int(tree["children_left"][node])
        else:
            node = int(tree["children_right"][node])

    v = np.asarray(tree["value"][node], dtype=np.float32).ravel()
    if v.size != N_CLASSES:
        # Defensive: some sklearn dumps can have shape issues if labels differed
        out = np.zeros(N_CLASSES, dtype=np.float32)
        out[: min(N_CLASSES, v.size)] = v[: min(N_CLASSES, v.size)]
        return out
    return v

def predict_bitstream(features):
    """
    Soft vote:
      - sum leaf class-count vectors across trees
      - normalize to probabilities
      - confidence = max prob * 100
    """
    x = _as_feature_vec(features)

    accum = np.zeros(N_CLASSES, dtype=np.float32)
    for tree in forest:
        accum += _leaf_value(tree, x)

    total = float(np.sum(accum))
    if total <= 0.0:
        return 0, 0.0

    probs = accum / total
    pred = int(np.argmax(probs))
    conf = float(np.max(probs) * 100.0)
    return pred, conf
