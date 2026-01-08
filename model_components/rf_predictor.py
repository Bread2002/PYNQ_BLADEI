# Copyright (c) 2025, Rye Stahle-Smith; All rights reserved.
# PYNQ BLADEI: Bitstream-Level Abnormality Detection for Embedded Inference
# January 6th, 2026
# Description: This helper script prepares the dual-head classification model for deployment.
#              It loads the Random Forest models and provides prediction functions.

import numpy as np                                          # For numerical operations
import json                                                 # For loading model JSON files
import os                                                   # For file path operations

HERE = os.path.dirname(__file__)                            # Get directory containing this script

# --------------------------
# Step 0: Load Metadata
# --------------------------
with open(os.path.join(HERE, "meta.json"), "r") as f:       # Open metadata file
    META = json.load(f)                                     # Parse JSON contents

TROJAN_CLASSES = META.get("trojan_classes", ["Benign", "Malicious"])  # Trojan class labels
FAMILY_CLASSES = META.get("family_classes", ["CRYPTO", "COMMS", "MCU/CPU", "BUS/DISPLAY", "ITC99", "ISCAS89"])  # Hardware family labels
N_FEATURES = int(META.get("n_features", 30))                # Expected feature vector dimension (after TSVD)
N_ORIGINAL_FEATURES = int(META.get("original_features", 278))  # Original feature dimension (before TSVD)

# --------------------------
# Step 1: Load TSVD Components
# --------------------------
tsvd_components = None                                      # Initialize TSVD components array
tsvd_path = os.path.join(HERE, "tsvd.json")                 # Path to TSVD JSON file
if os.path.exists(tsvd_path):                               # If TSVD file exists, open and parse it
    with open(tsvd_path, "r") as f:
        tsvd_data = json.load(f)
    tsvd_components = np.asarray(tsvd_data["components"], dtype=np.float32)  # Load TSVD components

# --------------------------
# Step 2: Load Scaler
# --------------------------
scaler_mean = None                                          # Initialize scaler mean array
scaler_scale = None                                         # Initialize scaler scale array
scaler_path = os.path.join(HERE, "scaler.json")             # Path to scaler JSON file
if os.path.exists(scaler_path):                             # If scaler file exists, open and parse it
    with open(scaler_path, "r") as f:
        scaler_data = json.load(f)
    scaler_mean = np.asarray(scaler_data["mean"], dtype=np.float32)   # Load mean values
    scaler_scale = np.asarray(scaler_data["scale"], dtype=np.float32) # Load scale values

# --------------------------
# Step 3: Load Trojan Detector
# --------------------------
trojan_forest = None                                        # Initialize JSON RF model
trojan_model_joblib = None                                  # Initialize joblib model
rf_trojan_path = os.path.join(HERE, "rf_trojan.json")       # Path to JSON RF model
trojan_joblib_path = os.path.join(HERE, "trojan_model.joblib")  # Path to joblib model

if os.path.exists(rf_trojan_path):                          # If JSON RF exists, load the quantized forest
    with open(rf_trojan_path, "r") as f:
        trojan_forest = json.load(f)
elif os.path.exists(trojan_joblib_path):                    # Otherwise, import joblib library and load the sklearn model
    import joblib
    trojan_model_joblib = joblib.load(trojan_joblib_path)

# --------------------------
# Step 4: Load Family Classifier
# --------------------------
family_forest = None                                        # Initialize JSON RF model
rf_family_path = os.path.join(HERE, "rf_family.json")       # Path to JSON RF model
family_joblib_path = os.path.join(HERE, "family_model.joblib")  # Path to joblib model

if os.path.exists(rf_family_path):                          # If JSON RF exists, load the quantized forest
    with open(rf_family_path, "r") as f:
        family_forest = json.load(f)
elif os.path.exists(family_joblib_path):                    # Otherwise, import joblib library and load the sklearn model
    import joblib
    family_model_joblib = joblib.load(family_joblib_path)


# --------------------------
# Step 5: Helper Functions
# --------------------------
def _as_feature_vec(x) -> np.ndarray:
    """Validate and convert raw feature vector (278 features)."""
    x = np.asarray(x, dtype=np.float32).ravel()             # Convert to 1D float32 array
    if x.size != N_ORIGINAL_FEATURES:                       # If size doesn't match expected, raise descriptive error
        raise ValueError(
            f"Feature length mismatch: got {x.size}, expected {N_ORIGINAL_FEATURES}. "
            f"Deploy-time feature extraction must match training-time features ({N_ORIGINAL_FEATURES})."
        )
    return x                                                # Return validated feature vector


def _apply_tsvd(x: np.ndarray) -> np.ndarray:
    """Apply TSVD dimensionality reduction (278 -> 30 features)."""
    if tsvd_components is None:                             # If TSVD not loaded, return original features
        return x
    return np.dot(x, tsvd_components.T)                     # Apply TSVD transformation


def _apply_scaler(x: np.ndarray) -> np.ndarray:
    """Apply StandardScaler normalization."""
    if scaler_mean is None or scaler_scale is None:         # If scaler not loaded, return unscaled features
        return x
    return (x - scaler_mean) / scaler_scale                 # Apply z-score normalization


def _leaf_value_binary(tree, x: np.ndarray) -> np.ndarray:
    """Traverse tree for binary classification, return class-count vector."""
    node = 0                                                # Start at root node
    while tree["children_left"][node] != -1:                # While not at a leaf node
        feature = int(tree["feature"][node])                # Get the split feature index
        threshold = float(tree["threshold"][node])          # Get the split threshold
        if x[feature] <= threshold:                         # If feature value is less than or equal to threshold, go to left child
            node = int(tree["children_left"][node])
        else:
            node = int(tree["children_right"][node])        # Otherwise, go to right child
    
    v = np.asarray(tree["value"][node], dtype=np.float32).ravel()  # Get leaf class counts
    return v                                                # Return class-count vector


def _leaf_value_multiclass(tree, x: np.ndarray, n_classes: int) -> np.ndarray:
    """Traverse tree for multi-class classification, return class-count vector."""
    node = 0                                                # Start at root node
    while tree["children_left"][node] != -1:                # While not at a leaf node
        feature = int(tree["feature"][node])                # Get the split feature index
        threshold = float(tree["threshold"][node])          # Get the split threshold
        if x[feature] <= threshold:                         # If feature value is less than or equal to threshold, go to left child
            node = int(tree["children_left"][node])
        else:
            node = int(tree["children_right"][node])        # Otherwise, go to right child
    
    v = np.asarray(tree["value"][node], dtype=np.float32).ravel()  # Get leaf class counts
    if v.size != n_classes:                                 # If size mismatch, create zero-padded array
        out = np.zeros(n_classes, dtype=np.float32)
        out[:min(n_classes, v.size)] = v[:min(n_classes, v.size)]  # Copy available values
        return out
    return v                                                # Return class-count vector


# --------------------------
# Step 6: Trojan Detection
# --------------------------
def predict_trojan(features) -> tuple:
    """
    Predict if bitstream is benign or malicious.
    Returns: (prediction_idx, confidence_percent, label_string)
    """
    x = _as_feature_vec(features)                           # Validate feature vector (278 features)
    x = _apply_tsvd(x)                                      # Apply TSVD dimensionality reduction (278 -> 30)
    x = _apply_scaler(x)                                    # Apply StandardScaler normalization
    
    if trojan_model_joblib is not None:                     # If using joblib model, get prediction
        pred = int(trojan_model_joblib.predict(x.reshape(1, -1))[0])
        if hasattr(trojan_model_joblib, 'predict_proba'):   # If model supports probabilities, compute confidence percentage
            proba = trojan_model_joblib.predict_proba(x.reshape(1, -1))[0]
            conf = float(np.max(proba) * 100.0)
        else:
            conf = 100.0                                    # Otherwise assume full confidence
        label = TROJAN_CLASSES[pred]                        # Get class label
        return pred, conf, label
    
    if trojan_forest is None:                               # If no model available, return unknown prediction
        return 0, 0.0, "Unknown"
    
    # RF soft vote
    accum = np.zeros(2, dtype=np.float32)                   # Initialize vote accumulator
    for tree in trojan_forest:                              # For each tree in the forest
        accum += _leaf_value_binary(tree, x)                # Add its class-count vote
    
    total = float(np.sum(accum))                            # Sum all votes
    if total <= 0.0:                                        # If no votes, return unknown prediction
        return 0, 0.0, "Unknown"
    
    probs = accum / total                                   # Normalize to probabilities
    pred = int(np.argmax(probs))                            # Get argmax prediction
    conf = float(np.max(probs) * 100.0)                     # Compute confidence percentage
    label = TROJAN_CLASSES[pred]                            # Get class label
    return pred, conf, label                                # Return prediction tuple


# --------------------------
# Step 7: Family Classification
# --------------------------
def predict_family(features) -> tuple:
    """
    Predict hardware family of the bitstream.
    Returns: (prediction_idx, confidence_percent, label_string)
    """
    x = _as_feature_vec(features)                           # Validate feature vector (278 features)
    x = _apply_tsvd(x)                                      # Apply TSVD dimensionality reduction (278 -> 30)
    x = _apply_scaler(x)                                    # Apply StandardScaler normalization
    
    if family_forest is None:                               # If no model available, return unknown prediction
        return 0, 0.0, "Unknown"
    
    n_classes = len(FAMILY_CLASSES)                         # Get number of family classes
    accum = np.zeros(n_classes, dtype=np.float32)           # Initialize vote accumulator
    for tree in family_forest:                              # For each tree in the forest
        accum += _leaf_value_multiclass(tree, x, n_classes) # Add its class-count vote
    
    total = float(np.sum(accum))                            # Sum all votes
    if total <= 0.0:                                        # If no votes, return unknown prediction
        return 0, 0.0, "Unknown"
    
    probs = accum / total                                   # Normalize to probabilities
    pred = int(np.argmax(probs))                            # Get argmax prediction
    conf = float(np.max(probs) * 100.0)                     # Compute confidence percentage
    label = FAMILY_CLASSES[pred]                            # Get class label
    return pred, conf, label                                # Return prediction tuple


# --------------------------
# Step 8: Full Prediction Pipeline
# --------------------------
def predict_bitstream(features) -> dict:
    """
    Full prediction pipeline: trojan detection + family classification.
    Returns dict with all predictions and confidence scores.
    """
    trojan_pred, trojan_conf, trojan_label = predict_trojan(features)   # Run trojan detection
    family_pred, family_conf, family_label = predict_family(features)   # Run family classification
    
    return {                                                # Return combined results dict
        "trojan": {
            "prediction": trojan_pred,                      # Trojan class index
            "confidence": trojan_conf,                      # Trojan confidence %
            "label": trojan_label,                          # Trojan class label
            "is_malicious": trojan_pred == 1                # Boolean malicious flag
        },
        "family": {
            "prediction": family_pred,                      # Family class index
            "confidence": family_conf,                      # Family confidence %
            "label": family_label                           # Family class label
        }
    }