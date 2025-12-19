# Copyright (c) 2025, Rye Stahle-Smith; All rights reserved.
# PYNQ BLADEI: Bitstream-Level Abnormality Detection for Embedded Inference
# December 18th, 2025
# Description: This script trains a supervised ML model to detect malicious FPGA bitstreams using byte-level and structural features.
import glob
import tarfile
import os
import sys
import json
import warnings
import random
import numpy as np
from collections import Counter
from scipy.stats import skew, kurtosis, entropy

from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# --------------------------
# Config
# --------------------------
N_CLASSES = 5
FEATURE_LEN = 266  # 256 byte histogram + 10 extended stats
RANDOM_SEED = 42

# --------------------------
# Step 0: Suppress Warnings
# --------------------------
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------
# Step 1: Set the Random Seed
# --------------------------
def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)

# --------------------------
# Step 2: Collect Bitstreams
# --------------------------
def collect_bitstreams():
    empty_files = glob.glob("trusthub_bitstreams/Empty/*.bit")
    benign_aes_files = glob.glob("trusthub_bitstreams/Benign/AES*.bit")
    benign_rs232_files = glob.glob("trusthub_bitstreams/Benign/RS232*.bit")
    mal_aes_files = glob.glob("trusthub_bitstreams/Malicious/AES*.bit")
    mal_rs232_files = glob.glob("trusthub_bitstreams/Malicious/RS232*.bit")
    all_files = empty_files + benign_aes_files + benign_rs232_files + mal_aes_files + mal_rs232_files
    print("=== Organizing bitstreams... ===")
    return empty_files, benign_aes_files, benign_rs232_files, mal_aes_files, mal_rs232_files, all_files

# --------------------------
# Step 3: Feature Extraction
# --------------------------
def extract_features(filepath: str) -> np.ndarray:
    with open(filepath, "rb") as f:
        data = f.read()

    size = len(data)
    if size == 0:
        return np.zeros(FEATURE_LEN, dtype=np.float32)

    counts = Counter(data)
    dense_vec = np.zeros(256, dtype=np.float32)
    for byte_val, count in counts.items():
        dense_vec[int(byte_val)] = float(count) / float(size)

    # 10 extended features
    byte_array = np.frombuffer(data, dtype=np.uint8)
    extended = np.array(
        [
            np.mean(byte_array),
            np.std(byte_array),
            skew(byte_array),
            kurtosis(byte_array),
            entropy(dense_vec),
            np.max(dense_vec),
            np.min(dense_vec[dense_vec > 0]) if np.any(dense_vec > 0) else 0.0,
            np.sum(dense_vec > 0.01),
            np.sum(byte_array),
            size / 1e6,
        ],
        dtype=np.float32,
    )

    feat = np.concatenate([dense_vec, extended]).astype(np.float32, copy=False)
    if feat.size != FEATURE_LEN:
        raise RuntimeError(f"Feature length bug: got {feat.size}, expected {FEATURE_LEN}")
    return feat

def display_progress(current, total):
    bar_length = 20
    percent = int((current / total) * 100)
    blocks = int((current / total) * bar_length)
    bar = "█" * blocks + "-" * (bar_length - blocks)
    sys.stdout.write(f"\rProgress: |{bar}| {percent}% ({current}/{total})")
    sys.stdout.flush()

def generate_features(all_files):
    print("=== Extracting sparse features... ===")
    X = np.zeros((len(all_files), FEATURE_LEN), dtype=np.float32)
    for i, fpath in enumerate(all_files, 1):
        X[i - 1] = extract_features(fpath)
        display_progress(i, len(all_files))
    print()
    return X

def define_labels(empty_files, benign_aes_files, benign_rs232_files, mal_aes_files, mal_rs232_files):
    print("=== Defining labels... ===")
    y = (
        [0] * len(empty_files)
        + [1] * len(benign_aes_files)
        + [2] * len(benign_rs232_files)
        + [3] * len(mal_aes_files)
        + [4] * len(mal_rs232_files)
    )
    return np.asarray(y, dtype=np.int64)

# --------------------------
# Step 4: Train/Test Split
# --------------------------
def split_dataset(X, y, test_size=0.25):
    print("=== Splitting the dataset for training/testing... ===")
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=RANDOM_SEED)

# --------------------------
# Step 5: Compare Classifiers (no TSVD, no SMOTE)
# --------------------------
def compare_classifiers(X_train, y_train, cv=5):
    classifiers = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            min_samples_split=5,
            random_state=RANDOM_SEED,
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=150,
            learning_rate=0.6,
            random_state=RANDOM_SEED,
        ),
        "Logistic Regression": LogisticRegression(max_iter=4000, class_weight="balanced"),
        "Naive Bayes": GaussianNB(),
        "SVM (RBF)": SVC(kernel="rbf", C=2.0, gamma="scale", class_weight="balanced", probability=True),
        "KNN": KNeighborsClassifier(n_neighbors=5, weights="distance"),
        "Decision Tree": DecisionTreeClassifier(max_depth=12, min_samples_split=3, class_weight="balanced"),
    }

    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, average="macro", zero_division=0),
        "recall": make_scorer(recall_score, average="macro", zero_division=0),
        "f1": make_scorer(f1_score, average="macro", zero_division=0),
    }

    print("=== Comparing classifiers using leakage-safe k-Fold CV (no TSVD/SMOTE) ===")
    cv_results = {}
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)

    for name, model in classifiers.items():
        results = cross_validate(model, X_train, y_train, cv=skf, scoring=scoring, n_jobs=-1)
        print(f"\n{name}")
        for metric in scoring:
            mean = results[f"test_{metric}"].mean()
            std = results[f"test_{metric}"].std()
            print(f"  {metric.capitalize()}: {mean:.2f} ± {std:.2f}")
        cv_results[name] = results["test_f1"].mean()

    return classifiers, cv_results

# --------------------------
# Step 6: Tune Random Forest (deployable)
# --------------------------
def tune_random_forest(X_train, y_train, cv=5):
    print("\n=== Hyperparameter Tuning for Random Forest ===")

    base = RandomForestClassifier(
        random_state=RANDOM_SEED,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )

    param_grid = {
        "n_estimators": [300, 500, 800],
        "max_depth": [None, 10, 14, 18],
        "min_samples_split": [2, 3, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2", None],
    }

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
    grid = GridSearchCV(
        base,
        param_grid,
        cv=skf,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
        error_score="raise",
    )
    grid.fit(X_train, y_train)

    print(f"\nBest parameters: {grid.best_params_}")
    print(f"Best CV F1 Score: {grid.best_score_:.4f}\n")
    return grid.best_estimator_

# --------------------------
# Step 7: Evaluate
# --------------------------
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    print("\n*** Final Evaluation on Hold-out Test Set using Random Forest ***\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("\n*** Confusion Matrix on Hold-out Test Set using Random Forest ***")
    cm = confusion_matrix(y_test, y_pred)
    print("\n\t\tPredicted\n\t\t0\t1\t2\t3\t4")
    for i, row in enumerate(cm):
        print(f"Actual {i} |\t" + "\t".join(str(int(v)) for v in row))

    print("\n*** Prediction Confidence Analysis ***")
    mismatch_count = 0
    for i, (true, pred, proba) in enumerate(zip(y_test, y_pred, y_pred_proba)):
        conf = float(np.max(proba))
        if int(true) != int(pred):
            print(f"Sample {i}: True={int(true)}, Pred={int(pred)}, Confidence={conf:.4f} [MISMATCH]")
            mismatch_count += 1
    if mismatch_count == 0:
        print("No mismatches found!")

    print("\n*** ML Training Complete! ***\n")

# --------------------------
# Step 8: Export RF in lightweight JSON form
# --------------------------
def export_rf(model: RandomForestClassifier, out_dir="./model_components", dtype=np.float16):
    print("=== Quantizing ML components... ===")
    os.makedirs(out_dir, exist_ok=True)

    meta = {
        "n_classes": N_CLASSES,
        "feature_len": FEATURE_LEN,
        "model_type": "RandomForestClassifier",
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    rf_data = []
    for tree in model.estimators_:
        t = tree.tree_
        rf_data.append(
            {
                "children_left": t.children_left.astype(np.int32).tolist(),
                "children_right": t.children_right.astype(np.int32).tolist(),
                "feature": t.feature.astype(np.int32).tolist(),
                "threshold": t.threshold.astype(np.float32).tolist(),  # keep stable
                # store per-node class counts as float16
                "value": t.value.squeeze(1).astype(dtype).tolist(),
            }
        )

    with open(os.path.join(out_dir, "rf_forest.json"), "w") as f:
        json.dump(rf_data, f)

    print("*** Quantization Complete! ***\n")

# --------------------------
# Step 9: Compress ML Pipeline
# --------------------------
def compress_to_tar_gz(output_file, targets):
    with tarfile.open(output_file, "w:gz") as tar:
        for target in targets:
            tar.add(target, arcname=os.path.basename(target))

def export_pipeline():
    print("=== Compressing pipeline for PYNQ Deployment... ===")
    targets = ["model_components", "deploy_model.py"]
    output_file = "PYNQ_BLADEI.tar.gz"
    compress_to_tar_gz(output_file, targets)
    print(f"*** Compression complete! Archive saved as '{output_file}'... ***")

# --------------------------
# Main Execution
# --------------------------
def main():
    set_seed(RANDOM_SEED)
    empty_files, benign_aes_files, benign_rs232_files, mal_aes_files, mal_rs232_files, all_files = collect_bitstreams()
    X = generate_features(all_files)
    y = define_labels(empty_files, benign_aes_files, benign_rs232_files, mal_aes_files, mal_rs232_files)

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    classifiers, cv_results = compare_classifiers(X_train, y_train, cv=5)

    # Always tune/export Random Forest (deployable)
    best_rf = tune_random_forest(X_train, y_train, cv=5)
    best_rf.fit(X_train, y_train)

    evaluate_model(best_rf, X_test, y_test)

    export_rf(best_rf, out_dir="./model_components", dtype=np.float16)
    export_pipeline()

if __name__ == "__main__":
    main()
