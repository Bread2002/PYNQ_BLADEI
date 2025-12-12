# Copyright (c) 2025, Rye Stahle-Smith; All rights reserved.
# PYNQ BLADEI: Bitstream-Level Abnormality Detection for Embedded Inference
# November 17th, 2025
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
from scipy.stats import skew, kurtosis
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE

# --------------------------
# Step 0: Suppress Warnings
# --------------------------
warnings.filterwarnings("ignore", category=FutureWarning)  # Suppress future warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Suppress user warnings

# --------------------------
# Step 1: Set the Random Seed
# --------------------------
def set_seed(seed=42):
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
def extract_sparse_features(filepath):
    with open(filepath, 'rb') as f:  # Read the data
        data = f.read()
    size = len(data)
    if size == 0:  # If the filesize is 0, return a null matrix
        return np.zeros(256)
    counts = Counter(data)
    dense_vec = np.zeros(256)
    for byte_val, count in counts.items():  # Normalize frequency of each byte value
        dense_vec[byte_val] = count / size
    return dense_vec

def display_progress(current, total):  # Helper function to visualize progress
    bar_length = 20
    percent = int((current / total) * 100)
    blocks = int((current / total) * bar_length)
    bar = '█' * blocks + '-' * (bar_length - blocks)
    sys.stdout.write(f'\rProgress: |{bar}| {percent}% ({current}/{total})')
    sys.stdout.flush()

def generate_features(all_files):
    print("=== Extracting sparse features... ===")
    feature_matrix = []
    for i, f in enumerate(all_files, 1):  # For each file in the dataset,
        feature_matrix.append(extract_sparse_features(f))  # Extract the features
        display_progress(i, len(all_files))  # Update progress
    print()
    return np.array(feature_matrix)  # Return independent variables (X)

def define_labels(empty_files, benign_aes_files, benign_rs232_files, mal_aes_files, mal_rs232_files):
    print("=== Defining labels... ===")
    y = [0]*len(empty_files) + [1]*len(benign_aes_files) + [2]*len(benign_rs232_files) + \
        [3]*len(mal_aes_files) + [4]*len(mal_rs232_files)
    return y  # Return dependent variable (y)

# --------------------------
# Step 3: Apply TSVD
# --------------------------
def apply_tsvd(X, n_components=30):
    print("=== Applying Truncated Singular Value Decomposition (TSVD)... ===")
    sparse_X = csr_matrix(X)  # Store X in a sparse matrix
    tsvd = TruncatedSVD(n_components=n_components, random_state=42)  # Initialize TSVD
    X_reduced = tsvd.fit_transform(sparse_X)  # Transform the sparse matrix with TSVD
    return X_reduced, tsvd

# --------------------------
# Step 5: Train/Test Split
# --------------------------
def split_dataset(X_reduced, y, test_size=0.25):
    print("=== Splitting the dataset for training/testing... ===")
    return train_test_split(X_reduced, y, test_size=test_size, stratify=y, random_state=42)

# --------------------------
# Step 6: Apply SMOTE
# --------------------------
def apply_smote(X_train, y_train, k_values=[2, 5, 7, 9, 11]):
    print("=== Comparing k_neighbors values for SMOTE... ===\n")
    best_k = None
    best_score = 0

    for k in k_values:  # For each k-value,
        try:
            smote = SMOTE(k_neighbors=k, random_state=42)  # Initialize SMOTE
            X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)  # Resample X and y with SMOTE

            model = RandomForestClassifier(n_estimators=100, random_state=42)  # Initialize a RF classifier
            scores = cross_val_score(model, X_train_smote, y_train_smote, cv=5, scoring='f1_macro')  # Cross validate using F1 Macro scoring
            mean_score = scores.mean()  # Find the mean score for F1 Macro
            
            print(f"SMOTE (k={k}): F1 Macro = {mean_score:.4f} ± {scores.std():.4f}")  # Display the k-value and associated mean score

            if mean_score > best_score:  # If the mean score is greater than the current best score,
                best_score = mean_score  # Update the best score
                best_k = k  # Store the best k-value

        except ValueError as e:
            print(f"SMOTE (k={k}) failed: {e}")

    print(f"\n=== Applying SMOTE with k_neighbors={best_k}... ===")
    smote = SMOTE(k_neighbors=best_k, random_state=42)  # Initialize SMOTE using the best k-value
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)  # Resample X and y with SMOTE
    return X_train_smote, y_train_smote, best_k

# --------------------------
# Step 7: Compare Classifiers
# --------------------------
def compare_classifiers(X_train_smote, y_train_smote):
    # Initialize a dictionary for various classifiers
    classifiers = {
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100),
        "AdaBoost": GradientBoostingClassifier(n_estimators=100),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": GaussianNB(),
        "SVM (RBF)": SVC(kernel='rbf'),
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier()
    }

    # Initialize a dictionary for scoring params
    scoring = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score, average='macro', zero_division=0),
        'recall': make_scorer(recall_score, average='macro', zero_division=0),
        'f1': make_scorer(f1_score, average='macro', zero_division=0)
    }

    print("=== Comparing classifiers using k-Fold Cross-Validation (kFCV)... ===")
    cv_results = {}
    k = 5
    for name, model in classifiers.items():
        results = cross_validate(model, X_train_smote, y_train_smote, cv=k, scoring=scoring)  # Cross validate using the current model
        print(f"\n{name}")  # Print the results
        for metric in scoring:
            mean = results[f'test_{metric}'].mean()
            std = results[f'test_{metric}'].std()
            print(f"  {metric.capitalize()}: {mean:.2f} ± {std:.2f}")
        cv_results[name] = results['test_f1'].mean()  # Store the results based on the mean F1 Score
    return classifiers, cv_results

# --------------------------
# Step 8: Evaluate ML Model
# --------------------------
def evaluate_best_model(classifiers, cv_results, X_train_smote, y_train_smote, X_test, y_test):
    best_model_name = max(cv_results, key=cv_results.get)  # Access the name of the best performing model
    best_model = classifiers[best_model_name]  # Store the best performing model
    best_model.fit(X_train_smote, y_train_smote)  # Fit X and y for the model

    y_pred = best_model.predict(X_test)  # Store all model predictions
    print(f"\n*** Final Evaluation on Hold-out Test Set using {best_model_name} ***\n")
    print(classification_report(y_test, y_pred, zero_division=0))  # Display the classification report

    print(f"\n*** Confusion Matrix on Hold-out Test Set using {best_model_name} ***")
    cm = confusion_matrix(y_test, y_pred)  # Initialize a confusion matrix
    print("\n\t\tPredicted\n\t\t0\t1\t2\t3\t4")  # Display the confusion matrix
    for i, row in enumerate(cm):
        print(f"Actual {i} |\t" + "\t".join(str(val) for val in row))
        
    print("\n*** ML Training Complete! ***\n")
    
    return best_model, best_model_name

# --------------------------
# Step 9: Quantize ML Model
# --------------------------
def quantize_model(best_model, tsvd):
    print(f"=== Quantizing ML components... ===")
    
    # Create the model_components directory, if it doesn't exist
    os.makedirs("./model_components", exist_ok=True)

    # Quantize and save TSVD components
    np.save("./model_components/tsvd_components.npy", tsvd.components_.astype(dtype))

    # Extract and quantize RF model
    rf_data = []
    for tree in best_model.estimators_:
        t = tree.tree_
        rf_data.append({
            "children_left": t.children_left.astype(np.int32).tolist(),
            "children_right": t.children_right.astype(np.int32).tolist(),
            "feature": t.feature.astype(np.int32).tolist(),
            "threshold": t.threshold.astype(dtype).tolist(),
            "value": t.value.squeeze(1).astype(dtype).tolist()
        })

    with open("./model_components/rf_forest.json", "w") as f:
        json.dump(rf_data, f)

    print("\n*** Quantization Complete! ***\n")
    
# --------------------------
# Step 10: Compress ML Pipeline
# --------------------------
def compress_to_tar_gz(output_file, targets):  # Helper function for compressing to tar.gz
    with tarfile.open(output_file, "w:gz") as tar:
        for target in targets:
            tar.add(target, arcname=os.path.basename(target))

def export_pipeline(best_model_name):
    print(f"=== Compressing pipeline for PYNQ Deployment... ===\n")
    targets = ["trusthub_bitstreams", "model_components", "deploy_model.py"]
    output_file = "PYNQ_BLADEI.tar.gz"
    compress_to_tar_gz(output_file, targets)
    print(f"*** Compression complete! Archive saved as '{output_file}'... ***")

# --------------------------
# Main Execution
# --------------------------
def main():
    set_seed(42)
    empty_files, benign_aes_files, benign_rs232_files, mal_aes_files, mal_rs232_files, all_files = collect_bitstreams()
    X = generate_features(all_files)
    y = define_labels(empty_files, benign_aes_files, benign_rs232_files, mal_aes_files, mal_rs232_files)
    X_reduced, tsvd = apply_tsvd(X)
    X_train, X_test, y_train, y_test = split_dataset(X_reduced, y)
    X_train_smote, y_train_smote, best_k = apply_smote(X_train, y_train)
    classifiers, cv_results = compare_classifiers(X_train_smote, y_train_smote)
    best_model, best_model_name = evaluate_best_model(classifiers, cv_results, X_train_smote, y_train_smote, X_test, y_test)  
    quantize_model(best_model, tsvd)
    
    use_armv7 = input("Are you deploying on an ARMv7 board? (y/n): ").strip().lower()
    if use_armv7 == 'y':
        export_pipeline(best_model_name)
    else:
        print("*** Skipping pipeline exportation... ***")

if __name__ == "__main__":
    main()
