# Copyright (c) 2025, Rye Stahle-Smith; All rights reserved.
# PYNQ BLADEI: Bitstream-Level Abnormality Detection for Embedded Inference
# January 6th, 2026
# Description: This script trains a supervised ML model to detect malicious FPGA bitstreams using byte-level and structural features.
#              Supports dual-head classification: Trojan Detection + Hardware Family Classification

import glob
import tarfile
import os
import sys
import json
import warnings
import random
import numpy as np
from collections import Counter
from scipy.sparse import csr_matrix
from scipy.stats import skew, kurtosis, entropy
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
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
# Step 1: Constants & Mappings
# --------------------------
FAMILY_MAPPING = {  # Map hardware families to filename prefixes
    "CRYPTO": ["AES", "BasicRSA"],
    "COMMS": ["RS232", "EthernetMAC10GE"],
    "MCU/CPU": ["PIC16F84"],
    "BUS/DISPLAY": ["wb_conmax", "vga_lcd"],
    "ITC99": ["b15", "b19"],
    "ISCAS89": ["s15850", "s35932", "s38417", "s38584"],
}

FAMILY_CLASSES = list(FAMILY_MAPPING.keys())  # List of family class names

# --------------------------
# Step 2: Set the Random Seed
# --------------------------
def set_seed(seed=42):
    random.seed(seed)  # Set Python random seed
    np.random.seed(seed)  # Set NumPy random seed

# --------------------------
# Step 3: Collect Bitstreams
# --------------------------
def get_family_from_filename(filename):
    """Determine the hardware family from the filename prefix."""
    basename = os.path.basename(filename)
    for family, prefixes in FAMILY_MAPPING.items():  # For each family,
        for prefix in prefixes:  # Check if filename starts with any prefix
            if basename.startswith(prefix):
                return family
    return "UNKNOWN"

def collect_bitstreams():
    benign_files = glob.glob("trusthub_bitstreams/Benign/*.bit")  # Collect benign bitstreams
    malicious_files = glob.glob("trusthub_bitstreams/Malicious/*.bit")  # Collect malicious bitstreams
    all_files = benign_files + malicious_files  # Combine all files
    
    print(f"=== Organizing bitstreams... ===")
    print(f"Benign Samples: {len(benign_files)}")
    print(f"Malicious Samples: {len(malicious_files)}")
    print(f"Total Samples: {len(all_files)}")
    
    print("\nFamily Distribution:")  # Display family distribution
    family_counts = {fam: 0 for fam in FAMILY_CLASSES}
    for f in all_files:
        family = get_family_from_filename(f)
        if family in family_counts:
            family_counts[family] += 1
    for fam, count in family_counts.items():
        print(f"  {fam}: {count}")
    
    return benign_files, malicious_files, all_files

# --------------------------
# Step 4: Feature Extraction
# --------------------------
def extract_enhanced_features(filepath):
    """Extract comprehensive features from a bitstream file (278 total)."""
    with open(filepath, 'rb') as f:  # Read the bitstream data
        data = f.read()
    
    byte_array = np.frombuffer(data, dtype=np.uint8)
    size = len(byte_array)
    
    if size == 0:  # If the filesize is 0, return a null array
        return np.zeros(278)
    
    # 1) Byte histogram (256 features)
    counts = Counter(byte_array)
    byte_hist = np.zeros(256)
    for byte_val, count in counts.items():  # Normalize frequency of each byte value
        byte_hist[byte_val] = count / size
    
    # 2) Extended statistical features (10 features)
    byte_entropy = entropy(byte_hist + 1e-10)  # Shannon entropy
    byte_mean = np.mean(byte_array)  # Mean byte value
    byte_std = np.std(byte_array)  # Standard deviation
    byte_skew = skew(byte_array) if size > 2 else 0.0  # Skewness
    byte_kurt = kurtosis(byte_array) if size > 3 else 0.0  # Kurtosis
    byte_min = np.min(byte_array)  # Minimum byte value
    byte_max = np.max(byte_array)  # Maximum byte value
    byte_median = np.median(byte_array)  # Median byte value
    zero_ratio = np.sum(byte_array == 0) / size  # Ratio of zero bytes
    ff_ratio = np.sum(byte_array == 255) / size  # Ratio of 0xFF bytes
    
    extended_stats = np.array([
        byte_entropy, byte_mean, byte_std, byte_skew, byte_kurt,
        byte_min, byte_max, byte_median, zero_ratio, ff_ratio
    ])
    
    # 3) Structural features (12 features)
    log_size = np.log1p(size)  # Log-transformed file size
    chunk_size = max(1, size // 4)  # Divide into 4 chunks
    chunks = [byte_array[i*chunk_size:(i+1)*chunk_size] for i in range(4)]
    chunk_means = [np.mean(c) if len(c) > 0 else 0.0 for c in chunks]  # Mean per chunk
    chunk_stds = [np.std(c) if len(c) > 0 else 0.0 for c in chunks]  # Std per chunk
    
    diff = np.diff(byte_array.astype(np.int16))  # Byte-to-byte differences
    transition_rate = np.sum(diff != 0) / max(1, len(diff))  # Rate of non-zero transitions
    avg_transition_mag = np.mean(np.abs(diff)) if len(diff) > 0 else 0.0  # Avg transition magnitude
    
    nibble_high = (byte_array >> 4) & 0x0F  # High nibble
    nibble_low = byte_array & 0x0F  # Low nibble
    nibble_balance = np.mean(nibble_high) - np.mean(nibble_low)  # Nibble balance
    
    structural = np.array([
        log_size, transition_rate, avg_transition_mag, nibble_balance
    ] + chunk_means + chunk_stds)
    
    return np.concatenate([byte_hist, extended_stats, structural])

def display_progress(current, total):
    """Helper function to visualize progress."""
    bar_length = 20
    percent = int((current / total) * 100)
    blocks = int((current / total) * bar_length)
    bar = '█' * blocks + '-' * (bar_length - blocks)
    sys.stdout.write(f'\rProgress: |{bar}| {percent}% ({current}/{total})')
    sys.stdout.flush()

def generate_features(all_files):
    print("\n=== Extracting features (256 histogram + 10 statistical + 12 structural)... ===")
    feature_matrix = []
    for i, f in enumerate(all_files, 1):  # For each file in the dataset,
        feature_matrix.append(extract_enhanced_features(f))  # Extract features
        display_progress(i, len(all_files))  # Update progress
    print()
    return np.array(feature_matrix)  # Return feature matrix (X)

def define_labels(benign_files, malicious_files, all_files):
    print("\n=== Defining labels... ===")
    
    # Trojan labels: 0 = Benign, 1 = Malicious
    y_trojan = [0]*len(benign_files) + [1]*len(malicious_files)
    print(f"Trojan Classes: Benign (0), Malicious (1)")
    
    # Family labels: index into FAMILY_CLASSES
    y_family = []
    for f in all_files:
        family = get_family_from_filename(f)
        if family in FAMILY_CLASSES:
            y_family.append(FAMILY_CLASSES.index(family))
        else:
            y_family.append(-1)
    
    print(f"Family Classes: {FAMILY_CLASSES}")
    
    return np.array(y_trojan), np.array(y_family)  # Return labels (y)

# --------------------------
# Step 5: Apply TSVD (Optional - for dimensionality reduction)
# --------------------------
def apply_tsvd(X, target_variance=0.95):
    """Apply TSVD with automatic component selection to achieve target explained variance."""
    print("\n=== Applying Truncated Singular Value Decomposition (TSVD)... ===")
    
    # First, find optimal number of components
    max_components = min(X.shape[0], X.shape[1]) - 1  # Max possible components
    sparse_X = csr_matrix(X)  # Store X in a sparse matrix
    
    # Try increasing components until we hit target variance
    for n_comp in range(10, max_components, 10):
        tsvd_test = TruncatedSVD(n_components=n_comp, random_state=42)
        tsvd_test.fit(sparse_X)
        explained = tsvd_test.explained_variance_ratio_.sum()
        if explained >= target_variance:
            break
    
    # Use the found number of components
    n_components = n_comp
    tsvd = TruncatedSVD(n_components=n_components, random_state=42)
    X_reduced = tsvd.fit_transform(sparse_X)
    explained_var = tsvd.explained_variance_ratio_.sum() * 100
    
    print(f"Reduced features from {X.shape[1]} to {n_components} components...")
    print(f"Explained Variance: {explained_var:.2f}%")
    return X_reduced, tsvd, n_components

# --------------------------
# Step 6: Apply StandardScaler
# --------------------------
def apply_scaling(X_train, X_test):
    print("\n=== Applying StandardScaler normalization... ===")
    scaler = StandardScaler()  # Initialize StandardScaler
    X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform training data
    X_test_scaled = scaler.transform(X_test)  # Transform test data
    return X_train_scaled, X_test_scaled, scaler

# --------------------------
# Step 7: Train/Test Split
# --------------------------
def split_dataset(X, y_trojan, y_family, test_size=0.20):
    print("\n=== Splitting the dataset for training/testing (80/20)... ===")
    X_train, X_test, y_trojan_train, y_trojan_test, y_family_train, y_family_test = train_test_split(
        X, y_trojan, y_family, test_size=test_size, stratify=y_trojan, random_state=42
    )
    print(f"Train Samples: {len(X_train)}")
    print(f"Test Samples: {len(X_test)}")
    return X_train, X_test, y_trojan_train, y_trojan_test, y_family_train, y_family_test

# --------------------------
# Step 8: Apply SMOTE (Optional - for oversampling)
# --------------------------
def apply_smote(X_train, y_train, k_values=[2, 5, 7, 9, 11]):
    """Apply SMOTE only if class imbalance exceeds threshold."""
    # Check class balance
    unique, counts = np.unique(y_train, return_counts=True)
    class_ratio = min(counts) / max(counts)
    
    print(f"\n=== Comparing k_neighbors values for SMOTE (imbalance ratio: {class_ratio:.2f})... ===\n")
    best_k = None
    best_score = 0

    for k in k_values:  # For each k-value
        try:
            smote = SMOTE(k_neighbors=k, random_state=42)  # Initialize SMOTE
            X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)  # Resample X and y with SMOTE

            model = RandomForestClassifier(n_estimators=100, random_state=42)  # Initialize a RF classifier
            scores = cross_val_score(model, X_train_smote, y_train_smote, cv=5, scoring='f1_macro')  # Cross validate using F1 Macro scoring
            mean_score = scores.mean()  # Find the mean score for F1 Macro
            
            print(f"SMOTE (k={k}): F1 Macro = {mean_score:.4f} ± {scores.std():.4f}")  # Display the k-value and associated mean score

            if mean_score > best_score:  # If the mean score is greater than the current best score
                best_score = mean_score  # Update the best score
                best_k = k  # Store the best k-value

        except ValueError as e:
            print(f"SMOTE (k={k}) failed: {e}")

    print(f"\n=== Applying SMOTE with k_neighbors={best_k}... ===")
    smote = SMOTE(k_neighbors=best_k, random_state=42)  # Initialize SMOTE using the best k-value
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)  # Resample X and y with SMOTE
    print(f"Training samples after SMOTE: {len(X_train_smote)}")
    return X_train_smote, y_train_smote, best_k

# --------------------------
# Step 9: Compare Classifiers
# --------------------------
def compare_classifiers(X_train, y_train, task_name="Trojan Detection"):
    """Compare multiple classifiers using k-Fold Cross-Validation."""
    # Initialize a dictionary for various classifiers
    classifiers = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Naive Bayes": GaussianNB(),
        "SVM (RBF)": SVC(kernel='rbf', random_state=42),
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42)
    }

    # Initialize a dictionary for scoring params
    scoring = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score, average='macro', zero_division=0),
        'recall': make_scorer(recall_score, average='macro', zero_division=0),
        'f1': make_scorer(f1_score, average='macro', zero_division=0)
    }

    print(f"\n=== Comparing classifiers for {task_name} using 5-Fold Cross-Validation... ===")
    cv_results = {}
    k = 5
    
    for name, model in classifiers.items():
        results = cross_validate(model, X_train, y_train, cv=k, scoring=scoring)  # Cross validate
        print(f"\n{name}")  # Print the results
        for metric in scoring:
            mean = results[f'test_{metric}'].mean()
            std = results[f'test_{metric}'].std()
            print(f"  {metric.capitalize()}: {mean:.2f} ± {std:.2f}")
        cv_results[name] = results['test_f1'].mean()  # Store results based on mean F1 Score
    
    best_name = max(cv_results, key=cv_results.get)  # Find best performing classifier
    best_score = cv_results[best_name]
    print(f"\n*** Best Classifier: {best_name} (F1 = {best_score:.4f}) ***")
    
    return classifiers, cv_results, best_name

# --------------------------
# Step 10: Train Trojan Detector with GridSearchCV
# --------------------------
def train_trojan_detector(X_train, y_train, X_test, y_test, best_classifier_name="Random Forest"):
    """Train trojan detector with GridSearchCV for hyperparameter optimization."""
    print(f"\n=== Training Trojan Detector ({best_classifier_name}) with GridSearchCV... ===")
    
    # Define parameter grids for each classifier type
    if best_classifier_name == "Logistic Regression":
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'class_weight': ['balanced', {0: 1, 1: 2}],
            'solver': ['lbfgs']
        }
        base_model = LogisticRegression(max_iter=1000, random_state=42)
    elif best_classifier_name == "Random Forest":
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5],
            'class_weight': ['balanced', {0: 1, 1: 2}]
        }
        base_model = RandomForestClassifier(random_state=42)
    elif best_classifier_name == "Gradient Boosting":
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        }
        base_model = GradientBoostingClassifier(random_state=42)
    elif best_classifier_name == "AdaBoost":
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.5, 1.0]
        }
        base_model = AdaBoostClassifier(random_state=42)
    elif best_classifier_name == "KNN":
        param_grid = {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
        base_model = KNeighborsClassifier()
    else:  # Default to Random Forest
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'class_weight': ['balanced']
        }
        base_model = RandomForestClassifier(random_state=42)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # 5-Fold CV
    scorer = make_scorer(f1_score, pos_label=1)  # Optimize for F1 on malicious class
    
    grid_search = GridSearchCV(base_model, param_grid, cv=cv, scoring=scorer, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)  # Fit the grid search
    
    best_model = grid_search.best_estimator_  # Get the best model
    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Best CV F1: {grid_search.best_score_:.4f}")
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    print(f"\n*** Trojan Detector - Test Set Evaluation ***\n")
    print(classification_report(y_test, y_pred, target_names=["Benign", "Malicious"], zero_division=0))
    
    # Display confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("*** Trojan Detector - Confusion Matrix ***")
    print("\n\t\t  Predicted")
    print("\t\t  Benign\tMalicious")
    print(f"Actual Benign    |\t{cm[0][0]}\t\t{cm[0][1]}")
    print(f"Actual Malicious |\t{cm[1][0]}\t\t{cm[1][1]}")
    
    return best_model

# --------------------------
# Step 11: Train Family Classifier with GridSearchCV
# --------------------------
def train_family_classifier(X_train, y_train, X_test, y_test, best_classifier_name="Random Forest"):
    """Train family classifier with GridSearchCV for hyperparameter optimization."""
    print(f"\n=== Training Family Classifier ({best_classifier_name}) with GridSearchCV... ===")
    
    # Define parameter grids for each classifier type
    if best_classifier_name == "Random Forest":
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'class_weight': ['balanced']
        }
        base_model = RandomForestClassifier(random_state=42)
    elif best_classifier_name == "Gradient Boosting":
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1]
        }
        base_model = GradientBoostingClassifier(random_state=42)
    else:  # Default to Random Forest
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'class_weight': ['balanced']
        }
        base_model = RandomForestClassifier(random_state=42)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # 5-Fold CV
    
    grid_search = GridSearchCV(base_model, param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)  # Fit the grid search
    
    best_model = grid_search.best_estimator_  # Get the best model
    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Best CV F1: {grid_search.best_score_:.4f}")
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    print(f"\n*** Family Classifier - Test Set Evaluation ***\n")
    print(classification_report(y_test, y_pred, target_names=FAMILY_CLASSES, zero_division=0))
    
    # Display confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("*** Family Classifier - Confusion Matrix ***\n")
    print("\t\t" + "\t".join([f[:7] for f in FAMILY_CLASSES]))
    for i, row in enumerate(cm):
        print(f"{FAMILY_CLASSES[i][:7]}\t|\t" + "\t".join(str(val) for val in row))
    
    return best_model

# --------------------------
# Step 12: Quantize ML Models
# --------------------------
def quantize_model(trojan_model, family_model, scaler, tsvd, n_components, dtype=np.float16):
    """Quantize and export trained ML models for edge deployment."""
    print(f"\n=== Quantizing ML components... ===")
    
    # Create the model_components directory if it doesn't exist
    os.makedirs("./model_components", exist_ok=True)

    # Quantize and save StandardScaler parameters
    scaler_data = {
        "mean": scaler.mean_.astype(dtype).tolist(),
        "scale": scaler.scale_.astype(dtype).tolist()
    }
    with open("./model_components/scaler.json", "w") as f:
        json.dump(scaler_data, f)

    # Save TSVD components (only if TSVD was used)
    if tsvd is not None:
        tsvd_data = {
            "components": tsvd.components_.astype(dtype).tolist(),
            "n_components": n_components
        }
        with open("./model_components/tsvd.json", "w") as f:
            json.dump(tsvd_data, f)
    else:
        # Remove tsvd.json if it exists from previous runs
        tsvd_path = "./model_components/tsvd.json"
        if os.path.exists(tsvd_path):
            os.remove(tsvd_path)

    def save_rf_model(model, filename):
        """Helper function to extract and save Random Forest trees."""
        rf_data = []
        for tree in model.estimators_:
            t = tree.tree_
            rf_data.append({
                "children_left": t.children_left.astype(np.int32).tolist(),
                "children_right": t.children_right.astype(np.int32).tolist(),
                "feature": t.feature.astype(np.int32).tolist(),
                "threshold": t.threshold.astype(dtype).tolist(),
                "value": t.value.squeeze(1).astype(dtype).tolist()
            })
        with open(f"./model_components/{filename}", "w") as f:
            json.dump(rf_data, f)
    
    # Save trojan detector model
    if hasattr(trojan_model, 'estimators_'):
        save_rf_model(trojan_model, "rf_trojan.json")
    else:  # For non-tree models use joblib
        import joblib
        joblib.dump(trojan_model, "./model_components/trojan_model.joblib")
    
    # Save family classifier model
    if hasattr(family_model, 'estimators_'):
        save_rf_model(family_model, "rf_family.json")
    else:  # For non-tree models use joblib
        import joblib
        joblib.dump(family_model, "./model_components/family_model.joblib")
    
    # Save metadata
    meta = {
        "trojan_classes": ["Benign", "Malicious"],
        "family_classes": FAMILY_CLASSES,
        "n_features": n_components,
        "original_features": 278  # 256 histogram + 10 statistical + 12 structural
    }
    with open("./model_components/meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("*** Quantization Complete! ***\n")

# --------------------------
# Step 13: Compress ML Pipeline
# --------------------------
def compress_to_tar_gz(output_file, targets):
    """Helper function for compressing files to tar.gz."""
    with tarfile.open(output_file, "w:gz") as tar:
        for target in targets:
            if os.path.exists(target):
                tar.add(target, arcname=os.path.basename(target))

def export_pipeline():
    """Compress the trained pipeline for PYNQ deployment."""
    print(f"=== Compressing pipeline for PYNQ Deployment... ===")
    targets = ["model_components", "deploy_model.py"]
    output_file = "PYNQ_BLADEI.tar.gz"
    compress_to_tar_gz(output_file, targets)
    print(f"*** Compression complete! Archive saved as '{output_file}'... ***")

# --------------------------
# Main Execution
# --------------------------
def main(use_tsvd=False, use_smote=False):
    set_seed(42)  # Set random seed for reproducibility
    
    # Step 3: Collect bitstreams
    benign_files, malicious_files, all_files = collect_bitstreams()
    
    # Step 4: Extract features and define labels
    X = generate_features(all_files)
    y_trojan, y_family = define_labels(benign_files, malicious_files, all_files)
    
    # Step 5: Optionally apply TSVD for dimensionality reduction
    if use_tsvd:
        X_for_split, tsvd, n_components = apply_tsvd(X, target_variance=0.95)
    else:
        print("\n=== Skipping TSVD (disabled)... ===")
        X_for_split = X
        tsvd = None
        n_components = X.shape[1]
    
    # Step 7: Train/Test split
    X_train, X_test, y_trojan_train, y_trojan_test, y_family_train, y_family_test = split_dataset(
        X_for_split, y_trojan, y_family
    )
    
    # Step 6: Apply StandardScaler
    X_train_scaled, X_test_scaled, scaler = apply_scaling(X_train, X_test)
    
    # Step 8: Optionally apply SMOTE for class balancing
    if use_smote:
        X_train_trojan, y_trojan_balanced, _ = apply_smote(X_train_scaled, y_trojan_train)
    else:
        print("\n=== Skipping SMOTE (disabled)... ===")
        X_train_trojan = X_train_scaled
        y_trojan_balanced = y_trojan_train
    
    # Step 9: Compare classifiers for Trojan Detection
    _, _, best_trojan_clf = compare_classifiers(
        X_train_trojan, y_trojan_balanced
    )
    
    # Step 10: Train Trojan Detector with best classifier
    trojan_model = train_trojan_detector(
        X_train_trojan, y_trojan_balanced, X_test_scaled, y_trojan_test, best_trojan_clf
    )

    # Step 11: Compare classifiers for Family Classification
    _, _, best_family_clf = compare_classifiers(
        X_train_scaled, y_family_train, "Family Classification"
    )
    
    # Step 12: Train Family Classifier with best classifier
    family_model = train_family_classifier(
        X_train_scaled, y_family_train, X_test_scaled, y_family_test, best_family_clf
    )
    
    # Step 13: Quantize and export models
    quantize_model(trojan_model, family_model, scaler, tsvd, n_components)
    
    # Step 14: Compress for deployment
    export_pipeline()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train BLADEI pipeline with optional TSVD and SMOTE")
    parser.add_argument("--tsvd", action="store_true", help="Enable TSVD for dimensionality reduction")
    parser.add_argument("--smote", action="store_true", help="Enable SMOTE for oversampling")
    args = parser.parse_args()
    main(use_tsvd=args.tsvd, use_smote=args.smote)