import numpy as np
import glob
import tarfile
import os
import sys
import json
from joblib import dump, load
from collections import Counter
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
# Step 1: Collect Bitstreams
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
# Step 2: Feature Extraction
# --------------------------
def extract_sparse_features(filepath):
    with open(filepath, 'rb') as f:
        data = f.read()
    size = len(data)
    if size == 0:
        return np.zeros(256)
    counts = Counter(data)
    dense_vec = np.zeros(256)
    for byte_val, count in counts.items():
        dense_vec[byte_val] = count / size
    return dense_vec

def display_progress(current, total):
    bar_length = 20
    percent = int((current / total) * 100)
    blocks = int((current / total) * bar_length)
    bar = '█' * blocks + '-' * (bar_length - blocks)
    sys.stdout.write(f'\rProgress: |{bar}| {percent}% ({current}/{total})')
    sys.stdout.flush()

def generate_features(all_files):
    print("=== Extracting sparse features... ===")
    feature_matrix = []
    for i, f in enumerate(all_files, 1):
        feature_matrix.append(extract_sparse_features(f))
        display_progress(i, len(all_files))
    print()
    return np.array(feature_matrix)

def define_labels(empty_files, benign_aes_files, benign_rs232_files, mal_aes_files, mal_rs232_files):
    print("=== Defining labels... ===")
    y = [0]*len(empty_files) + [1]*len(benign_aes_files) + [2]*len(benign_rs232_files) + \
        [3]*len(mal_aes_files) + [4]*len(mal_rs232_files)
    return y

# --------------------------
# Step 3: TSVD
# --------------------------
def apply_tsvd(X, n_components=30):
    print("=== Applying Truncated Singular Value Decomposition (TSVD)... ===")
    sparse_X = csr_matrix(X)
    tsvd = TruncatedSVD(n_components=n_components, random_state=42)
    X_reduced = tsvd.fit_transform(sparse_X)
    return X_reduced, tsvd

# --------------------------
# Step 4: Train/Test Split
# --------------------------
def split_dataset(X_reduced, y, test_size=0.25):
    print("=== Splitting the dataset for training/testing... ===")
    return train_test_split(X_reduced, y, test_size=test_size, stratify=y, random_state=42)

# --------------------------
# Step 5: Apply SMOTE
# --------------------------
def apply_smote(X_train, y_train, k_values=[2, 5, 7, 9, 11]):
    print("=== Comparing k_neighbors values for SMOTE... ===\n")
    smote_results = {}
    best_k = None
    best_score = 0

    for k in k_values:
        try:
            smote = SMOTE(k_neighbors=k, random_state=42)
            X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

            model = RandomForestClassifier(n_estimators=100, random_state=42)
            scores = cross_val_score(model, X_train_smote, y_train_smote, cv=5, scoring='f1_macro')
            mean_score = scores.mean()
            
            smote_results[k] = mean_score
            print(f"SMOTE (k={k}): F1 Macro = {mean_score:.4f} ± {scores.std():.4f}")

            if mean_score > best_score:
                best_score = mean_score
                best_k = k

        except ValueError as e:
            print(f"SMOTE (k={k}) failed: {e}")

    print(f"\n=== Applying SMOTE with k_neighbors={best_k}... ===")
    smote = SMOTE(k_neighbors=best_k, random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    return X_train_smote, y_train_smote, best_k

# --------------------------
# Step 6: Compare Classifiers
# --------------------------
def compare_classifiers(X_train_smote, y_train_smote):
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
        results = cross_validate(model, X_train_smote, y_train_smote, cv=k, scoring=scoring)
        print(f"\n{name}")
        for metric in scoring:
            mean = results[f'test_{metric}'].mean()
            std = results[f'test_{metric}'].std()
            print(f"  {metric.capitalize()}: {mean:.2f} ± {std:.2f}")
        cv_results[name] = results['test_f1'].mean()
    return classifiers, cv_results

# --------------------------
# Step 7: Evaluate Model
# --------------------------
def evaluate_best_model(classifiers, cv_results, X_train_smote, y_train_smote, X_test, y_test):
    best_model_name = max(cv_results, key=cv_results.get)
    best_model = classifiers[best_model_name]
    best_model.fit(X_train_smote, y_train_smote)

    y_pred = best_model.predict(X_test)
    print(f"\n*** Final Evaluation on Hold-out Test Set using {best_model_name} ***\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    print(f"\n*** Confusion Matrix on Hold-out Test Set using {best_model_name} ***")
    cm = confusion_matrix(y_test, y_pred)
    print("\n\t\tPredicted\n\t\t0\t1\t2\t3\t4")
    for i, row in enumerate(cm):
        print(f"Actual {i} |\t" + "\t".join(str(val) for val in row))
    return best_model, best_model_name

# --------------------------
# Step 8: Serialize Model
# --------------------------
def serialize_model(best_model, tsvd):
    print(f"=== Serializing {type(best_model).__name__}... ===")
    os.makedirs("./model_components", exist_ok=True)
    dump(best_model, './model_components/random_forest_model.joblib')
    dump(tsvd, './model_components/tsvd.joblib')

    tsvd = load("./model_components/tsvd.joblib")
    np.save("./model_components/tsvd_components.npy", tsvd.components_)

    rf = load("./model_components/random_forest_model.joblib")

    def extract_tree(tree):
        return {
            "children_left": tree.children_left.tolist(),
            "children_right": tree.children_right.tolist(),
            "feature": tree.feature.tolist(),
            "threshold": tree.threshold.tolist(),
            "value": tree.value.squeeze(1).tolist()
        }

    forest_json = [extract_tree(estimator.tree_) for estimator in rf.estimators_]

    with open("./model_components/rf_forest.json", "w") as f:
        json.dump(forest_json, f)

    print("*** Serialization complete! ***")

# --------------------------
# Step 9: Compress Pipeline
# --------------------------
def compress_to_tar_gz(output_file, targets):
    with tarfile.open(output_file, "w:gz") as tar:
        for target in targets:
            tar.add(target, arcname=os.path.basename(target))

def export_pipeline(best_model_name):
    print(f"=== Compressing {best_model_name} for PYNQ Deployment... ===\n")
    targets = ["trusthub_bitstreams", "model_components", "VirtualEnv", "deploy_model.ipynb", "requirements.txt"]
    output_file = "PYNQ_BLADEI.tar.gz"
    compress_to_tar_gz(output_file, targets)
    print(f"\n*** Compression complete! Archive saved as '{output_file}'... ***")

# --------------------------
# Main Execution
# --------------------------
def main():
    empty_files, benign_aes_files, benign_rs232_files, mal_aes_files, mal_rs232_files, all_files = collect_bitstreams()
    X = generate_features(all_files)
    y = define_labels(empty_files, benign_aes_files, benign_rs232_files, mal_aes_files, mal_rs232_files)
    X_reduced, tsvd = apply_tsvd(X)
    X_train, X_test, y_train, y_test = split_dataset(X_reduced, y)
    X_train_smote, y_train_smote, best_k = apply_smote(X_train, y_train)
    classifiers, cv_results = compare_classifiers(X_train_smote, y_train_smote)
    best_model, best_model_name = evaluate_best_model(classifiers, cv_results, X_train_smote, y_train_smote, X_test, y_test)
    serialize_model(best_model, tsvd)
    use_armv7 = input("Are you deploying on an ARMv7 board? (y/n): ").strip().lower()
    if use_armv7 == 'y':
        export_pipeline(best_model_name)
    else:
        print("*** Skipping model compression... ***")

if __name__ == "__main__":
    main()
