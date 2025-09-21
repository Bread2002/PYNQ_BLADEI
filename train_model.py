# Copyright (c) 2025, Rye Stahle-Smith; All rights reserved.
# PYNQ BLADEI: Bitstream-Level Abnormality Detection for Embedded Inference
# September 21st, 2025
# Description: This script trains a supervised ML model to detect malicious FPGA bitstreams using byte-level and structural features, and optionally trains a CNN-based NLP model to cross-check and confirm the ML predictions.
import glob
import tarfile
import os
import sys
import json
import warnings
import torch
import numpy as np
import torch.nn as nn
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
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import logging, BertTokenizer, BertModel

# --------------------------
# Step 0: Suppress Warnings
# --------------------------
logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

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
# Step 3: Apply TSVD
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
# Step 7: Evaluate ML Model
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
        
    print("\n*** ML Training Complete! ***\n")
    
    return best_model, best_model_name

# --------------------------
# Step 8: Initialize NLP for Prediction Confirmation (CNN-based)
# --------------------------
def features_to_text(feature_vector, model_prediction):
    top_features_idx = np.argsort(feature_vector)[-5:][::-1]  # top 5 features
    feature_descriptions = [f"byte_{i} high frequency" for i in top_features_idx]
    text = f"Predicted class: {model_prediction}. " + ", ".join(feature_descriptions)
    return text

class SimpleTokenizer:
    def __init__(self, vocab=None):
        if vocab is None:
            self.vocab = {"[PAD]": 0, "[UNK]": 1}
        else:
            self.vocab = vocab

    def build_vocab(self, texts):
        idx = len(self.vocab)
        for text in texts:
            for word in text.lower().split():
                if word not in self.vocab:
                    self.vocab[word] = idx
                    idx += 1

    def encode(self, text, max_length=50):
        tokens = [self.vocab.get(word, self.vocab["[UNK]"]) for word in text.lower().split()]
        tokens = tokens[:max_length]
        padding = [self.vocab["[PAD]"]] * (max_length - len(tokens))
        return tokens + padding

class ReportDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=50):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = torch.tensor(self.tokenizer.encode(text, self.max_length), dtype=torch.long)
        return {
            'input_ids': encoding,
            'labels': torch.tensor(label, dtype=torch.long)
        }

class SimpleNLPClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_classes=5):
        super(SimpleNLPClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)

def train_nlp_model(texts, labels, num_classes=5, epochs=20, batch_size=16):
    print("\n=== Training CNN-based NLP Secondary System... ===")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Tokenizer
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(texts)

    dataset = ReportDataset(texts, labels, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimpleNLPClassifier(len(tokenizer.vocab), num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            labels_batch = batch['labels'].to(device)
            outputs = model(input_ids)
            loss = loss_fn(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs} completed. Loss: {avg_loss:.4f}")
        
    print("\n*** NLP Training Complete! ***\n")
    
    return model, tokenizer

# --------------------------
# Step 9: Serialize ML+NLP Model
# --------------------------
def serialize_model(best_model, tsvd, nlp_model=None, tokenizer=None):
    print(f"=== Serializing ML components... ===")
    os.makedirs("./model_components", exist_ok=True)

    # Save ML models
    dump(best_model, './model_components/random_forest_model.joblib')
    dump(tsvd, './model_components/tsvd.joblib')

    # Save TSVD components
    tsvd = load("./model_components/tsvd.joblib")
    np.save("./model_components/tsvd_components.npy", tsvd.components_)

    # Save RF structure
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

    # Save NLP components
    if nlp_model and tokenizer:
        print("=== Serializing NLP components... ===")

        weights = {
            "embedding": nlp_model.embedding.weight.detach().cpu().tolist(),
            "conv1_weight": nlp_model.conv1.weight.detach().cpu().tolist(),
            "conv1_bias": nlp_model.conv1.bias.detach().cpu().tolist(),
            "fc_weight": nlp_model.fc.weight.detach().cpu().tolist(),
            "fc_bias": nlp_model.fc.bias.detach().cpu().tolist(),
        }

        np.savez("./model_components/cnn_weights.npz", **{k: np.array(v) for k, v in weights.items()})

        with open("./model_components/cnn_tokenizer.json", "w") as f:
            json.dump(tokenizer.vocab, f)

    print("\n*** Serialization Complete! ***\n")
    
# --------------------------
# Step 10: Compress ML+NLP Pipeline
# --------------------------
def compress_to_tar_gz(output_file, targets):
    with tarfile.open(output_file, "w:gz") as tar:
        for target in targets:
            tar.add(target, arcname=os.path.basename(target))

def export_pipeline(best_model_name):
    print(f"=== Compressing ML and NLP components for PYNQ Deployment... ===\n")
    targets = ["trusthub_bitstreams", "model_components", "VirtualEnv", "deploy_model.py", "requirements.txt"]
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
        
    train_nlp = input("Do you want to train the NLP model for cross-checking? (y/n): ").strip().lower()
    if train_nlp == 'y':
        texts = [features_to_text(x, pred) for x, pred in zip(X_train_smote, y_train_smote)]
        labels = list(y_train_smote)
        nlp_model, tokenizer = train_nlp_model(texts, labels)
        serialize_model(best_model, tsvd, nlp_model, tokenizer)
    else:
        print("*** Skipping NLP model training... ***")
        serialize_model(best_model, tsvd)
    
    use_armv7 = input("Are you deploying on an ARMv7 board? (y/n): ").strip().lower()
    if use_armv7 == 'y':
        export_pipeline(best_model_name)
    else:
        print("*** Skipping pipeline exportation... ***")

if __name__ == "__main__":
    main()
