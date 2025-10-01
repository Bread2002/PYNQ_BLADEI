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
import random
import torch
import numpy as np
import torch.nn as nn
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
from torch.utils.data import Dataset, DataLoader

# --------------------------
# Step 0: Suppress Warnings
# --------------------------
warnings.filterwarnings("ignore", category=FutureWarning)  # Suppress future warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Suppress user warnings

# --------------------------
# Step 1: Set the Random Seed
# --------------------------
def set_seed(seed=42):
    random.seed(seed)                # Python random module
    np.random.seed(seed)             # Numpy
    torch.manual_seed(seed)          # PyTorch CPU
    torch.cuda.manual_seed(seed)     # PyTorch GPU
    torch.cuda.manual_seed_all(seed) # If multiple GPUs

    # Deterministic cudnn (may slow down training a bit)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
# Step 9: Initialize NLP for Prediction Confirmation (CNN-based)
# --------------------------
def features_to_text(feature_vector, top_n=5):
    N = len(feature_vector)
    fv = np.array(feature_vector, dtype=float)

    # Compute various global statistics
    entropy = -np.sum([p*np.log2(p) for p in fv if p > 0])
    unique_bytes = np.count_nonzero(fv)
    mean_val = np.mean(fv)
    var_val = np.var(fv)
    skew_val = skew(fv)
    kurt_val = kurtosis(fv)

    # Divide vector into quarters for byte activity
    quarter = N // 4
    low_bytes = fv[:quarter].sum()
    mid_bytes = fv[quarter:2*quarter].sum()
    high_bytes = fv[2*quarter:].sum()

    # Identify primary byte and next-highest frequency
    byte_0_share = fv[0] if N > 0 else 0
    sorted_freqs = np.sort(fv)[::-1]
    next_max = sorted_freqs[1] if len(sorted_freqs) > 1 else 0

    # Compute additional differentiating statistics
    nonzero_indices = np.where(fv > 0)[0]
    max_gap = max(np.diff(nonzero_indices)) if len(nonzero_indices) > 1 else 0
    pct_sparse = np.mean(fv < 0.001)
    energy_center = np.sum(np.arange(N) * fv) / (np.sum(fv) + 1e-9)
    q25, q50, q75 = np.percentile(fv, [25, 50, 75])

    # Compute byte-group ratios and textual dominance descriptions
    low_mid_ratio = (low_bytes / (mid_bytes + 1e-9))
    mid_high_ratio = (mid_bytes / (high_bytes + 1e-9))
    if low_mid_ratio >= 46000:
        low_mid_desc = "strong low-byte dominance"
    else:
        low_mid_desc = "moderate low-byte dominance"

    # Helper function for bucketed entropy levels
    def bucket_entropy(val):
        if val < 0.05: return "flat entropy"
        elif val < 0.2: return "ultra low entropy"
        elif val < 0.5: return "very low entropy"
        elif val < 1.5: return "low entropy"
        elif val < 3.0: return "moderate entropy"
        else: return "high entropy"

    # Helper function for bucketed unique byte counts
    def bucket_unique(n):
        coverage_pct = n / N * 100
        if n < 5: return f"extremely few unique bins ({n}/{N})"
        elif n < 20: return f"very few unique bins ({n}/{N})"
        elif n < 100: return f"limited unique bins ({n}/{N})"
        elif n < N//2: return f"many unique bins ({n}/{N})"
        else: return f"full coverage ({n}/{N})"

    # Helper function for bucketed primary byte dominance
    def bucket_dom(val):
        if val > 0.99: return "byte_0 absolute lock"
        elif val > 0.95: return "byte_0 overwhelming"
        elif val > 0.75: return "byte_0 strong"
        elif val > 0.50: return "byte_0 moderate"
        else: return "byte_0 weak"

    # Helper function for bucketed maximum gap between non-zero bytes
    def bucket_gap(gap):
        if gap == 0: return "dense byte usage"
        elif gap < N//10: return "moderately gapped"
        else: return "highly gapped"

    # Helper function for bucketed sparsity of the byte vector
    def bucket_sparse(pct):
        if pct > 0.95: return "almost fully sparse"
        elif pct > 0.75: return "mostly sparse"
        elif pct > 0.50: return "partially sparse"
        else: return "dense"

    # Helper function for bucketed energy centroid of byte distribution
    def bucket_energy(val):
        if val < N/4: return "low-byte centroid"
        elif val < N/2: return "mid-byte centroid"
        elif val < 3*N/4: return "upper-mid centroid"
        else: return "high-byte centroid"

    # Helper function for bucketed quartiles of byte frequencies
    def bucket_quartiles(q25, q50, q75):
        return f"Q25 {q25:.2%}, Q50 {q50:.2%}, Q75 {q75:.2%}"

    # Build a summary of statistics and descriptive labels
    summary = [
        bucket_entropy(entropy),
        bucket_unique(unique_bytes),
        bucket_dom(byte_0_share),
        bucket_gap(max_gap),
        bucket_sparse(pct_sparse),
        bucket_energy(energy_center),
        f"next max {next_max:.2%}",
        f"mean {mean_val:.2%}, var {var_val:.4f}, skew {skew_val:.2f}, kurt {kurt_val:.2f}",
        bucket_quartiles(q25, q50, q75),
        f"{low_mid_desc}, mid/high ratio {mid_high_ratio:.2f}"
    ]

    # Identify the top 'N' most frequent bytes
    top_features_idx = np.argsort(fv)[-top_n:][::-1]
    top_features = [f"byte_{i} frequent ({fv[i]:.2%})" for i in top_features_idx]

    # Compute pairwise comparisons between top features
    pairwise = []
    for i in range(len(top_features_idx)):
        for j in range(i+1, len(top_features_idx)):
            idx_i = top_features_idx[i]
            idx_j = top_features_idx[j]
            ratio = fv[idx_i] / (fv[idx_j] + 1e-9)
            diff = fv[idx_i] - fv[idx_j]

            # Assign descriptive labels based on ratio thresholds
            if ratio >= 1000:
                dominance = f"byte_{idx_i} >> byte_{idx_j}"
            elif ratio >= 2:
                dominance = f"byte_{idx_i} > byte_{idx_j}"
            else:
                dominance = f"byte_{idx_i} ~ byte_{idx_j}"
            pairwise.append(f"{dominance} (ratio {ratio:.2f}, diff {diff:.2%})")

    # Return combined textual explanation
    return ", ".join(summary) + ". " + " | ".join(top_features + pairwise)

# Converts raw text into numerical form so it can be used by the model
class SimpleTokenizer:
    def __init__(self, vocab=None):
        if vocab is None:  # Initialize with default tokens if no vocab is provided
            self.vocab = {"[PAD]": 0, "[UNK]": 1}
        else:
            self.vocab = vocab

    def build_vocab(self, texts):
        idx = len(self.vocab)  # Start indexing after special tokens
        for text in texts:
            for word in text.lower().split():  # Split into words by whitespace
                if word not in self.vocab:  # Add unseen words to the vocab
                    self.vocab[word] = idx
                    idx += 1

    def encode(self, text, max_length=50):
        tokens = [self.vocab.get(word, self.vocab["[UNK]"]) for word in text.lower().split()]  # Map words to IDs
        tokens = tokens[:max_length]  # Truncate if too long
        padding = [self.vocab["[PAD]"]] * (max_length - len(tokens))  # Add padding if too short
        return tokens + padding

# Prepares text and label pairs so they can be fed into the model
class ReportDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=50):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)  # Return the total number of samples

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = torch.tensor(self.tokenizer.encode(text, self.max_length), dtype=torch.long)  # Convert to tensor
        return {
            'input_ids': encoding,
            'labels': torch.tensor(label, dtype=torch.long)
        }

# A simple CNN that classifies text into categories
class SimpleNLPClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_classes=5):
        super(SimpleNLPClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)  # Word embeddings
        self.conv1 = nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1)  # Convolution over embeddings
        self.relu = nn.ReLU()  # Non-linearity
        self.pool = nn.AdaptiveMaxPool1d(1)  # Global pooling
        self.dropout = nn.Dropout(0.3)  # Regularization
        self.fc = nn.Linear(128, num_classes)  # Final classification layer

    def forward(self, input_ids):
        x = self.embedding(input_ids)  # Map tokens to vectors
        x = x.permute(0, 2, 1)  # Rearrange for 1D convolution
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x).squeeze(-1)  # Pool and flatten
        x = self.dropout(x)
        return self.fc(x)  # Predict class scores

def nlp_cross_check(model, tokenizer, features, max_length=50):
    text = features_to_text(features)
    token_ids = tokenizer.encode(text, max_length=max_length)
    
    token_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
    logits = model(token_ids)
    nlp_pred = int(torch.argmax(logits, dim=1)[0])
    return nlp_pred

def train_nlp_model(texts, labels, num_classes=5, epochs=50, batch_size=16):
    print("\n=== Training CNN-based NLP Secondary System... ===")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = SimpleTokenizer()  # Initialize the tokenizer
    tokenizer.build_vocab(texts)  # Build a vocab for the tokenizer

    dataset = ReportDataset(texts, labels, tokenizer)  # Prepare text and label pairs
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # Create batches and shuffle the data

    model = SimpleNLPClassifier(len(tokenizer.vocab), num_classes=num_classes).to(device)  # Define the model and move it to the device
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # Initialize the optimizer for training
    loss_fn = nn.CrossEntropyLoss()  # Define the loss function for classification

    for epoch in range(epochs):  # For each epoch,
        model.train()  # Set the model to training mode
        total_loss = 0
        for batch in loader:  # For each batch,
            optimizer.zero_grad()  # Reset gradients
            input_ids = batch['input_ids'].to(device)  # Move inputs to the device
            labels_batch = batch['labels'].to(device)  # Move labels to the device
            outputs = model(input_ids)  # Forward pass
            loss = loss_fn(outputs, labels_batch)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update the model weights
            total_loss += loss.item()  # Accumulate loss for reporting
        avg_loss = total_loss / len(loader)  # Calculate the average loss per batch
        print(f"Epoch {epoch+1}/{epochs} completed. Loss: {avg_loss:.4f}")  # Display progress
    
    return model, tokenizer

def evaluate_nlp_model(model, tokenizer, X_test, y_test, num_classes=5):
    nlp_predictions = []
    for i in range(len(X_test)):
        nlp_pred = nlp_cross_check(model, tokenizer, X_test[i])
        nlp_predictions.append(nlp_pred)

    print("\n*** NLP System Evaluation on Hold-out Test Set ***\n")
    print(classification_report(y_test, nlp_predictions, zero_division=0))

    cm = confusion_matrix(y_test, nlp_predictions)
    print("\n*** Confusion Matrix on Hold-out Test Set using NLP ***")
    print("\n\t\tPredicted")
    print("\t\t" + "\t".join(str(i) for i in range(num_classes)))
    for i, row in enumerate(cm):
        print(f"Actual {i} |\t" + "\t".join(str(val) for val in row))

    print("\n*** NLP Training Complete! ***\n")

    return

# --------------------------
# Step 10: Quantize ML+NLP Model
# --------------------------
def quantize_model(best_model, tsvd, nlp_model=None, tokenizer=None, dtype=np.float16):
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

    # Quantize NLP model
    if nlp_model and tokenizer:
        print("=== Saving quantized NLP components... ===")
        weights = {
            "embedding": nlp_model.embedding.weight.detach().cpu().numpy().astype(dtype),
            "conv1_weight": nlp_model.conv1.weight.detach().cpu().numpy().astype(dtype),
            "conv1_bias": nlp_model.conv1.bias.detach().cpu().numpy().astype(dtype),
            "fc_weight": nlp_model.fc.weight.detach().cpu().numpy().astype(dtype),
            "fc_bias": nlp_model.fc.bias.detach().cpu().numpy().astype(dtype),
        }
        np.savez("./model_components/cnn_weights.npz", **weights)

        with open("./model_components/cnn_tokenizer.json", "w") as f:
            json.dump(tokenizer.vocab, f)

    print("\n*** Quantization Complete! ***\n")
    
# --------------------------
# Step 11: Compress ML+NLP Pipeline
# --------------------------
def compress_to_tar_gz(output_file, targets):  # Helper function for compressing to tar.gz
    with tarfile.open(output_file, "w:gz") as tar:
        for target in targets:
            tar.add(target, arcname=os.path.basename(target))

def export_pipeline(best_model_name):
    print(f"=== Compressing pipeline for PYNQ Deployment... ===\n")
    targets = ["trusthub_bitstreams", "model_components", "VirtualEnv", "deploy_model.py"]
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
        
    train_nlp = input("Do you want to train the NLP model for cross-checking? (y/n): ").strip().lower()
    if train_nlp == 'y':
        texts = [features_to_text(x) for x in X_train_smote]
        labels = list(y_train_smote)
        nlp_model, tokenizer = train_nlp_model(texts, labels)
        evaluate_nlp_model(nlp_model, tokenizer, X_test, y_test)
        quantize_model(best_model, tsvd, nlp_model, tokenizer)
    else:
        print("*** Skipping NLP model training... ***")
        quantize_model(best_model, tsvd)
    
    use_armv7 = input("Are you deploying on an ARMv7 board? (y/n): ").strip().lower()
    if use_armv7 == 'y':
        export_pipeline(best_model_name)
    else:
        print("*** Skipping pipeline exportation... ***")

if __name__ == "__main__":
    main()
