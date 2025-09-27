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
from torch.utils.data import Dataset, DataLoader

# --------------------------
# Step 0: Suppress Warnings
# --------------------------
warnings.filterwarnings("ignore", category=FutureWarning)  # Suppress future warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Suppress user warnings

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
# Step 6: Compare Classifiers
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
# Step 7: Evaluate ML Model
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
# Step 8: Initialize NLP for Prediction Confirmation (CNN-based)
# --------------------------
def features_to_text(feature_vector, model_prediction):
    top_features_idx = np.argsort(feature_vector)[-5:][::-1]  # Access the top 5 features
    feature_descriptions = [f"byte_{i} frequency: {feature_vector[i]:.2%}" for i in top_features_idx]  # Assess each frequency per byte 
    text = f"Predicted Class: {model_prediction}. " + ", ".join(feature_descriptions)  # Finalize the description text
    return text

# Converts raw text into numerical form so it can be used by a model
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

def train_nlp_model(texts, labels, num_classes=5, epochs=20, batch_size=16):
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
        
    print("\n*** NLP Training Complete! ***\n")
    
    return model, tokenizer

# --------------------------
# Step 9: Quantize ML+NLP Model
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
# Step 10: Compress ML+NLP Pipeline
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
