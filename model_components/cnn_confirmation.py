import os
import json
import numpy as np
from scipy.stats import skew, kurtosis

class NumPyCNN:
    # Loads pretrained weights for embedding, convolution, and fully connected layers
    def __init__(self, weights, vocab_size, embed_dim=128, num_classes=5):
        self.embedding = weights["embedding"]
        self.conv1_weight = weights["conv1_weight"]
        self.conv1_bias = weights["conv1_bias"]
        self.fc_weight = weights["fc_weight"]
        self.fc_bias = weights["fc_bias"]

        self.embed_dim = embed_dim
        self.num_classes = num_classes

    def forward(self, input_ids):
        # Lookup embeddings for input tokens
        x = self.embedding[input_ids]
        x = np.transpose(x, (0, 2, 1))  # Rearrange to (batch, channels, seq)

        # Perform 1D convolution manually
        batch, in_ch, seq = x.shape
        out_ch, _, k = self.conv1_weight.shape
        y = np.zeros((batch, out_ch, seq))
        pad = k // 2
        for b in range(batch):
            for oc in range(out_ch):
                for i in range(seq):
                    start = max(0, i - pad)
                    end = min(seq, i + pad + 1)
                    kernel_start = pad - (i - start)
                    kernel_end = kernel_start + (end - start)
                    y[b, oc, i] = (
                        np.sum(
                            x[b, :, start:end] *
                            self.conv1_weight[oc, :, kernel_start:kernel_end]
                        )
                        + self.conv1_bias[oc]
                    )
                    
        y = np.maximum(y, 0)  # ReLU activation
        y = np.max(y, axis=2)  # Global max pooling

        # Linear layer for classification
        logits = np.dot(y, self.fc_weight.T) + self.fc_bias
        return logits


class SimpleTokenizer:
    def __init__(self, vocab=None):
        if vocab is None:  # Initialize with default tokens if no vocab is provided
            self.vocab = {"[PAD]": 0, "[UNK]": 1}
        else:
            self.vocab = vocab
        self.inv_vocab = {idx: token for token, idx in self.vocab.items()}

    def encode(self, text, max_length=50):
        tokens = text.lower().split()
        ids = [self.vocab.get(tok, self.vocab["[UNK]"]) for tok in tokens]  # Map words to IDs
        # Pad or truncate to fixed length
        if len(ids) < max_length:
            ids.extend([self.vocab["[PAD]"]] * (max_length - len(ids)))
        else:
            ids = ids[:max_length]
        return ids

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

def load_nlp_model(weight_path='./model_components/cnn_weights.npz',
                   tokenizer_path='./model_components/cnn_tokenizer.json'):
    # Load tokenizer vocabulary
    with open(tokenizer_path, "r") as f:
        vocab = json.load(f)
    tokenizer = SimpleTokenizer(vocab=vocab)

    # Load CNN weights from disk
    weights = np.load(weight_path)
    weights = {k: weights[k] for k in weights.files}

    # Rebuild model using loaded parameters
    vocab_size, embed_dim = weights["embedding"].shape
    num_classes = weights["fc_weight"].shape[0]
    model = NumPyCNN(weights, vocab_size, embed_dim, num_classes)

    return model, tokenizer


def nlp_cross_check(model, tokenizer, features, max_length=50):
    # Turn feature vector into descriptive text
    text = features_to_text(features)
    token_ids = tokenizer.encode(text, max_length=max_length)
    token_ids = np.array(token_ids, dtype=np.int64).reshape(1, -1)

    # Run forward pass through the NumPy CNN
    logits = model.forward(token_ids)
    
    # Compute probabilities via softmax
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    nlp_pred = int(np.argmax(probs, axis=1)[0])
    nlp_confidence = float(np.max(probs))  # Assess the NLP confidence
    
    return nlp_pred, nlp_confidence, text