import os
import json
import numpy as np

class NumPyCNN:
    def __init__(self, weights, vocab_size, embed_dim=128, num_classes=5):
        self.embedding = weights["embedding"]
        self.conv1_weight = weights["conv1_weight"]
        self.conv1_bias = weights["conv1_bias"]
        self.fc_weight = weights["fc_weight"]
        self.fc_bias = weights["fc_bias"]

        self.embed_dim = embed_dim
        self.num_classes = num_classes

    def forward(self, input_ids):
        x = self.embedding[input_ids]
        x = np.transpose(x, (0, 2, 1))

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
                    
        y = np.maximum(y, 0)

        y = np.max(y, axis=2)

        logits = np.dot(y, self.fc_weight.T) + self.fc_bias
        return logits

class SimpleTokenizer:
    def __init__(self, vocab=None):
        if vocab is None:
            self.vocab = {"[PAD]": 0, "[UNK]": 1}
        else:
            self.vocab = vocab
        self.inv_vocab = {idx: token for token, idx in self.vocab.items()}

    def encode(self, text, max_length=50):
        tokens = text.lower().split()
        ids = [self.vocab.get(tok, self.vocab["[UNK]"]) for tok in tokens]
        # pad / truncate
        if len(ids) < max_length:
            ids.extend([self.vocab["[PAD]"]] * (max_length - len(ids)))
        else:
            ids = ids[:max_length]
        return ids

def features_to_text(feature_vector, ml_prediction):
    top_features_idx = np.argsort(feature_vector)[-5:][::-1]  # top 5 features
    feature_descriptions = [f"byte_{i} high frequency" for i in top_features_idx]
    text = f"Predicted class: {ml_prediction}. " + ", ".join(feature_descriptions)
    return text

def load_nlp_model(weight_path='./model_components/cnn_weights.npz',
                   tokenizer_path='./model_components/cnn_tokenizer.json'):
    with open(tokenizer_path, "r") as f:
        vocab = json.load(f)
    tokenizer = SimpleTokenizer(vocab=vocab)

    weights = np.load(weight_path)
    weights = {k: weights[k] for k in weights.files}

    vocab_size, embed_dim = weights["embedding"].shape
    num_classes = weights["fc_weight"].shape[0]
    model = NumPyCNN(weights, vocab_size, embed_dim, num_classes)

    return model, tokenizer

def nlp_cross_check(model, tokenizer, features, ml_prediction, max_length=50):
    text = features_to_text(features, ml_prediction)
    token_ids = tokenizer.encode(text, max_length=max_length)
    token_ids = np.array(token_ids, dtype=np.int64).reshape(1, -1)

    logits = model.forward(token_ids)
    nlp_pred = int(np.argmax(logits, axis=1)[0])
    return nlp_pred
