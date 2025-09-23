# Copyright (c) 2025, Rye Stahle-Smith; All rights reserved.
# PYNQ BLADEI: Bitstream-Level Abnormality Detection for Embedded Inference
# September 21st, 2025
# Description: This script runs real-time classification of FPGA bitstreams on PYNQ-supported boards using the serialized ML and NLP components.
import os
import random
import platform
import psutil
import time
import warnings
import numpy as np
from collections import Counter
from model_components.cnn_confirmation import load_nlp_model, nlp_cross_check
from model_components.rf_predictor import predict_bitstream

# --------------------------
# Step 0: Suppress Warnings
# --------------------------
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------
# Step 1: Collect Bitstream Files
# --------------------------
def collect_bitstreams(base_path="trusthub_bitstreams"):
    categories = ["Empty", "Benign", "Malicious"]
    bitstream_files = []
    for category in categories:
        folder_path = os.path.join(base_path, category)
        for file in os.listdir(folder_path):
            if file.endswith(".bit"):
                bitstream_files.append(os.path.join(folder_path, file))
    return bitstream_files, categories

# --------------------------
# Step 2: Map Labels
# --------------------------
def get_label_map():
    return {
        0: "Empty (Class 0)",
        1: "Benign AES (Class 1)",
        2: "Benign RS232 (Class 2)",
        3: "Malicious AES (Class 3)",
        4: "Malicious RS232 (Class 4)"
    }

# --------------------------
# Step 3: Determine Actual Class
# --------------------------
def get_actual_class(folder, filename):
    if folder == "Empty":
        return "Empty (Class 0)"
    elif folder == "Benign":
        if filename.startswith("AES"):
            return "Benign AES (Class 1)"
        elif filename.startswith("RS232"):
            return "Benign RS232 (Class 2)"
        else:
            return "Benign (Unknown Class)"
    elif folder == "Malicious":
        if filename.startswith("AES"):
            return "Malicious AES (Class 3)"
        elif filename.startswith("RS232"):
            return "Malicious RS232 (Class 4)"
        else:
            return "Malicious (Unknown Class)"
    else:
        return "Unknown"
    
# --------------------------
# Step 4: Run Prediction Trials
# --------------------------
def run_trials(bitstream_files, label_map, nlp_model=None, tokenizer=None, num_trials=5):
    total_time_ms = 0
    for trial in range(num_trials):
        bitstream_path = random.choice(bitstream_files)
        filename = os.path.basename(bitstream_path)
        folder = os.path.basename(os.path.dirname(bitstream_path))

        print(f"*** Trial {trial + 1}: Processing {filename}... ***")
        actual_class = get_actual_class(folder, filename)

        # Measure Load Time
        start_load = time.time()
        with open(bitstream_path, 'rb') as f:
            data = f.read()
        end_load = time.time()

        # Feature Extraction
        start_feat = time.time()
        size = len(data)
        if size == 0:
            features = np.zeros(256)
        else:
            counts = Counter(data)
            dense_vec = np.zeros(256)
            for byte_val, count in counts.items():
                dense_vec[byte_val] = count / size
            features = dense_vec
        end_feat = time.time()

        # ML Prediction
        start_pred = time.time()
        ml_prediction = predict_bitstream(features)
        end_pred = time.time()

        # NLP Cross-check
        if nlp_model and tokenizer:
            start_conf = time.time()
            nlp_prediction = nlp_cross_check(nlp_model, tokenizer, features, ml_prediction)
            end_conf = time.time()

        print(f"Actual Class:    {actual_class}")
        print(f"ML Predicted:    {label_map.get(ml_prediction, 'Unknown')}")
        
        if nlp_model and tokenizer:
            print(f"NLP Cross-Check: {'Match' if nlp_prediction == ml_prediction else 'Mismatch'}")

        load_time_ms = (end_load - start_load) * 1000
        feat_time_ms = (end_feat - start_feat) * 1000
        pred_time_ms = (end_pred - start_pred) * 1000
        
        if nlp_model and tokenizer:
            conf_time_ms = (end_conf - start_conf) * 1000
            total_time_ms += load_time_ms + feat_time_ms + pred_time_ms + conf_time_ms
        else:
            total_time_ms += load_time_ms + feat_time_ms + pred_time_ms

        print(f"\n=== Latency Summary ===")
        print(f"Load Bitstream:      {load_time_ms:.2f} ms")
        print(f"Feature Extraction:  {feat_time_ms:.2f} ms")
        if nlp_model and tokenizer:
            print(f"Prediction:          {pred_time_ms:.2f} ms")
            print(f"NLP Confirmation:    {conf_time_ms:.2f} ms\n")
        else:
            print(f"Prediction:          {pred_time_ms:.2f} ms\n")

    print(f"Average Latency: {total_time_ms / num_trials / 1000:.2f} s\n")

# --------------------------
# Step 6: Print System Info
# --------------------------
def print_system_info():
    print("=== System Information: ===")
    print("System:", platform.system())
    print("Node Name:", platform.node())
    print("Release:", platform.release())
    print("Version:", platform.version())
    print("Machine:", platform.machine())
    print("Processor:", platform.processor())

    print("\n=== CPU Information: ===")
    print("CPU Cores:", psutil.cpu_count(logical=False))
    print("Logical Processors:", psutil.cpu_count(logical=True))
    print("CPU Usage per Core:", psutil.cpu_percent(percpu=True))
    print("Total RAM:", psutil.virtual_memory().total / 1024**2, "MB")

# --------------------------
# Main Execution
# --------------------------
def main():
    bitstream_files, categories = collect_bitstreams()
    label_map = get_label_map()
    
    deploy_nlp = input("Do you want to deploy the NLP model for cross-checking? (y/n): ").strip().lower()
    if deploy_nlp == 'y':
        nlp_model, tokenizer = load_nlp_model()
        run_trials(bitstream_files, label_map, nlp_model, tokenizer)
    else:
        run_trials(bitstream_files, label_map)
        
    print_system_info()

if __name__ == "__main__":
    main()
