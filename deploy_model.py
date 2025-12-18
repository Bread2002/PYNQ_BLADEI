# Copyright (c) 2025, Rye Stahle-Smith; All rights reserved.
# PYNQ BLADEI: Bitstream-Level Abnormality Detection for Embedded Inference
# December 18th, 2025
# Description: This script simulates a cloud-to-edge bitstream deployment pipeline and utilizes BLADEI to accept, block, or quarantine the generated bitstream prior to deployment.
import os
import sys
import random
import platform
import psutil
import time
import warnings
import shutil
import numpy as np
from collections import Counter
from model_components.rf_predictor import predict_bitstream

# --------------------------
# Step 0: Suppress Warnings
# --------------------------
warnings.filterwarnings("ignore", category=FutureWarning)  # Supress future warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Supress user warnings

# --------------------------
# Step 1: Get Bitstream from Command Line Argument
# --------------------------
def get_bitstream_from_args():
    if len(sys.argv) < 2:
        print("ERROR: No bitstream file provided.")
        print("Usage: python3 deploy_model.py <path_to_bitstream>")
        sys.exit(1)
    
    bitstream_path = sys.argv[1]
    
    if not os.path.isfile(bitstream_path):
        print(f"ERROR: Bitstream file not found: {bitstream_path}")
        sys.exit(1)
    
    return bitstream_path

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
def get_actual_class(filename):
    if filename.startswith("empty"):
        return "Empty (Class 0)"
    elif filename.startswith("AES") and "TjFree" in filename:
        return "Benign AES (Class 1)"
    elif filename.startswith("AES") and "TjIn" in filename:
        return "Malicious AES (Class 3)"
    elif filename.startswith("RS232") and "TjFree" in filename:
        return "Benign RS232 (Class 2)"
    elif filename.startswith("RS232") and "TjIn" in filename:
        return "Malicious RS232 (Class 4)"
    else:
        return "Unknown"

# --------------------------
# Step 4: Ensure Quarantine Folder
# --------------------------
def ensure_quarantine_folder(base_path):
    quarantine_path = os.path.join(base_path, "Quarantine")  # Initialize quarantine path
    if not os.path.isdir(quarantine_path):
        os.makedirs(quarantine_path, exist_ok=True)  # Create quarantine folder if it does not exist
    return quarantine_path

# --------------------------
# Step 2: Map Labels
# --------------------------
def run_trial(bitstream_path, label_map):
    filename = os.path.basename(bitstream_path)
    base_path = os.path.dirname(bitstream_path)
    quarantine_path = ensure_quarantine_folder(base_path)  # Ensure quarantine folder exists

    # Remove extension to present as an HDL name
    hdl_name = os.path.splitext(filename)[0]

    # Process the bitstream
    print("======= BLADEI Vetting: =======")
    print(f"INFO: Processing bitstream: {filename}")
    actual_class = get_actual_class(filename)  # Determine the actual class

    # Measure the load time
    start_load = time.time()
    with open(bitstream_path, 'rb') as f:
        data = f.read()
    end_load = time.time()

    # Measure the time to extract features
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

    # Measure the ML prediction time
    start_pred = time.time()
    ml_prediction, ml_conf = predict_bitstream(features)
    end_pred = time.time()

    # Compute each time in ms
    load_time_ms = (end_load - start_load) * 1000
    feat_time_ms = (end_feat - start_feat) * 1000
    pred_time_ms = (end_pred - start_pred) * 1000

    total_time_ms = (load_time_ms + feat_time_ms + pred_time_ms)  # Track total runtime

    # Display the prediction results
    predicted_label = label_map.get(int(ml_prediction), "Unknown")  # Map prediction to label
    print(f"\nActual Class: {actual_class}")
    print(f"Predicted Class: {predicted_label} [{ml_conf:.2f}% Confidence]")

    # Quarantine malicious bitstream types
    if ("Malicious" in predicted_label) or ("Empty" in predicted_label):
        quarantined_file = os.path.join(quarantine_path, filename)
        try:
            shutil.move(bitstream_path, quarantined_file)  # Move bitstream to quarantine
            print(f"\nACTION: Bitstream quarantined -> {quarantined_file}")
            print("ACTION: Deployment blocked.\n")
        except Exception as e:
            print(f"\nERROR: Failed to quarantine bitstream: {e}\n")
    else:
        print("\nACTION: Bitstream passed vetting. Proceed to deployment.\n")

    # Display the latency summary
    print(f"======= Latency Summary: =======")
    print(f"Load Bitstream:\t\t{load_time_ms:.2f} ms")
    print(f"Feature Extraction:\t{feat_time_ms:.2f} ms")
    print(f"Prediction:\t\t{pred_time_ms:.2f} ms\n")
    print(f"Total Latency: {total_time_ms / 1000:.2f} s\n")

# --------------------------
# Step 7: Print System Info
# --------------------------
def print_system_info():
    print("======= System Information: =======")
    print("System:", platform.system())
    print("Node Name:", platform.node())
    print("Release:", platform.release())
    print("Version:", platform.version())
    print("Machine:", platform.machine())
    print("Processor:", platform.processor())

    print("\n======= CPU Information: =======")
    print("CPU Cores:", psutil.cpu_count(logical=False))
    print("Logical Processors:", psutil.cpu_count(logical=True))
    print("CPU Usage per Core:", psutil.cpu_percent(percpu=True))
    print("Total RAM:", psutil.virtual_memory().total / 1024**2, "MB")

# --------------------------
# Main Execution
# --------------------------
def main():
    bitstream_path = get_bitstream_from_args()
    label_map = get_label_map()

    run_trial(bitstream_path, label_map)

    print_system_info()

if __name__ == "__main__":
    main()
