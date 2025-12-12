# Copyright (c) 2025, Rye Stahle-Smith; All rights reserved.
# PYNQ BLADEI: Bitstream-Level Abnormality Detection for Embedded Inference
# November 17th, 2025
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
# Step 1: Collect Bitstream Files
# --------------------------
def collect_bitstreams(base_path="trusthub_bitstreams"):
    categories = ["Empty", "Benign", "Malicious"]  # Initialize categories
    bitstream_files = {category: [] for category in categories}  # Initialize category dictionary
    for category in categories:  # For each category, initialize the associated file path
        folder_path = os.path.join(base_path, category)
        if not os.path.isdir(folder_path):
            continue
        for file in os.listdir(folder_path):
            if file.endswith(".bit"):
                bitstream_files[category].append(os.path.join(folder_path, file))
    return bitstream_files

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
# Step 4: Ensure Quarantine Folder
# --------------------------
def ensure_quarantine_folder(base_path="trusthub_bitstreams"):
    quarantine_path = os.path.join(base_path, "Quarantine")  # Initialize quarantine path
    if not os.path.isdir(quarantine_path):
        os.makedirs(quarantine_path, exist_ok=True)  # Create quarantine folder if it does not exist
    return quarantine_path

# --------------------------
# Step 5: Pick a Random Bitstream
# --------------------------
def pick_random_bitstream(bitstream_files):
    all_files = []
    for files in bitstream_files.values():
        all_files.extend(files)

    selected_path = random.choice(all_files)  # Pick a random file
    return selected_path

# --------------------------
# Step 6: Resolve Benign Counterpart (For Malicious Demo)
# --------------------------
def resolve_benign_counterpart(bitstream_files, malicious_path):
    filename = os.path.basename(malicious_path)
    if filename.startswith("AES"):
        benign_pool = [f for f in bitstream_files.get("Benign", []) if os.path.basename(f).startswith("AES")]
    elif filename.startswith("RS232"):
        benign_pool = [f for f in bitstream_files.get("Benign", []) if os.path.basename(f).startswith("RS232")]
    else:
        benign_pool = bitstream_files.get("Benign", [])

    if benign_pool:
        return random.choice(benign_pool)
    return None

# --------------------------
# Step 7: Mock Cloud-to-Edge Pipeline
# --------------------------
def display_progress(current, total):  # Helper function to visualize progress
    bar_length = 20
    percent = int((current / total) * 100)
    blocks = int((current / total) * bar_length)
    bar = '█' * blocks + '-' * (bar_length - blocks)
    sys.stdout.write(f'\rProgress: |{bar}| {percent}% ({current}/{total})')
    sys.stdout.flush()

def mock_cloud_pipeline(display_hdl_name, announce_injection=False):
    print("\n======= Cloud Submission Pipeline (Simulated): =======")
    print(f"INFO: Processing user submission -> {display_hdl_name}.v")
    time.sleep(0.15)

    print("INFO: Establishing secure session with cloud FPGA service...")
    time.sleep(0.75)

    print(f"INFO: HDL source identified: {display_hdl_name}.v\n")
    time.sleep(0.40)

    print("INFO: Uploading HDL package to remote workspace...")
    total = 30
    for i in range(total + 1):
        display_progress(i, total)
        time.sleep(0.06)
    print("\nINFO: Upload complete.\n")

    print("INFO: Job queued for synthesis/implementation...")
    total = 12
    for i in range(total + 1):
        display_progress(i, total)
        time.sleep(0.10)
    print("\nINFO: Build server allocated.\n")

    print("INFO: Vivado Batch: synthesization (synth_design)...")
    total = 35
    for i in range(total + 1):
        display_progress(i, total)
        time.sleep(0.06)
    print("\nINFO: Synthesis complete.\n")

    print("INFO: Vivado Batch: implementation (opt/place/route_design)...")
    total = 45
    for i in range(total + 1):
        display_progress(i, total)
        time.sleep(0.055)
    print("\nINFO: Implementation complete.\n")

    print("INFO: Vivado Batch: bitstream generation (write_bitstream)...")
    total = 18
    for i in range(total + 1):
        display_progress(i, total)
        time.sleep(0.09)

    # If malicious was selected, we act like a benign build was used until bitgen completes
    if announce_injection:
        print("\nALERT: Bitstream intercepted and malicious payload injected by an unknown user.")
        print("INFO: Bitstream generated.\n")
        time.sleep(0.75)
    else:
        print("\nINFO: Bitstream generated.\n")

    print("INFO: Delivering artifact to device for on-device analysis...")
    total = 22
    for i in range(total + 1):
        display_progress(i, total)
        time.sleep(0.06)
    print("\nINFO: Artifact delivered. Proceeding to vetting via BLADEI...\n")

# --------------------------
# Step 9: Run Prediction Trial
# --------------------------
def run_trial(bitstream_files, label_map, base_path="trusthub_bitstreams"):
    quarantine_path = ensure_quarantine_folder(base_path)  # Ensure quarantine folder exists

    bitstream_path = pick_random_bitstream(bitstream_files)  # Pick a random file
    filename = os.path.basename(bitstream_path)
    folder = os.path.basename(os.path.dirname(bitstream_path))

    # Remove extension to present as an HDL name
    hdl_name = os.path.splitext(filename)[0]

    # If malicious is selected, display a benign counterpart name early (but keep malicious selected in background)
    announce_injection = False
    display_hdl_name = hdl_name
    if folder == "Malicious":
        benign_counterpart = resolve_benign_counterpart(bitstream_files, bitstream_path)
        if benign_counterpart is not None:
            display_hdl_name = os.path.splitext(os.path.basename(benign_counterpart))[0]
        announce_injection = True  # Injection message appears during bitstream generation

    input("Press 'ENTER' to the begin simulation... ")

    # Mock cloud submission + Vivado flow
    mock_cloud_pipeline(display_hdl_name, announce_injection=announce_injection)

    # After injection notification, we process the originally selected bitstream (malicious stays malicious)
    print("======= BLADEI Vetting: =======")
    print(f"INFO: Processing bitstream...")
    actual_class = get_actual_class(folder, filename)  # Determine the actual class

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
# Step 10: Print System Info
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
    bitstream_files = collect_bitstreams()
    label_map = get_label_map()

    run_trial(bitstream_files, label_map)

    print_system_info()

if __name__ == "__main__":
    main()
