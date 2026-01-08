# Copyright (c) 2025, Rye Stahle-Smith; All rights reserved.
# PYNQ BLADEI: Bitstream-Level Abnormality Detection for Embedded Inference
# January 6th, 2026
# Description: This script simulates a cloud-to-edge bitstream deployment pipeline.
#              Accepts, blocks, or quarantines the generated bitstream prior to deployment.
#              Utilizes dual-head classification: Trojan Detection + Hardware Family Classification

import os                                                   # For file and directory operations
import sys                                                  # For command line arguments and exit codes
import platform                                             # For system information retrieval
import psutil                                               # For CPU and memory stats
import time                                                 # For timing measurements
import warnings                                             # For suppressing warnings
import shutil                                               # For file quarantine operations
import numpy as np                                          # For numerical operations
from collections import Counter                             # For byte frequency counting
from scipy.stats import skew, kurtosis, entropy             # For statistical feature extraction
from model_components.rf_predictor import predict_bitstream # For ML prediction pipeline

# --------------------------
# Step 0: Suppress Warnings
# --------------------------
warnings.filterwarnings("ignore", category=FutureWarning)   # Ignore future deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning)     # Ignore user-level warnings

# --------------------------
# Step 1: Get Bitstream from Command Line Argument
# --------------------------
def get_bitstream_from_args():
    if len(sys.argv) < 2:                                   # If no command line arguments provided, print error message and exit
        print("ERROR: No bitstream file provided.")
        print("Usage: python3 deploy_model.py <path_to_bitstream>")
        sys.exit(1)

    bitstream_path = sys.argv[1]                            # Get the bitstream path from first argument

    if not os.path.isfile(bitstream_path):                  # If the file doesn't exist, exit with error code
        print(f"ERROR: Bitstream file not found: {bitstream_path}")
        sys.exit(1)

    return bitstream_path                                   # Return the validated path

# --------------------------
# Step 2: Ensure Quarantine Folder
# --------------------------
def ensure_quarantine_folder(base_path):
    quarantine_path = os.path.join(base_path, "Quarantine") # Create quarantine folder path
    if not os.path.isdir(quarantine_path):                  # If it doesn't exist, create it recursively
        os.makedirs(quarantine_path, exist_ok=True)
    return quarantine_path                                  # Return the quarantine folder path

# --------------------------
# Step 3: Feature Extraction (must match train_model.py: 278 features)
# --------------------------
def extract_features(filepath):
    """
    Extract features from a bitstream file.
    Must match train_model.py: 256 histogram + 10 statistical + 12 structural = 278 features
    """
    with open(filepath, 'rb') as f:                         # Open the bitstream file in binary mode
        data = f.read()                                     # Read all bytes into memory
    
    byte_array = np.frombuffer(data, dtype=np.uint8)        # Convert raw bytes to numpy array
    size = len(byte_array)                                  # Get the total byte count
    
    if size == 0:                                           # If file is empty
        return np.zeros(278, dtype=np.float32)              # Return zero feature vector
    
    # 1) Byte histogram (256 features) - normalized frequency
    counts = Counter(byte_array)                            # Count occurrences of each byte value
    byte_hist = np.zeros(256, dtype=np.float32)             # Initialize histogram array
    for byte_val, count in counts.items():                  # For each unique byte value
        byte_hist[byte_val] = count / size                  # Store its normalized frequency
    
    # 2) Extended statistical features (10 features)
    byte_entropy = entropy(byte_hist + 1e-10)               # Shannon entropy of byte distribution
    byte_mean = np.mean(byte_array)                         # Mean byte value
    byte_std = np.std(byte_array)                           # Standard deviation of bytes
    byte_skew = skew(byte_array) if size > 2 else 0.0       # Skewness of byte distribution
    byte_kurt = kurtosis(byte_array) if size > 3 else 0.0   # Kurtosis of byte distribution
    byte_min = float(np.min(byte_array))                    # Minimum byte value
    byte_max = float(np.max(byte_array))                    # Maximum byte value
    byte_median = float(np.median(byte_array))              # Median byte value
    zero_ratio = np.sum(byte_array == 0) / size             # Ratio of zero bytes
    ff_ratio = np.sum(byte_array == 255) / size             # Ratio of 0xFF bytes
    
    extended_stats = np.array([                             # Pack all statistical features
        byte_entropy, byte_mean, byte_std, byte_skew, byte_kurt,
        byte_min, byte_max, byte_median, zero_ratio, ff_ratio
    ], dtype=np.float32)
    
    # 3) Structural features (12 features)
    log_size = np.log1p(size)                               # Log-transformed file size
    chunk_size = max(1, size // 4)                          # Divide file into 4 chunks
    chunks = [byte_array[i*chunk_size:(i+1)*chunk_size] for i in range(4)]
    chunk_means = [np.mean(c) if len(c) > 0 else 0.0 for c in chunks]  # Chunk means (4)
    chunk_stds = [np.std(c) if len(c) > 0 else 0.0 for c in chunks]    # Chunk stds (4)
    
    diff = np.diff(byte_array.astype(np.int16))             # Compute byte-to-byte differences
    transition_rate = np.sum(diff != 0) / max(1, len(diff)) # Rate of non-zero transitions
    avg_transition_mag = np.mean(np.abs(diff)) if len(diff) > 0 else 0.0  # Avg transition magnitude
    
    nibble_high = (byte_array >> 4) & 0x0F                  # Extract high nibbles (4 bits)
    nibble_low = byte_array & 0x0F                          # Extract low nibbles (4 bits)
    nibble_balance = np.mean(nibble_high) - np.mean(nibble_low)  # Compute nibble balance
    
    structural = np.array([                                 # Pack all structural features
        log_size, transition_rate, avg_transition_mag, nibble_balance
    ] + chunk_means + chunk_stds, dtype=np.float32)
    
    return np.concatenate([byte_hist, extended_stats, structural]).astype(np.float32)  # Return 278-dim vector

# --------------------------
# Step 4: Run BLADEI Vetting
# --------------------------
def run_trial(bitstream_path):
    filename = os.path.basename(bitstream_path)             # Extract the filename from full path
    base_path = os.path.dirname(bitstream_path)             # Get the directory containing the file
    quarantine_path = ensure_quarantine_folder(base_path)   # Ensure quarantine folder exists

    print("======= BLADEI Vetting: =======")
    print(f"Processing bitstream: {filename}\n")

    # Load bitstream
    start_load = time.time()                                # Start timing file load
    with open(bitstream_path, 'rb') as f:                   # Open bitstream in binary mode
        data = f.read()                                     # Read all bytes into memory
    end_load = time.time()                                  # Stop timing

    # Extract features
    start_feat = time.time()                                # Start timing feature extraction
    features = extract_features(bitstream_path)             # Extract 278-dimensional feature vector
    end_feat = time.time()                                  # Stop timing

    # Predict (dual-head: trojan + family)
    start_pred = time.time()                                # Start timing ML prediction
    result = predict_bitstream(features)                    # Run dual-head classifier
    end_pred = time.time()                                  # Stop timing

    # Timing
    load_time_ms = (end_load - start_load) * 1000           # Convert load time to milliseconds
    feat_time_ms = (end_feat - start_feat) * 1000           # Convert feature extraction time
    pred_time_ms = (end_pred - start_pred) * 1000           # Convert prediction time
    total_time_s = (load_time_ms + feat_time_ms + pred_time_ms) / 1000  # Total in seconds

    # Display results
    trojan_result = result["trojan"]                        # Get trojan detection results
    family_result = result["family"]                        # Get family classification results
    
    # Display separate predictions for each classifier
    trojan_label = trojan_result["label"]                   # "Benign" or "Malicious"
    trojan_conf = trojan_result["confidence"]               # Trojan confidence percentage
    family_label = family_result["label"]                   # Hardware family name
    family_conf = family_result["confidence"]               # Family confidence percentage
    
    print(f"Trojan Detection: {trojan_label} [{trojan_conf:.2f}% Confidence]")
    print(f"Family Classification: {family_label} [{family_conf:.2f}% Confidence]\n")

    # Quarantine decision
    if trojan_result["is_malicious"]:                       # If classified as malicious, quarantine the bitstream
        quarantined_file = os.path.join(quarantine_path, filename)
        try:
            shutil.move(bitstream_path, quarantined_file)
            print(f"ACTION: Bitstream quarantined -> {quarantined_file}")
            print("ACTION: Deployment blocked.")
        except Exception as e:                              # Handle any file operation errors
            print(f"ERROR: Failed to quarantine bitstream: {e}")
    else:                                                   # If classified as benign, approve deployment
        print("ACTION: Bitstream passed vetting. Proceed to deployment.")

    # Latency summary
    print(f"\n======= Latency Summary: =======")
    print(f"Load Bitstream:\t\t{load_time_ms:.2f} ms")
    print(f"Feature Extraction:\t{feat_time_ms:.2f} ms")
    print(f"Prediction:\t\t{pred_time_ms:.2f} ms")
    print(f"\nTotal Latency: {total_time_s:.2f} s")

    return result                                           # Return full classification results

# --------------------------
# Step 5: Print System Info
# --------------------------
def print_system_info():
    print("\n======= System Information: =======")
    print(f"System: {platform.system()}")                   # Operating system name
    print(f"Node Name: {platform.node()}")                  # Network hostname
    print(f"Release: {platform.release()}")                 # OS release version
    print(f"Version: {platform.version()}")                 # OS version string
    print(f"Machine: {platform.machine()}")                 # Hardware architecture
    print(f"Processor: {platform.processor()}")             # Processor type

    print("\n======= CPU Information: =======")
    print(f"CPU Cores: {psutil.cpu_count(logical=False)}")  # Physical CPU cores
    print(f"Logical Processors: {psutil.cpu_count(logical=True)}")  # Logical CPU threads
    print(f"CPU Usage per Core: {psutil.cpu_percent(percpu=True)}")  # Per-core CPU usage
    print(f"Total RAM: {psutil.virtual_memory().total / 1024**2} MB")  # Total RAM in MB


# --------------------------
# Main Execution
# --------------------------
def main():
    bitstream_path = get_bitstream_from_args()              # Get bitstream path from command line
    
    result = run_trial(bitstream_path)                      # Run BLADEI vetting pipeline
    print_system_info()                                     # Print system information
    
    # Return exit code based on classification
    if result["trojan"]["is_malicious"]:                    # If classified as malicious
        sys.exit(1)                                         # Exit with error code 1
    sys.exit(0)                                             # Otherwise exit success


if __name__ == "__main__":
    main()
