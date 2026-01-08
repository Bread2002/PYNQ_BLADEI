# **BLADEI: Bitstream-Level Abnormality Detection for Embedded Inference**
### *Copyright (c) 2025, Rye Stahle-Smith* 

---

## 📌 Project Overview

This repository contains an embedded deployment pipeline for detecting **malicious FPGA bitstreams** using a trained machine learning (ML) model. Bitstreams are configuration files that can be weaponized to introduce hardware Trojans, posing serious risks in shared or cloud-hosted reconfigurable systems. This project leverages a lightweight, byte-level classification approach and enables **on-device malware detection** for **PYNQ-supported FPGA boards**, without requiring reverse engineering techniques or access to original source code or netlists. The pipeline features **dual-head classification** for both Trojan detection and hardware family identification across six categories. Benchmark designs from Trust-Hub (AES, RS232, ITC'99, ISCAS'89, etc.) were synthesized, implemented, and used for training and validation.

---

## ⚙️ Features

- 🔍 **Byte-frequency analysis** of binary `.bit` files
- 🧩 Lightweight **byte-level + statistical** feature extraction
- 📊 Real-time inference using a supervised **Random Forest** with a custom, dependency-light predictor
- ⚡ Deployment-ready for **ARMv7 (e.g., PYNQ-Z1/Z2, Zynq-7000 SoC)** and **ARMv8 (e.g., Zynq UltraScale+ MPSoC, RFSoC, Kria) boards**
- 🧪 Verified with state-of-the-art (SOTA) bitstreams derived from **Trust-Hub** benchmarks

---

## 📂 Repository Structure
pynq-maldetect/<br>
├── trusthub_bitstreams.zip ***# Sample `.bit` files (Benign or Malicious)***<br>
├── model_components/ ***# Output directory for trained models***<br>
├── LICENSE.md<br>
├── PYNQ_BLADEI.tar.gz ***# Pre-trained models***
├── README.md<br>
├── deploy_model.py ***# Model deployment for on-device inference***<br>
├── requirements.txt ***# Python dependencies for training***<br>
└── train_model.py ***# Model training and export for PYNQ***<br>

> ⚠️ **Notice:**
> Due to file size constraints, the sample dataset (`trusthub_bitstreams/`) is hosted separately on the [Releases](https://github.com/Bread2002/PYNQ_BLADEI/releases/tag/v3.0.0) page. The file is password-protected, however, access is available upon request: ryes@email.sc.edu

---

## 🚀 Setup Instructions

This project is divided into two parts:

- 🧠 **Model Training and Export**
- ⚙️ **On-Device Inference**

---

### 🧠 `train_model.py` — Model Training and Export

> **Requirements:**
> - Python 3.8+
> - Python Packages: `scikit-learn`, `numpy`, `scipy`, `imblearn`

> ⚠️ **Note:**
> Training should be performed on a general-purpose machine (laptop, workstation, or server) for **both ARMv7 and ARMv8** targets. While some ARMv8 boards *may* be capable of training, it is not the recommended workflow. Training is heavier, package availability can be inconsistent, and it’s typically slower and less reproducible than running on a PC.  

1. Clone the Repository:
   ```bash
   git clone https://github.com/Bread2002/PYNQ_BLADEI.git
   cd PYNQ_BLADEI
   ```
   
2. Install Dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Training Script:
   ```bash
   python train_model.py
   ```
   
4. Optional Flags for TSVD and SMOTE:
   ```bash
   python train_model.py --tsvd  # Enable TSVD dimensionality reduction
   python train_model.py --smote  # Enable SMOTE oversampling
   python train_model.py --tsvd --smote  # Enable both
   ```

#### Features:
- **278-dimensional feature extraction** from `.bit` files:
  - 256-bin normalized byte histogram
  - 10 statistical features (mean, std, skew, kurtosis, entropy, etc.)
  - 12 structural features (header patterns, segment analysis)
- **Dual-head classification**:
  - Trojan Detector (Benign vs Malicious)
  - Family Classifier (CRYPTO, COMMS, MCU/CPU, BUS/DISPLAY, ITC99, and ISCAS89)
- **Automated model selection** via classifier comparison and GridSearchCV hyperparameter tuning
- Optional TSVD for dimensionality reduction (`--tsvd`)
- Optional SMOTE for class imbalance handling (`--smote`)
- Quantized model export as JSON for lightweight edge deployment

---

### ⚙️ `deploy_model.py` — On-Device Inference

> **Requirements:**
> - A supported FPGA board with PYNQ v3.1
> - Quantized model components (via on-board training or exported archive)

1. Import the Archive to your PYNQ board via Jupyter Notebook or SSH/SFTP

2. Decompress the Archive:
    ```bash
    mkdir PYNQ_BLADEI
    tar -xvzf PYNQ_BLADEI.tar.gz -C ./PYNQ_BLADEI
    rm PYNQ_BLADEI.tar.gz
    cd PYNQ_BLADEI
    ```

3. Create a Deployment Directory:
    ```bash
    mkdir -p mock_deployment  # Add your bitstreams here for deployment
    ```

4. Run the Deployment Script:
   ```bash
   python deploy_model.py ./mock_deployment/bitstream.bit
   ```

#### Features:
- Loads `.bit` files from local storage or command line argument
- Extracts 278-dimensional feature vector (byte histogram + statistical + structural)
- Displays prediction results with confidence scores
- Latency breakdown
  - Load time
  - Feature extraction time
  - Inference time
- Automatically quarantines malicious bitstreams

---

## 📈 Sample Output
### Benign Bitstream
======= BLADEI Vetting: =======<br>
Processing bitstream: AES-T2000_TjFree_20251218_152520.bit<br>

Trojan Detection: Benign [64.8% Confidence]<br>
Family Classification: CRYPTO [100.0% Confidence]<br>

ACTION: Bitstream passed vetting. Proceed to deployment.<br>

======= Latency Summary: =======<br>
Load Bitstream: 21.25 ms<br>
Feature Extraction: 16462.59 ms<br>
ML Prediction: 114.74 ms<br>

Total Latency: 16.60 s<br>

======= System Information: =======<br>
System: Linux<br>
Node Name: pynq<br>
Release: 6.6.10-xilinx-v2024.1-g08e597ec1786<br>
Version: #1 SMP PREEMPT Sat Apr 27 05:22:24 UTC 2024<br>
Machine: armv7l<br>
Processor: armv7l<br>

======= CPU Information: =======<br>
CPU Cores: 2<br>
Logical Processors: 2<br>
CPU Usage per Core: [0.4, 99.9]<br>
Total RAM: 491.6640625 MB<br>

### Malicious Bitstream
======= BLADEI Vetting: =======<br>
Processing bitstream: b15-T300_TjIn_20260106_114122.bit<br>

Trojan Detection: Malicious [71.5% Confidence]<br>
Family Classification: ITC99 [95.0% Confidence]<br>

ACTION: Bitstream quarantined -> ./mock_deployment/Quarantine/AES-T500_TjIn_20251218_163136.bit<br>
ACTION: Deployment blocked.<br>

======= Latency Summary: =======<br>
Load Bitstream: 21.64 ms<br>
Feature Extraction: 16408.97 ms<br>
ML Prediction: 131.15 ms<br>

Total Latency: 16.56 s<br>

======= System Information: =======<br>
System: Linux<br>
Node Name: pynq<br>
Release: 6.6.10-xilinx-v2024.1-g08e597ec1786<br>
Version: #1 SMP PREEMPT Sat Apr 27 05:22:24 UTC 2024<br>
Machine: armv7l<br>
Processor: armv7l<br>

======= CPU Information: =======<br>
CPU Cores: 2<br>
Logical Processors: 2<br>
CPU Usage per Core: [0.5, 98.8]<br>
Total RAM: 491.6640625 MB<br>

---

## 🤝 Acknowledgments
The authors were pleased to have this work accepted for presentation at the 37th annual ACM/ IEEE Supercomputing Conference. This work was supported by the McNair Junior Fellowship and Office of Undergraduate Research at the University of South Carolina. OpenAl's ChatGPT assisted with language and grammar correction. While this project utilizes benchmark designs from Trust-Hub, a resource sponsored by the National Science Foundation (NSF), all technical content and analysis were independently developed by the authors. This research also utilized PYNQ, provided by AMD and Xilinx, whose tools and hardware facilitated the synthesis and deployment stages of this study. Access to the FPGA devices was made possible through the AMD University Program.

---

## 🛠️ Future Work
- Develop a real-time, simulated cloud-to-edge deployment pipeline (HDL → Synthesis → BLADEI → FPGA)
- Explore deep learning architectures (CNN, RNN, LSTM/ Transformer, etc.) for improved feature learning and detection accuracy
- ~~Expand the current dataset with more SOTA benchmarks (ISCAS'89, ITC'99, etc.)~~
- ~~Develop a mock cloud-to-edge bitstream deployment pipeline~~
- ~~Improve detection latency with quantized models~~
- ~~Expand support for additional FPGA boards~~

---

## 🖊️ References
> - AMD. (2024). PYNQ: Python Productivity for Zynq. Retrieved from https://www.pynq.io
> - Benz, F., Seffrin, A., & Huss, S. A. (2012). BIL: A Tool-Chain for Bitstream Reverse-Engineering. Proceedings of the IEEE International Conference on Field Programmable Logic and Applications (FPL), 735–738. IEEE.
> - Chawla, N., Bowyer, K., Hall, L., & Kegelmeyer, W. (2002). SMOTE: Synthetic Minority Over-sampling Technique. Journal of Artificial Intelligence Research, 16, 321–357.
> - Elnaggar, R., & Chakrabarty, K. (2018). Machine Learning for Hardware Security: Opportunities and Risks. Journal of Electronic Testing, 34(2), 183–201.
> - Elnaggar, R., Chaudhuri, J., Karri, R., & Chakrabarty, K. (2023). Learning Malicious Circuits in FPGA Bitstreams. IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, 42(3), 726–739. Retrieved from https://ieeexplore.ieee.org/document/9828544/
> - Hayashi, V. T., & Ruggiero, W. V. (2025). Hardware Trojan Detection in Open-Source Hardware Designs Using Machine Learning. IEEE Access. Retrieved from https://ieeexplore.ieee.org/document/10904479/
> - Imbalanced-learn Developers. (2024). SMOTE. Retrieved from https://bit.ly/3IXc0l7
> - Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., … Duchesnay, E. (2011). scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825–2830. Retrieved from https://dl.acm.org/doi/10.5555/1953048.2078195
> - Salmani, H., Tehranipoor, M., & Karri, R. (2013). On Design Vulnerability Analysis and Trust Benchmark Development. Proceedings of the IEEE International Conference on Computer Design (ICCD). IEEE.
> - Scikit-learn Developers. (2025a). Cross Validation. Retrieved from https://bit.ly/3gct8QG
> - Scikit-learn Developers. (2025b). Truncated SVD. Retrieved from https://bit.ly/4mmi4BT
> - Seo, Y., Yoon, J., Jang, J., Cho, M., Kim, H.-K., & Kwon, T. (2018). Poster: Towards Reverse Engineering FPGA Bitstreams for Hardware Trojan Detection. Proceedings of the Network and Distributed System Security Symposium (NDSS), 18–21. Internet Society.
> - Shakya, B., He, T., Salmani, H., Forte, D., Bhunia, S., & Tehranipoor, M. (2017). Benchmarking of Hardware Trojans and Maliciously Affected Circuits. Journal of Hardware and Systems Security.
> - Yoon, J., Seo, Y., Jang, J., Cho, M., Kim, J., Kim, H., & Kwon, T. (2018). A Bitstream Reverse Engineering Tool for FPGA Hardware Trojan Detection. Proceedings of the 2018 ACM SIGSAC Conference on Computer and Communications Security, 2318–2320. Presented at the Toronto, Canada. doi:10.1145/3243734.3278487
