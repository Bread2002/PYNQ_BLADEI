# **BLADEI: Bitstream-Level Abnormality Detection for Embedded Inference**
### *Copyright (c) 2025, Rye Stahle-Smith* 

---

## 📌 Project Overview

This repository contains an embedded deployment pipeline for detecting **malicious FPGA bitstreams** using a trained machine learning (ML) model. Bitstreams are configuration files that can be weaponized to introduce hardware Trojans, posing serious risks in shared or cloud-hosted reconfigurable systems. This project leverages a lightweight, byte-level classification approach and enables **on-device malware detection** for **PYNQ-supported FPGA boards**, without requiring reverse engineering techniques or access to original source code or netlists. Benchmark designs, including AES-128 and RS232 variants, were obtained from Trust-Hub, then synthesized, implemented, and categorized as benign, malicious, or empty `.bit` files

---

## ⚙️ Features

- 🔍 **Byte-frequency analysis** of binary `.bit` files
- 📉 Dimensionality reduction and class balancing via **TSVD** and **SMOTE**
- 📊 Real-time inference using trained **scikit-learn classifiers** (e.g., Random Forest)
- ⚡ Deployment-ready for **ARMv7 (e.g., PYNQ-Z1/Z2, Zynq-7000 SoC)** and **ARMv8 (e.g., Zynq UltraScale+ MPSoC, RFSoC, Kria) boards**
- 🧪 Verified with state-of-the-art (SOTA) bitstreams derived from **Trust-Hub** benchmarks

---

## 📂 Repository Structure
pynq-maldetect/<br>
├── trusthub_bitstreams/ ***# Sample `.bit` files (Benign, Malicious, Empty)***<br>
├── model_components/ ***# Quantized ML model components***<br>
├── train_model.py ***# Model training and export for PYNQ***<br>
├── deploy_model.py ***# Model deployment for on-device inference***<br>
├── requirements.txt ***# Python dependencies***<br>
├── LICENSE.md<br>
└── README.md<br>

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
> Training should be performed on a general-purpose machine (laptop, workstation, or server) for **both ARMv7 and ARMv8** targets. While some ARMv8 boards *may* be capable of training, it is not the intended workflow here—training is heavier, package availability can be inconsistent, and it’s typically slower and less reproducible than running on a PC.  

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

#### ***Features:***
- Byte-level and structural feature extraction from `.bit` files  
- Dimensionality reduction via TSVD  
- Class balancing with SMOTE  
- Training multiple classifiers (e.g., Random Forest, SVM)  
- Evaluation using k-Fold Cross-Validation  
- Model and TSVD components exported as a `.tar.gz` archive for PYNQ deployment on ARMv7 boards

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

3. Run the Deployment Script:
   ```bash
   python deploy_model.py
   ```

#### Features:
- Loads `.bit` files from local storage  
- Extracts sparse and structural features  
- Applies TSVD transformation  
- Predicts class (`Benign`, `Malicious`, or `Empty`) using the trained model
- Displays prediction result with latency breakdown:
  - Load time  
  - Feature extraction time  
  - Inference time

---

## 📈 Sample Output of Mock Deployment Pipeline
======= BLADEI Vetting: =======<br>
Processing bitstream: AES-T2000_TjFree_20251218_152520.bit<br>

Actual Class: Benign AES (Class 1)<br>
Predicted Class: Benign AES (Class 1) [80.00% Confidence]<br>

ACTION: Bitstream passed vetting. Proceed to deployment.<br>

======= Latency Summary: =======<br>
Load Bitstream:		21.85 ms<br>
Feature Extraction:	3217.94 ms<br>
Prediction:		14.83 ms<br>

Total Latency: 3.25 s<br>

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
CPU Usage per Core: [99.5, 0.5]<br>
Total RAM: 491.6640625 MB<br>

---

## 🤝 Acknowledgments
The authors were pleased to have this work accepted for presentation at the 37th annual ACM/ IEEE Supercomputing Conference. This work was supported by the McNair Junior Fellowship and Office of Undergraduate Research at the University of South Carolina. OpenAl's ChatGPT assisted with language and grammar correction. While this project utilizes benchmark designs from Trust-Hub, a resource sponsored by the National Science Foundation (NSF), all technical content and analysis were independently developed by the authors. This research also utilized PYNQ, provided by AMD and Xilinx, whose tools and hardware facilitated the synthesis and deployment stages of this study. Access to the FPGA devices was made possible through the AMD University Program.

---

## 🛠️ Future Work
- Expand the current dataset with more SOTA benchmarks (ISCAS'85, ISCAS'89, ITC'02, and ITC'99)
- Add a CNN-based image classification model to authenticate ML predictions
- ~~Implement a mock cloud-to-edge bitstream deployment pipeline~~
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
