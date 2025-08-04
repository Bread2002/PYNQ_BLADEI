# **Machine Learning-Based Detection of Simulated Malware in FPGA Bitstreams**
### *Copyright (c) 2025, Rye Stahle-Smith* 

---

## 📌 Project Overview

This repository contains an embedded deployment pipeline for detecting **malicious FPGA bitstreams** using a trained machine learning model. Bitstreams are configuration files that can be weaponized to introduce hardware Trojans, posing serious risks in shared or cloud-hosted reconfigurable systems. This project leverages a lightweight, byte-level classification approach and enables **on-device malware detection** for the **Xilinx PYNQ-Z1** board, without requiring reverse engineering techniques or access to original source code or netlists. Benchmark designs, including AES-128 and RS232 variants, were obtained from Trust-Hub, then synthesized, implemented, and categorized as benign, malicious, or empty `.bit` files.

---

## ⚙️ Features

- 🔍 **Byte-frequency analysis** of binary `.bit` files
- 📉 Dimensionality reduction and class balancing via **TSVD** and **SMOTE**
- 📊 Real-time inference using trained **scikit-learn classifiers** (e.g., Random Forest)
- ⚡ Deployment-ready for **PYNQ-Z1 (Zynq-7000 SoC)**
- 🧪 Verified with state-of-the-art (SOTA) bitstreams derived from **Trust-Hub** benchmarks

---

## 📂 Repository Structure
pynq-maldetect/<br>
├── trusthub_bitstreams/ ***# Sample `.bit` files (Benign, Malicious, Empty)***<br>
├── model_components/ ***# Serialized sklearn model components***<br>
├── VirtualEnv/ ***# Virtual environment***<br>
├── train_model.ipynb ***# Model training and export for PYNQ***<br>
├── deploy_model.ipynb ***# Model deployment for on-device inference***<br>
├── requirements.txt ***# Python dependencies***<br>
└── README.md<br>

---

## 🚀 Setup Instructions

This project is divided into two parts:

- 🧠 **Model Training and Export** (Run on your local machine or server with a standard CPU)
- ⚙️ **On-Device Inference** (Run on the PYNQ-Z1 board)

---

### 🧠 `train_model.ipynb` — Model Training and Export (CPU Only)

> **Requirements:**
> - Python 3.8+
> - `scikit-learn`, `numpy`, `pandas`, `matplotlib`, `imblearn`  
> - *Do not attempt this on PYNQ. Training is too resource-intensive and `scikit-learn` is not supported for ARM.*

1. Clone the Repository:
   ```bash
   git clone https://github.com/Bread2002/pynq-maldetect.git
   cd pynq-maldetect
   ```

2. Run the Training Script:
   ```bash
    jupyter notebook train_model.ipynb
   ```

#### ***Features:***
- Byte-level and structural feature extraction from `.bit` files  
- Dimensionality reduction via TSVD  
- Class balancing with SMOTE  
- Training multiple classifiers (e.g., Random Forest, SVM)  
- Evaluation using Stratified k-Fold Cross-Validation  
- Model and TSVD components exported as a `.tar.gz` archive for use on PYNQ

---

### ⚙️ `deploy_model.ipynb` — On-Device Inference (PYNQ-Z1)

> **Requirements:**
> - PYNQ-Z1 board with Python 3.x
> - Pre-trained model archive (pynq_maldetect.tar.gz)

1. Import the Archive to your PYNQ board via Jupyter Notebook

2. Export the Archive:
    ```bash
    mkdir pynq_maldetect
    tar -xvzf pynq_maldetect.tar.gz -C ./pynq_maldetect
    rm pynq_maldetect.tar.gz
    cd pynq_maldetect
    ```

3. Install Dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Deployment Script:
   ```bash
   jupyter notebook deploy_model.ipynb
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

## 📈 Example Output
*** Trial 1: Processing RS232_T1500_Trojan.bit... ***<br>
Actual Class:    Malicious (Class 2)<br>
Predicted Class: Malicious (Class 2)<br>

=== Latency Summary ===<br>
Load Bitstream:      195.81 ms<br>
Feature Extraction:  3096.98 ms<br>
Prediction:          23.86 ms<br>

*** Trial 2: Processing empty27.bit... ***<br>
Actual Class:    Empty (Class 0)<br>
Predicted Class: Empty (Class 0)<br>

=== Latency Summary ===<br>
Load Bitstream:      223.81 ms<br>
Feature Extraction:  3098.21 ms<br>
Prediction:          13.48 ms<br>

*** Trial 3: Processing empty27.bit... ***<br>
Actual Class:    Empty (Class 0)<br>
Predicted Class: Empty (Class 0)<br>

=== Latency Summary ===<br>
Load Bitstream:      24.16 ms<br>
Feature Extraction:  3095.33 ms<br>
Prediction:          18.85 ms<br>

*** Trial 4: Processing AES_T700.bit... ***<br>
Actual Class:    Benign (Class 1)<br>
Predicted Class: Benign (Class 1)<br>

=== Latency Summary ===<br>
Load Bitstream:      188.02 ms<br>
Feature Extraction:  3096.70 ms<br>
Prediction:          21.86 ms<br>

*** Trial 5: Processing RS232_T1900.bit... ***<br>
Actual Class:    Benign (Class 1)<br>
Predicted Class: Benign (Class 1)<br>

=== Latency Summary ===<br>
Load Bitstream:      188.15 ms<br>
Feature Extraction:  3097.69 ms<br>
Prediction:          21.89 ms<br>


Average Latency: 3.28 s<br>

---

## 🤝 Acknowledgments
This work was supported by the McNair Junior Fellowship and Office of Undergraduate Research at the University of South Carolina. The authors used OpenAl's ChatGPT to assist with language and grammar correction. While this project utilizes benchmark designs from Trust-Hub, a resource sponsored by the National Science Foundation (NSF), all technical content and analysis were independently developed by the authors.
This research also made use of the PYNQ-Z1 FPGA platform, provided by AMD and Xilinx, whose tools and hardware enabled the synthesis and deployment stages of this study.

---

## 🛠️ Future Work
- Improve detection latency with quantized ML models
- Integrate live USB bitstream capture
- Expand support for additional FPGA boards

## 🖊️ References
> - AMD. (2024). PYNQTM: Python Productivity for Zynq. Retrieved from https://www.pynq.io
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
