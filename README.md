# **BLADEI: Bitstream-Level Abnormality Detection for Embedded Inference**
### *Copyright (c) 2025, Rye Stahle-Smith* 

---

## 📌 Project Overview

This repository contains an embedded deployment pipeline for detecting **malicious FPGA bitstreams** using a trained machine learning (ML) model. Bitstreams are configuration files that can be weaponized to introduce hardware Trojans, posing serious risks in shared or cloud-hosted reconfigurable systems. This project leverages a lightweight, byte-level classification approach and enables **on-device malware detection** for **PYNQ-supported FPGA boards**, without requiring reverse engineering techniques or access to original source code or netlists. Benchmark designs, including AES-128 and RS232 variants, were obtained from Trust-Hub, then synthesized, implemented, and categorized as benign, malicious, or empty `.bit` files. Additionally, a CNN-based natural language processing (NLP) model is trained to **cross-check ML predictions** by converting top byte-frequency features into textual representations, providing an extra layer of confirmation for bitstream classification.

---

## ⚙️ Features

- 🔍 **Byte-frequency analysis** of binary `.bit` files
- 📉 Dimensionality reduction and class balancing via **TSVD** and **SMOTE**
- 📊 Real-time inference using trained **scikit-learn classifiers** (e.g., Random Forest)
- 📝 CNN-based **NLP cross-checking** for validating ML predictions using text representations of top features
- ⚡ Deployment-ready for **ARMv7 (e.g., PYNQ-Z1/Z2, Zynq-7000 SoC)** and **ARMv8 (e.g., Zynq UltraScale+ MPSoC, RFSoC, Kria) boards**
- 🧪 Verified with state-of-the-art (SOTA) bitstreams derived from **Trust-Hub** benchmarks

---

## 📂 Repository Structure
pynq-maldetect/<br>
├── trusthub_bitstreams/ ***# Sample `.bit` files (Benign, Malicious, Empty)***<br>
├── model_components/ ***# Quantized ML+NLP model components***<br>
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
> - Python Packages: `scikit-learn`, `numpy`, `scipy`, `pandas`, `joblib`, `imblearn`, `pytorch`

> ⚠️ **Note:**
> On **ARMv7 (32-bit)** boards (e.g., PYNQ-Z1/Z2), training is not supported. These boards lack prebuilt wheels and have insufficient resources for model training. Use a general-purpose CPU (e.g., laptop, workstation, or server) instead. If you take this route, *skip Steps 2 and 4* and ***comment out the "PYNQ-specific Packages" from `requirements.txt`***.<br>
> On **ARMv8 (64-bit)** boards (e.g., Zynq UltraScale+, Kria, RFSoC), you may train directly on the board if sufficient resources are available.

1. Clone the Repository:
   ```bash
   git clone https://github.com/Bread2002/PYNQ_BLADEI.git
   cd PYNQ_BLADEI
   ```

2. Source the PYNQ Virtual Environment:
   ```bash
   source /usr/local/share/pynq-venv/bin/activate
   ```
   
3. Install Dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Deactivate the PYNQ Virtual Environment:
   ```bash
   deactivate
   ```

5. Run the Training Script:
   ```bash
   python train_model.py
   ```

#### ***Features:***
- Byte-level and structural feature extraction from `.bit` files  
- Dimensionality reduction via TSVD  
- Class balancing with SMOTE  
- Training multiple classifiers (e.g., Random Forest, SVM)  
- CNN-based NLP cross-checking to validate ML predictions
- Evaluation using k-Fold Cross-Validation  
- Model and TSVD components exported as a `.tar.gz` archive for PYNQ deployment on ARMv7 boards

---

### ⚙️ `deploy_model.py` — On-Device Inference

> **Requirements:**
> - A supported FPGA board with PYNQ v3.1
> - Quantized model components (via on-board training or exported archive)

> ⚠️ **Note:**
> If you are on an **ARMv8 (64-bit)** board (e.g., UltraScale+, Kria, RFSoC), you may have trained directly on the device. In this case, *skip to Step 3*.<br>
> If you are on an **ARMv7 (32-bit)** board (e.g., PYNQ-Z1/Z2), *begin at Step 1*. Since you cannot train on the board, you must import the archive.


1. Import the Archive to your PYNQ board via Jupyter Notebook

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
- Confirms prediction via CNN-based NLP system
- Displays prediction result with latency breakdown:
  - Load time  
  - Feature extraction time  
  - Inference time
  - NLP cross-check time

---

## 📈 Example Output
*** Trial 1: Processing empty7.bit... ***<br>
Actual Class:    Empty (Class 0)<br>
ML Predicted:    Empty (Class 0)<br>
NLP Cross-Check: Match<br>

=== Latency Summary ===<br>
Load Bitstream:      7.34 ms<br>
Feature Extraction:  1791.22 ms<br>
Prediction:          5.83 ms<br>
NLP Confirmation:    536.88 ms<br>

*** Trial 2: Processing RS232_T600.bit... ***<br>
Actual Class:    Benign RS232 (Class 2)<br>
ML Predicted:    Benign RS232 (Class 2)<br>
NLP Cross-Check: Match<br>

=== Latency Summary ===<br>
Load Bitstream:      8.25 ms<br>
Feature Extraction:  1913.59 ms<br>
Prediction:          5.90 ms<br>
NLP Confirmation:    538.15 ms<br>

*** Trial 3: Processing AES_T2000_Trojan.bit... ***<br>
Actual Class:    Malicious AES (Class 3)<br>
ML Predicted:    Malicious AES (Class 3)<br>
NLP Cross-Check: Match<br>

=== Latency Summary ===<br>
Load Bitstream:      7.76 ms<br>
Feature Extraction:  1840.20 ms<br>
Prediction:          6.33 ms<br>
NLP Confirmation:    536.09 ms<br>

*** Trial 4: Processing AES_T600.bit... ***<br>
Actual Class:    Benign AES (Class 1)<br>
ML Predicted:    Benign AES (Class 1)<br>
NLP Cross-Check: Match<br>

=== Latency Summary ===<br>
Load Bitstream:      7.25 ms<br>
Feature Extraction:  1832.26 ms<br>
Prediction:          6.30 ms<br>
NLP Confirmation:    536.31 ms<br>

*** Trial 5: Processing AES_T2100_Trojan.bit... ***<br>
Actual Class:    Malicious AES (Class 3)<br>
ML Predicted:    Malicious AES (Class 3)<br>
NLP Cross-Check: Match<br>

=== Latency Summary ===<br>
Load Bitstream:      7.45 ms<br>
Feature Extraction:  1821.59 ms<br>
Prediction:          6.22 ms<br>
NLP Confirmation:    536.80 ms<br>

Average Latency: 2.39 s<br>

---

## 🤝 Acknowledgments
The authors were pleased to have this work accepted for presentation at the 37th annual ACM/ IEEE Supercomputing Conference. This work was supported by the McNair Junior Fellowship and Office of Undergraduate Research at the University of South Carolina. OpenAl's ChatGPT assisted with language and grammar correction. While this project utilizes benchmark designs from Trust-Hub, a resource sponsored by the National Science Foundation (NSF), all technical content and analysis were independently developed by the authors. This research also utilized PYNQ, provided by AMD and Xilinx, whose tools and hardware facilitated the synthesis and deployment stages of this study. Access to the FPGA devices was made possible through the AMD University Program.

---

## 🛠️ Future Work
- Provide human-readable justifications to interpret detection results
- Integrate live USB bitstream capture
- ~~Improve detection latency with quantized models~~
- ~~Add NLP-based confirmation for ML predictions~~
- ~~Expand support for additional FPGA boards~~

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
