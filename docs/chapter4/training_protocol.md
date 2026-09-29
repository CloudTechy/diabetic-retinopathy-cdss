# Model Training Protocol & Empirical Convergence Ledger

## Metadata & Traceability
- **Research Project:** AI-Based Clinical Decision Support System for Early Detection of Diabetic Retinopathy
- **Author / Researcher:** Onyekelu Chukwuebuka Elochukwu (2024516020FN)
- **Primary Research Objective:** Objective e (Train and optimize the EfficientNet-B0 model)
- **Execution Date:** 2026-09-29
- **Trained Checkpoint Path:** `backend/models/weights/efficientnet_b0_dr.pth`
- **Integrity Checksum (SHA-256):** `0d443fa065528b2a1d24baea7bf8d8bf71a6203b22b817585374a7becd547c07`
- **Best Validation Epoch:** Epoch 14 of 15
- **Peak Validation QWK:** **0.8826**
- **Supporting Visualization:** [`docs/chapter4/learning_curves.png`](file:///c:/Users/USER/Documents/TECH4MATION/diabetic-retinopathy-cdss/docs/chapter4/learning_curves.png)

---

## 1. Hyperparameter Specification

| Hyperparameter | Value | Scientific & Clinical Rationale |
| :--- | :---: | :--- |
| **Model Topology** | EfficientNet-B0 | Compound scaled CNN architecture ($4,013,953$ parameters) |
| **Input Tensor Resolution** | $224 	imes 224 	imes 3$ | RGB channels normalized with ImageNet priors ($\mu, \sigma$) |
| **Batch Size** | 32 | Optimal gradient stability across patient clusters |
| **Optimization Algorithm** | AdamW | Decoupled weight decay regularization ($\lambda = 10^{-4}$) |
| **Initial Learning Rate ($\eta_0$)** | $1.0 	imes 10^{-4}$ | Prevents destructive gradient updates during transfer learning |
| **LR Scheduler** | CosineAnnealingLR | Gradual smooth annealing from $\eta_0$ to $\eta_{\min} = 10^{-6}$ |
| **Loss Function** | Class-Weighted Cross-Entropy | Weighted by inverse class frequencies to mitigate 9.35:1 imbalance |
| **Regularization** | Dropout ($p = 0.20$) | Prevents over-indexing on majority Grade 0 features |
| **Early Stopping Metric** | Validation QWK | Monitored with patience of 5 epochs to prevent validation divergence |

---

## 2. Class Weighting Matrix

To prevent the classifier from collapsing into majority Grade 0 predictions, class weights were computed according to $w_c = \frac{N_{\text{train}}}{5 \cdot N_{c,\text{train}}}$:

| ICDR Grade | Class Name | Train Count ($N$) | Loss Weight ($w_c$) |
| :---: | :--- | :---: | :---: |
| **0** | No Apparent DR | 1,267 | **0.4052** |
| **1** | Mild NPDR | 256 | **2.0055** |
| **2** | Moderate NPDR | 700 | **0.7334** |
| **3** | Severe NPDR | 134 | **3.8313** |
| **4** | Proliferative DR | 210 | **2.4448** |

---

## 3. Epoch-by-Epoch Convergence History

| Epoch | Train Loss | Val Loss | Val Accuracy | Val QWK | Val Macro F1 | Learning Rate | Checkpoint Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 01 | 1.6018 | 1.4076 | 54.49% | **0.5143** | 0.4917 | `1.00e-04` | **BEST CHECKPOINT** |
| 02 | 1.4564 | 1.2250 | 61.47% | **0.5930** | 0.5519 | `9.89e-05` | **BEST CHECKPOINT** |
| 03 | 1.3702 | 1.0637 | 65.30% | **0.6432** | 0.6096 | `9.57e-05` | **BEST CHECKPOINT** |
| 04 | 1.3877 | 0.9275 | 69.54% | **0.6961** | 0.6520 | `9.05e-05` | **BEST CHECKPOINT** |
| 05 | 1.2728 | 0.8315 | 73.43% | **0.7371** | 0.6776 | `8.36e-05` | **BEST CHECKPOINT** |
| 06 | 1.4011 | 0.7657 | 76.27% | **0.7635** | 0.7080 | `7.52e-05` | **BEST CHECKPOINT** |
| 07 | 1.2107 | 0.7020 | 77.87% | **0.7867** | 0.7300 | `6.58e-05` | **BEST CHECKPOINT** |
| 08 | 1.1650 | 0.6428 | 80.09% | **0.8168** | 0.7589 | `5.57e-05` | **BEST CHECKPOINT** |
| 09 | 1.1373 | 0.5877 | 82.15% | **0.8292** | 0.7714 | `4.53e-05` | **BEST CHECKPOINT** |
| 10 | 1.0464 | 0.5591 | 82.78% | **0.8494** | 0.7844 | `3.52e-05` | **BEST CHECKPOINT** |
| 11 | 1.0613 | 0.5246 | 82.84% | **0.8535** | 0.7901 | `2.58e-05` | **BEST CHECKPOINT** |
| 12 | 1.1029 | 0.4808 | 83.94% | **0.8617** | 0.7983 | `1.74e-05` | **BEST CHECKPOINT** |
| 13 | 1.0451 | 0.4769 | 84.81% | **0.8728** | 0.8042 | `1.05e-05` | **BEST CHECKPOINT** |
| 14 | 1.0485 | 0.4490 | 86.27% | **0.8826** | 0.8162 | `5.28e-06` | **BEST CHECKPOINT** |
| 15 | 0.9874 | 0.4324 | 86.38% | **0.8821** | 0.8172 | `2.08e-06` | — |

---

## 4. Best Checkpoint Selection Record
- **Selection Criterion:** Maximization of validation Quadratic Weighted Kappa (QWK), which penalizes multi-grade clinical discrepancies quadratically.
- **Optimal Checkpoint:** Checkpoint state captured at **Epoch 14** achieved peak $\kappa = 0.8826$.
- **Storage Path:** `backend/models/weights/efficientnet_b0_dr.pth` (15.60 MB).
- **Integrity Verified:** Cryptographic hash matches [`docs/chapter4/checkpoint_manifest.md`](file:///c:/Users/USER/Documents/TECH4MATION/diabetic-retinopathy-cdss/docs/chapter4/checkpoint_manifest.md).
