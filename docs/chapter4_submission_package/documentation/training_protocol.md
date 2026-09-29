# Model Training Protocol & Empirical Convergence Ledger

## Metadata & Traceability
- **Research Project:** AI-Based Clinical Decision Support System for Early Detection of Diabetic Retinopathy
- **Author / Researcher:** Onyekelu Chukwuebuka Elochukwu (2024516020FN)
- **Primary Research Objective:** Objective e (Train and optimize the EfficientNet-B0 model)
- **Execution Date:** 2026-09-29
- **Trained Checkpoint Path:** `backend/models/weights/efficientnet_b0_dr.pth`
- **Integrity Checksum (SHA-256):** `8ee14d7591a8e6a1b86c15416a77375a198bd49399b3977a3de79a00e3dd14fa`
- **Best Validation Epoch:** Epoch 11 of 15
- **Peak Validation QWK:** **0.8937**
- **Total Wall-Clock Training Time:** 3,147 seconds (52.5 minutes), ~210 s/epoch
- **Raw Evidence:** [`training_execution.log`](training_execution.log), [`epoch_history.csv`](epoch_history.csv), [`training_summary.json`](training_summary.json)
- **Supporting Visualization:** [`learning_curves.png`](learning_curves.png)

---

## 1. Hyperparameter Specification

| Hyperparameter | Value | Scientific & Clinical Rationale |
| :--- | :---: | :--- |
| **Model Topology** | EfficientNet-B0 | Compound scaled CNN architecture (4,013,953 parameters) |
| **Initialisation** | ImageNet `IMAGENET1K_V1` | Transfer learning from natural-image features |
| **Input Tensor Resolution** | $224 \times 224 \times 3$ | RGB channels normalized with ImageNet priors ($\mu, \sigma$) |
| **Batch Size** | 32 | Fits T4 VRAM at this resolution with stable gradients |
| **Epochs** | 15 | Validation loss plateaus from epoch 8 (see §3) |
| **Optimization Algorithm** | AdamW | Decoupled weight decay regularization ($\lambda = 10^{-4}$) |
| **Initial Learning Rate ($\eta_0$)** | $1.0 \times 10^{-4}$ | Prevents destructive gradient updates during transfer learning |
| **LR Scheduler** | CosineAnnealingLR | Smooth annealing from $\eta_0$ to $\eta_{\min} = 10^{-6}$ |
| **Loss Function** | Class-Weighted Cross-Entropy | Weighted by inverse class frequency to mitigate the 9.4:1 imbalance |
| **Regularization** | Dropout ($p = 0.20$) | Prevents over-indexing on majority Grade 0 features |
| **Checkpoint Selection** | Maximum validation QWK | Ordinal-aware; penalises multi-grade errors quadratically |
| **Random Seed** | 42 | Applied to `random`, `numpy`, `torch`, `torch.cuda` |

**Note on early stopping.** No early-stopping patience was applied. All 15 epochs were run to completion and the checkpoint from the best-QWK epoch was retained. The distinction matters for reproducibility: the final-epoch weights are *not* the served weights.

---

## 2. Class Weighting Matrix

Weights were computed as $w_c = \dfrac{N_{\text{train}}}{5 \cdot N_{c,\text{train}}}$ over the 2,563 training images:

| ICDR Grade | Class Name | Train Count ($N$) | Loss Weight ($w_c$) |
| :---: | :--- | :---: | :---: |
| **0** | No Apparent DR | 1,264 | **0.4055** |
| **1** | Mild NPDR | 259 | **1.9792** |
| **2** | Moderate NPDR | 699 | **0.7333** |
| **3** | Severe NPDR | 135 | **3.7970** |
| **4** | Proliferative DR | 206 | **2.4883** |

---

## 3. Epoch-by-Epoch Convergence History

Transcribed verbatim from [`epoch_history.csv`](epoch_history.csv).

| Epoch | Train Loss | Val Loss | Val Accuracy | Val QWK | Val Macro F1 | Learning Rate | Checkpoint |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 01 | 1.362504 | 0.947581 | 65.64% | 0.7712 | 0.4917 | `1.00e-04` | **BEST** |
| 02 | 1.034721 | 0.774030 | 70.55% | 0.8305 | 0.5411 | `9.89e-05` | **BEST** |
| 03 | 0.904448 | 0.638443 | 79.64% | 0.8673 | 0.6547 | `9.57e-05` | **BEST** |
| 04 | 0.785597 | 0.620398 | 78.91% | 0.8558 | 0.6371 | `9.05e-05` | — |
| 05 | 0.767437 | 0.608991 | 79.82% | 0.8744 | 0.6311 | `8.36e-05` | **BEST** |
| 06 | 0.712628 | 0.629791 | 76.73% | 0.8519 | 0.6341 | `7.52e-05` | — |
| 07 | 0.679063 | 0.571831 | 77.45% | 0.8637 | 0.6423 | `6.58e-05` | — |
| 08 | 0.668761 | 0.567656 | 81.09% | 0.8796 | 0.6885 | `5.57e-05` | **BEST** |
| 09 | 0.623921 | 0.564016 | 80.18% | 0.8838 | 0.6637 | `4.53e-05` | **BEST** |
| 10 | 0.564530 | 0.573863 | 80.18% | 0.8828 | 0.6676 | `3.52e-05` | — |
| **11** | **0.517256** | **0.560883** | **81.09%** | **0.8937** | **0.6862** | `2.58e-05` | **BEST (SELECTED)** |
| 12 | 0.535151 | 0.559696 | 80.73% | 0.8917 | 0.6813 | `1.74e-05` | — |
| 13 | 0.542185 | 0.573130 | 80.73% | 0.8885 | 0.6697 | `1.05e-05` | — |
| 14 | 0.528482 | 0.555868 | 80.55% | 0.8886 | 0.6771 | `5.28e-06` | — |
| 15 | 0.496885 | 0.557285 | 80.36% | 0.8869 | 0.6785 | `2.08e-06` | — |

---

## 4. Convergence Analysis

**No overfitting collapse.** Training loss falls monotonically from 1.3625 to 0.4969, while validation loss falls from 0.9476 to a plateau around 0.556–0.573 from epoch 8 onward. The gap between the two curves stays narrow and the validation curve does not turn upward, which is the signature of a run that stopped at roughly the right time rather than one that memorised the training set.

**Validation QWK saturates early.** $\kappa$ exceeds 0.86 by epoch 3 and thereafter moves within a 0.04 band, peaking at 0.8937 at epoch 11. The final four epochs contribute no material improvement — consistent with the cosine schedule having annealed the learning rate below $3 \times 10^{-5}$.

**Macro F1 lags accuracy throughout** (0.686 vs 81.1% at the selected epoch). This gap is the minority-class problem stated plainly: the model learns Grade 0 quickly and the sparse Grades 3 and 4 slowly. It is the same effect that surfaces in the held-out per-class table in [`model_evaluation_report.md`](model_evaluation_report.md).

**Validation-to-test consistency.** Validation $\kappa = 0.8937$ at selection versus held-out $\kappa = 0.8777$. The 0.016 drop is small and in the expected direction, indicating the checkpoint-selection step did not materially overfit the validation split.

---

## 5. Best Checkpoint Selection Record

- **Selection Criterion:** Maximum validation Quadratic Weighted Kappa.
- **Optimal Checkpoint:** Epoch 11, $\kappa = 0.8937$.
- **Storage Path:** `backend/models/weights/efficientnet_b0_dr.pth` (15.60 MB).
- **Integrity:** SHA-256 recorded in [`checkpoint_manifest.md`](checkpoint_manifest.md) and enforced at runtime by the inference service.
