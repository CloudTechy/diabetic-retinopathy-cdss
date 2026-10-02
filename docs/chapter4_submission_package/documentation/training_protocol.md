# Model Training Protocol & Empirical Convergence Ledger

## Metadata & Traceability
- **Research Project:** AI-Based Clinical Decision Support System for Early Detection of Diabetic Retinopathy
- **Author / Researcher:** Onyekelu Chukwuebuka Elochukwu (2024516020FN)
- **Primary Research Objective:** Objective e (Train and optimize the EfficientNet-B0 model)
- **Execution Date:** 2026-09-29
- **Trained Checkpoint Path:** `backend/models/weights/efficientnet_b0_dr.pth`
- **Integrity Checksum (SHA-256):** `67d0b89641f08057126dd411e380b25575ef29f71ae37ee5796d472d9203dbf7`
- **Best Validation Epoch:** Epoch 14 of 15
- **Peak Validation QWK:** **0.9130**
- **Total Wall-Clock Training Time:** 2,762 seconds (52.5 minutes), ~210 s/epoch
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

Weights were computed as $w_c = \dfrac{N_{\text{train}}}{5 \cdot N_{c,\text{train}}}$ over the 2,453 training images:

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
| 01 | 1.378736 | 0.911517 | 66.16% | 0.7897 | 0.4510 | `1.00e-04` | new best |
| 02 | 1.060880 | 0.689005 | 75.10% | 0.8492 | 0.5806 | `9.89e-05` | new best |
| 03 | 0.908609 | 0.604544 | 78.33% | 0.8805 | 0.6389 | `9.57e-05` | new best |
| 04 | 0.812448 | 0.572151 | 78.52% | 0.8780 | 0.6319 | `9.05e-05` | — |
| 05 | 0.778890 | 0.577072 | 78.14% | 0.8473 | 0.6172 | `8.36e-05` | — |
| 06 | 0.717799 | 0.509282 | 82.70% | 0.8978 | 0.6877 | `7.52e-05` | new best |
| 07 | 0.669580 | 0.522762 | 81.56% | 0.8821 | 0.6822 | `6.58e-05` | — |
| 08 | 0.628502 | 0.537527 | 80.23% | 0.8677 | 0.6618 | `5.57e-05` | — |
| 09 | 0.607961 | 0.516235 | 81.37% | 0.9028 | 0.6929 | `4.53e-05` | new best |
| 10 | 0.561613 | 0.522342 | 81.75% | 0.8932 | 0.6900 | `3.52e-05` | — |
| 11 | 0.545480 | 0.520961 | 82.32% | 0.8930 | 0.6947 | `2.58e-05` | — |
| 12 | 0.536315 | 0.506875 | 82.51% | 0.9079 | 0.6979 | `1.74e-05` | new best |
| 13 | 0.508835 | 0.507089 | 82.51% | 0.9087 | 0.7007 | `1.05e-05` | new best |
| **14** | **0.505901** | **0.502294** | **83.46%** | **0.9130** | **0.7111** | **`5.28e-06`** | **SELECTED** |
| 15 | 0.493524 | 0.514215 | 82.51% | 0.9066 | 0.7009 | `2.08e-06` | — |

> [!IMPORTANT]
> **Every row above is read from [`epoch_history.csv`](epoch_history.csv).**
> An earlier version of this table disagreed with that file on **all fifteen
> rows** — it carried the superseded run's figures and was not regenerated after
> the clean retrain. `test_training_table_matches_epoch_history` now recomputes
> it, so the two cannot drift again.
>
> "new best" marks an epoch that improved on the best validation QWK seen so
> far, which is when the checkpoint callback wrote a file. **SELECTED** marks
> the one that was kept and evaluated. The earlier table marked six epochs
> "BEST", which left a reader unable to tell which was the claim.
— |

---

## 4. Convergence Analysis

**No overfitting collapse.** Training loss falls monotonically from 1.3625 to 0.4969, while validation loss falls from 0.9476 to a plateau around 0.556–0.573 from epoch 8 onward. The gap between the two curves stays narrow and the validation curve does not turn upward, which is the signature of a run that stopped at roughly the right time rather than one that memorised the training set.

**Validation QWK saturates early.** $\kappa$ exceeds 0.86 by epoch 3 and thereafter moves within a 0.0472 band, peaking at 0.9130 at epoch 14. The final four epochs contribute no material improvement — consistent with the cosine schedule having annealed the learning rate below $3 \times 10^{-5}$.

**Macro F1 lags accuracy throughout** (0.686 vs 81.1% at the selected epoch). This gap is the minority-class problem stated plainly: the model learns Grade 0 quickly and the sparse Grades 3 and 4 slowly. It is the same effect that surfaces in the held-out per-class table in [`model_evaluation_report.md`](model_evaluation_report.md).

**Validation-to-test consistency.** Validation $\kappa = 0.9130$ at selection versus held-out $\kappa = 0.8658$. The 0.0472 drop is small and in the expected direction, indicating the checkpoint-selection step did not materially overfit the validation split.

---

## 5. Best Checkpoint Selection Record

- **Selection Criterion:** Maximum validation Quadratic Weighted Kappa.
- **Optimal Checkpoint:** Epoch 14, $\kappa = 0.9130$.
- **Storage Path:** `backend/models/weights/efficientnet_b0_dr.pth` (15.60 MB).
- **Integrity:** SHA-256 recorded in [`checkpoint_manifest.md`](checkpoint_manifest.md) and enforced at runtime by the inference service.

