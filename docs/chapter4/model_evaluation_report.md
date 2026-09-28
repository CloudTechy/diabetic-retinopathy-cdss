# Held-Out Model Evaluation & Statistical Results Report

## Metadata & Traceability
- **Research Project:** AI-Based Clinical Decision Support System for Early Detection of Diabetic Retinopathy
- **Author / Researcher:** Onyekelu Chukwuebuka Elochukwu (2024516020FN)
- **Related Research Objective:** Objective h (Evaluate model performance)
- **Git Commit:** `22cda2c` (Baseline)
- **Date Generated:** 2026-09-28
- **Evaluation Dataset:** Untouched Held-Out Test Set ($N = 1,200$ independent patient fundus photos)
- **Model Checkpoint:** `backend/models/weights/efficientnet_b0_dr.pth` (Frozen, eval mode)
- **Evidence Files:** [`docs/chapter4/held_out_predictions.csv`](file:///c:/Users/USER/Documents/TECH4MATION/diabetic-retinopathy-cdss/docs/chapter4/held_out_predictions.csv), [`docs/chapter4/confusion_matrix.png`](file:///c:/Users/USER/Documents/TECH4MATION/diabetic-retinopathy-cdss/docs/chapter4/confusion_matrix.png)
- **Hardware/Software Environment:** Python 3.13, PyTorch 2.6 / torchvision, scikit-learn, Pillow

---

## 1. Global Performance Metrics

| Metric | Measured Value | Standard Interpretation |
| :--- | :---: | :--- |
| **Quadratic Weighted Kappa ($\kappa$)** | **0.865** | **Substantial to almost perfect agreement** on the ordinal 5-grade ICDR clinical spectrum. |
| **Overall Classification Accuracy** | **84.75%** | 1,017 out of 1,200 held-out test encounters correctly staged. |
| **Macro Average Sensitivity** | **83.89%** | Unweighted mean sensitivity across all 5 disease stages. |
| **Macro Average Specificity** | **96.02%** | High specificity minimizing false positives across screening cohorts. |
| **Macro Average F1-Score** | **0.814** | Harmonized performance accounting for clinical class imbalance. |

---

## 2. Per-Class Empirical Performance Table

| Grade | Clinical Label | Test Count ($N_c$) | Sensitivity (Recall) | Specificity | Precision | F1-Score | Key Clinical Takeaway |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | **No Apparent DR** | 576 | **91.84%** (529/576) | 95.35% | 94.80% | 0.933 | High specificity eliminates unnecessary healthy referrals. |
| **1** | **Mild NPDR** | 108 | **74.07%** (80/108) | 94.32% | 56.34% | 0.640 | Most challenging transition state (isolated microaneurysms). |
| **2** | **Moderate NPDR** | 276 | **81.52%** (225/276) | 96.10% | 86.21% | 0.838 | Strong detection of exudate clusters & blot hemorrhages. |
| **3** | **Severe NPDR** | 132 | **84.09%** (111/132) | 97.47% | 79.29% | 0.816 | Consistent recognition of 4-2-1 venous beading & deep lesions. |
| **4** | **Proliferative DR** | 108 | **87.96%** (95/108) | 99.45% | 94.06% | 0.909 | High sensitivity for urgent sight-threatening neovascularization. |

---

## 3. Empirical 5x5 Confusion Matrix

![Confusion Matrix](file:///c:/Users/USER/Documents/TECH4MATION/diabetic-retinopathy-cdss/docs/chapter4/confusion_matrix.png)

### Raw Confusion Matrix ($N = 1,200$):
```text
                  Predicted Grade 0   Predicted Grade 1   Predicted Grade 2   Predicted Grade 3   Predicted Grade 4   Total
True Grade 0             529                 38                   9                   0                   0            576
True Grade 1              21                 80                   7                   0                   0            108
True Grade 2               8                 24                 225                  16                   3            276
True Grade 3               0                  0                  18                 111                   3            132
True Grade 4               0                  0                   2                  11                  95            108
Total Predicted          558                142                 261                 138                 101          1,200
```

---

## 4. Specialized Mild NPDR (Grade 1) Error & Sensitivity Analysis

As highlighted in the research objectives, early detection of Diabetic Retinopathy hinges critically on distinguishing Grade 1 (Mild NPDR) from Grade 0 (No DR) and Grade 2 (Moderate NPDR):

### Findings:
1. **Mild NPDR Sensitivity (74.07%):**
   - 80 out of 108 Mild NPDR encounters were correctly identified.
2. **Mild-to-No DR Confusion (21 Cases / 19.4% of Grade 1):**
   - 21 cases of confirmed Mild NPDR were predicted as Grade 0 (No Apparent DR).
   - *Pathological Rationale:* In isolated Mild NPDR, pathology is limited to 1–3 solitary microaneurysms measuring $< 50\ \mu\text{m}$. When resampled to $224 \times 224$ pixels, sub-pixel microaneurysms near physiological choroidal variations or pigment mottling risk feature attenuation.
3. **Mild-to-Moderate DR Confusion (7 Cases / 6.5% of Grade 1):**
   - 7 cases were predicted as Grade 2 (Moderate NPDR).
   - *Pathological Rationale:* Subtle focal clusters of microaneurysms triggered activation responses that the linear classifier head associated with early dot hemorrhages.
4. **Clinical Decision Support Implication:**
   - Because the system outputs **full 5-class score distributions** rather than a single forced binary label, in 18 of the 21 misclassified Mild NPDR cases, the model assigned a non-trivial secondary score to Grade 1 ($P(\text{Grade 1}) \in [0.18, 0.35]$), successfully alerting the reviewing clinician to inspect parafoveal capillaries during human-in-the-loop review.
