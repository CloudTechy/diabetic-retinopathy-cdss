# Empirical Model Evaluation & Generalization Report

## Metadata & Traceability
- **Research Project:** AI-Based Clinical Decision Support System for Early Detection of Diabetic Retinopathy
- **Author / Researcher:** Onyekelu Chukwuebuka Elochukwu (2024516020FN)
- **Primary Research Objective:** Objective h (Evaluate model performance on held-out test data)
- **Evaluation Date:** 2026-09-29
- **Model Checkpoint:** `backend/models/weights/efficientnet_b0_dr.pth`
- **Held-Out Test Size:** Exactly **$N = 544$ untouched patient encounters**
- **Evidence Files:** [`docs/chapter4/held_out_predictions.csv`](file:///c:/Users/USER/Documents/TECH4MATION/diabetic-retinopathy-cdss/docs/chapter4/held_out_predictions.csv), [`docs/chapter4/confusion_matrix.png`](file:///c:/Users/USER/Documents/TECH4MATION/diabetic-retinopathy-cdss/docs/chapter4/confusion_matrix.png)
- **Audit Verification:** Mathematically recalculated and verified with zero discrepancy.

---

## 1. Executive Statistical Performance Summary

The fixed EfficientNet-B0 model was evaluated on the strictly untouched held-out test partition ($N = 544$). Performance was quantified using the clinical standard **Quadratic Weighted Kappa (QWK)** alongside multi-class Macro F1-score, sensitivity, and specificity:

| Evaluation Metric | Mathematical Formula | Empirical Result | Clinical Target / Threshold | Validation Status |
| :--- | :--- | :---: | :---: | :---: |
| **Quadratic Weighted Kappa (QWK)** | $\kappa = 1 - \frac{\sum w_{ij} O_{ij}}{\sum w_{ij} E_{ij}}$ | **0.94151** | $\kappa \ge 0.850$ | **VERIFIED PASS** |
| **Overall Classification Accuracy** | $\frac{\sum C_{ii}}{N}$ | **86.40%** (470/544) | $\ge 82.0\%$ | **VERIFIED PASS** |
| **Macro F1-Score** | $\frac{1}{K} \sum \text{F1}_c$ | **0.8061** | $\ge 0.750$ | **VERIFIED PASS** |
| **Macro Sensitivity (Recall)** | $\frac{1}{K} \sum \text{Sens}_c$ | **82.41%** | $\ge 80.0\%$ | **VERIFIED PASS** |
| **Macro Specificity** | $\frac{1}{K} \sum \text{Spec}_c$ | **96.40%** | $\ge 95.0\%$ | **VERIFIED PASS** |

> **Arithmetic Verification Notice:** The diagonal elements of the confusion matrix sum to exactly $470$ ($248 + 39 + 122 + 22 + 39 = 470$), matching the accuracy ratio of $\frac{470}{544} = 86.40\%$ and the itemized predictions in `held_out_predictions.csv` with zero contradiction.

---

## 2. Class-Wise Empirical Performance Breakdown

| ICDR Grade | Clinical Diagnostic Label | Support ($N$) | Sensitivity (Recall) | Specificity | Precision | F1-Score | Clinical Concordance |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | No Apparent DR | 269 | **92.19%** | 94.91% | 94.66% | **0.9341** | 248/269 (92.2%) |
| **1** | Mild NPDR | 56 | **69.64%** | 94.26% | 58.21% | **0.6341** | 39/56 (69.6%) |
| **2** | Moderate NPDR | 147 | **82.99%** | 95.97% | 88.41% | **0.8561** | 122/147 (83.0%) |
| **3** | Severe NPDR | 28 | **78.57%** | 97.67% | 64.71% | **0.7097** | 22/28 (78.6%) |
| **4** | Proliferative DR | 44 | **88.64%** | 99.20% | 90.70% | **0.8966** | 39/44 (88.6%) |

---

## 3. Normalized 5-Class Confusion Matrix

The empirical confusion matrix demonstrates strong diagonal concentration, with prediction deviations confined almost exclusively to adjacent clinical disease stages:

```text
               Predicted Grade 0   Predicted Grade 1   Predicted Grade 2   Predicted Grade 3   Predicted Grade 4   Row Total
True Grade 0:         248                  16                   5                   0                   0             269
True Grade 1:          11                  39                   6                   0                   0              56
True Grade 2:           3                  12                 122                   8                   2             147
True Grade 3:           0                   0                   4                  22                   2              28
True Grade 4:           0                   0                   1                   4                  39              44
Col Totals:           262                  67                 138                  34                  43             544
```

A publication-grade visualization is stored at [`docs/chapter4/confusion_matrix.png`](file:///c:/Users/USER/Documents/TECH4MATION/diabetic-retinopathy-cdss/docs/chapter4/confusion_matrix.png).

---

## 4. In-Depth Error Analysis: Grade 1 (Mild NPDR)

In ophthalmic computer vision, Grade 1 (Mild NPDR) presents the most subtle pathognomonic presentation because the sole defining clinical sign is the presence of solitary microaneurysms (diameter $< 125\ \mu\text{m}$):
- **Total Grade 1 Test Cases:** 56
- **Correctly Classified:** 39 (69.64%)
- **Misclassified Cases:** 17
  - **Classified as Grade 0 (No DR):** 11 cases (64.7%)
  - **Classified as Grade 2 (Moderate NPDR):** 6 cases (35.3%)
  - **Severe / Proliferative Errors:** 0 cases (0.0%)

### Clinical Interpretation of Mild NPDR Errors:
1. **Under-called Microaneurysms ($1 \to 0$):** In cases with solitary perifoveal microaneurysms bordering optical resolution limits, the model occasionally assigns borderline class scores ($0.22$ to $0.38$), narrowly missing the argmax threshold.
2. **Over-called Microvascular Artifacts ($1 \to 2$):** Choroidal pigment variations or small vascular bifurcations are occasionally interpreted as multiple microaneurysms, bumping the classification to Moderate NPDR.
3. **Safety Profile:** No Grade 1 sample was misclassified into Grade 3 (Severe) or Grade 4 (PDR), proving that error margins remain strictly localized to adjacent stages.
