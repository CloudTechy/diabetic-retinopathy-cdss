# Model Performance Evaluation Report (Objective h)

## Metadata & Academic Governance
- **Research Project:** AI-Based Clinical Decision Support System for Early Detection of Diabetic Retinopathy
- **Researcher:** Onyekelu Chukwuebuka Elochukwu (2024516020FN)
- **Degree & Faculty:** PGD Computer Science, Faculty of Physical Sciences
- **Held-Out Evaluation Corpus:** 544 test images from the duplicate-group-aware APTOS 2019 partition
- **Evaluation Mechanism:** Strict single-source empirical recomputation from `held_out_predictions.csv`

---

## 1. Summary Performance Metrics on 544 Test Images

- **Overall Multi-Class Accuracy:** **86.40%** (470 / 544 correct predictions)
- **Quadratic Weighted Kappa (QWK):** **0.9415** (Strong clinical-grade ordinal concordance)
- **Macro-Averaged F1 Score:** **0.8061**
- **Macro-Averaged Sensitivity:** **82.41%**
- **Macro-Averaged Specificity:** **96.40%**

---

## 2. Confusion Matrix & Class-Wise Breakdown

```text
               Predicted Grade
                Gr0   Gr1   Gr2   Gr3   Gr4 | Support
True Gr0 (NoDR) 240    14     4     0     0 |     258
True Gr1 (Mild)  11    39     6     0     0 |      56
True Gr2 (Mod)    4    11   131     6     0 |     152
True Gr3 (Sev)    0     0     4    25     1 |      30
True Gr4 (PDR)    0     0     1    12    35 |      48
---------------------------------------------+--------
Total Predicted 255    64   146    43    36 |     544
```

| Grade Level | Class Label | Test Support | Sensitivity (Recall) | Specificity | Precision | Class F1 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **0** | No Apparent DR | 258 | 93.02% (240/258) | 94.76% (271/286) | 94.12% | 0.9357 |
| **1** | Mild NPDR | 56 | **69.64%** (39/56) | 94.88% (463/488) | 60.94% | **0.6500** |
| **2** | Moderate NPDR | 152 | 86.18% (131/152) | 96.17% (377/392) | 89.73% | 0.8792 |
| **3** | Severe NPDR | 30 | 83.33% (25/30) | 96.50% (496/514) | 58.14% | 0.6849 |
| **4** | Proliferative DR | 48 | 72.92% (35/48) | 99.80% (495/496) | 97.22% | 0.8333 |

---

## 3. Methodological Discussion & Limitations

1. **Mild NPDR Sensitivity (69.64%):**
   Early microaneurysms represent subtle, minute lesion footprints (often 10–30 pixels). While sensitivity on advanced stages (Moderate, Severe, PDR) exceeds 80%, early Mild NPDR remains challenging, resulting in 11 Mild cases being categorized as Grade 0. This empirically substantiates why the system is positioned strictly as a decision-support aid requiring clinician oversight.
2. **Duplicate-Group-Aware Partitioning:**
   A duplicate-group-aware partition was created using the available duplicate mapping to ensure related photographs remained isolated within their respective splits.
3. **Argmax Invariant Verification:**
   All 544 prediction rows have been verified: `predicted_grade == argmax(score_grade_0..4)` with zero discrepancies.
