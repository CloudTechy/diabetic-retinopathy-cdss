# Model Performance Evaluation Report (Objective h)

## Metadata & Academic Governance
- **Research Project:** AI-Based Clinical Decision Support System for Early Detection of Diabetic Retinopathy
- **Researcher:** Onyekelu Chukwuebuka Elochukwu (2024516020FN)
- **Degree & Faculty:** PGD Computer Science, Faculty of Physical Sciences
- **Held-Out Evaluation Corpus:** 549 test images from the stratified APTOS 2019 partition
- **Evaluated Checkpoint:** SHA-256 `8ee14d7591a8e6a1b86c15416a77375a198bd49399b3977a3de79a00e3dd14fa`
- **Evaluation Mechanism:** Every metric below is recomputed from [`held_out_predictions.csv`](held_out_predictions.csv) by [`backend/scripts/analyze_clinical_metrics.py`](../../backend/scripts/analyze_clinical_metrics.py), which depends only on the Python standard library.

---

## 1. Headline Metrics on 549 Held-Out Images

| Metric | Value |
| :--- | :---: |
| **Quadratic Weighted Kappa ($\kappa$)** | **0.8777** |
| **Exact 5-Class Accuracy** | **78.69%** (432 / 549) |
| **Within-One-Grade Agreement** | **92.71%** (509 / 549) |
| **Top-2 Accuracy** | **94.35%** |
| **Macro-Averaged F1** | **0.6525** |
| Over-called (predicted grade above truth) | 11.7% |
| Under-called (predicted grade below truth) | 9.7% |

**On reading these numbers.** Exact 5-class accuracy is the weakest available summary of this model and is reported here only for completeness. The held-out cohort is 49.2% Grade 0, so accuracy is dominated by the majority class, and the ICDR scale is ordinal — confusing Grade 2 with Grade 3 is not the same error as confusing Grade 0 with Grade 4. $\kappa = 0.8777$ and the operating points in §3 are the metrics that carry clinical meaning, and they are what the discussion should be built on.

$\kappa = 0.8777$ sits within the range reported in the published literature for EfficientNet-B0 at $224 \times 224$ input resolution on APTOS 2019 without ensembling, test-time augmentation, or ordinal-regression heads. No claim of state-of-the-art performance is made or implied.

---

## 2. Confusion Matrix & Per-Class Breakdown

```text
                    Predicted Grade
                 Gr0   Gr1   Gr2   Gr3   Gr4 | Support
True Gr0 (NoDR)  267     3     0     0     0 |     270
True Gr1 (Mild)    3    40    11     0     1 |      55
True Gr2 (Mod)     3    25    80    15    27 |     150
True Gr3 (Sev)     0     1     4    17     7 |      29
True Gr4 (PDR)     0     1     7     9    28 |      45
-----------------------------------------------+--------
Total Predicted  273    70   102    41    63 |     549
```

| Grade | Class Label | Support | Sensitivity | 95% CI (Wilson) | Specificity | Precision | F1 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | No Apparent DR | 270 | **98.9%** | 96.8 – 99.6 | 97.8% | 97.8% | 0.983 |
| **1** | Mild NPDR | 55 | 72.7% | 59.8 – 82.7 | 93.9% | 57.1% | 0.640 |
| **2** | Moderate NPDR | 150 | **53.3%** | 45.4 – 61.1 | 94.5% | 78.4% | 0.635 |
| **3** | Severe NPDR | 29 | 58.6% | 40.7 – 74.5 | 95.4% | 41.5% | 0.486 |
| **4** | Proliferative DR | 45 | 62.2% | 47.6 – 74.9 | 93.1% | 44.4% | 0.518 |

Confidence intervals are Wilson score intervals. They are wide for Grades 3 and 4 because those classes carry only 29 and 45 held-out cases respectively; the point estimates for those grades should be treated as indicative rather than precise.

---

## 3. Clinical Operating Points

A screening service does not act on a five-way grade. It makes a referral decision. Collapsing the ordinal output at the two clinically meaningful thresholds gives the following.

### 3.1 Referable DR (Grade $\ge 2$) — the primary screening endpoint

| Metric | Value | 95% CI |
| :--- | :---: | :---: |
| **Sensitivity** | **86.6%** | 81.5 – 90.5 |
| **Specificity** | **96.3%** | 93.7 – 97.9 |
| Positive Predictive Value | 94.2% | — |
| Negative Predictive Value | 91.2% | — |
| Referable cases missed | **30 / 224** | — |

### 3.2 Sight-Threatening DR (Grade $\ge 3$)

| Metric | Value | 95% CI |
| :--- | :---: | :---: |
| **Sensitivity** | **82.4%** | 72.2 – 89.4 |
| **Specificity** | **91.0%** | 88.0 – 93.2 |
| Positive Predictive Value | 58.6% | — |
| Negative Predictive Value | **97.1%** | — |
| Sight-threatening cases missed | **13 / 74** | — |

### 3.3 Any DR (Grade $\ge 1$)

| Metric | Value | 95% CI |
| :--- | :---: | :---: |
| Sensitivity | 97.8% | 95.4 – 99.0 |
| Specificity | 98.9% | 96.8 – 99.6 |
| Cases missed | 6 / 279 | — |

### 3.4 Composition of the 30 referable misses

| True grade of missed case | Count |
| :--- | :---: |
| Grade 2 (Moderate NPDR) | 28 |
| Grade 3 (Severe NPDR) | 1 |
| Grade 4 (Proliferative DR) | 1 |

This distribution matters more than the count. Of the 224 referable cases, exactly **two** sight-threatening cases (one Severe NPDR, one PDR) were released as non-referable; the remaining 28 misses were Moderate NPDR, the mildest referable grade and the one with the longest safe interval to re-screening. The model's failures are concentrated where their clinical cost is lowest.

---

## 4. Error Structure

**The model errs toward over-referral.** Over-calling exceeds under-calling (11.7% vs 9.7%), and the single largest off-diagonal cell is 27 Moderate NPDR cases predicted as Proliferative DR. In a screening context this direction is the safer one: an over-called patient receives an unnecessary ophthalmology appointment, whereas an under-called patient is sent home with untreated disease.

**Grade 2 is the weakest class** at 53.3% sensitivity. Its 70 errors distribute as 27 → Grade 4, 25 → Grade 1, 15 → Grade 3, 3 → Grade 0. Notably, in 23 of the 27 Grade-2-called-Grade-4 cases, Grade 2 was still the model's *second* choice, and 122 of the 150 Grade 2 cases (81.3%) were assigned *some* referable grade. The model reliably recognises that these eyes need referral; it is the precise severity stratification it gets wrong.

**Grade 0 is near-perfect** at 98.9% sensitivity with 97.8% precision. Because Grade 0 is 49.2% of a screening population, this is what would make the system operationally useful: it clears the healthy majority with high confidence and concentrates clinician attention on the remainder.

---

## 5. Data Integrity Audit

### 5.1 Argmax invariant
All 549 prediction rows satisfy `predicted_grade == argmax(score_grade_0..4)`, with zero discrepancies.

### 5.2 Byte-level duplicate leakage — disclosed

The pipeline partitions by `duplicate_group_id`, derived from APTOS's `duplicated_info.csv`. **That file is not distributed with the Kaggle competition download.** When it is absent, the fallback assigns every image its own group, so the de-duplication step runs but groups nothing. The committed manifest confirms this: 3,662 groups for 3,662 images.

Auditing the committed SHA-256 column directly instead reveals:

| Finding | Value |
| :--- | :---: |
| Held-out images byte-identical to a training image | **27 / 549 (4.92%)** |
| Validation images byte-identical to a training image | 17 / 550 |
| Distribution of affected held-out images | Gr1: 6, Gr2: 16, Gr3: 2, Gr4: 3 |

**Measured effect on the reported result: none.**

| Subset | N | Accuracy | QWK |
| :--- | :---: | :---: | :---: |
| Affected (duplicated) images | 27 | 77.78% | — |
| Clean images | 522 | **78.74%** | **0.877818** |
| Full held-out cohort | 549 | 78.69% | 0.877747 |

The model performs marginally *worse* on the duplicated images than on the clean ones, and the clean-subset $\kappa$ (0.877818) is indistinguishable from the full-cohort $\kappa$ (0.877747). The reported metrics are therefore not inflated by memorised duplicates. The condition is disclosed here rather than silently corrected, and the clean-subset figures are available for any reviewer who prefers them. Re-running the split with `duplicated_info.csv` present is recorded as future work in [`known_limitations.md`](known_limitations.md).

---

## 6. Reproduction

```bash
# Recompute every number in this report from the committed predictions.
python backend/scripts/analyze_clinical_metrics.py
```

Requires only the Python standard library. Writes [`clinical_metrics.json`](clinical_metrics.json).
