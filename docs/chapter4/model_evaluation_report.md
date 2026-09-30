# Model Performance Evaluation Report (Objective h)

## Metadata & Academic Governance
- **Research Project:** AI-Based Clinical Decision Support System for Early Detection of Diabetic Retinopathy
- **Researcher:** Onyekelu Chukwuebuka Elochukwu (2024516020FN)
- **Degree & Faculty:** PGD Computer Science, Faculty of Physical Sciences
- **Held-Out Evaluation Corpus:** 549 test images
- **Evaluated Checkpoint:** SHA-256 `8ee14d7591a8e6a1b86c15416a77375a198bd49399b3977a3de79a00e3dd14fa`
- **Evaluation Mechanism:** Every metric below is recomputed from [`held_out_predictions.csv`](held_out_predictions.csv) by [`analyze_clinical_metrics.py`](../../backend/scripts/analyze_clinical_metrics.py), which depends only on the Python standard library.

---

> [!WARNING]
> ## These results are superseded. Do not cite them in Chapter Four.
>
> The partition they were produced from is contaminated. Independent audit of
> [`dataset_split_manifest.csv`](dataset_split_manifest.csv) found:
>
> - 3,662 records over only **3,534 unique image hashes**
> - **48 exact-duplicate groups spanning more than one partition** — 27 test images byte-identical to a training image, 17 validation to training, 6 test to validation
> - **30 exact-duplicate groups carrying conflicting severity labels**
>
> A test set in that condition cannot be described as held out. Comparing scores
> on the contaminated subset against the rest does **not** repair it: validation
> contamination influences which checkpoint is selected, and conflicting labels
> put contradictory supervision into training.
>
> A leakage-free split has been implemented in
> [`build_clean_split.py`](../../backend/scripts/build_clean_split.py) — de-duplication
> keyed on the SHA-256 of image bytes, label-conflicting groups excluded, one
> representative per group, and an assertion of zero hash overlap between
> partitions. It yields **3,504 unique, non-conflicting images**.
>
> **The model must be retrained once on that split and every artefact
> regenerated.** The figures below are retained only so the contaminated and
> clean runs can be compared, and are labelled throughout as provisional.

---

## 1. Headline Metrics (provisional, contaminated split)

| Metric | Value |
| :--- | :---: |
| **Quadratic Weighted Kappa ($\kappa$)** | 0.8777 |
| **Exact 5-Class Accuracy** | 78.69% (432 / 549) |
| **Within-One-Grade Agreement** | 92.71% (509 / 549) |
| **Macro-Averaged F1** | 0.6525 |
| Over-called (predicted grade above truth) | 11.7% |
| Under-called (predicted grade below truth) | 9.7% |

**On reading these numbers.** The held-out cohort is 49.2% Grade 0, so exact accuracy is dominated by the majority class, and the ICDR scale is ordinal — confusing Grade 2 with Grade 3 is not the same error as confusing Grade 0 with Grade 4. $\kappa$ and the per-class breakdown in §2 are the informative figures.

$\kappa = 0.8777$ sits within the range reported for EfficientNet-B0 at $224 \times 224$ on APTOS 2019 without ensembling, test-time augmentation or ordinal-regression heads. No claim of state-of-the-art performance is made.

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

Confidence intervals are Wilson score intervals. They are wide for Grades 3 and 4 because those classes carry only 29 and 45 held-out cases; those point estimates are indicative rather than precise.

---

## 3. Classification Behaviour

This section describes what the classifier does with the image data. It makes no statement about clinical management, referral, or patient outcome — none of which is established by retrospective image classification, and none of which is within this system's scope.

### 3.1 Mild NPDR (Grade 1) recognition — 72.7%

Grade 1 is the transition the ICDR scale places between a normal fundus and established retinopathy, and it is the class the research objectives single out. Sensitivity is **72.7%** (40 / 55).

Its 15 errors distribute as **11 → Grade 2**, **3 → Grade 0**, **1 → Grade 4**.

### 3.2 Mild NPDR confused with No DR — 3 cases

Only **3** Grade 1 images were assigned Grade 0. In each, the Grade 1 score was retained as the second-ranked class (0.092, 0.206, 0.298), so the distinction was represented in the output distribution rather than absent from it.

A plausible mechanism is resolution: early microaneurysms measure roughly 25–50 µm and occupy few pixels on a multi-megapixel sensor, and resizing to $224 \times 224$ applies smoothing that can attenuate an isolated lesion's local contrast. This is **consistent with** the error pattern, not demonstrated by it — establishing it would require an ablation at higher input resolution, which was not run.

### 3.3 Moderate NPDR (Grade 2) is the weakest class — 53.3%

Its 70 errors distribute as 27 → Grade 4, 25 → Grade 1, 15 → Grade 3, 3 → Grade 0.

Grade 2 is the ICDR scale's widest and least sharply bounded category — "more than microaneurysms but less than severe" — and its boundaries with Grade 1 and Grade 3 turn on lesion count and distribution, which is the information most degraded by downsampling.

In 23 of the 27 Grade-2-called-Grade-4 cases, Grade 2 remained the second-ranked class.

### 3.4 Grade 0 recognition — 98.9%

Sensitivity 98.9% with 97.8% precision, the strongest of the five classes. Grade 0 is 49.2% of this cohort, so this is also the class with the most training support.

### 3.5 Ordinal error structure

Predicted grades exceed the reference grade in 11.7% of cases and fall below it in 9.7%. Within-one-grade agreement is 92.71%, and top-2 accuracy is 94.35% — the reference grade is within the model's two highest-scoring classes for most images it grades incorrectly.

---

## 4. Secondary Exploratory Analysis: Binary Collapse

> [!IMPORTANT]
> **Exploratory only.** The figures in this section are arithmetic collapses of
> the five-class output. They are **not** system referral decisions, **not**
> evidence of clinical safety, and **not** a demonstration of clinical
> usefulness. The system produces preliminary decision-support information for
> a reviewing clinician; it does not make referral determinations, and no
> retrospective image-classification result can establish that it should.
>
> Reported because the binary collapse is conventional in the DR literature and
> makes this work comparable to it — nothing more.

| Collapse | Sensitivity | 95% CI | Specificity | 95% CI |
| :--- | :---: | :---: | :---: | :---: |
| Grade $\ge 1$ | 97.8% | 95.4 – 99.0 | 98.9% | 96.8 – 99.6 |
| Grade $\ge 2$ | 86.6% | 81.5 – 90.5 | 96.3% | 93.7 – 97.9 |
| Grade $\ge 3$ | 82.4% | 72.2 – 89.4 | 91.0% | 88.0 – 93.2 |

Distribution of the 30 grade-$\ge 2$ cases assigned a grade below 2: 28 were Grade 2, 1 was Grade 3, 1 was Grade 4.

---

## 5. Data Integrity Audit

### 5.1 Argmax invariant
All 549 prediction rows satisfy `predicted_grade == argmax(score_grade_0..4)`, with zero discrepancies.

### 5.2 Partition contamination — the reason these results are provisional

| Finding | Value |
| :--- | :---: |
| Records / unique image hashes | 3,662 / **3,534** |
| Duplicate groups spanning partitions | **48** |
| Test images byte-identical to a training image | 27 |
| Validation images byte-identical to a training image | 17 |
| Test images byte-identical to a validation image | 6 |
| Duplicate groups with conflicting labels | **30** |

Reproduce with `python backend/scripts/analyze_clinical_metrics.py` (leakage audit section).

An earlier revision of this report argued the contamination was immaterial because accuracy on the 27 affected test images (77.78%) was no higher than on the remaining 522 (78.74%). **That argument is insufficient and has been withdrawn.** It addresses only test-set contamination, and not:

- **validation contamination** (17 images), which influences which epoch's checkpoint is selected; or
- **conflicting labels** (30 groups), which place contradictory supervision in training.

Neither effect is observable in a held-out score comparison. The correct remedy is a clean split and a retrain, not a post-hoc argument.

---

## 6. Reproduction

```bash
# Recompute every number in this report from the committed predictions
python backend/scripts/analyze_clinical_metrics.py

# Build the corrected, leakage-free split (requires the APTOS images)
python backend/scripts/build_clean_split.py aptos2019/train_images
```
