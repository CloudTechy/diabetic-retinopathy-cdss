# Model Performance Evaluation Report (Objective h)

## Metadata & Academic Governance
- **Research Project:** AI-Based Clinical Decision Support System for Early Detection of Diabetic Retinopathy
- **Researcher:** Onyekelu Chukwuebuka Elochukwu (2024516020FN)
- **Degree & Faculty:** PGD Computer Science, Faculty of Physical Sciences
- **Held-Out Evaluation Corpus:** 525 test images, leakage-free
- **Evaluated Checkpoint:** SHA-256 `67d0b89641f08057126dd411e380b25575ef29f71ae37ee5796d472d9203dbf7`
- **Evaluation Mechanism:** Every metric below is recomputed from [`held_out_predictions.csv`](held_out_predictions.csv) by [`analyze_clinical_metrics.py`](../../backend/scripts/analyze_clinical_metrics.py), which depends only on the Python standard library.

---

## 1. Partition Integrity

This evaluation uses the corrected split produced by [`build_clean_split.py`](../../backend/scripts/build_clean_split.py). The audit trail is [`dataset_split_audit.json`](dataset_split_audit.json).

| Property | Value |
| :--- | :---: |
| Labelled records in APTOS 2019 | 3,662 |
| Unique image hashes | 3,534 |
| Exact duplicate copies removed | 128 |
| Duplicate groups with conflicting labels, **excluded** | 30 (62 images) |
| **Unique, non-conflicting images retained** | **3,504** |
| Train / Validation / Test | 2,453 / 526 / **525** |
| **Hash overlap between partitions** | **0 (asserted before training)** |
| Held-out images byte-identical to a training image | **0 / 525 (0.0%)** |

The final row is recomputed independently of the split builder by the leakage audit inside `analyze_clinical_metrics.py`. The previous partition, on which all superseded results were produced, returned 27 / 549.

---

## 2. Headline Metrics

| Metric | Value |
| :--- | :---: |
| **Quadratic Weighted Kappa ($\kappa$)** | **0.8658** |
| **Exact 5-Class Accuracy** | **84.00%** (441 / 525) |
| **Within-One-Grade Agreement** | 93.71% (492 / 525) |
| **Macro-Averaged F1** | 0.7031 |
| Over-called (predicted grade above truth) | 8.6% |
| Under-called (predicted grade below truth) | 7.4% |

**On reading these numbers.** The held-out cohort is 51.4% Grade 0, so exact accuracy is dominated by the majority class, and the ICDR scale is ordinal — confusing Grade 2 with Grade 3 is not the same error as confusing Grade 0 with Grade 4. $\kappa$ and the per-class breakdown in §3 are the informative figures.

$\kappa = 0.8658$ sits within the range reported for EfficientNet-B0 at $224 \times 224$ on APTOS 2019 without ensembling, test-time augmentation or ordinal-regression heads. The QWK difference of 0.0472 between the validation peak (0.9130) and the test set indicates slight overfitting but remains stable. No claim of state-of-the-art performance is made.

### 2.1 Comparison with the superseded contaminated run

Both runs are reported because the comparison is the evidence that the contamination mattered — not because the clean run is being presented as an improvement.

| Metric | Contaminated (N=549) | Clean (N=525) | Change |
| :--- | ---: | ---: | :--- |
| Accuracy | 78.69% | **84.00%** | +5.31 |
| Macro F1 | 0.6525 | **0.7031** | +0.0506 |
| Within-one-grade | 92.71% | **93.71%** | +1.00 |
| QWK | 0.8777 | 0.8658 | −0.0119 |
| Moderate NPDR sensitivity | 53.3% | **75.5%** | +22.2 |
| Held-out images copied from training | 27 / 549 | **0 / 525** | — |

> [!IMPORTANT]
> **This is not a controlled comparison and must not be reported as one.** The two runs use different test sets — 549 versus 525 images, of different composition — so the difference confounds the change in partition with the change in cohort. It is indicative of the contamination's effect, not a measurement of it.
>
> It is also **not** evidence that removing leakage improves a model. The most likely driver of the gains is the exclusion of 30 duplicate groups carrying contradictory severity labels, which removed contradictory supervision from training. That is a hypothesis consistent with the Grade 2 result, not a demonstrated cause.

---

## 3. Confusion Matrix & Per-Class Breakdown

```text
                    Predicted Grade
                 Gr0   Gr1   Gr2   Gr3   Gr4 | Support
True Gr0 (NoDR)  266     2     1     0     1 |     270
True Gr1 (Mild)    5    33    12     0     0 |      50
True Gr2 (Mod)     0    11   105     8    15 |     139
True Gr3 (Sev)     0     0     5    15     6 |      26
True Gr4 (PDR)     1     6     9     2    22 |      40
-----------------------------------------------+--------
Total Predicted  272    52   132    25    44 |     525
```

| Grade | Class Label | Support | Sensitivity | 95% CI (Wilson) | Specificity | Precision | F1 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | No Apparent DR | 270 | **98.5%** | 96.3 – 99.4 | 97.7% | 97.8% | 0.982 |
| **1** | Mild NPDR | 50 | 66.0% | 52.2 – 77.6 | 96.0% | 63.5% | 0.647 |
| **2** | Moderate NPDR | 139 | **75.5%** | 67.8 – 81.9 | 93.0% | 79.5% | 0.775 |
| **3** | Severe NPDR | 26 | 57.7% | 38.9 – 74.5 | 98.0% | 60.0% | 0.588 |
| **4** | Proliferative DR | 40 | 55.0% | 39.8 – 69.3 | 95.5% | 50.0% | 0.524 |

Confidence intervals are Wilson score intervals. They are wide for Grades 3 and 4 because those classes carry only 26 and 40 held-out cases; those point estimates are indicative rather than precise.

---

## 4. Classification Behaviour

This section describes what the classifier does with the image data. It makes no statement about clinical management, referral, or patient outcome — none of which is established by retrospective image classification, and none of which is within this system's scope.

### 4.1 Mild NPDR (Grade 1) recognition — 66.0%

Grade 1 is the transition the ICDR scale places between a normal fundus and established retinopathy, and it is the class the research objectives single out. Sensitivity is **66.0%** (33 / 50), with a wide interval of 52.2 – 77.6.

Its 17 errors distribute as **12 → Grade 2** and **5 → Grade 0**. No Grade 1 image was placed at Grade 3 or Grade 4.

### 4.2 Mild NPDR confused with No DR — 5 cases

Five Grade 1 images were assigned Grade 0. This is the error the objectives care about most, because it is the one that would place an eye with early retinopathy in the same bucket as a normal one.

A plausible mechanism is resolution: early microaneurysms measure roughly 25–50 µm and occupy few pixels on a multi-megapixel sensor, and resizing to $224 \times 224$ applies smoothing that can attenuate an isolated lesion's local contrast. This is **consistent with** the error pattern, not demonstrated by it — establishing it would require an ablation at higher input resolution, which was not run.

### 4.3 Moderate NPDR (Grade 2) — 75.5%

Grade 2 was the weakest class in the superseded run at 53.3% sensitivity. It is now **75.5%** (105 / 139), the largest single change between the two runs.

Its 34 errors distribute as 15 → Grade 4, 11 → Grade 1, 8 → Grade 3. No Grade 2 image was placed at Grade 0.

Grade 2 is the ICDR scale's widest and least sharply bounded category — "more than microaneurysms but less than severe" — and 30 of the duplicate groups excluded from the clean split disagreed about exactly these boundaries: 9 disagreed between Grades 2 and 3, 8 between Grades 2 and 4, 5 between Grades 1 and 2.

### 4.4 Proliferative DR (Grade 4) — 55.0%

The weakest class in the clean run, at 55.0% (22 / 40) with an interval of 39.8 – 69.3.

Its errors are also the most ordinally distant in the matrix: **6 → Grade 1** and **1 → Grade 0**, alongside 9 → Grade 2 and 2 → Grade 3. Those seven far-displaced errors are the principal reason $\kappa$ declined slightly while every other headline metric rose — QWK penalises displacement quadratically, so a handful of three- and four-grade errors outweighs many more one-grade improvements elsewhere.

### 4.5 Grade 0 recognition — 98.5%

Sensitivity 98.5% with 97.8% precision, the strongest of the five classes. Grade 0 is 51.4% of this cohort, so it is also the class with the most training support.

### 4.6 Ordinal error structure

Predicted grades exceed the reference grade in 8.6% of cases and fall below it in 7.4%. Within-one-grade agreement is 93.71%.

---

## 5. Secondary Exploratory Analysis: Binary Collapse

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

| Collapse | Sensitivity | 95% CI | Specificity | 95% CI | Cases below the cut |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Grade $\ge 1$ | 97.7% | 95.0 – 98.9 | 98.5% | 96.3 – 99.4 | 6 |
| Grade $\ge 2$ | 91.2% | 86.5 – 94.4 | 95.6% | 92.8 – 97.4 | 18 |
| Grade $\ge 3$ | 68.2% | 56.2 – 78.2 | 94.8% | 92.3 – 96.5 | 21 |

---

## 6. Data Integrity Audit

### 6.1 Argmax invariant
All 525 prediction rows satisfy `predicted_grade == argmax(score_grade_0..4)`, with zero discrepancies.

### 6.2 Leakage
Zero held-out images are byte-identical to any training image (0 / 525). Partition hash overlap was asserted to be zero by `build_clean_split.py` before training began, and is confirmed after the fact by an independent recomputation.

---

## 7. Training Provenance

| Property | Value |
| :--- | :--- |
| Architecture | EfficientNet-B0, ImageNet `IMAGENET1K_V1` initialisation |
| Parameters | 4,013,953 |
| Epochs | 15 |
| Best epoch | **14** (validation $\kappa$ = 0.9130) |
| Optimiser / schedule | AdamW, lr 1e-4, weight decay 1e-4, CosineAnnealingLR |
| Hardware | Tesla T4, PyTorch 2.11.0+cu128 |
| Wall-clock | 2,761.6 s (~46 min) |
| Checkpoint | 15.60 MB, SHA-256 `67d0b896…` |

Transcript: [`training_execution.log`](training_execution.log). Per-epoch history: [`epoch_history.csv`](epoch_history.csv).

---

## 8. Reproduction

```bash
# Recompute every number in this report from the committed predictions
python backend/scripts/analyze_clinical_metrics.py

# Rebuild the leakage-free split (requires the APTOS images)
python backend/scripts/build_clean_split.py aptos2019/train_images
```

