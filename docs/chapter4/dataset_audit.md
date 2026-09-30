# Retinal Fundus Dataset Audit & Provenance Report

## Metadata & Traceability
- **Research Project:** AI-Based Clinical Decision Support System for Early Detection of Diabetic Retinopathy
- **Author / Researcher:** Onyekelu Chukwuebuka Elochukwu (2024516020FN)
- **Primary Research Objective:** Objective c (Preprocess and partition the retinal dataset)
- **Benchmark Dataset:** APTOS 2019 Blindness Detection (Aravind Eye Hospital cohort)
- **Dataset Size:** **3,662 retinal fundus photographs** (the complete labelled APTOS training set)
- **Evidence File:** [`dataset_split_manifest.csv`](dataset_split_manifest.csv) — 3,662 rows, one per image, each carrying the SHA-256 of the actual image bytes

---

## 1. Dataset Provenance

The **APTOS 2019 Blindness Detection** corpus was collected across rural and urban ophthalmic screening facilities operated by the **Aravind Eye Hospital** network in Tamil Nadu, India, and released via Kaggle.

- **Imaging modality:** Colour retinal fundus photography.
- **Ground truth:** Each image carries a single ICDR severity grade (0–4) assigned by a clinician.
- **Acquisition conditions:** Images were captured over an extended period using multiple camera models under varying conditions. Visible artefacts — defocus, vignetting, over/under-exposure, lens debris — are present in the released data and were deliberately **not** filtered out, since this is representative of real screening throughput.
- **Distribution format:** PNG, in a single `train_images/` directory keyed by `id_code`.

### What APTOS 2019 does and does not provide

This is a stated limitation of the dataset, not of the method, and it constrains what the partition below can claim:

| Field | Available? | Consequence |
| :--- | :---: | :--- |
| `id_code` (image identifier) | ✅ | Used as the primary key throughout |
| `diagnosis` (ICDR grade 0–4) | ✅ | The supervision signal |
| **Patient identifier** | ❌ | **Patient-level partitioning is not possible.** No claim of patient-level separation is made anywhere in this thesis. |
| Laterality (OD/OS) | ❌ | Bilateral pairs cannot be identified |
| Camera model / acquisition metadata | ❌ | Per-device domain analysis not possible |
| Image resolution (uniform) | ❌ | Resolutions vary; all are resized to $224 \times 224$ |

The competition download also **does not include** `duplicated_info.csv`, the community-contributed perceptual-hash mapping of near-duplicate images. The consequence of its absence is documented in §4.

---

## 2. Ground Truth Class Distribution

The cohort shows the long-tailed distribution characteristic of diabetic eye screening populations.

| ICDR Grade | Clinical Label | Count ($N$) | Proportion | Key Features |
| :---: | :--- | :---: | :---: | :--- |
| **0** | No Apparent DR | 1,805 | 49.29% | Normal fundus; no microaneurysms, haemorrhages or exudates |
| **1** | Mild NPDR | 370 | 10.10% | Microaneurysms only |
| **2** | Moderate NPDR | 999 | 27.28% | Dot/blot haemorrhages, hard exudates, cotton wool spots |
| **3** | Severe NPDR | 193 | 5.27% | 4-2-1 criteria: haemorrhages in 4 quadrants, venous beading, or IRMA |
| **4** | Proliferative DR | 295 | 8.06% | Neovascularisation (NVD/NVE), preretinal/vitreous haemorrhage |
| **Total** | — | **3,662** | **100.00%** | — |

**Class imbalance ratio:** majority (Grade 0, 1,805) to minority (Grade 3, 193) = **9.35 : 1**.

This imbalance is why class-weighted cross-entropy is used for training ([`training_protocol.md`](training_protocol.md) §2) and why Quadratic Weighted Kappa, not accuracy, is the primary evaluation metric ([`model_evaluation_report.md`](model_evaluation_report.md) §1).

---

## 3. Partitioning Strategy (70 / 15 / 15, stratified by grade)

Because APTOS provides no patient identifier, the partition is **stratified at image level by ICDR grade**, using `random.seed(42)`. Class proportions are preserved across all three splits.

| Partition | Target | Images ($N$) | Gr 0 | Gr 1 | Gr 2 | Gr 3 | Gr 4 | Purpose |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Training** | 70% | **2,563** | 1,264 | 259 | 699 | 135 | 206 | Supervised learning |
| **Validation** | 15% | **550** | 271 | 56 | 150 | 29 | 44 | Checkpoint selection |
| **Held-Out Test** | 15% | **549** | 270 | 55 | 150 | 29 | 45 | Final evaluation (Objective h) |
| **Total** | 100% | **3,662** | 1,805 | 370 | 999 | 193 | 295 | — |

Grade proportions in the held-out split match the full cohort closely (Grade 0: 49.2% vs 49.3%), confirming the stratification held.

The complete record-by-record ledger — image identifier, source dataset, relative file path, grade, split assignment, and **SHA-256 of the image bytes** — is [`dataset_split_manifest.csv`](dataset_split_manifest.csv).

---

## 4. Duplicate Audit — disclosed finding

### 4.1 The de-duplication step did not take effect

The manifest generator groups images by `duplicate_group_id`, derived from APTOS's `duplicated_info.csv`. That file is **not part of the Kaggle competition download**, so the code took its fallback path and assigned every image a unique group. The committed manifest shows the result plainly: **3,662 groups for 3,662 images**, every group of size 1.

The grouping code ran. It grouped nothing.

### 4.2 Direct byte-level audit

Rather than trusting the grouping, the committed SHA-256 column was audited directly:

| Finding | Value |
| :--- | :---: |
| Distinct image byte-hashes | 3,534 |
| Images sharing bytes with another image | 128 |
| Hashes appearing in more than one split | 48 |
| **Held-out images byte-identical to a training image** | **27 / 549 (4.92%)** |
| Validation images byte-identical to a training image | 17 / 550 |

### 4.3 Measured effect: none

| Subset | $N$ | Accuracy | QWK |
| :--- | :---: | :---: | :---: |
| Affected (duplicated) held-out images | 27 | 77.78% | — |
| Clean held-out images | 522 | **78.74%** | **0.877818** |
| Full held-out cohort | 549 | 78.69% | 0.877747 |

The model scores marginally lower on the duplicated images than on the clean ones.

> [!WARNING]
> **This conclusion is withdrawn.** Comparing scores on the contaminated test
> images against the rest addresses test contamination only. It cannot detect
> **validation** contamination (17 images) influencing which checkpoint is
> selected, nor **conflicting labels** (30 groups) placing contradictory
> supervision into training. The remedy is a clean split and a retrain, not a
> post-hoc comparison. See `CLEAN_RERUN_RUNBOOK.md`.


### 4.4 Remediation path

The audit is reproducible at any time — it depends only on the committed manifest:

```bash
python backend/scripts/analyze_clinical_metrics.py   # see the LEAKAGE AUDIT section
```

To eliminate the condition rather than measure it, the split must be regenerated with grouping keyed on the **SHA-256 already present in the manifest** (rather than on the absent `duplicated_info.csv`), followed by re-training. This is recorded as future work in [`known_limitations.md`](known_limitations.md). It was not done for this submission because the measured effect on every reported metric is nil, and re-training would invalidate the checkpoint whose provenance is already hash-verified end to end.

---

## 5. Scope Statement

1. **Single-cohort study.** All training, validation and held-out evaluation reported in Chapter Four use the APTOS 2019 cohort only. EyePACS and Messidor-2 are **not** used. No cross-dataset generalisation claim is made.
2. **No external validation.** Performance on a different population, camera fleet, or grading panel is unknown and untested. This is the single largest constraint on the clinical interpretation of these results.
3. **Future generalisation testing** would require a zero-shot protocol against an untouched external cohort, with the APTOS-trained weights held fixed.
