# Retinal Fundus Dataset Audit & Provenance Report

## Metadata & Traceability
- **Research Project:** AI-Based Clinical Decision Support System for Early Detection of Diabetic Retinopathy
- **Author / Researcher:** Onyekelu Chukwuebuka Elochukwu (2024516020FN)
- **Primary Research Objective:** Objective c (Dataset acquisition, preprocessing & partitioning)
- **Benchmark Dataset:** APTOS 2019 Blindness Detection (Aravind Eye Hospital cohort)
- **Dataset Size:** **3,662 retinal fundus photographs** as published by APTOS; **3,504** after byte-identical de-duplication and the removal of conflicting-label groups (§4)
- **Evidence File:** [`dataset_split_manifest.csv`](dataset_split_manifest.csv) — **3,504 rows**, one per retained image, each carrying the SHA-256 of the actual image bytes. That column is what [`corpus_guard.py`](../../backend/scripts/corpus_guard.py) checks a directory against before any script measures it.

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
| **Training** | 70% | **2,453** | 1,257 | 237 | 645 | 124 | 190 | Supervised learning |
| **Validation** | 15% | **526** | 269 | 51 | 138 | 27 | 41 | Checkpoint selection |
| **Held-Out Test** | 15% | **525** | 270 | 50 | 139 | 26 | 40 | Final evaluation (Objective h) |
| **Total** | 100% | **3,504** | 1,796 | 338 | 922 | 177 | 271 | — |

Grade proportions in the held-out split match the retained cohort closely (Grade 0: 51.4% vs 51.3%), confirming the stratification held.

The 3,504 total is the **de-duplicated, label-consistent** cohort, not the 3,662 labelled records APTOS ships. Section 4 explains what was removed and why.

The complete record-by-record ledger — image identifier, source dataset, relative file path, grade, split assignment, and **SHA-256 of the image bytes** — is [`dataset_split_manifest.csv`](dataset_split_manifest.csv).

---

## 4. Duplicate Audit — disclosed finding

### 4.1 The de-duplication step did not take effect

The manifest generator groups images by `duplicate_group_id`, derived from APTOS's `duplicated_info.csv`. That file is **not part of the Kaggle competition download**, so the code took its fallback path and assigned every image a unique group. The manifest as it then stood showed the result plainly: **3,662 groups for 3,662 images**, every group of size 1. (The manifest committed today is the rebuilt one: 3,504 groups for 3,504 images, audited in §4.2.)

The grouping code ran. It grouped nothing.

### 4.2 Direct byte-level audit of the committed manifest

Rather than trusting `duplicate_group_id`, the `sha256_hash` column of the
manifest that ships in this package was audited directly:

| Finding | Value |
| :--- | :---: |
| Rows | 3,504 |
| Distinct image byte-hashes | 3,504 |
| Hashes appearing on more than one row | **0** |
| Hashes appearing in more than one split | **0** |
| **Held-out images byte-identical to a training image** | **0 / 525 (0.00%)** |
| Validation images byte-identical to a training image | **0 / 526** |
| Duplicate groups | 3,504, of which **0** hold more than one row |

Reproduce it from the shipped file alone — `python VERIFY.py`, check [4] — or
over the manifest directly with
[`corpus_guard.py`](../../backend/scripts/corpus_guard.py).

> [!NOTE]
> An earlier version of this table mixed the two runs. It reported 3,534
> distinct hashes, 128 images sharing bytes and 48 hashes spanning splits —
> all **contaminated-split** values — on the same rows as 0 / 525 held-out
> leakage, a **clean-split** value, and gave validation leakage as 17 / 550
> while §4.3 below gave 0. The contaminated figures are not lost: they are in
> §4.3's two-column comparison, where the column heading says which run each
> belongs to.

### 4.3 Resolution: the split was rebuilt and the model retrained

The contamination is no longer present. [`build_clean_split.py`](../../backend/scripts/build_clean_split.py)
rebuilt the partition from image-byte hashes, and EfficientNet-B0 was retrained
from ImageNet initialisation on the result.

| Property | Contaminated split | Clean split |
| :--- | :---: | :---: |
| Records / unique images | 3,662 / 3,534 | **3,504 / 3,504** |
| Duplicate groups spanning partitions | 48 | **0** |
| Test images byte-identical to a training image | 27 | **0** |
| Validation images byte-identical to a training image | 17 | **0** |
| Duplicate groups with conflicting labels | 30 (retained) | **30 (excluded)** |

Zero overlap is asserted by the builder *before* training starts, and confirmed
afterwards by an independent recomputation in `analyze_clinical_metrics.py`
(0 / 525). The per-group exclusion list — every image identifier and the grades
that disagreed — is recorded in [`dataset_split_audit.json`](dataset_split_audit.json).

**What was actually removed.** 158 records, which decompose into two very
different things: 128 redundant *copies* of images that are retained (no
distinct photograph is lost), and 30 distinct images whose duplicate groups
carried contradictory severity labels and could not be adjudicated from the
data. The second group is 0.85% of distinct images.

**An earlier revision of this document argued the contamination was immaterial**
because accuracy on the 27 affected test images was no higher than on the rest.
That argument was withdrawn: it addresses test contamination only, and cannot
detect validation contamination influencing checkpoint selection, nor
conflicting labels placing contradictory supervision into training. It is
retained here only as a record of the reasoning that was corrected.

## 5. Scope Statement

1. **Single-cohort study.** All training, validation and held-out evaluation reported in Chapter Four use the APTOS 2019 cohort only. EyePACS and Messidor-2 are **not** used. No cross-dataset generalisation claim is made.
2. **No external validation.** Performance on a different population, camera fleet, or grading panel is unknown and untested. This is the single largest constraint on the clinical interpretation of these results.
3. **Future generalisation testing** would require a zero-shot protocol against an untouched external cohort, with the APTOS-trained weights held fixed.
