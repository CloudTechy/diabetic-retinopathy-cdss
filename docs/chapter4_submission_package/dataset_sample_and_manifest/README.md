# Dataset Manifest & Integrity Verification

## Contents

| File | Description |
| :--- | :--- |
| `dataset_split_manifest.csv` | 3,662 rows — one per APTOS 2019 image — recording image id, source, grade, split assignment, and the **SHA-256 of the actual image bytes**. |
| `verify_manifest_hashes.py` | Verifies the manifest against your own copy of the APTOS images. |

---

## Why no images are included

The APTOS 2019 images are **not redistributed here**, for two reasons:

1. **Licence.** The Kaggle competition rules restrict redistribution of the dataset.
2. **It would defeat the check.** A byte-level integrity verification is only meaningful against the authentic files. An earlier revision of this package shipped 15 small placeholder PNGs in a `sample_test_images/` directory; they were 640×480 synthetic images, **eight of which were byte-identical to one another**, and they matched none of the real APTOS files. They have been removed, along with the `hash_verification_output.txt` that reported them as "VERIFIED MATCH".

---

## How to verify

```bash
# 1. Obtain the dataset (Kaggle account with competition rules accepted)
kaggle competitions download -c aptos2019-blindness-detection -p ./aptos2019
unzip ./aptos2019/aptos2019-blindness-detection.zip -d ./aptos2019

# 2. Verify a sample, one split, or everything
python verify_manifest_hashes.py ./aptos2019/train_images --sample 25
python verify_manifest_hashes.py ./aptos2019/train_images --split test
python verify_manifest_hashes.py ./aptos2019/train_images
```

Exit code `0` and `[SUCCESS]` mean every image checked matched its recorded digest — i.e. the manifest describes real image bytes.

---

## Manifest schema

| Column | Description |
| :--- | :--- |
| `image_id` | APTOS `id_code`; the file is `<image_id>.png` |
| `duplicate_group_id` | Intended duplicate grouping. **See the caveat below.** |
| `source_dataset` | Always `APTOS 2019 (Aravind Eye Hospital)` |
| `file_path` | Path relative to the dataset root |
| `true_grade` | ICDR severity 0–4 |
| `true_label` | Human-readable grade label |
| `split` | `train` (2,563) / `val` (550) / `test` (549) |
| `sha256_hash` | SHA-256 of the image file's bytes |

### Caveat on `duplicate_group_id`

This column is present but **carries no grouping information**: there are 3,662 distinct group ids for 3,662 images. The generator derives groups from APTOS's `duplicated_info.csv`, which is not part of the Kaggle competition download, so it fell through to assigning every image its own group.

Auditing `sha256_hash` directly instead shows **27 of 549 held-out images (4.92%) are byte-identical to a training image**. The measured effect on reported metrics is nil — accuracy 77.78% on the affected images vs 78.74% on the clean 522, and clean-subset $\kappa$ = 0.877818 vs 0.877747 full-cohort.

This is disclosed rather than hidden. Reproduce the audit with:

```bash
python backend/scripts/analyze_clinical_metrics.py   # see the LEAKAGE AUDIT section
```

Full discussion: `documentation/dataset_audit.md` §4.
