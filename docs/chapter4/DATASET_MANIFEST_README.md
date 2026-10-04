# Dataset Manifest & Integrity Verification

## Contents

| File | Description |
| :--- | :--- |
| `dataset_split_manifest.csv` | **3,504 rows** — one per *retained* APTOS 2019 image — recording image id, source, grade, split assignment, and the **SHA-256 of the actual image bytes**. APTOS publishes 3,662 labelled records; 158 are removed by byte-identical de-duplication and the exclusion of the 30 conflicting-label groups. |
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
python docs/chapter4/verify_manifest_hashes.py ./aptos2019/train_images --sample 25
python docs/chapter4/verify_manifest_hashes.py ./aptos2019/train_images --split test
python docs/chapter4/verify_manifest_hashes.py ./aptos2019/train_images
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
| `split` | `train` (2,453) / `val` (526) / `test` (525) — summing to the 3,504 retained records |
| `sha256_hash` | SHA-256 of the image file's bytes |

### Caveat on `duplicate_group_id`

Each group holds exactly **one** row, and that is the intended end state rather than a failure. `build_clean_split.py` hashes every file, groups byte-identical images, excludes the 30 groups carrying conflicting labels, and keeps a single representative of each remaining group — so a duplicate group cannot span partitions by construction. (An earlier generator derived groups from APTOS's `duplicated_info.csv`, which is not part of the Kaggle competition download, and fell through to assigning every image its own group: 3,662 groups for 3,662 images, grouping nothing. That is superseded.)

Auditing `sha256_hash` directly instead shows **0 of 525 held-out images (0.00%) are byte-identical to a training image**, and 0 of 526 validation images. No hash appears in more than one split.

Because nothing leaks, there is **no clean subset to compare against** and no leakage-adjusted metric to report; `clinical_metrics.json` carries `leakage_adjusted: null` for that reason. An earlier version of this paragraph quoted 77.78% against 78.74% on "the clean 522" with $\kappa$ 0.877818 vs 0.865832. Those figures came from the superseded contaminated split, where 3 held-out images did leak; they do not describe the committed split and have been removed.

This is disclosed rather than hidden. Reproduce the audit with:

```bash
python backend/scripts/analyze_clinical_metrics.py   # see the LEAKAGE AUDIT section
```

Full discussion: [`dataset_audit.md`](dataset_audit.md) §4.
