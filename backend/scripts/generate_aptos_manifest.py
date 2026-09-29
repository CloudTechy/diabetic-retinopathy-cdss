#!/usr/bin/env python3
"""
APTOS 2019 Dataset Manifest Generator — Genuine Byte-Level Hashing
PGD Computer Science, Faculty of Physical Sciences

Reads the official APTOS 2019 train.csv + duplicated_info.csv,
hashes every actual image binary on disk, groups by perceptual
duplicate hash, and produces a stratified 70/15/15 patient-level split.
"""

import os
import csv
import random
import hashlib
from collections import defaultdict, Counter
from pathlib import Path


def sha256_file(path: str) -> str:
    """Compute SHA-256 digest of a file's raw bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def build_aptos_manifest():
    random.seed(42)

    # --- Paths ---
    root = Path(__file__).resolve().parents[2]  # repo root
    dataset_dir = root / "storage" / "datasets" / "aptos2019"
    images_dir = dataset_dir / "train_images"
    train_csv = dataset_dir / "train.csv"
    dup_csv = dataset_dir / "duplicated_info.csv"

    if not train_csv.exists():
        raise FileNotFoundError(f"Missing {train_csv}. Download the APTOS 2019 dataset first.")

    # --- Read official labels ---
    with open(train_csv, "r", encoding="utf-8") as f:
        samples = list(csv.DictReader(f))
    print(f"Total APTOS 2019 samples in train.csv: {len(samples)}")
    raw_counts = Counter(int(s["diagnosis"]) for s in samples)
    for g in range(5):
        print(f"  Grade {g}: {raw_counts.get(g, 0)}")

    # --- Read duplicate / perceptual hash info for grouping ---
    id_to_phash = {}
    if dup_csv.exists():
        with open(dup_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                p = row.get("path", "")
                img_id = Path(p).stem
                h = row.get("Hash", "")
                if h and img_id:
                    id_to_phash[img_id] = h
        print(f"Loaded perceptual hashes for {len(id_to_phash)} images from duplicated_info.csv")
    else:
        print("Warning: duplicated_info.csv not found. Each image treated as independent group.")

    # --- Build duplicate groups ---
    # Using perceptual hash to group images that may be bilateral/duplicate.
    # NOTE: Perceptual similarity does not definitively establish patient identity,
    # so we use "duplicate_group_id" instead of "patient_id".
    phash_to_group = {}
    group_counter = 1
    image_groups = defaultdict(list)

    for s in samples:
        img_id = s["id_code"]
        diag = int(s["diagnosis"])
        phash = id_to_phash.get(img_id)

        if phash:
            if phash not in phash_to_group:
                phash_to_group[phash] = f"DG-{group_counter:04d}"
                group_counter += 1
            gid = phash_to_group[phash]
        else:
            gid = f"DG-{group_counter:04d}"
            group_counter += 1

        image_groups[gid].append({
            "image_id": img_id,
            "duplicate_group_id": gid,
            "source_dataset": "APTOS 2019 (Aravind Eye Hospital)",
            "true_grade": diag,
            "file_path": f"train_images/{img_id}.png",
        })

    print(f"Total duplicate groups: {len(image_groups)}")

    # --- Stratified group-level split (70/15/15) ---
    groups_by_grade = defaultdict(list)
    for gid, imgs in image_groups.items():
        dom_grade = imgs[0]["true_grade"]
        groups_by_grade[dom_grade].append((gid, imgs))

    train_records, val_records, test_records = [], [], []

    for grade in range(5):
        groups = groups_by_grade[grade]
        random.shuffle(groups)

        n = len(groups)
        n_train = int(round(n * 0.70))
        n_val = int(round(n * 0.15))

        for _, imgs in groups[:n_train]:
            for img in imgs:
                img["split"] = "train"
                train_records.append(img)

        for _, imgs in groups[n_train:n_train + n_val]:
            for img in imgs:
                img["split"] = "val"
                val_records.append(img)

        for _, imgs in groups[n_train + n_val:]:
            for img in imgs:
                img["split"] = "test"
                test_records.append(img)

    all_records = train_records + val_records + test_records
    print(f"\nPartitioned: Train={len(train_records)}, Val={len(val_records)}, "
          f"Test={len(test_records)} (Total={len(all_records)})")

    # --- Compute genuine SHA-256 byte hashes of actual image files ---
    icdr_labels = [
        "Grade 0: No Apparent DR",
        "Grade 1: Mild NPDR",
        "Grade 2: Moderate NPDR",
        "Grade 3: Severe NPDR",
        "Grade 4: Proliferative DR",
    ]

    output_rows = []
    missing_count = 0
    for r in all_records:
        grade = r["true_grade"]
        img_path = images_dir / f"{r['image_id']}.png"

        if img_path.exists():
            file_hash = sha256_file(str(img_path))
        else:
            file_hash = "FILE_NOT_FOUND"
            missing_count += 1

        output_rows.append({
            "image_id": r["image_id"],
            "duplicate_group_id": r["duplicate_group_id"],
            "source_dataset": r["source_dataset"],
            "file_path": r["file_path"],
            "true_grade": grade,
            "true_label": icdr_labels[grade],
            "split": r["split"],
            "sha256_hash": file_hash,
        })

    if missing_count > 0:
        print(f"\nWARNING: {missing_count} of {len(all_records)} image files not found on disk.")
        print("  Hashes for missing files are set to 'FILE_NOT_FOUND'.")
        print("  Download the full APTOS 2019 dataset to storage/datasets/aptos2019/train_images/")
    else:
        print(f"\nAll {len(all_records)} image files found. SHA-256 byte hashes computed from actual binaries.")

    # --- Write manifest ---
    out_dir = root / "docs" / "chapter4"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_out = out_dir / "dataset_split_manifest.csv"

    fieldnames = [
        "image_id", "duplicate_group_id", "source_dataset", "file_path",
        "true_grade", "true_label", "split", "sha256_hash",
    ]

    # Sort deterministically
    output_rows.sort(key=lambda r: (r["split"], r["true_grade"], r["image_id"]))

    with open(manifest_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Saved {len(output_rows)} records to {manifest_out}")

    # --- Summary ---
    print("\n--- Partition Summary ---")
    for s_name, recs in [("Train", train_records), ("Validation", val_records), ("Held-Out Test", test_records)]:
        c = Counter(r["true_grade"] for r in recs)
        print(f"{s_name:15s} (N={len(recs):4d}): "
              f"Gr0={c[0]:4d}, Gr1={c[1]:3d}, Gr2={c[2]:3d}, Gr3={c[3]:3d}, Gr4={c[4]:3d}")


if __name__ == "__main__":
    build_aptos_manifest()
