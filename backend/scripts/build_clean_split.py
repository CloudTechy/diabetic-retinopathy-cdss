#!/usr/bin/env python3
"""
Build a leakage-free, label-consistent APTOS split. Single source of truth.

The previous manifest was contaminated. Independent audit of it found:

    3,662 records but only 3,534 unique image hashes
    48 exact-duplicate groups spanning more than one partition
       27 test images byte-identical to a training image
       17 validation images byte-identical to a training image
        6 test images byte-identical to a validation image
    30 exact-duplicate groups carrying CONFLICTING severity labels

A test set in that condition cannot be described as held out. Comparing scores
on the contaminated subset against the rest does not repair it: validation
contamination influences which checkpoint gets selected, and conflicting labels
put contradictory supervision into training.

The cause was a de-duplication step keyed on APTOS's `duplicated_info.csv`,
which is not part of the Kaggle competition download. When absent, the fallback
gave every image its own group, so the step ran and grouped nothing.

This module keys de-duplication on the SHA-256 of the image bytes instead,
which is always available because we compute it ourselves.

Procedure, in order:

    1. Hash every image by its actual bytes.
    2. Group exact duplicates by that hash.
    3. Drop groups whose members disagree on the severity label. They are not
       adjudicable from the data, and guessing would be fabrication.
    4. Keep one representative per surviving group.
    5. Stratify the unique images 70/15/15 by ICDR grade.
    6. Assert zero hash overlap between the three partitions, and fail loudly
       if that does not hold.

Expected result on APTOS 2019: 3,504 unique, non-conflicting images.

    Grade 0  No DR              1,796
    Grade 1  Mild NPDR            338
    Grade 2  Moderate NPDR        922
    Grade 3  Severe NPDR          177
    Grade 4  Proliferative DR     271

Usage:
    python backend/scripts/build_clean_split.py aptos2019/train_images \\
        --labels aptos2019/train.csv --out output/dataset_split_manifest.csv
"""

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict

ICDR_LABELS = [
    "Grade 0: No Apparent DR",
    "Grade 1: Mild NPDR",
    "Grade 2: Moderate NPDR",
    "Grade 3: Severe NPDR",
    "Grade 4: Proliferative DR",
]

FIELDNAMES = [
    "image_id", "duplicate_group_id", "source_dataset", "file_path",
    "true_grade", "true_label", "split", "sha256_hash",
]


def sha256_file(path, chunk=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def build_clean_split(images_dir, labels_csv, seed=42,
                      train_frac=0.70, val_frac=0.15, log=print):
    """
    Returns (rows, report). `rows` is the manifest; `report` records exactly
    what was excluded and why, so the audit is part of the output rather than
    something a reader has to reconstruct.
    """
    with open(labels_csv, newline="", encoding="utf-8") as fh:
        samples = list(csv.DictReader(fh))
    log(f"Labelled records in {os.path.basename(labels_csv)}: {len(samples)}")

    # 1-2. Hash every image and group exact duplicates by content.
    by_hash = defaultdict(list)
    missing = []
    for s in samples:
        image_id = s["id_code"]
        path = os.path.join(images_dir, f"{image_id}.png")
        if not os.path.exists(path):
            missing.append(image_id)
            continue
        by_hash[sha256_file(path)].append({
            "image_id": image_id,
            "true_grade": int(s["diagnosis"]),
            "file_path": f"train_images/{image_id}.png",
        })

    if missing:
        log(f"WARNING: {len(missing)} labelled images absent from disk")

    log(f"Unique image hashes: {len(by_hash)} "
        f"({sum(len(v) for v in by_hash.values()) - len(by_hash)} exact duplicates)")

    # 3. Drop label-conflicting duplicate groups.
    conflicting, consistent = {}, {}
    for h, members in by_hash.items():
        if len({m["true_grade"] for m in members}) > 1:
            conflicting[h] = members
        else:
            consistent[h] = members

    if conflicting:
        log(f"EXCLUDED {len(conflicting)} duplicate groups with conflicting labels "
            f"({sum(len(v) for v in conflicting.values())} images). "
            f"They cannot be adjudicated from the data.")

    # 4. One representative per group, chosen deterministically.
    unique = []
    for h, members in consistent.items():
        rep = sorted(members, key=lambda m: m["image_id"])[0]
        unique.append({**rep, "sha256_hash": h,
                       "duplicate_group_id": f"DG-{h[:12]}",
                       "n_in_group": len(members)})

    log(f"Unique, non-conflicting images retained: {len(unique)}")
    dist = Counter(u["true_grade"] for u in unique)
    for g in range(5):
        log(f"   {ICDR_LABELS[g]:<28} {dist[g]:>5}")

    # 5. Stratified split over UNIQUE images.
    rng = random.Random(seed)
    by_grade = defaultdict(list)
    for u in unique:
        by_grade[u["true_grade"]].append(u)

    rows = []
    for grade in range(5):
        items = sorted(by_grade[grade], key=lambda m: m["image_id"])
        rng.shuffle(items)
        n = len(items)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        for split, chunk in (("train", items[:n_train]),
                             ("val", items[n_train:n_train + n_val]),
                             ("test", items[n_train + n_val:])):
            for m in chunk:
                rows.append({
                    "image_id": m["image_id"],
                    "duplicate_group_id": m["duplicate_group_id"],
                    "source_dataset": "APTOS 2019 (Aravind Eye Hospital)",
                    "file_path": m["file_path"],
                    "true_grade": grade,
                    "true_label": ICDR_LABELS[grade],
                    "split": split,
                    "sha256_hash": m["sha256_hash"],
                })

    rows.sort(key=lambda r: (r["split"], r["true_grade"], r["image_id"]))

    # 6. The assertion the previous pipeline never made.
    hashes = defaultdict(set)
    for r in rows:
        hashes[r["split"]].add(r["sha256_hash"])
    overlaps = {
        "train_val": len(hashes["train"] & hashes["val"]),
        "train_test": len(hashes["train"] & hashes["test"]),
        "val_test": len(hashes["val"] & hashes["test"]),
    }
    if any(overlaps.values()):
        raise SystemExit(f"LEAKAGE: partitions share image hashes: {overlaps}")

    counts = Counter(r["split"] for r in rows)
    log(f"\nSplit: train {counts['train']} / val {counts['val']} / test {counts['test']}")
    log("Hash overlap between partitions: 0 (verified)")

    report = {
        "labelled_records": len(samples),
        "images_missing_on_disk": len(missing),
        "unique_hashes": len(by_hash),
        "exact_duplicate_images": sum(len(v) for v in by_hash.values()) - len(by_hash),
        "conflicting_label_groups_excluded": len(conflicting),
        "conflicting_label_images_excluded": sum(len(v) for v in conflicting.values()),
        "unique_non_conflicting_retained": len(unique),
        "grade_distribution": {ICDR_LABELS[g]: dist[g] for g in range(5)},
        "split_counts": dict(counts),
        "partition_hash_overlap": overlaps,
        "seed": seed,
        "excluded_conflicting_groups": [
            {"sha256": h, "image_ids": [m["image_id"] for m in v],
             "grades": sorted({m["true_grade"] for m in v})}
            for h, v in sorted(conflicting.items())
        ],
    }
    return rows, report


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images_dir")
    parser.add_argument("--labels", default=None,
                        help="train.csv (default: alongside images_dir)")
    parser.add_argument("--out", default="output/dataset_split_manifest.csv")
    parser.add_argument("--report", default=None,
                        help="Where to write the audit JSON (default: beside --out)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    labels = args.labels or os.path.join(os.path.dirname(args.images_dir.rstrip("/\\")),
                                         "train.csv")
    if not os.path.isdir(args.images_dir):
        raise SystemExit(f"Not a directory: {args.images_dir}")
    if not os.path.exists(labels):
        raise SystemExit(f"Labels not found: {labels}")

    print("=" * 78)
    print("BUILD CLEAN SPLIT - de-duplicated by image bytes, label-consistent")
    print("=" * 78)

    rows, report = build_clean_split(args.images_dir, labels, seed=args.seed)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    report_path = args.report or os.path.join(
        os.path.dirname(os.path.abspath(args.out)), "dataset_split_audit.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print(f"\nManifest: {args.out}  ({len(rows)} rows)")
    print(f"Audit:    {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
