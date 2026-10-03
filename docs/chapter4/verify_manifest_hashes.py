#!/usr/bin/env python3
"""
Verify the dataset manifest against a local APTOS 2019 image directory.

Postgraduate Diploma in Computer Science - Faculty of Physical Sciences

This script proves that `dataset_split_manifest.csv` records the SHA-256 of the
ACTUAL APTOS image bytes, rather than hashes of some derived or synthesised
string. It is the check that distinguishes a genuine manifest from a fabricated
one, so it is worth running.

The APTOS 2019 images are NOT redistributed in this repository. The competition
rules restrict redistribution of the dataset, and shipping placeholder stand-ins
would defeat the purpose of a byte-level integrity check. Point this script at
your own copy instead.

Usage:
    # 1. Obtain the dataset (requires a Kaggle account with the rules accepted)
    kaggle competitions download -c aptos2019-blindness-detection -p ./aptos2019
    unzip ./aptos2019/aptos2019-blindness-detection.zip -d ./aptos2019

    # 2. Verify (whole dataset, or a sample)
    python verify_manifest_hashes.py ./aptos2019/train_images
    python verify_manifest_hashes.py ./aptos2019/train_images --sample 25
    python verify_manifest_hashes.py ./aptos2019/train_images --split test

Exit code is 0 only if every image checked matched its recorded digest.
"""

import argparse
import csv
import hashlib
import os
import random
import sys


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Verify dataset_split_manifest.csv against real APTOS image bytes."
    )
    parser.add_argument("images_dir", help="Path to the APTOS train_images/ directory")
    parser.add_argument("--sample", type=int, default=0,
                        help="Check only N randomly chosen images (default: all)")
    parser.add_argument("--split", choices=["train", "val", "test"],
                        help="Restrict the check to one partition")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    manifest_path = os.path.join(script_dir, "dataset_split_manifest.csv")

    if not os.path.exists(manifest_path):
        print(f"Error: manifest not found at {manifest_path}")
        return 2
    if not os.path.isdir(args.images_dir):
        print(f"Error: image directory not found at {args.images_dir}")
        print("Download APTOS 2019 first - see the module docstring.")
        return 2

    with open(manifest_path, "r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    if args.split:
        rows = [r for r in rows if r["split"] == args.split]
    if args.sample and args.sample < len(rows):
        random.seed(args.seed)
        rows = random.sample(rows, args.sample)

    print("=" * 96)
    print("APTOS 2019 MANIFEST - BYTE-LEVEL SHA-256 INTEGRITY VERIFICATION")
    print("=" * 96)
    print(f"Manifest : {manifest_path}")
    print(f"Images   : {args.images_dir}")
    print(f"Checking : {len(rows)} image(s)"
          + (f" from split '{args.split}'" if args.split else ""))
    print("-" * 96)

    matched = mismatched = missing = 0
    for row in rows:
        img_id = row["image_id"]
        expected = (row.get("sha256_hash") or row.get("sha256") or "").strip().lower()
        path = os.path.join(args.images_dir, f"{img_id}.png")

        if not os.path.exists(path):
            print(f"{img_id:<16} | MISSING ON DISK")
            missing += 1
            continue

        actual = sha256_file(path)
        if actual == expected:
            matched += 1
        else:
            mismatched += 1
            print(f"{img_id:<16} | MISMATCH")
            print(f"{'':<16} |   expected {expected}")
            print(f"{'':<16} |   actual   {actual}")

    print("-" * 96)
    print(f"Matched: {matched}   Mismatched: {mismatched}   Missing: {missing}")
    print("=" * 96)

    if mismatched == 0 and missing == 0 and matched > 0:
        print("[SUCCESS] Every image checked matches its recorded SHA-256 digest.")
        print("          The manifest records genuine image bytes.")
        return 0

    if mismatched:
        print("[FAILURE] One or more digests did not match. Either the manifest does not")
        print("          describe these files, or the files have been modified.")
    if missing:
        print("[WARNING] Some images were absent from the supplied directory.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
