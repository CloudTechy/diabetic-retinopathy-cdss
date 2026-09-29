#!/usr/bin/env python3
"""
Verify SHA-256 Hashes of Sample Images against Manifest
Postgraduate Diploma in Computer Science - Faculty of Physical Sciences
"""

import os
import csv
import hashlib

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    manifest_path = os.path.join(script_dir, "dataset_split_manifest.csv")
    images_dir = os.path.join(script_dir, "sample_test_images")

    if not os.path.exists(manifest_path):
        print(f"Error: Manifest not found at {manifest_path}")
        return

    if not os.path.exists(images_dir):
        print(f"Error: Sample images directory not found at {images_dir}")
        return

    # Load manifest hashes
    manifest_hashes = {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            manifest_hashes[row["image_id"]] = row.get("sha256_hash", "") or row.get("sha256", "")

    sample_files = [f for f in os.listdir(images_dir) if f.endswith(".png")]
    print(f"Found {len(sample_files)} sample test images on disk.")
    print("=" * 80)
    print(f"{'Image ID':<16} | {'On-Disk SHA-256':<64} | Status")
    print("-" * 80)

    all_matched = True
    for fname in sorted(sample_files):
        img_id = fname.replace(".png", "")
        fpath = os.path.join(images_dir, fname)
        with open(fpath, "rb") as f:
            computed_hash = hashlib.sha256(f.read()).hexdigest()

        expected_hash = manifest_hashes.get(img_id, "")
        if computed_hash == expected_hash:
            status = "MATCHED (100%)"
        else:
            status = f"MISMATCH (expected {expected_hash[:8]}...)"
            all_matched = False

        print(f"{img_id:<16} | {computed_hash} | {status}")

    print("=" * 80)
    if all_matched:
        print("[SUCCESS] All sample test images perfectly match the manifest SHA-256 digests.")
    else:
        print("[FAILURE] Some hashes did not match.")

if __name__ == "__main__":
    main()
