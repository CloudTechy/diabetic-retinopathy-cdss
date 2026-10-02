#!/usr/bin/env python3
"""
Refuse a corpus whose files claim an identity they do not have.

WHAT THIS CATCHES

`storage/datasets/aptos2019/train_images/` on the development machine holds 15
files named after real APTOS images — `d1f1ea894da1.png`, `e62490b7d0e9.png` and
so on, every one of them an `image_id` that appears in
`dataset_split_manifest.csv`. None of them is that image. All 15 are 640x480,
4–5 KB, and **eight are byte-identical to each other**. Real APTOS photographs
are roughly 1.8 MB at resolutions up to 2848 px.

They are placeholders wearing the names of the held-out cohort. Any script
pointed at that directory produces output that looks exactly like evidence:
real-looking image ids, plausible gate metrics, a grade per image. Nothing in
the pipeline noticed, because every existing check asked whether an `image_id`
appears in the manifest — never whether the FILE is the image that id names.

The manifest already carries the answer: a SHA-256 of the real image bytes, per
row. This compares them.

WHY IT SAMPLES BY DEFAULT

Hashing all 3,662 APTOS images means reading ~6.6 GB. A directory substituted
wholesale fails on the first file checked, so a seeded sample of 64 settles that
case in a second. A single swapped file inside an otherwise genuine corpus is a
different threat, and `--full` is the answer to it; the scripts that touch only
a handful of images check all of them.

Usage:
    python backend/scripts/corpus_guard.py aptos2019/train_images
    python backend/scripts/corpus_guard.py aptos2019/train_images --full
"""

import argparse
import csv
import hashlib
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
DEFAULT_MANIFEST = os.path.join(REPO_ROOT, "docs", "chapter4",
                                "dataset_split_manifest.csv")

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(manifest_path=None):
    """image_id -> expected sha256 of the image bytes."""
    path = manifest_path or DEFAULT_MANIFEST
    if not os.path.exists(path):
        return {}
    with open(path, newline="", encoding="utf-8") as fh:
        return {row["image_id"]: row["sha256_hash"]
                for row in csv.DictReader(fh)
                if row.get("image_id") and row.get("sha256_hash")}


def check_corpus(images_dir, manifest_path=None, sample=64, seed=42,
                 only=None):
    """
    Returns (checked, impostors, recognised) where `impostors` is a list of
    (image_id, expected, actual) for files whose name is a manifest image_id and
    whose bytes are not that image.

    `only` restricts the check to specific filenames - used by callers that read
    a known handful of images and can afford to verify every one.
    """
    expected = load_manifest(manifest_path)
    if not expected:
        return 0, [], 0

    names = sorted(only) if only else sorted(
        n for n in os.listdir(images_dir) if n.lower().endswith(IMAGE_SUFFIXES))

    recognised = [n for n in names if os.path.splitext(n)[0] in expected]
    if not recognised:
        return 0, [], 0

    subset = recognised
    if sample and len(recognised) > sample:
        subset = random.Random(seed).sample(recognised, sample)

    impostors = []
    for name in subset:
        image_id = os.path.splitext(name)[0]
        actual = _sha256(os.path.join(images_dir, name))
        if actual != expected[image_id]:
            impostors.append((image_id, expected[image_id], actual))

    return len(subset), impostors, len(recognised)


def assert_corpus_is_authentic(images_dir, manifest_path=None, sample=64,
                               only=None, quiet=False):
    """
    Raise SystemExit when the directory contains a file that claims a manifest
    identity it does not have. Callers put this before any measurement.
    """
    checked, impostors, recognised = check_corpus(
        images_dir, manifest_path, sample=sample, only=only)

    if not recognised:
        if not quiet:
            print(f"  [corpus] no manifest image ids found in {images_dir} - "
                  f"authenticity not checked")
        return

    if impostors:
        lines = [
            "",
            "=" * 74,
            "CORPUS REJECTED - files claim an identity they do not have",
            "=" * 74,
            f"Directory : {images_dir}",
            f"Checked   : {checked} of {recognised} files bearing a manifest image_id",
            f"Impostors : {len(impostors)}",
            "",
        ]
        for image_id, want, got in impostors[:8]:
            lines.append(f"  {image_id}")
            lines.append(f"      manifest sha256 {want[:32]}...")
            lines.append(f"      file     sha256 {got[:32]}...")
        if len(impostors) > 8:
            lines.append(f"  ... and {len(impostors) - 8} more")
        lines += [
            "",
            "These files are named after real images in dataset_split_manifest.csv",
            "but are not those images. Measuring them would produce output that",
            "looks like evidence - real image ids, plausible metrics, a grade per",
            "image - and means nothing.",
            "",
            "Point --images-dir at the genuine APTOS train_images/ directory.",
            "=" * 74,
            "",
        ]
        raise SystemExit("\n".join(lines))

    if not quiet:
        scope = "all" if checked == recognised else f"{checked} sampled of"
        print(f"  [corpus] authentic: {scope} {recognised} manifest images "
              f"match their recorded SHA-256")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images_dir")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--full", action="store_true",
                        help="Hash every file bearing a manifest image_id")
    parser.add_argument("--sample", type=int, default=64)
    args = parser.parse_args()

    if not os.path.isdir(args.images_dir):
        raise SystemExit(f"No directory at {args.images_dir}")

    assert_corpus_is_authentic(
        args.images_dir, args.manifest,
        sample=0 if args.full else args.sample)
    return 0


if __name__ == "__main__":
    sys.exit(main())
