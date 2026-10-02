#!/usr/bin/env python3
"""
Measure the Gate 1 and Gate 3 admission thresholds over the real corpus so they
can be chosen from evidence instead of guessed.

Covers both uncalibrated values:

  LAPLACIAN_BLUR_THRESHOLD = 60.0   rejected 10/10 genuine held-out images
  MIN_IMAGE_DIMENSION      = 512    rejected 1 genuine held-out image
  CONTRAST_THRESHOLD       = 18.0   rejected 5/10 once the first two were fixed

WHY THIS EXISTS

`LAPLACIAN_BLUR_THRESHOLD` was set to 60.0 a priori. The first genuine run of
the validation pipeline rejected 10 of 10 unmodified held-out APTOS images,
because real images in this corpus score 5.7-22.0. As configured the system
would refuse to grade every genuine fundus photograph.

A sharpness threshold is not a universal constant. Laplacian variance depends
on sensor resolution, optics, compression and any resizing applied upstream, so
it must be calibrated against the imaging characteristics of the corpus the
system will actually see.

WHAT THIS DOES, AND DOES NOT DO

It measures. It reports the distribution, and what each candidate threshold
would reject. It does NOT write the threshold into the configuration, because
choosing the operating point is a judgement about how much blur is tolerable -
and a script that silently picked the number that made the tests pass would be
fitting the threshold to the result, which is the failure this whole exercise
exists to prevent.

DEVELOPMENT SUBSET ONLY

The threshold is measured over the TRAINING and VALIDATION partitions, read
from `docs/chapter4/dataset_split_manifest.csv`. The held-out test partition is
excluded: choosing an operating point on the test set leaks it, for the same
reason the split exists at all. Pass --allow-test-split only if you have a
reason you are willing to write down.

Pick a percentile, justify it in `validation_module_spec.md`, and set it by
hand.

Usage:
    python backend/scripts/calibrate_blur_threshold.py aptos2019/train_images
    python backend/scripts/calibrate_blur_threshold.py <dir> --sample 800
"""

import argparse
import json
import os
import random
import statistics
import sys

# Refuse a directory whose files claim a manifest identity they do not have.
# A local directory of 640x480 placeholders named after real held-out images
# produced output that looked exactly like evidence; nothing noticed, because
# every check asked whether an image_id was in the manifest and none asked
# whether the FILE was that image. See corpus_guard.py.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus_guard import assert_corpus_is_authentic  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
BACKEND_ROOT = os.path.join(REPO_ROOT, "backend")
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

OUT = os.path.join(REPO_ROOT, "docs", "chapter4", "blur_threshold_calibration.json")


def percentile(sorted_values, q):
    """Linear-interpolated percentile; q in [0, 100]."""
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images_dir", help="APTOS train_images/ directory")
    parser.add_argument("--sample", type=int, default=0,
                        help="Measure a random sample of N images (0 = all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full-corpus-check", action="store_true",
                        help="Hash every manifest image rather than a sample of 64")

    parser.add_argument("--out", default=OUT,
                        help="Where to write the report (default: the evidence folder)")
    parser.add_argument("--force", action="store_true",
                        help="Write to the evidence folder even from a small corpus")
    parser.add_argument("--allow-test-split", action="store_true",
                        help="Include held-out test images (leaks the test set)")
    parser.add_argument("--manifest", default=None,
                        help="Split manifest (default: docs/chapter4/dataset_split_manifest.csv)")
    args = parser.parse_args()

    import numpy as np
    from PIL import Image
    from app.core.config import settings
    from app.services.validation.gate3_quality import compute_laplacian_variance

    def gate3_laplacian(pil_image):
        """
        Replicate gate3_quality.py's Laplacian path exactly.

        The value is resolution-dependent by definition, so measuring it any
        other way would calibrate a threshold against a quantity the gate never
        computes. Bilinear resize to 1024 on the long edge when larger,
        full-resolution luma otherwise - matching the gate line for line.
        """
        rgb = pil_image.convert("RGB")
        width, height = rgb.size
        if max(height, width) > 1024:
            scale = 1024.0 / max(height, width)
            scaled = rgb.convert("L").resize(
                (int(width * scale), int(height * scale)), Image.Resampling.BILINEAR)
            return compute_laplacian_variance(np.asarray(scaled, dtype=np.float32))
        arr = np.asarray(rgb, dtype=np.float32)
        gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        return compute_laplacian_variance(gray)

    from app.services.validation.downsample import downsample_for_analysis

    def gate3_contrast(pil_image):
        """
        Replicate gate3_quality.py's contrast path exactly: nearest-neighbour
        subsample to VALIDATION_ANALYSIS_MAX_DIM, luma, drop the camera's black
        surround at gray > 15, standard deviation of what remains.
        """
        rgb = pil_image.convert("RGB")
        analysis = downsample_for_analysis(rgb, settings.VALIDATION_ANALYSIS_MAX_DIM)
        arr = np.asarray(analysis, dtype=np.float32)
        gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        mask = gray > 15.0
        fg = gray[mask] if np.any(mask) else gray.ravel()
        return float(np.std(fg))

    if not os.path.isdir(args.images_dir):
        raise SystemExit(f"Not a directory: {args.images_dir}")

    assert_corpus_is_authentic(
        args.images_dir, sample=0 if args.full_corpus_check else 64)

    names = sorted(n for n in os.listdir(args.images_dir)
                   if n.lower().endswith((".png", ".jpg", ".jpeg")))

    # Restrict to the development subset (train + validation).
    manifest = args.manifest or os.path.join(
        REPO_ROOT, "docs", "chapter4", "dataset_split_manifest.csv")
    split_of = {}
    if os.path.exists(manifest):
        import csv as _csv
        with open(manifest, newline="", encoding="utf-8") as fh:
            for row in _csv.DictReader(fh):
                split_of[row["image_id"]] = row["split"]

    if not split_of:
        print("WARNING: no split manifest found. Measuring every image in the")
        print("         directory, which may include the held-out test set.")
    elif args.allow_test_split:
        print("WARNING: --allow-test-split given. Held-out test images are")
        print("         included; this leaks the test partition.")
    else:
        dev = {n for n in names if split_of.get(os.path.splitext(n)[0]) in ("train", "val")}
        excluded = len(names) - len(dev)
        names = sorted(dev)
        print(f"Development subset: {len(names)} images (train + validation).")
        print(f"Excluded:           {excluded} held-out test or unlisted images.")
        if not names:
            raise SystemExit(
                "No training or validation images found. Check that the manifest "
                "matches this image directory.")

    if args.sample and args.sample < len(names):
        random.Random(args.seed).shuffle(names)
        names = sorted(names[:args.sample])

    print("=" * 74)
    print("GATE 3 BLUR THRESHOLD CALIBRATION")
    print("=" * 74)
    print(f"Corpus:            {args.images_dir}")
    print(f"Images measured:   {len(names)}")
    print(f"Current threshold: {settings.LAPLACIAN_BLUR_THRESHOLD}")
    print()

    values = []
    short_edges = []
    contrasts = []
    for i, name in enumerate(names, 1):
        try:
            with Image.open(os.path.join(args.images_dir, name)) as im:
                width, height = im.size
                short_edges.append(min(width, height))
                values.append(gate3_laplacian(im))
                contrasts.append(gate3_contrast(im))
        except Exception as exc:                      # noqa: BLE001
            print(f"  [skip] {name}: {exc}")
        if i % 500 == 0:
            print(f"  {i}/{len(names)}...")

    if not values:
        raise SystemExit("No images could be measured.")

    values.sort()
    pcts = {f"p{q}": round(percentile(values, q), 3)
            for q in (0.1, 0.5, 1, 2, 5, 10, 25, 50, 75, 90, 99)}

    print()
    print("DISTRIBUTION OF LAPLACIAN VARIANCE ACROSS THE REAL CORPUS")
    print(f"  min {min(values):.2f}   median {statistics.median(values):.2f}   "
          f"max {max(values):.2f}   mean {statistics.mean(values):.2f}")
    for k, v in pcts.items():
        print(f"    {k:>5}: {v:9.3f}")

    current = settings.LAPLACIAN_BLUR_THRESHOLD
    rejected_now = sum(1 for v in values if v < current)
    print()
    print(f"At the CURRENT threshold of {current}: "
          f"{rejected_now}/{len(values)} ({100*rejected_now/len(values):.1f}%) "
          f"of genuine images would be REJECTED.")

    print()
    print("WHAT EACH CANDIDATE WOULD REJECT")
    print(f"  {'threshold':>10}  {'rejected':>9}  {'share':>7}")
    for q in (0.1, 0.5, 1, 2, 5, 10):
        t = percentile(values, q)
        n = sum(1 for v in values if v < t)
        print(f"  {t:10.2f}  {n:9d}  {100*n/len(values):6.2f}%")

    # ------------------------------------------------------------------
    # Gate 1: minimum image dimension
    # ------------------------------------------------------------------
    short_edges.sort()
    min_dim = settings.MIN_IMAGE_DIMENSION
    too_small = sum(1 for e in short_edges if e < min_dim)
    dim_pcts = {f"p{q}": int(percentile(short_edges, q))
                for q in (0.1, 0.5, 1, 2, 5, 10, 50, 100)}

    print()
    print("=" * 74)
    print("GATE 1 MINIMUM-DIMENSION CALIBRATION")
    print("=" * 74)
    print(f"Current MIN_IMAGE_DIMENSION: {min_dim}")
    print(f"  shortest edge: min {min(short_edges)}   median "
          f"{int(percentile(short_edges, 50))}   max {max(short_edges)}")
    for k, v in dim_pcts.items():
        print(f"    {k:>5}: {v:7d} px")
    print()
    print(f"At the CURRENT minimum of {min_dim} px: {too_small}/{len(short_edges)} "
          f"({100*too_small/len(short_edges):.1f}%) of genuine images would be "
          f"REJECTED before reaching the classifier.")
    if too_small:
        print("  The model was trained on these images after resizing to 224x224,")
        print("  so rejecting them at the door contradicts the training set. Either")
        print("  lower the minimum to admit them, or state why images the model was")
        print("  fitted on must not be graded.")

    # ------------------------------------------------------------------
    # Gate 3: contrast dynamic range
    # ------------------------------------------------------------------
    contrasts.sort()
    c_thr = settings.CONTRAST_THRESHOLD
    low_contrast = sum(1 for c in contrasts if c < c_thr)
    c_pcts = {f"p{q}": round(percentile(contrasts, q), 3)
              for q in (0.1, 0.5, 1, 2, 5, 10, 25, 50, 75, 90, 99)}

    def crowding(threshold, window=0.5):
        """How many images sit within +/- window of a candidate cut-point."""
        return sum(1 for c in contrasts if abs(c - threshold) <= window)

    print()
    print("=" * 74)
    print("GATE 3 CONTRAST CALIBRATION")
    print("=" * 74)
    print(f"Current CONTRAST_THRESHOLD: {c_thr}")
    print(f"  min {min(contrasts):.2f}   median {percentile(contrasts, 50):.2f}   "
          f"max {max(contrasts):.2f}")
    for k, v in c_pcts.items():
        print(f"    {k:>5}: {v:9.3f}")
    print()
    print(f"At the CURRENT threshold of {c_thr}: {low_contrast}/{len(contrasts)} "
          f"({100*low_contrast/len(contrasts):.1f}%) of genuine images would be REJECTED.")
    print(f"  Images within +/-0.5 of {c_thr}: {crowding(c_thr)} "
          f"({100*crowding(c_thr)/len(contrasts):.1f}%)")
    print()
    print("CROWDING AT EACH CANDIDATE - a cut-point in a dense region is fragile")
    print(f"  {'threshold':>10}  {'rejected':>9}  {'within +/-0.5':>14}")
    for q in (0.1, 0.5, 1, 2, 5, 10):
        t = percentile(contrasts, q)
        n = sum(1 for c in contrasts if c < t)
        print(f"  {t:10.2f}  {n:9d}  {crowding(t):14d}")
    print()
    print("  The decision-preservation check found 8 images flipping 18.0 -> 17.9")
    print("  under analysis subsampling. Two causes: the gate compared a value")
    print("  rounded to 1dp (fixed - it now decides on full precision), and 18.0")
    print("  sat in a crowded part of the distribution. Prefer a cut-point with")
    print("  few images beside it.")

    print()
    print("HOW TO CHOOSE")
    print("  The threshold's job is to reject images too blurred to grade, not to")
    print("  reject the corpus. A low percentile (0.5-2%) keeps that promise while")
    print("  admitting the images the model was trained on. Whatever you pick,")
    print("  record the percentile AND this distribution in")
    print("  docs/chapter4/validation_module_spec.md - a bare number with no")
    print("  derivation is how the current threshold got there.")
    print()
    print("  A deliberately blurred image (Gaussian radius 12) measured 0.9 in")
    print("  validation_test_results.csv, against 5.7-22.0 for unmodified images,")
    print("  so the metric separates the two populations. Only the cut-point is wrong.")

    report = {
        "corpus": args.images_dir,
        "subset": ("train+val (development)" if not args.allow_test_split
                   else "ALL IMAGES INCLUDING HELD-OUT TEST"),
        "manifest": manifest if os.path.exists(manifest) else None,
        "images_measured": len(values),
        "sampled": bool(args.sample),
        "current_threshold": current,
        "rejected_at_current_threshold": rejected_now,
        "rejected_share_at_current_threshold": round(rejected_now / len(values), 4),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
        "mean": round(statistics.mean(values), 3),
        "median": round(statistics.median(values), 3),
        "percentiles": pcts,
        "contrast": {
            "current_threshold": c_thr,
            "rejected_at_current_threshold": low_contrast,
            "rejected_share_at_current_threshold": round(low_contrast / len(contrasts), 4),
            "min": round(min(contrasts), 3),
            "median": round(percentile(contrasts, 50), 3),
            "max": round(max(contrasts), 3),
            "percentiles": c_pcts,
            "crowding_within_half_unit": {
                f"p{q}": crowding(percentile(contrasts, q))
                for q in (0.1, 0.5, 1, 2, 5, 10)
            },
        },
        "min_image_dimension": {
            "current_threshold": min_dim,
            "rejected_at_current_threshold": too_small,
            "rejected_share_at_current_threshold": round(too_small / len(short_edges), 4),
            "shortest_edge_min": min(short_edges),
            "shortest_edge_median": int(percentile(short_edges, 50)),
            "shortest_edge_max": max(short_edges),
            "percentiles": dim_pcts,
        },
        "note": ("Measurement only. The threshold is not written to configuration "
                 "by this script; the operating point is a judgement that must be "
                 "made and justified explicitly."),
    }
    # A calibration derived from a handful of images is not evidence, and the
    # default destination is the Chapter Four evidence folder. A smoke test on
    # a dozen generated images once wrote there; refusing small corpora by
    # default is cheaper than noticing afterwards.
    MIN_CORPUS = 500
    destination = args.out
    if destination == OUT and len(values) < MIN_CORPUS and not args.force:
        destination = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "blur_threshold_calibration.SMALL_CORPUS.json")
        print()
        print(f"REFUSING to write the evidence folder: only {len(values)} images "
              f"measured (minimum {MIN_CORPUS}).")
        print("A calibration from this few images is not a percentile estimate.")
        print(f"Writing to {destination} instead. Pass --force to override.")

    os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
    with open(destination, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nWritten: {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
