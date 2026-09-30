#!/usr/bin/env python3
"""
Measure the Laplacian-variance distribution over the real corpus so the Gate 3
blur threshold can be chosen from evidence instead of guessed.

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

    if not os.path.isdir(args.images_dir):
        raise SystemExit(f"Not a directory: {args.images_dir}")

    names = sorted(n for n in os.listdir(args.images_dir)
                   if n.lower().endswith((".png", ".jpg", ".jpeg")))
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
    for i, name in enumerate(names, 1):
        try:
            with Image.open(os.path.join(args.images_dir, name)) as im:
                values.append(gate3_laplacian(im))
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
        "note": ("Measurement only. The threshold is not written to configuration "
                 "by this script; the operating point is a judgement that must be "
                 "made and justified explicitly."),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nWritten: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
