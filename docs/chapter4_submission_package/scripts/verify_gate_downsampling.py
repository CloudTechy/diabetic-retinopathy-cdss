#!/usr/bin/env python3
"""
Confirm on REAL fundus images that downsampling changed no gate verdict.

The unit tests cover synthetic images placed deliberately near each threshold.
This runs the same comparison across an actual APTOS directory, because the
claim being made — "this optimisation is decision-preserving" — is about real
clinical images, and synthetic evidence alone does not establish it.

For every image it evaluates Gates 2 and 3 twice, once with
VALIDATION_ANALYSIS_MAX_DIM set to 0 (full resolution, the pre-optimisation
behaviour) and once at the configured value, then reports any verdict that
differs along with the largest metric deviations observed.

Usage:
    python backend/scripts/verify_gate_downsampling.py aptos2019/train_images
    python backend/scripts/verify_gate_downsampling.py aptos2019/train_images --limit 500
    python backend/scripts/verify_gate_downsampling.py aptos2019/train_images --max-dim 1024

Exit code is 0 only if zero verdicts changed.
"""

import argparse
import json
import os
import random
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BACKEND_ROOT = os.path.join(REPO_ROOT, "backend")
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images_dir", help="Directory of fundus images")
    parser.add_argument("--limit", type=int, default=0, help="Check at most N images (0 = all)")
    parser.add_argument("--max-dim", type=int, default=None,
                        help="Analysis resolution to test (default: the configured value)")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed when --limit is used")
    args = parser.parse_args()

    from PIL import Image
    from app.core.config import settings
    from app.services.validation.gate2_relevance import evaluate_gate2
    from app.services.validation.gate3_quality import evaluate_gate3

    max_dim = args.max_dim if args.max_dim is not None else settings.VALIDATION_ANALYSIS_MAX_DIM

    if not os.path.isdir(args.images_dir):
        raise SystemExit(f"Not a directory: {args.images_dir}")

    files = sorted(f for f in os.listdir(args.images_dir)
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))
    if not files:
        raise SystemExit(f"No images found in {args.images_dir}")
    if args.limit and args.limit < len(files):
        random.seed(args.seed)
        files = sorted(random.sample(files, args.limit))

    print("=" * 84)
    print("GATE DOWNSAMPLING - DECISION-PRESERVATION CHECK ON REAL IMAGES")
    print("=" * 84)
    print(f"Images        : {args.images_dir}  ({len(files)} to check)")
    print(f"Comparing     : full resolution  vs  longest side {max_dim} (nearest)")
    print("-" * 84)

    g2_flips, g3_flips = [], []
    worst = {"coverage": 0.0, "rb": 0.0, "contrast": 0.0, "extreme": 0.0, "laplacian": 0.0}
    checked = 0
    t0 = time.perf_counter()

    for n, fname in enumerate(files, 1):
        path = os.path.join(args.images_dir, fname)
        try:
            with Image.open(path) as im:
                image = im.convert("RGB")
        except Exception as exc:
            print(f"  [skip] {fname}: {exc}")
            continue

        settings.VALIDATION_ANALYSIS_MAX_DIM = 0
        f2, f3 = evaluate_gate2(image), evaluate_gate3(image)
        settings.VALIDATION_ANALYSIS_MAX_DIM = max_dim
        r2, r3 = evaluate_gate2(image), evaluate_gate3(image)

        if (f2.passed, f2.error_code) != (r2.passed, r2.error_code):
            g2_flips.append((fname, f2.passed, r2.passed, f2.error_code, r2.error_code))
            print(f"  GATE2 FLIP {fname}: {f2.passed}->{r2.passed} "
                  f"({f2.error_code}->{r2.error_code}) coverage "
                  f"{f2.mask_coverage}->{r2.mask_coverage}")
        if (f3.passed, f3.error_code) != (r3.passed, r3.error_code):
            g3_flips.append((fname, f3.passed, r3.passed, f3.error_code, r3.error_code))
            print(f"  GATE3 FLIP {fname}: {f3.passed}->{r3.passed} "
                  f"({f3.error_code}->{r3.error_code}) contrast "
                  f"{f3.contrast_dynamic_range}->{r3.contrast_dynamic_range}")

        worst["coverage"] = max(worst["coverage"], abs(f2.mask_coverage - r2.mask_coverage))
        worst["rb"] = max(worst["rb"], abs(f2.red_to_blue_ratio - r2.red_to_blue_ratio))
        worst["contrast"] = max(worst["contrast"],
                                abs(f3.contrast_dynamic_range - r3.contrast_dynamic_range))
        worst["extreme"] = max(worst["extreme"],
                               abs(f3.extreme_pixel_ratio - r3.extreme_pixel_ratio))
        worst["laplacian"] = max(worst["laplacian"],
                                 abs(f3.laplacian_variance - r3.laplacian_variance))
        checked += 1
        if n % 250 == 0:
            print(f"  {n}/{len(files)} ...")

    elapsed = time.perf_counter() - t0
    print("-" * 84)
    print(f"Checked {checked} images in {elapsed:.0f}s")
    print(f"Gate 2 verdict changes : {len(g2_flips)}")
    print(f"Gate 3 verdict changes : {len(g3_flips)}")
    print()
    print("Largest metric deviations observed (full resolution vs reduced):")
    print(f"  aperture coverage    {worst['coverage']:.5f}   (thresholds 0.20 / 0.98)")
    print(f"  red/blue ratio       {worst['rb']:.5f}   (threshold 1.15)")
    print(f"  contrast std         {worst['contrast']:.5f}   (threshold 18.0)")
    print(f"  extreme pixel ratio  {worst['extreme']:.5f}   (threshold 0.35)")
    print(f"  laplacian variance   {worst['laplacian']:.5f}   (must be exactly 0.0)")
    print("=" * 84)

    out = os.path.join(REPO_ROOT, "docs", "chapter4", "gate_downsampling_verification.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({
            "images_dir": args.images_dir,
            "images_checked": checked,
            "analysis_max_dim": max_dim,
            "resampling": "nearest",
            "gate2_verdict_changes": len(g2_flips),
            "gate3_verdict_changes": len(g3_flips),
            "gate2_flips": g2_flips,
            "gate3_flips": g3_flips,
            "max_abs_deviation": {k: round(v, 6) for k, v in worst.items()},
            "elapsed_s": round(elapsed, 1),
        }, fh, indent=2)
    print(f"Written: {out}")

    if worst["laplacian"] != 0.0:
        print("\n[FAILURE] Laplacian variance moved. It must be independent of this setting.")
        return 1
    if g2_flips or g3_flips:
        print(f"\n[FAILURE] {len(g2_flips) + len(g3_flips)} verdict(s) changed. "
              "Raise --max-dim, or set VALIDATION_ANALYSIS_MAX_DIM=0 to disable.")
        return 1

    print(f"\n[SUCCESS] {checked} images, zero verdict changes. "
          "The optimisation is decision-preserving on this corpus.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
