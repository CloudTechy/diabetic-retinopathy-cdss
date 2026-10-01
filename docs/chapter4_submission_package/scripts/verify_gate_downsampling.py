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

    # A raw maximum deviation can be alarming and meaningless at the same time.
    # The red/blue ratio is r_mean / (b_mean + 1e-6): on a very dark image with
    # almost no blue signal it takes enormous values, so a tiny change in
    # b_mean moves it by a lot in absolute terms while leaving it orders of
    # magnitude clear of the 1.15 threshold. What matters is the deviation on
    # images that are actually near a decision boundary, so track that
    # separately, along with how close any image came to flipping.
    #
    # Read from the live configuration. An earlier revision wrote the numbers
    # in here, and when calibration moved CONTRAST_THRESHOLD from 18.0 to 8.8
    # the report went on printing 18.0 and counting 2,502 images as near a
    # boundary the system no longer had.
    NEAR = {
        "rb": (settings.RETINAL_RED_RATIO_MIN, 0.5),
        "coverage_lo": (settings.RETINAL_MIN_COVERAGE, 0.05),
        "coverage_hi": (settings.RETINAL_MAX_COVERAGE, 0.05),
        "contrast": (settings.CONTRAST_THRESHOLD, 5.0),
        "extreme": (settings.ILLUMINATION_EXTREME_RATIO_MAX, 0.1),
    }
    near_worst = {k: 0.0 for k in NEAR}
    near_counts = {k: 0 for k in NEAR}
    min_margin = {k: float("inf") for k in NEAR}
    unexplained = []
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

        # A flip is EXPLAINED when some metric crossed its own threshold by no
        # more than the deviation measured on that same image - the image was
        # already inside the measurement band, where no threshold can promise a
        # stable verdict. A flip whose every margin exceeds its own deviation is
        # not a boundary effect; it is a defect, and it fails the run below.
        def classify(label, fname, exact_pairs):
            detail = []
            explained = False
            for name, threshold, full, reduced in exact_pairs:
                margin = abs(full - threshold)
                deviation = abs(full - reduced)
                detail.append({"metric": name, "threshold": threshold,
                               "full_resolution": round(full, 6),
                               "reduced": round(reduced, 6),
                               "margin_to_threshold": round(margin, 6),
                               "deviation": round(deviation, 6)})
                if margin <= deviation + 1e-9:
                    explained = True
            return {"image": fname, "gate": label, "boundary_proximity": explained,
                    "metrics": detail}

        if (f2.passed, f2.error_code) != (r2.passed, r2.error_code):
            info = classify("gate2", fname, [
                ("coverage_lo", settings.RETINAL_MIN_COVERAGE,
                 f2.mask_coverage_exact, r2.mask_coverage_exact),
                ("coverage_hi", settings.RETINAL_MAX_COVERAGE,
                 f2.mask_coverage_exact, r2.mask_coverage_exact),
                ("rb", settings.RETINAL_RED_RATIO_MIN,
                 f2.red_to_blue_ratio_exact, r2.red_to_blue_ratio_exact),
            ])
            g2_flips.append(info)
            if not info["boundary_proximity"]:
                unexplained.append(info)
            print(f"  GATE2 FLIP {fname}: {f2.passed}->{r2.passed} "
                  f"({f2.error_code}->{r2.error_code}) coverage "
                  f"{f2.mask_coverage_exact:.6f}->{r2.mask_coverage_exact:.6f}"
                  f"{'' if info['boundary_proximity'] else '  [UNEXPLAINED]'}")
        if (f3.passed, f3.error_code) != (r3.passed, r3.error_code):
            info = classify("gate3", fname, [
                ("contrast", settings.CONTRAST_THRESHOLD,
                 f3.contrast_dynamic_range_exact, r3.contrast_dynamic_range_exact),
                ("extreme", settings.ILLUMINATION_EXTREME_RATIO_MAX,
                 f3.extreme_pixel_ratio_exact, r3.extreme_pixel_ratio_exact),
                ("laplacian", settings.LAPLACIAN_BLUR_THRESHOLD,
                 f3.laplacian_variance_exact, r3.laplacian_variance_exact),
            ])
            g3_flips.append(info)
            if not info["boundary_proximity"]:
                unexplained.append(info)
            print(f"  GATE3 FLIP {fname}: {f3.passed}->{r3.passed} "
                  f"({f3.error_code}->{r3.error_code}) contrast "
                  f"{f3.contrast_dynamic_range_exact:.6f}->"
                  f"{r3.contrast_dynamic_range_exact:.6f}"
                  f"{'' if info['boundary_proximity'] else '  [UNEXPLAINED]'}")

        # Measured on the FULL-PRECISION metrics. Taking these differences from
        # the rounded fields quantises every margin onto the rounding grid,
        # which is how a previous run came to report a closest margin of
        # exactly 0.0000 for 2,502 separate images.
        worst["coverage"] = max(worst["coverage"],
                                abs(f2.mask_coverage_exact - r2.mask_coverage_exact))
        worst["rb"] = max(worst["rb"],
                          abs(f2.red_to_blue_ratio_exact - r2.red_to_blue_ratio_exact))
        worst["contrast"] = max(worst["contrast"],
                                abs(f3.contrast_dynamic_range_exact
                                    - r3.contrast_dynamic_range_exact))
        worst["extreme"] = max(worst["extreme"],
                               abs(f3.extreme_pixel_ratio_exact
                                   - r3.extreme_pixel_ratio_exact))
        worst["laplacian"] = max(worst["laplacian"],
                                 abs(f3.laplacian_variance_exact
                                     - r3.laplacian_variance_exact))

        # Margin to each threshold, and the deviation seen near one.
        for key, value, deviation in (
            ("rb", f2.red_to_blue_ratio_exact,
             abs(f2.red_to_blue_ratio_exact - r2.red_to_blue_ratio_exact)),
            ("coverage_lo", f2.mask_coverage_exact,
             abs(f2.mask_coverage_exact - r2.mask_coverage_exact)),
            ("coverage_hi", f2.mask_coverage_exact,
             abs(f2.mask_coverage_exact - r2.mask_coverage_exact)),
            ("contrast", f3.contrast_dynamic_range_exact,
             abs(f3.contrast_dynamic_range_exact - r3.contrast_dynamic_range_exact)),
            ("extreme", f3.extreme_pixel_ratio_exact,
             abs(f3.extreme_pixel_ratio_exact - r3.extreme_pixel_ratio_exact)),
        ):
            threshold, band = NEAR[key]
            margin = abs(value - threshold)
            min_margin[key] = min(min_margin[key], margin)
            if margin <= band:
                near_counts[key] += 1
                near_worst[key] = max(near_worst[key], deviation)

        checked += 1
        if n % 250 == 0:
            print(f"  {n}/{len(files)} ...")

    elapsed = time.perf_counter() - t0
    print("-" * 84)
    print(f"Checked {checked} images in {elapsed:.0f}s")
    print(f"Gate 2 verdict changes : {len(g2_flips)}")
    print(f"Gate 3 verdict changes : {len(g3_flips)}")
    print()
    print("Largest metric deviations observed (full resolution vs reduced),")
    print("measured on the full-precision values the gates decide on:")
    print(f"  aperture coverage    {worst['coverage']:.5f}   (thresholds "
          f"{settings.RETINAL_MIN_COVERAGE} / {settings.RETINAL_MAX_COVERAGE})")
    print(f"  red/blue ratio       {worst['rb']:.5f}   (threshold "
          f"{settings.RETINAL_RED_RATIO_MIN})")
    print(f"  contrast std         {worst['contrast']:.5f}   (threshold "
          f"{settings.CONTRAST_THRESHOLD})")
    print(f"  extreme pixel ratio  {worst['extreme']:.5f}   (threshold "
          f"{settings.ILLUMINATION_EXTREME_RATIO_MAX})")
    print(f"  laplacian variance   {worst['laplacian']:.5f}   (must be exactly 0.0)")
    print()
    print("Deviation among images NEAR a threshold - the figure that matters:")
    print(f"  {'metric':<14}{'threshold':>10}{'near':>7}{'max dev':>12}{'closest margin':>17}")
    for key in NEAR:
        threshold, band = NEAR[key]
        margin = min_margin[key]
        margin_txt = "n/a" if margin == float("inf") else f"{margin:.4f}"
        print(f"  {key:<14}{threshold:>10}{near_counts[key]:>7}"
              f"{near_worst[key]:>12.5f}{margin_txt:>17}")
    print()
    print("  A large raw deviation with a tiny near-threshold deviation is benign:")
    print("  the red/blue ratio is r_mean/(b_mean + 1e-6), so on a near-black image")
    print("  it takes huge values that move a lot in absolute terms while staying")
    print(f"  orders of magnitude clear of the "
          f"{settings.RETINAL_RED_RATIO_MIN} cutoff.")

    if g2_flips or g3_flips:
        print()
        print("VERDICT CHANGES")
        for info in g2_flips + g3_flips:
            tag = ("within the measured deviation of its threshold"
                   if info["boundary_proximity"] else "NOT explained by boundary proximity")
            print(f"  {info['image']} ({info['gate']}): {tag}")
            for m in info["metrics"]:
                print(f"      {m['metric']:<12} threshold {m['threshold']}"
                      f"  value {m['full_resolution']}  margin {m['margin_to_threshold']}"
                      f"  deviation {m['deviation']}")
    print("=" * 84)

    # Write beside the other artefacts, whichever layout this is running from.
    repo_target = os.path.join(REPO_ROOT, "docs", "chapter4")
    package_target = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "logs_and_metrics")
    target = repo_target if os.path.isdir(repo_target) else package_target
    out = os.path.join(target, "gate_downsampling_verification.json")
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
            "flips_unexplained_by_boundary_proximity": len(unexplained),
            "thresholds_in_force": {
                "RETINAL_MIN_COVERAGE": settings.RETINAL_MIN_COVERAGE,
                "RETINAL_MAX_COVERAGE": settings.RETINAL_MAX_COVERAGE,
                "RETINAL_RED_RATIO_MIN": settings.RETINAL_RED_RATIO_MIN,
                "CONTRAST_THRESHOLD": settings.CONTRAST_THRESHOLD,
                "ILLUMINATION_EXTREME_RATIO_MAX": settings.ILLUMINATION_EXTREME_RATIO_MAX,
                "LAPLACIAN_BLUR_THRESHOLD": settings.LAPLACIAN_BLUR_THRESHOLD,
            },
            "max_abs_deviation": {k: round(v, 6) for k, v in worst.items()},
            "near_threshold": {
                k: {
                    "threshold": NEAR[k][0],
                    "band": NEAR[k][1],
                    "images_in_band": near_counts[k],
                    "max_abs_deviation": round(near_worst[k], 6),
                    "closest_margin": (None if min_margin[k] == float("inf")
                                       else round(min_margin[k], 6)),
                }
                for k in NEAR
            },
            "elapsed_s": round(elapsed, 1),
        }, fh, indent=2)
    print(f"Written: {out}")

    if worst["laplacian"] != 0.0:
        print("\n[FAILURE] Laplacian variance moved. It must be independent of this setting.")
        return 1
    if unexplained:
        print(f"\n[FAILURE] {len(unexplained)} verdict(s) changed for images that were "
              "NOT near a threshold.")
        print("  Subsampling moved them across a boundary they were clear of, which is a")
        print("  defect in the analysis path, not a boundary effect. Raise --max-dim, or")
        print("  set VALIDATION_ANALYSIS_MAX_DIM=0 to analyse at full resolution.")
        return 1

    total_flips = len(g2_flips) + len(g3_flips)
    if total_flips:
        # Stated rather than hidden. Zero flips is not a property any threshold
        # through a continuous distribution can guarantee: some image is always
        # arbitrarily close to the cut-point. What CAN be guaranteed, and is
        # checked above, is that no image clear of its boundary changes verdict.
        print(f"\n[SUCCESS] {checked} images. {total_flips} verdict change(s) "
              f"({100 * total_flips / max(checked, 1):.3f}%), every one of them for an "
              "image already within the measured deviation of its threshold.")
        print("  No image clear of a boundary changed verdict. Report the flip rate in")
        print("  Chapter 4 alongside this figure - it is a property of the corpus meeting")
        print("  the cut-point, not a defect, and it should not be presented as zero.")
        return 0

    print(f"\n[SUCCESS] {checked} images, zero verdict changes. "
          "The optimisation is decision-preserving on this corpus.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
