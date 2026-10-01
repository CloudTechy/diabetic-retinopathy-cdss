#!/usr/bin/env python3
"""
Set the Gate 1 and Gate 3 admission thresholds from a measured distribution.

THE DIVISION OF LABOUR

You choose a percentile. That is the judgement, and it is the thing that gets
written down: "we admit the sharpest 99% of the development corpus". This
script derives the numbers that percentile implies and records where they came
from.

What it will not do is search for a value that makes a test pass. The
percentile is declared first, on the command line, and the thresholds follow
from the corpus. If the resulting configuration still rejects images you think
it should admit, that is a finding about the percentile, not a reason to try
another one until the output looks better.

PREREQUISITE

    python backend/scripts/calibrate_blur_threshold.py <aptos>/train_images

which measures the TRAINING and VALIDATION partitions only and writes
docs/chapter4/blur_threshold_calibration.json.

Usage:
    python backend/scripts/apply_validation_thresholds.py --percentile 1.0
    python backend/scripts/apply_validation_thresholds.py --percentile 0.5 --dry-run
"""

import argparse
import json
import os
import re
import sys
from datetime import date

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))

CALIBRATION = os.path.join(REPO_ROOT, "docs", "chapter4",
                           "blur_threshold_calibration.json")
CONFIG = os.path.join(REPO_ROOT, "backend", "app", "core", "config.py")
SPEC = os.path.join(REPO_ROOT, "docs", "chapter4", "validation_module_spec.md")

ALLOWED = {"0.1", "0.5", "1", "1.0", "2", "2.0", "5", "5.0", "10", "10.0"}


def _pct_key(value: str) -> str:
    """Normalise '1.0' -> 'p1', '0.5' -> 'p0.5' to match the report's keys."""
    f = float(value)
    return f"p{int(f)}" if f == int(f) else f"p{f}"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--percentile", required=True,
                        help="Development-corpus percentile to admit from "
                             "(one of: " + ", ".join(sorted(ALLOWED)) + ")")
    parser.add_argument("--rationale", default=None,
                        help="One sentence recorded in the spec alongside the value")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would change and write nothing")
    args = parser.parse_args()

    if args.percentile not in ALLOWED:
        raise SystemExit(
            f"--percentile must be one of {sorted(ALLOWED)}; the calibration "
            f"report only carries those. Re-run the calibration if you need another.")

    if not os.path.exists(CALIBRATION):
        raise SystemExit(
            f"No calibration found at {CALIBRATION}.\n"
            "Run calibrate_blur_threshold.py against the APTOS images first - "
            "this script sets thresholds from measurements, it does not invent them.")

    with open(CALIBRATION, encoding="utf-8") as fh:
        cal = json.load(fh)

    subset = cal.get("subset", "unknown")
    if "HELD-OUT" in subset.upper():
        raise SystemExit(
            "The calibration included the held-out test partition. Choosing an\n"
            "operating point on the test set leaks it. Re-run the calibration\n"
            "without --allow-test-split.")

    n = cal["images_measured"]
    key = _pct_key(args.percentile)
    if key not in cal["percentiles"]:
        raise SystemExit(f"Percentile {key} absent from the report "
                         f"(has: {sorted(cal['percentiles'])}).")

    blur = round(float(cal["percentiles"][key]), 1)
    dim_block = cal.get("min_image_dimension") or {}
    dim = dim_block.get("percentiles", {}).get(key)
    if dim is None:
        raise SystemExit(
            "The calibration carries no resolution distribution. Re-run "
            "calibrate_blur_threshold.py - it measures both thresholds now.")
    dim = int(dim)

    c_block = cal.get("contrast") or {}
    contrast = c_block.get("percentiles", {}).get(key)
    if contrast is None:
        raise SystemExit(
            "The calibration carries no contrast distribution. Re-run "
            "calibrate_blur_threshold.py - it measures all three thresholds now.")
    contrast = round(float(contrast), 1)

    old_blur = cal.get("current_threshold")
    old_dim = dim_block.get("current_threshold")
    old_contrast = c_block.get("current_threshold")
    crowd = (c_block.get("crowding_within_half_unit") or {}).get(key)

    print("=" * 74)
    print("APPLY VALIDATION THRESHOLDS")
    print("=" * 74)
    print(f"Calibration:  {n} images, {subset}")
    print(f"Percentile:   {args.percentile}%  (admitting the top {100 - float(args.percentile):.1f}%)")
    print()
    print(f"  LAPLACIAN_BLUR_THRESHOLD   {old_blur}  ->  {blur}")
    print(f"  MIN_IMAGE_DIMENSION        {old_dim}  ->  {dim}")
    print(f"  CONTRAST_THRESHOLD         {old_contrast}  ->  {contrast}")
    if crowd is not None:
        print(f"      ({crowd} images sit within +/-0.5 of that cut-point)")
    print()
    print(f"By construction this admits {100 - float(args.percentile):.1f}% of the "
          f"development corpus on each metric.")
    print("It does NOT guarantee any particular validation-evidence outcome.")
    print("Re-run generate_validation_evidence.py to see what it actually does.")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    # ---- config ---------------------------------------------------------
    with open(CONFIG, encoding="utf-8") as fh:
        cfg = fh.read()
    cfg_new = re.sub(r"LAPLACIAN_BLUR_THRESHOLD:\s*float\s*=\s*[\d.]+",
                     f"LAPLACIAN_BLUR_THRESHOLD: float = {blur}", cfg)
    cfg_new = re.sub(r"MIN_IMAGE_DIMENSION:\s*int\s*=\s*\d+",
                     f"MIN_IMAGE_DIMENSION: int = {dim}", cfg_new)
    cfg_new = re.sub(r"CONTRAST_THRESHOLD:\s*float\s*=\s*[\d.]+",
                     f"CONTRAST_THRESHOLD: float = {contrast}", cfg_new)
    if cfg_new == cfg:
        raise SystemExit("Could not locate the settings in config.py; aborting.")
    with open(CONFIG, "w", encoding="utf-8") as fh:
        fh.write(cfg_new)
    print(f"\nUpdated {os.path.relpath(CONFIG, REPO_ROOT)}")

    # ---- spec -----------------------------------------------------------
    rationale = args.rationale or (
        f"Admit the sharpest and largest {100 - float(args.percentile):.1f}% of the "
        f"development corpus; the remainder is treated as too degraded to grade.")
    record = f"""
### Calibrated admission thresholds

| Setting | Value | Derivation |
| :--- | ---: | :--- |
| `LAPLACIAN_BLUR_THRESHOLD` | **{blur}** | {args.percentile}th percentile of Laplacian variance |
| `MIN_IMAGE_DIMENSION` | **{dim}** px | {args.percentile}th percentile of shortest edge |
| `CONTRAST_THRESHOLD` | **{contrast}** | {args.percentile}th percentile of foreground std |

**Declared percentile:** {args.percentile}% — {rationale}

**Measured over:** {n} images, {subset}, from
[`dataset_split_manifest.csv`](dataset_split_manifest.csv). The held-out test
partition is excluded: choosing an operating point on it would leak it.

**Distribution:** [`blur_threshold_calibration.json`](blur_threshold_calibration.json)
carries the full percentile table for both metrics, so this value can be
checked against the data it came from.

**Previous values:** `LAPLACIAN_BLUR_THRESHOLD` {old_blur}, `MIN_IMAGE_DIMENSION`
{old_dim}, `CONTRAST_THRESHOLD` {old_contrast}. All three were chosen a priori and
never measured against this corpus. The blur threshold rejected 10 of 10 genuine
held-out images; with it corrected, the contrast threshold still rejected 5 of 10.

*Applied {date.today().isoformat()} by `apply_validation_thresholds.py`.*
"""

    with open(SPEC, encoding="utf-8") as fh:
        spec = fh.read().replace("\r\n", "\n")

    marker = "### Calibrated admission thresholds"
    if marker in spec:
        start = spec.index(marker)
        end = spec.find("\n## ", start)
        spec = spec[:start] + record.lstrip("\n") + (spec[end:] if end != -1 else "")
    else:
        anchor = "> cut-point is in the wrong place."
        if anchor in spec:
            cut = spec.index(anchor) + len(anchor)
            spec = spec[:cut] + "\n" + record + spec[cut:]
        else:
            spec = spec.rstrip() + "\n\n" + record

    with open(SPEC, "w", encoding="utf-8") as fh:
        fh.write(spec)
    print(f"Updated {os.path.relpath(SPEC, REPO_ROOT)}")

    print("\nNext: python backend/scripts/generate_validation_evidence.py <aptos>/train_images")
    return 0


if __name__ == "__main__":
    sys.exit(main())
