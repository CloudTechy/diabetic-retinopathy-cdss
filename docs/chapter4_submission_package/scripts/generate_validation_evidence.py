#!/usr/bin/env python3
"""
Produce genuine evidence of the three validation gates' behaviour.

This replaces a `validation_test_results.csv` that was fabricated. That file
cited ten "test cases": two were training images presented as held-out, six
used identifiers that do not exist in APTOS at all (`NONRET-XRAY-01`,
`CORRUPT-BYTE-01`, and similar), and every metric in it was a demonstration
constant copied from the frontend's mock data. Zero of its ten rows referenced
an actual held-out image.

Everything this script emits is measured by running the real gate functions
over real bytes.

Two kinds of case are produced, and the distinction is recorded in the output
so no reader has to guess:

  ACCEPT cases   Unmodified held-out images drawn from the committed manifest,
                 sampled across all five ICDR grades. These should pass.

  REJECT cases   Derived from a named held-out image by a stated, reproducible
                 transformation - Gaussian blur, channel inversion to destroy
                 the retinal spectral signature, truncation to corrupt the
                 file, and so on. The `derivation` column records exactly what
                 was done, so a reviewer can reproduce the input. A derived
                 image is a real test; an invented image ID is not.

Usage:
    python backend/scripts/generate_validation_evidence.py aptos2019/train_images
    python backend/scripts/generate_validation_evidence.py <dir> --per-grade 3
"""

import argparse
import csv
import io
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
BACKEND_ROOT = os.path.join(REPO_ROOT, "backend")
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

CHAPTER4 = os.path.join(REPO_ROOT, "docs", "chapter4")
MANIFEST = os.path.join(CHAPTER4, "dataset_split_manifest.csv")
OUT_CSV = os.path.join(CHAPTER4, "validation_test_results.csv")

FIELDS = [
    "test_case", "image_id", "source", "derivation", "expected",
    "gate1_status", "gate2_status", "gate2_aspect_ratio", "gate2_mask_coverage",
    "gate2_red_blue_ratio", "gate3_status", "gate3_laplacian_variance",
    "gate3_contrast_std", "gate3_extreme_ratio", "overall_status",
    "failed_gate", "error_code",
]


def load_manifest():
    with open(MANIFEST, newline="", encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if r["split"] == "test"]


def derive(kind, image_bytes):
    """Return (bytes, human-readable derivation) for a negative case."""
    from PIL import Image, ImageFilter
    import numpy as np

    if kind == "truncated":
        cut = len(image_bytes) // 3
        return image_bytes[:cut], f"first {cut} bytes only (file truncated)"

    if kind == "not_an_image":
        return b"%PDF-1.4\n% not an image\n", "PDF header substituted for image bytes"

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    if kind == "blurred":
        img = img.filter(ImageFilter.GaussianBlur(radius=12.0))
        note = "Gaussian blur, radius 12.0"
    elif kind == "channel_swapped":
        arr = np.asarray(img)[:, :, ::-1]          # RGB -> BGR
        img = Image.fromarray(arr)
        note = "red and blue channels swapped (destroys retinal spectral signature)"
    elif kind == "flat":
        arr = np.full_like(np.asarray(img), 128)
        img = Image.fromarray(arr)
        note = "uniform mid-grey (no contrast, no aperture)"
    elif kind == "letterboxed":
        w, h = img.size
        canvas = Image.new("RGB", (w, int(h * 0.45)), (0, 0, 0))
        canvas.paste(img.resize((w, int(h * 0.45))), (0, 0))
        img, note = canvas, "resized to a 2.2:1 panorama (outside camera proportions)"
    else:
        raise ValueError(kind)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), note


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images_dir", help="APTOS train_images/ directory")
    parser.add_argument("--per-grade", type=int, default=2,
                        help="Accept cases to sample per ICDR grade (default 2)")
    args = parser.parse_args()

    from app.services.validation.gate1_integrity import evaluate_gate1
    from app.services.validation.gate2_relevance import evaluate_gate2
    from app.services.validation.gate3_quality import evaluate_gate3

    if not os.path.isdir(args.images_dir):
        raise SystemExit(f"Not a directory: {args.images_dir}")

    test_rows = load_manifest()
    by_grade = {}
    for r in test_rows:
        by_grade.setdefault(int(r["true_grade"]), []).append(r)

    cases = []
    for grade in sorted(by_grade):
        for rec in by_grade[grade][:args.per_grade]:
            cases.append(("accept", rec, None))

    negatives = ["blurred", "channel_swapped", "flat", "letterboxed",
                 "truncated", "not_an_image"]
    donors = [by_grade[g][args.per_grade] for g in sorted(by_grade)
              if len(by_grade[g]) > args.per_grade]
    for i, kind in enumerate(negatives):
        cases.append(("reject", donors[i % len(donors)], kind))

    print("=" * 96)
    print("VALIDATION GATE EVIDENCE - measured by running the real gates")
    print("=" * 96)

    out = []
    for n, (expected, rec, kind) in enumerate(cases, 1):
        path = os.path.join(args.images_dir, f"{rec['image_id']}.png")
        if not os.path.exists(path):
            print(f"  [skip] {rec['image_id']} not on disk")
            continue
        with open(path, "rb") as fh:
            data = fh.read()

        derivation = "none (unmodified held-out image)"
        if kind:
            data, derivation = derive(kind, data)

        g1, pil = evaluate_gate1(data, f"{rec['image_id']}.png")
        g2 = g3 = None
        if pil is not None:
            g2 = evaluate_gate2(pil)
            if g2.passed:
                g3 = evaluate_gate3(pil)

        failed = 1 if not g1.passed else (2 if g2 and not g2.passed else
                                          (3 if g3 and not g3.passed else ""))
        overall = "ACCEPTED" if (g1.passed and g2 and g2.passed and g3 and g3.passed) else "REJECTED"
        code = (g1.error_code or (g2.error_code if g2 else None)
                or (g3.error_code if g3 else None) or "")

        out.append({
            "test_case": f"VAL-{n:02d}",
            "image_id": rec["image_id"],
            "source": f"APTOS held-out, {rec['true_label']}",
            "derivation": derivation,
            "expected": expected.upper(),
            "gate1_status": "PASS" if g1.passed else "FAIL",
            "gate2_status": ("PASS" if g2.passed else "FAIL") if g2 else "NOT REACHED",
            "gate2_aspect_ratio": f"{g2.aspect_ratio:.3f}" if g2 else "",
            "gate2_mask_coverage": f"{g2.mask_coverage:.4f}" if g2 else "",
            "gate2_red_blue_ratio": f"{g2.red_to_blue_ratio:.3f}" if g2 else "",
            "gate3_status": ("PASS" if g3.passed else "FAIL") if g3 else "NOT REACHED",
            "gate3_laplacian_variance": f"{g3.laplacian_variance:.1f}" if g3 else "",
            "gate3_contrast_std": f"{g3.contrast_dynamic_range:.1f}" if g3 else "",
            "gate3_extreme_ratio": f"{g3.extreme_pixel_ratio:.3f}" if g3 else "",
            "overall_status": overall,
            "failed_gate": failed,
            "error_code": code,
        })

        agrees = (overall == "ACCEPTED") == (expected == "accept")
        print(f"  VAL-{n:02d}  {rec['image_id']}  {expected:<6} -> {overall:<9} "
              f"{'OK' if agrees else 'UNEXPECTED'}  {derivation[:44]}")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(out)

    unexpected = [r for r in out
                  if (r["overall_status"] == "ACCEPTED") != (r["expected"] == "ACCEPT")]
    print("-" * 96)
    print(f"{len(out)} cases written to {OUT_CSV}")
    if unexpected:
        print(f"{len(unexpected)} case(s) behaved unexpectedly - report them, do not hide them:")
        for r in unexpected:
            print(f"  {r['test_case']} {r['image_id']} expected {r['expected']} "
                  f"got {r['overall_status']} ({r['error_code']})")
        return 1
    print("All cases behaved as expected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
