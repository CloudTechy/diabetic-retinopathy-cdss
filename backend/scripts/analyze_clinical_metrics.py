#!/usr/bin/env python3
"""
Derive the clinical screening metrics for Chapter 4 from the held-out
predictions, and audit the split for byte-level leakage.

Everything this emits is recomputed from two committed artefacts:
  docs/chapter4/held_out_predictions.csv   (one row per held-out image)
  docs/chapter4/dataset_split_manifest.csv (SHA-256 of every image file)

It depends only on the standard library, so an external reviewer can run it
without installing PyTorch or scikit-learn:

    python backend/scripts/analyze_clinical_metrics.py

Why this exists separately from evaluate_model.py: overall accuracy is a poor
summary of a screening model on a cohort that is ~49% grade 0. The decision a
screening service actually makes is binary — refer this patient, or do not —
so the referable and sight-threatening operating points are the numbers that
carry clinical meaning, and they are reported here with confidence intervals.
"""

import csv
import json
import math
import os
from collections import Counter, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))


def _locate(filename, subdirs):
    """
    Find an artefact whether this script runs from the repository or from a
    standalone submission package.

    The archive is repository-relative, so docs/chapter4/ is the one location
    in both. The remaining candidates serve a superseded archive layout that
    split the artefacts across logs_and_metrics/ and dataset_sample_and_manifest/.
    """
    candidates = [os.path.join(REPO_ROOT, "docs", "chapter4", filename)]
    package_root = os.path.dirname(_HERE)
    for sub in subdirs:
        candidates.append(os.path.join(package_root, sub, filename))
    candidates.append(os.path.join(package_root, filename))

    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]  # report the canonical location in the error


CHAPTER4 = os.path.join(REPO_ROOT, "docs", "chapter4")
PREDICTIONS = _locate("held_out_predictions.csv", ["logs_and_metrics"])
MANIFEST = _locate("dataset_split_manifest.csv",
                   ["dataset_sample_and_manifest", "logs_and_metrics"])
OUT_JSON = os.path.join(os.path.dirname(PREDICTIONS), "clinical_metrics.json")

ICDR = [
    "Grade 0: No Apparent DR",
    "Grade 1: Mild NPDR",
    "Grade 2: Moderate NPDR",
    "Grade 3: Severe NPDR",
    "Grade 4: Proliferative DR",
]


def wilson_ci(successes, total, z=1.96):
    """Wilson score interval — valid for the small per-class counts here."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (round(100 * (centre - margin), 1), round(100 * (centre + margin), 1))


def quadratic_weighted_kappa(y_true, y_pred, n_classes=5):
    """QWK computed from first principles; no scikit-learn dependency."""
    observed = [[0] * n_classes for _ in range(n_classes)]
    for t, p in zip(y_true, y_pred):
        observed[t][p] += 1

    n = len(y_true)
    true_counts = Counter(y_true)
    pred_counts = Counter(y_pred)

    num = 0.0
    den = 0.0
    for i in range(n_classes):
        for j in range(n_classes):
            w = ((i - j) ** 2) / ((n_classes - 1) ** 2)
            expected = true_counts[i] * pred_counts[j] / n
            num += w * observed[i][j]
            den += w * expected
    return 1 - num / den if den else 0.0


def binary_operating_point(y_true, y_pred, threshold, name):
    """Collapse the ordinal grades to the referral decision at `threshold`."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t >= threshold and p >= threshold)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t >= threshold and p < threshold)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t < threshold and p >= threshold)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t < threshold and p < threshold)

    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0

    return {
        "name": name,
        "threshold": threshold,
        "n_positive": tp + fn,
        "n_negative": tn + fp,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        # Full precision. An earlier version stored two decimals, and every
        # consumer that rounded to one decimal rounded twice: 249/255 is
        # 97.647%, stored 97.65, shown 97.7. Presentation rounds exactly once.
        "sensitivity_pct": 100 * sens,
        "sensitivity_ci95": wilson_ci(tp, tp + fn),
        "specificity_pct": 100 * spec,
        "specificity_ci95": wilson_ci(tn, tn + fp),
        "ppv_pct": 100 * ppv,
        "npv_pct": 100 * npv,
    }


def per_class_metrics(y_true, y_pred):
    rows = []
    n = len(y_true)
    for g in range(5):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == g and p == g)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == g and p != g)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != g and p == g)
        tn = n - tp - fn - fp

        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0

        rows.append({
            "grade": g,
            "label": ICDR[g],
            "support": tp + fn,
            "sensitivity_pct": 100 * sens,
            "sensitivity_ci95": wilson_ci(tp, tp + fn),
            "specificity_pct": 100 * spec,
            "precision_pct": 100 * prec,
            "f1": f1,
        })
    return rows


def audit_leakage():
    """
    Report held-out images whose bytes are identical to a training image.

    The pipeline groups by APTOS's duplicated_info.csv, which is not part of
    the Kaggle competition download. When it is absent every image becomes its
    own group and the de-duplication silently does nothing, so this audit is
    run directly against the committed SHA-256 column instead of trusting that
    the grouping worked.
    """
    if not os.path.exists(MANIFEST):
        return None

    with open(MANIFEST, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    by_split = defaultdict(set)
    for r in rows:
        by_split[r["split"]].add(r["sha256_hash"])

    train_hashes = by_split["train"]
    test_rows = [r for r in rows if r["split"] == "test"]
    leaked = [r for r in test_rows if r["sha256_hash"] in train_hashes]
    leaked_ids = {r["image_id"] for r in leaked}

    # build_clean_split.py collapses each byte-identical group to a single
    # representative, so in a correctly built manifest EVERY group holds exactly
    # one row. An earlier version reported `any(c > 1)` and printed
    # "duplicate grouping effective: False" - true, and the opposite of how it
    # reads: the grouping had already done its work at split time. Report the
    # property that matters instead, which is that no group spans partitions.
    groups = Counter(r.get("duplicate_group_id", "") for r in rows)
    spanning = {gid: count for gid, count in groups.items() if count > 1}
    group_splits = defaultdict(set)
    for r in rows:
        group_splits[r.get("duplicate_group_id", "")].add(r["split"])
    cross_partition = sorted(gid for gid, splits in group_splits.items()
                             if len(splits) > 1)

    return {
        "n_test": len(test_rows),
        "n_test_leaked": len(leaked),
        "leaked_pct": round(100 * len(leaked) / len(test_rows), 2) if test_rows else 0.0,
        "leaked_by_grade": dict(sorted(Counter(int(r["true_grade"]) for r in leaked).items())),
        "leaked_image_ids": sorted(leaked_ids),
        "duplicate_groups": len(groups),
        "groups_with_more_than_one_row": len(spanning),
        "groups_collapsed_to_one_representative": not spanning,
        "groups_spanning_more_than_one_partition": len(cross_partition),
        "cross_partition_group_ids": cross_partition[:20],
        "n_val_leaked_from_train": len(by_split["val"] & train_hashes),
    }


def main():
    with open(PREDICTIONS, newline="", encoding="utf-8") as fh:
        preds = list(csv.DictReader(fh))

    y_true = [int(r["true_grade"]) for r in preds]
    y_pred = [int(r["predicted_grade"]) for r in preds]
    n = len(y_true)

    exact = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    within_one = sum(1 for t, p in zip(y_true, y_pred) if abs(t - p) <= 1)
    over = sum(1 for t, p in zip(y_true, y_pred) if p > t)
    under = sum(1 for t, p in zip(y_true, y_pred) if p < t)

    confusion = [[0] * 5 for _ in range(5)]
    for t, p in zip(y_true, y_pred):
        confusion[t][p] += 1

    leakage = audit_leakage()

    # Re-score on the subset with no byte-identical training counterpart, so
    # the headline figure cannot be attributed to memorised duplicates.
    clean = None
    if leakage and leakage["n_test_leaked"]:
        leaked_ids = set(leakage["leaked_image_ids"])
        pairs = [(t, p, r["image_id"]) for t, p, r in zip(y_true, y_pred, preds)]
        ct = [t for t, p, i in pairs if i not in leaked_ids]
        cp = [p for t, p, i in pairs if i not in leaked_ids]
        lt = [t for t, p, i in pairs if i in leaked_ids]
        lp = [p for t, p, i in pairs if i in leaked_ids]
        clean = {
            "n_clean": len(ct),
            "clean_accuracy_pct": round(100 * sum(1 for t, p in zip(ct, cp) if t == p) / len(ct), 2),
            "clean_qwk": round(quadratic_weighted_kappa(ct, cp), 6),
            "clean_referable": binary_operating_point(ct, cp, 2, "Referable DR (clean subset)"),
            "n_leaked": len(lt),
            "leaked_accuracy_pct": round(100 * sum(1 for t, p in zip(lt, lp) if t == p) / len(lt), 2) if lt else None,
        }

    result = {
        "n_test": n,
        "exact_accuracy_pct": 100 * exact / n,
        "within_one_grade_pct": 100 * within_one / n,
        "over_called_pct": 100 * over / n,
        "under_called_pct": 100 * under / n,
        "quadratic_weighted_kappa": quadratic_weighted_kappa(y_true, y_pred),
        "confusion_matrix": confusion,
        "per_class": per_class_metrics(y_true, y_pred),
        "operating_points": [
            binary_operating_point(y_true, y_pred, 2, "Referable DR"),
            binary_operating_point(y_true, y_pred, 3, "Sight-threatening DR"),
            binary_operating_point(y_true, y_pred, 1, "Any DR"),
        ],
        "leakage_audit": leakage,
        "leakage_adjusted": clean,
    }

    with open(OUT_JSON, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(result, fh, indent=2)

    # Console summary
    print("=" * 72)
    print(f"HELD-OUT COHORT: N = {n}")
    print("=" * 72)
    print(f"Exact accuracy       {result['exact_accuracy_pct']:.2f}%")
    print(f"Within-1-grade       {result['within_one_grade_pct']:.2f}%")
    print(f"QWK                  {result['quadratic_weighted_kappa']:.6f}")
    print(f"Over-called          {result['over_called_pct']:.1f}%   "
          f"Under-called {result['under_called_pct']:.1f}%")

    print(f"\n{'Grade':<6}{'N':>5}{'Sens%':>9}{'95% CI':>15}{'Spec%':>9}{'Prec%':>9}{'F1':>8}")
    for r in result["per_class"]:
        ci = f"{r['sensitivity_ci95'][0]}-{r['sensitivity_ci95'][1]}"
        print(f"{r['grade']:<6}{r['support']:>5}{r['sensitivity_pct']:>9.1f}{ci:>15}"
              f"{r['specificity_pct']:>9.1f}{r['precision_pct']:>9.1f}{r['f1']:>8.3f}")

    for op in result["operating_points"]:
        print(f"\n--- {op['name']} (grade >= {op['threshold']}) ---")
        print(f"  Sensitivity {op['sensitivity_pct']:.1f}%  95% CI {op['sensitivity_ci95']}")
        print(f"  Specificity {op['specificity_pct']:.1f}%  95% CI {op['specificity_ci95']}")
        print(f"  PPV {op['ppv_pct']:.1f}%   NPV {op['npv_pct']:.1f}%   missed = {op['fn']}")

    if leakage:
        print(f"\n--- LEAKAGE AUDIT ---")
        if leakage["groups_collapsed_to_one_representative"]:
            print(f"  duplicate groups: {leakage['duplicate_groups']}, "
                  f"each collapsed to one representative")
            print(f"    -> a group cannot span partitions by construction")
        else:
            print(f"  duplicate groups: {leakage['duplicate_groups']}, of which "
                  f"{leakage['groups_with_more_than_one_row']} hold more than one row")
            print(f"    groups spanning more than one partition: "
                  f"{leakage['groups_spanning_more_than_one_partition']}")
        print(f"  held-out images byte-identical to a training image: "
              f"{leakage['n_test_leaked']}/{leakage['n_test']} ({leakage['leaked_pct']}%)")
        if clean:
            print(f"  accuracy on the {clean['n_leaked']} affected images: {clean['leaked_accuracy_pct']}%")
            print(f"  accuracy on the {clean['n_clean']} clean images:     {clean['clean_accuracy_pct']}%")
            print(f"  clean-subset QWK: {clean['clean_qwk']:.6f}")

    print(f"\nWritten: {OUT_JSON}")


if __name__ == "__main__":
    main()
