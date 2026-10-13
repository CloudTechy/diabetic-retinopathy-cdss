#!/usr/bin/env python3
"""
Post-hoc multiclass probability calibration (temperature scaling) for the
five-class EfficientNet-B0 diabetic-retinopathy classifier.

    python backend/scripts/calibrate_probabilities.py --images-dir C:/tmp/aptos2019/train_images

TWO PHASES, IN THIS ORDER, AND WHAT EACH ONE MAY READ

  Phase 1 - fit (validation partition only, 526 images)
    The manifest is filtered to split == "val" as it is read; no test row is
    held in memory during this phase and no test image is opened. Logits are
    produced for the 526 validation images, written to
    validation_predictions.csv, and one scalar T > 0 is fitted by minimising
    the validation multiclass NLL of softmax(z / T). T is written to
    temperature_scaling_artifact.json (and a byte-identical runtime copy beside
    the weights), and its SHA-256 is recorded. T is now frozen.

  Phase 2 - evaluate (test partition, 525 images, untouched until now)
    The frozen artifact is re-read from disk through the runtime engine's own
    loader. The test manifest rows are read, their logits produced and written
    to test_predictions.csv, and T is applied ONCE. Uncalibrated and calibrated
    NLL, ECE, MCE and multiclass Brier score are computed, together with the
    invariance checks (top-one predictions, accuracy, confusion matrix, QWK).
    Nothing in this phase writes or alters T; the artifact's SHA-256 is
    re-checked at the end and the run refuses if it changed.

  There is no test-set "deploy or not" gate: a choice between T = 1 and the
  fitted T made on test metrics would itself be selecting the temperature with
  test labels. The validation-fitted T is what the runtime serves.

  Phase 3 - runtime evidence
    The full runtime engine (checkpoint + runtime artifact copy) grades a
    sample of test images through EfficientNetB0InferenceService.predict(), and
    the class_scores it returns - the vector the interface receives - are
    compared with softmax(test logits / T).

The model is loaded through the runtime engine's loader, the preprocessing is
the engine's build_preprocess(), and T is applied with the engine's own
calibrated_probabilities(), so the evidence and the served probabilities come
from the same code.

FILES WRITTEN (all under --out, default docs/chapter4/probability_calibration/):

  validation_predictions.csv          phase 1: per-image logits, softmax, prediction (val)
  temperature_scaling_artifact.json   phase 1: the fitted parameter (no test-set statistic)
  test_predictions.csv                phase 2: per-image logits, uncalibrated and calibrated probabilities
  test_metrics.json                   phase 2: NLL / ECE / MCE / Brier before and after, invariance checks
  reliability_diagram_test.png        phase 2: test-set reliability diagram, before and after
  leakage_audit.json                  order of operations, split overlap, artifact hashes
  runtime_integration_evidence.json   phase 3: engine output vs softmax(logits / T)
  calibration_execution.log           verbatim stdout of the run
  calibration_evaluation_report.md    generated from the files above

and backend/models/calibration/temperature_scaling_artifact.json, the
byte-identical runtime copy.
"""

import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(BACKEND_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from corpus_guard import assert_corpus_is_authentic  # noqa: E402

NUM_CLASSES = 5
GRADE_LABELS = ["No Apparent DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]
# 15 equal-width bins is the primary ECE/MCE binning (fixed before any test
# result was seen); 10 bins is reported beside it as a sensitivity check.
PRIMARY_BINS = 15
SECONDARY_BINS = 10
T_BOUNDS = (0.05, 10.0)
ARTIFACT_VERSION = "ts-v2"
SCRIPT_NAME = os.path.basename(__file__)
RUNTIME_SAMPLE = 25

DEFAULT_MANIFEST = os.path.join(REPO_ROOT, "docs", "chapter4", "dataset_split_manifest.csv")
DEFAULT_HELD_OUT = os.path.join(REPO_ROOT, "docs", "chapter4", "held_out_predictions.csv")
DEFAULT_EVAL_SUMMARY = os.path.join(REPO_ROOT, "docs", "chapter4", "evaluation_summary.json")
DEFAULT_OUT = os.path.join(REPO_ROOT, "docs", "chapter4", "probability_calibration")
RUNTIME_ARTIFACT = os.path.join(BACKEND_DIR, "models", "calibration", "temperature_scaling_artifact.json")

# Keys the fitted-parameter JSON may carry. Anything else - in particular any
# test-set statistic - is refused before the file is written.
ARTIFACT_KEYS = {"version", "method", "temperature", "fit_split", "fit_counts", "fit",
                 "checkpoint", "manifest_sha256", "validation_ids_sha256", "script",
                 "created_at_utc", "note"}


# ---------------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------------

class Tee:
    """Write stdout to the terminal and to calibration_execution.log verbatim."""

    def __init__(self, path):
        self._file = open(path, "w", encoding="utf-8", newline="\n")
        self._stdout = sys.stdout

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)
        self._file.flush()

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        self._file.close()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ids_digest(ids):
    """SHA-256 of the sorted, newline-joined image ids: names a partition exactly."""
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def write_json(path, payload):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


def banner(text):
    print()
    print("=" * 78)
    print(text)
    print("=" * 78)


def rel(path):
    return os.path.relpath(path, REPO_ROOT).replace("\\", "/")


class AuditTrail:
    """Ordered record of what each phase read and wrote, for leakage_audit.json."""

    def __init__(self):
        self.events = []

    def __call__(self, phase, event, **detail):
        self.events.append({"step": len(self.events) + 1, "phase": phase, "event": event, **detail})


# ---------------------------------------------------------------------------
# manifest and image verification
# ---------------------------------------------------------------------------

def read_split(manifest_path, split):
    """Rows of ONE split. The other splits are discarded as the file is read."""
    with open(manifest_path, newline="", encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if r["split"] == split]


def read_ids_by_split(manifest_path):
    """image_id -> split only; carries no grades. Used for the overlap check."""
    with open(manifest_path, newline="", encoding="utf-8") as fh:
        out = {}
        for r in csv.DictReader(fh):
            out.setdefault(r["split"], set()).add(r["image_id"])
        return out


def grade_counts(rows):
    counts = {str(g): 0 for g in range(NUM_CLASSES)}
    for r in rows:
        counts[str(int(r["true_grade"]))] += 1
    return counts


def verify_images(images_dir, rows):
    """Every file must hash to the manifest's sha256_hash before it is used."""
    missing, mismatched = [], []
    for r in rows:
        path = os.path.join(images_dir, os.path.basename(r["file_path"]))
        if not os.path.exists(path):
            missing.append(r["image_id"])
        elif sha256_file(path) != r["sha256_hash"]:
            mismatched.append(r["image_id"])
    if missing or mismatched:
        raise SystemExit(
            f"REFUSED: {len(missing)} images missing from {images_dir}, {len(mismatched)} whose "
            f"bytes do not match the manifest: {(missing + mismatched)[:5]}")
    assert_corpus_is_authentic(images_dir, sample=0,
                               only=[os.path.basename(r["file_path"]) for r in rows])
    return len(rows)


# ---------------------------------------------------------------------------
# logits through the runtime engine's own code path
# ---------------------------------------------------------------------------

def load_weights(checkpoint_path):
    from app.services.ai_service import EfficientNetB0InferenceService
    service = EfficientNetB0InferenceService(checkpoint_path=checkpoint_path)
    model, device = service.load_weights()
    return service, model, device


def compute_logits(model, device, rows, images_dir, batch_size):
    import torch
    from PIL import Image
    from app.services.ai_service import build_preprocess

    preprocess = build_preprocess()
    out = np.zeros((len(rows), NUM_CLASSES), dtype=np.float64)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            tensors = []
            for r in chunk:
                with Image.open(os.path.join(images_dir, os.path.basename(r["file_path"]))) as im:
                    tensors.append(preprocess(im.convert("RGB")))
            out[start:start + len(chunk)] = model(torch.stack(tensors).to(device)).cpu().double().numpy()
            done = start + len(chunk)
            if done % (batch_size * 4) == 0 or done == len(rows):
                print(f"    {done}/{len(rows)} images")
    return out


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def softmax(logits, temperature=1.0):
    z = np.asarray(logits, dtype=np.float64) / float(temperature)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(probs, labels):
    p = np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)
    return float(-np.mean(np.log(p)))


def brier(probs, labels):
    """Multiclass Brier score: mean over images of sum_k (p_k - y_k)^2 (range 0..2)."""
    onehot = np.eye(NUM_CLASSES)[labels]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def confusion(labels, preds):
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    return cm


def qwk(labels, preds):
    """Quadratic weighted kappa from first principles (no sklearn)."""
    n, k = len(labels), NUM_CLASSES
    obs = confusion(labels, preds).astype(np.float64)
    expected = np.outer(np.bincount(labels, minlength=k), np.bincount(preds, minlength=k)) / n
    w = np.array([[((i - j) ** 2) / ((k - 1) ** 2) for j in range(k)] for i in range(k)])
    denom = float(np.sum(w * expected))
    return float(1.0 - np.sum(w * obs) / denom) if denom > 0 else 0.0


def reliability_bins(scores, hits, n_bins):
    """Equal-width bins on the top-one probability: (lo, hi], bin 0 also includes 0."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(scores)
    bins, ece, mce = [], 0.0, 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (scores > lo) & (scores <= hi) if i > 0 else (scores >= lo) & (scores <= hi)
        count = int(mask.sum())
        acc = conf = gap = None
        if count:
            acc, conf = float(hits[mask].mean()), float(scores[mask].mean())
            gap = abs(acc - conf)
            ece += (count / n) * gap
            mce = max(mce, gap)
        bins.append({"bin": i, "lower": float(lo), "upper": float(hi), "count": count,
                     "accuracy": acc, "mean_confidence": conf, "gap": gap})
    return bins, float(ece), float(mce)


def evaluate(probs, labels):
    """Every metric the report quotes, from one probability matrix."""
    preds = probs.argmax(axis=1)
    top = probs.max(axis=1)
    hits = (preds == labels).astype(np.float64)
    bins_p, ece_p, mce_p = reliability_bins(top, hits, PRIMARY_BINS)
    bins_s, ece_s, mce_s = reliability_bins(top, hits, SECONDARY_BINS)
    return {
        "n": int(len(labels)),
        "nll": nll(probs, labels),
        "brier": brier(probs, labels),
        f"ece_{PRIMARY_BINS}": ece_p, f"mce_{PRIMARY_BINS}": mce_p,
        f"ece_{SECONDARY_BINS}": ece_s, f"mce_{SECONDARY_BINS}": mce_s,
        "accuracy": float(hits.mean()),
        "correct": int(hits.sum()),
        "qwk": qwk(labels, preds),
        "confusion_matrix": confusion(labels, preds).tolist(),
        "mean_top_one_probability": float(top.mean()),
        f"bins_{PRIMARY_BINS}": bins_p,
        f"bins_{SECONDARY_BINS}": bins_s,
    }


# ---------------------------------------------------------------------------
# the fit - validation records only
# ---------------------------------------------------------------------------

def fit_temperature(logits, labels, split_tags):
    """
    Fit one scalar T > 0 by minimising the mean multiclass NLL on the records given.

    `split_tags` is checked record by record: if any tag is not "val" the fit
    is REFUSED. backend/tests/test_probability_calibration.py passes test-tagged
    records to this function and asserts the refusal.
    """
    from scipy.optimize import minimize_scalar

    tags = list(split_tags)
    if not (len(tags) == len(labels) == len(logits)):
        raise ValueError("logits, labels and split_tags must have the same length")
    offenders = sorted({t for t in tags if t != "val"})
    if offenders:
        raise ValueError(
            f"fit_temperature refuses records tagged {offenders}; only 'val' records may be "
            "used to fit the temperature (the test partition must never reach the fit)")
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)

    def objective(t):
        return nll(softmax(logits, t), labels)

    result = minimize_scalar(objective, bounds=T_BOUNDS, method="bounded",
                             options={"xatol": 1e-8, "maxiter": 500})
    t_star = float(result.x)
    if not (math.isfinite(t_star) and T_BOUNDS[0] < t_star < T_BOUNDS[1]):
        raise RuntimeError(f"fitted temperature {t_star} is not inside {T_BOUNDS}")
    return t_star, {
        "objective": "mean multiclass negative log-likelihood of softmax(logits / T) on the validation split",
        "optimiser": "scipy.optimize.minimize_scalar(method='bounded')",
        "bounds": list(T_BOUNDS),
        "xatol": 1e-8,
        "function_evaluations": int(result.nfev),
        "converged": bool(result.success),
        "validation_nll_at_T_1": objective(1.0),
        "validation_nll_at_fitted_T": float(result.fun),
        "deterministic": "the objective is a closed-form function of fixed logits and labels and "
                         "the bounded method draws no random numbers; a rerun gives the same T",
    }


# ---------------------------------------------------------------------------
# writers and plot
# ---------------------------------------------------------------------------

def write_validation_predictions(path, rows, logits, probs):
    header = (["image_id", "split", "sha256_hash", "true_grade"]
              + [f"logit_{k}" for k in range(NUM_CLASSES)]
              + [f"prob_uncalibrated_{k}" for k in range(NUM_CLASSES)] + ["predicted_grade"])
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        for i, r in enumerate(rows):
            w.writerow([r["image_id"], r["split"], r["sha256_hash"], int(r["true_grade"])]
                       + [repr(float(v)) for v in logits[i]]
                       + [repr(float(v)) for v in probs[i]] + [int(probs[i].argmax())])


def write_test_predictions(path, rows, logits, before, after):
    header = (["image_id", "split", "sha256_hash", "true_grade"]
              + [f"logit_{k}" for k in range(NUM_CLASSES)]
              + [f"prob_uncalibrated_{k}" for k in range(NUM_CLASSES)]
              + [f"prob_calibrated_{k}" for k in range(NUM_CLASSES)]
              + ["predicted_grade_uncalibrated", "predicted_grade_calibrated"])
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        for i, r in enumerate(rows):
            w.writerow([r["image_id"], r["split"], r["sha256_hash"], int(r["true_grade"])]
                       + [repr(float(v)) for v in logits[i]]
                       + [repr(float(v)) for v in before[i]]
                       + [repr(float(v)) for v in after[i]]
                       + [int(before[i].argmax()), int(after[i].argmax())])


def plot_reliability(before, after, temperature, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

    key = f"bins_{PRIMARY_BINS}"
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.6), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.1]})
    for col, (m, title) in enumerate(((before, "Uncalibrated (T = 1)"),
                                      (after, f"Calibrated (validation-fitted T = {temperature:.4f})"))):
        ax, hx = axes[0][col], axes[1][col]
        bins = m[key]
        width = 1.0 / len(bins)
        centres = [(b["lower"] + b["upper"]) / 2 for b in bins]
        ax.plot([0, 1], [0, 1], linestyle="--", color="#64748b", linewidth=1.0)
        for c, b in zip(centres, bins):
            if b["count"]:
                ax.bar(c, b["accuracy"], width=width * 0.92, color="#94a3b8", edgecolor="white", zorder=2)
                ax.plot([c], [b["mean_confidence"]], marker="o", color="#1d4ed8", markersize=4.5, zorder=3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{title}\nNLL {m['nll']:.4f}  ECE {m[f'ece_{PRIMARY_BINS}']:.4f}  "
                     f"MCE {m[f'mce_{PRIMARY_BINS}']:.4f}  Brier {m['brier']:.4f}", loc="left", fontsize=9.5)
        if col == 0:
            from matplotlib.lines import Line2D
            from matplotlib.patches import Patch
            ax.set_ylabel("observed accuracy")
            ax.legend(handles=[
                Line2D([], [], linestyle="--", color="#64748b", label="perfect calibration"),
                Patch(color="#94a3b8", label="observed accuracy in bin"),
                Line2D([], [], marker="o", color="#1d4ed8", linestyle="none",
                       label="mean top-one probability in bin")],
                loc="upper left", frameon=False, fontsize=7.5)
        hx.bar(centres, [b["count"] for b in bins], width=width * 0.92, color="#94a3b8", edgecolor="white")
        hx.set_xlabel(f"top-one probability ({PRIMARY_BINS} equal-width bins)")
        if col == 0:
            hx.set_ylabel("images")
    fig.suptitle(f"Test-set reliability diagram (n = {before['n']}, held-out partition; "
                 "T fitted on the validation partition only)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_in_subprocess(metrics_path, png_path):
    """
    Draw the diagram from test_metrics.json in a fresh interpreter.

    torch and matplotlib each bring an OpenMP runtime in some distributions
    (Anaconda), and loading both in one process aborts. The diagram needs only
    the JSON, so it is drawn where torch is never imported.
    """
    import subprocess
    subprocess.run([sys.executable, os.path.abspath(__file__), "--plot", metrics_path, png_path], check=True)


def cross_check_held_out(path, rows, probs):
    """Agreement of this run's test probabilities with the committed held_out_predictions.csv."""
    if not os.path.exists(path):
        return {"present": False}
    with open(path, newline="", encoding="utf-8") as fh:
        committed = {r["image_id"]: r for r in csv.DictReader(fh)}
    compared = agree = 0
    worst = 0.0
    for i, r in enumerate(rows):
        c = committed.get(r["image_id"])
        if c is None:
            continue
        compared += 1
        agree += int(int(c["predicted_grade"]) == int(probs[i].argmax()))
        worst = max(worst, max(abs(float(c[f"score_grade_{k}"]) - probs[i, k]) for k in range(NUM_CLASSES)))
    return {"present": True, "rows_compared": compared, "predicted_grade_agreements": agree,
            "max_abs_probability_difference": worst,
            "note": "committed file is from the Colab T4 (CUDA) run, rounded to 6 dp; this run is CPU"}


# ---------------------------------------------------------------------------
# phases
# ---------------------------------------------------------------------------

def phase_fit(args, audit, model, device, ckpt_sha):
    banner("PHASE 1 - FIT on the validation partition only")
    val_rows = read_split(args.manifest, "val")
    audit("fit", "read manifest rows with split == 'val' only", n=len(val_rows))
    print(f"validation rows: {len(val_rows)}  per grade {grade_counts(val_rows)}")
    verified = verify_images(args.images_dir, val_rows)
    audit("fit", "verified validation image bytes against manifest SHA-256", n=verified)

    val_logits = compute_logits(model, device, val_rows, args.images_dir, args.batch_size)
    val_labels = np.array([int(r["true_grade"]) for r in val_rows])
    val_probs = softmax(val_logits)
    val_path = os.path.join(args.out, "validation_predictions.csv")
    write_validation_predictions(val_path, val_rows, val_logits, val_probs)
    audit("fit", "wrote validation_predictions.csv", sha256=sha256_file(val_path))

    temperature, fit = fit_temperature(val_logits, val_labels, [r["split"] for r in val_rows])
    audit("fit", "fitted T by minimising validation NLL", temperature=temperature)
    print(f"fitted temperature T = {temperature:.10f}")
    print(f"validation NLL: T=1 {fit['validation_nll_at_T_1']:.6f} -> fitted {fit['validation_nll_at_fitted_T']:.6f}")

    artifact = {
        "version": ARTIFACT_VERSION,
        "method": "temperature_scaling",
        "temperature": temperature,
        "fit_split": "val",
        "fit_counts": {"n": len(val_rows), "per_grade": grade_counts(val_rows)},
        "fit": fit,
        "checkpoint": {"filename": os.path.basename(args.checkpoint_path), "sha256": ckpt_sha},
        "manifest_sha256": sha256_file(args.manifest),
        "validation_ids_sha256": ids_digest(r["image_id"] for r in val_rows),
        "script": {"name": SCRIPT_NAME, "sha256": sha256_file(os.path.abspath(__file__))},
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat(),
        "note": "fitted on the validation partition only, before any test image was read; "
                "carries no test-set statistic; apply as softmax(logits / temperature)",
    }
    extra = set(artifact) - ARTIFACT_KEYS
    if extra:
        raise SystemExit(f"REFUSED: artifact carries unexpected keys {sorted(extra)}")
    art_path = os.path.join(args.out, "temperature_scaling_artifact.json")
    write_json(art_path, artifact)
    os.makedirs(os.path.dirname(args.runtime_artifact), exist_ok=True)
    with open(art_path, "rb") as src, open(args.runtime_artifact, "wb") as dst:
        dst.write(src.read())
    frozen_sha = sha256_file(art_path)
    if sha256_file(args.runtime_artifact) != frozen_sha:
        raise SystemExit("REFUSED: runtime artifact copy is not byte-identical")
    audit("fit", "froze T: wrote temperature_scaling_artifact.json and runtime copy",
          sha256=frozen_sha, runtime_copy=rel(args.runtime_artifact))
    print(f"artifact frozen: {rel(art_path)}  sha256 {frozen_sha}")
    return {"rows": val_rows, "logits": val_logits, "labels": val_labels,
            "artifact_path": art_path, "artifact_sha256": frozen_sha}


def phase_evaluate(args, audit, model, device, ckpt_sha, fitted):
    from app.services.ai_service import calibrated_probabilities, load_calibration_artifact

    banner("PHASE 2 - EVALUATE: frozen T applied once to the untouched test partition")
    artifact = load_calibration_artifact(fitted["artifact_path"], ckpt_sha)
    temperature = float(artifact["temperature"])
    audit("evaluate", "re-read frozen artifact through the runtime loader", temperature=temperature,
          sha256=sha256_file(fitted["artifact_path"]))

    test_rows = read_split(args.manifest, "test")
    audit("evaluate", "read manifest rows with split == 'test' (first access to the test partition)",
          n=len(test_rows))
    verified = verify_images(args.images_dir, test_rows)
    audit("evaluate", "verified test image bytes against manifest SHA-256", n=verified)
    print(f"test rows: {len(test_rows)}  per grade {grade_counts(test_rows)}")

    test_logits = compute_logits(model, device, test_rows, args.images_dir, args.batch_size)
    test_labels = np.array([int(r["true_grade"]) for r in test_rows])
    before_p = softmax(test_logits)
    after_p = softmax(test_logits, temperature)       # the one application of T
    audit("evaluate", "applied frozen T once to the test logits", temperature=temperature)

    # the runtime function must agree with the vectorised application, row by row
    runtime_diff = max(float(np.max(np.abs(np.array(calibrated_probabilities(z.tolist(), temperature)) - p)))
                       for z, p in zip(test_logits, after_p))
    if runtime_diff > 1e-12:
        raise SystemExit(f"REFUSED: runtime calibrated_probabilities() differs by {runtime_diff}")

    before, after = evaluate(before_p, test_labels), evaluate(after_p, test_labels)
    changed = int((before_p.argmax(axis=1) != after_p.argmax(axis=1)).sum())
    invariance = {
        "top_one_predictions_changed": changed,
        "accuracy_identical": before["accuracy"] == after["accuracy"],
        "confusion_matrix_identical": before["confusion_matrix"] == after["confusion_matrix"],
        "qwk_identical": before["qwk"] == after["qwk"],
    }
    summary = {}
    if os.path.exists(args.eval_summary):
        with open(args.eval_summary, encoding="utf-8") as fh:
            committed = json.load(fh)
        summary = {"committed_evaluation_summary": rel(args.eval_summary),
                   "confusion_matrix_matches_committed": committed.get("confusion_matrix") == after["confusion_matrix"],
                   "accuracy_matches_committed": round(after["accuracy"], 6) == committed.get("accuracy"),
                   "qwk_matches_committed": round(after["qwk"], 6) == committed.get("qwk")}
    invariance.update(summary)
    print(json.dumps(invariance, indent=2))
    if changed or not all(v for k, v in invariance.items() if k.endswith(("identical",))):
        raise SystemExit("REFUSED: temperature scaling changed a top-one prediction")

    test_path = os.path.join(args.out, "test_predictions.csv")
    write_test_predictions(test_path, test_rows, test_logits, before_p, after_p)
    audit("evaluate", "wrote test_predictions.csv", sha256=sha256_file(test_path))

    for name, m in (("uncalibrated", before), ("calibrated", after)):
        print(f"test {name:12s}: NLL {m['nll']:.6f}  ECE{PRIMARY_BINS} {m[f'ece_{PRIMARY_BINS}']:.6f}  "
              f"MCE{PRIMARY_BINS} {m[f'mce_{PRIMARY_BINS}']:.6f}  ECE{SECONDARY_BINS} {m[f'ece_{SECONDARY_BINS}']:.6f}  "
              f"MCE{SECONDARY_BINS} {m[f'mce_{SECONDARY_BINS}']:.6f}  Brier {m['brier']:.6f}  "
              f"acc {m['accuracy']:.6f}  QWK {m['qwk']:.6f}")

    cross = cross_check_held_out(args.held_out, test_rows, before_p)
    print(f"cross-check vs held_out_predictions.csv: {cross}")

    metrics = {
        "split": "test",
        "n": len(test_rows),
        "temperature": temperature,
        "temperature_source": f"{rel(fitted['artifact_path'])} (fit_split=val; sha256 {fitted['artifact_sha256']})",
        "binning": {"primary": f"{PRIMARY_BINS} equal-width bins on the top-one probability",
                    "secondary": f"{SECONDARY_BINS} equal-width bins", "bin_rule": "(lower, upper]; bin 0 includes 0"},
        "brier_definition": "mean over images of sum over the 5 classes of (p_k - y_k)^2",
        "uncalibrated": before,
        "calibrated": after,
        "change": {k: after[k] - before[k] for k in
                   ("nll", "brier", f"ece_{PRIMARY_BINS}", f"mce_{PRIMARY_BINS}",
                    f"ece_{SECONDARY_BINS}", f"mce_{SECONDARY_BINS}")},
        "invariance": invariance,
        "runtime_function_max_abs_difference": runtime_diff,
        "held_out_cross_check": cross,
    }
    metrics_path = os.path.join(args.out, "test_metrics.json")
    write_json(metrics_path, metrics)
    plot_in_subprocess(metrics_path, os.path.join(args.out, "reliability_diagram_test.png"))
    audit("evaluate", "wrote test_metrics.json and reliability_diagram_test.png")
    return {"rows": test_rows, "logits": test_logits, "temperature": temperature, "metrics": metrics}


def phase_runtime(args, audit, ckpt_path, evaluated):
    """Grade test images through the full runtime engine; compare with softmax(logits / T)."""
    from PIL import Image
    from app.core.config import settings
    from app.services.ai_service import EfficientNetB0InferenceService

    banner(f"PHASE 3 - RUNTIME: EfficientNetB0InferenceService.predict() on {RUNTIME_SAMPLE} test images")
    old_attr = settings.STORAGE_ATTRIBUTIONS_PATH
    settings.STORAGE_ATTRIBUTIONS_PATH = tempfile.mkdtemp(prefix="calib_gradcam_")
    try:
        engine = EfficientNetB0InferenceService(checkpoint_path=ckpt_path,
                                                calibration_artifact_path=args.runtime_artifact)
        engine.load_model()
        T = evaluated["temperature"]
        if engine.temperature != T:
            raise SystemExit(f"REFUSED: runtime engine loaded T={engine.temperature}, evaluation used {T}")
        rows_out, worst_cal, worst_logit = [], 0.0, 0.0
        for i, r in enumerate(evaluated["rows"][:RUNTIME_SAMPLE]):
            with Image.open(os.path.join(args.images_dir, os.path.basename(r["file_path"]))) as im:
                out = engine.predict(im.convert("RGB"))
            served = np.array([cs["score"] for cs in out.class_scores])
            expected = softmax(evaluated["logits"][i:i + 1], T)[0]
            raw = softmax(evaluated["logits"][i:i + 1])[0]
            d_cal = float(np.max(np.abs(served - expected)))
            d_logit = float(np.max(np.abs(np.array(out.logits) - evaluated["logits"][i])))
            worst_cal, worst_logit = max(worst_cal, d_cal), max(worst_logit, d_logit)
            rows_out.append({"image_id": r["image_id"], "primary_grade": out.primary_grade,
                             "served_class_scores": served.tolist(),
                             "expected_softmax_logits_over_T": expected.tolist(),
                             "uncalibrated_softmax": raw.tolist(),
                             "max_abs_difference_served_vs_expected": d_cal,
                             "max_abs_difference_served_vs_uncalibrated": float(np.max(np.abs(served - raw)))})
    finally:
        settings.STORAGE_ATTRIBUTIONS_PATH = old_attr
    evidence = {
        "engine": "app.services.ai_service.EfficientNetB0InferenceService (load_model + predict)",
        "runtime_artifact": rel(args.runtime_artifact),
        "runtime_artifact_sha256": sha256_file(args.runtime_artifact),
        "temperature_loaded_by_engine": engine.temperature,
        "model_version_reported": out.model_version,
        "images": len(rows_out),
        "max_abs_difference_served_vs_softmax_logits_over_T": worst_cal,
        "max_abs_logit_difference_single_vs_batched": worst_logit,
        "tolerance": 1e-5,
        "served_vector_is_calibrated": worst_cal <= 1e-5,
        "per_image": rows_out,
    }
    write_json(os.path.join(args.out, "runtime_integration_evidence.json"), evidence)
    audit("runtime", "graded test images through the runtime engine", images=len(rows_out),
          max_abs_difference=worst_cal)
    print(f"served vs softmax(logits/T): max abs diff {worst_cal:.2e} over {len(rows_out)} images")
    if worst_cal > 1e-5:
        raise SystemExit("REFUSED: the runtime engine does not serve softmax(logits / T)")
    return evidence


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def generate_report(out_dir, artifact, metrics, audit_doc, runtime):
    b, a = metrics["uncalibrated"], metrics["calibrated"]
    inv = metrics["invariance"]
    t = artifact["temperature"]
    P, S = PRIMARY_BINS, SECONDARY_BINS

    def row(label, key, fmt="{:.4f}"):
        return f"| {label} | {fmt.format(b[key])} | {fmt.format(a[key])} | {a[key] - b[key]:+.4f} |"

    L = [
        "# Probability calibration (temperature scaling): validation-fitted, test-evaluated",
        "",
        "Generated by `backend/scripts/calibrate_probabilities.py`. Every number is read from the JSON and CSV files in this directory.",
        "",
        "## 1. Procedure",
        "",
        f"1. **Fit (validation partition only).** Logits for the {artifact['fit_counts']['n']} validation images were produced with the fixed checkpoint (SHA-256 `{artifact['checkpoint']['sha256']}`) and written to `validation_predictions.csv`. One scalar $T$ was fitted by minimising the validation multiclass NLL of $\\mathrm{{softmax}}(z/T)$ with `{artifact['fit']['optimiser']}` on bounds {artifact['fit']['bounds']}. No test row was read and no test image was opened in this phase.",
        f"2. **Freeze.** $T$ was written to `temperature_scaling_artifact.json` (SHA-256 `{audit_doc['artifact_sha256_at_freeze']}`) and copied byte-identically to `backend/models/calibration/temperature_scaling_artifact.json` for the runtime engine.",
        f"3. **Evaluate (test partition, {metrics['n']} images).** Only after the freeze were the test rows read, their logits produced (`test_predictions.csv`), and the frozen $T$ applied once.",
        "4. **Runtime.** The full inference engine, loading the runtime copy of the artifact, graded test images; its served class scores were compared with $\\mathrm{softmax}(z/T)$ (section 5).",
        "",
        "There is no test-set gate deciding whether to use the fitted $T$: such a gate would select the temperature with test labels.",
        "",
        "## 2. Fitted parameter",
        "",
        f"- **Validation-fitted temperature:** $T = {t:.6f}$",
        f"- Validation NLL: {artifact['fit']['validation_nll_at_T_1']:.6f} at $T=1$, {artifact['fit']['validation_nll_at_fitted_T']:.6f} at the fitted $T$ ({artifact['fit']['function_evaluations']} function evaluations; converged: {artifact['fit']['converged']}).",
        "- Validation per grade: " + ", ".join(f"grade {g}: {n}" for g, n in artifact['fit_counts']['per_grade'].items()) + ".",
        "",
        "## 3. Test set, before and after the frozen temperature",
        "",
        "| Metric | Uncalibrated ($T=1$) | Calibrated | Change |",
        "| :--- | ---: | ---: | ---: |",
        row("Multiclass NLL", "nll"),
        row(f"ECE ({P} bins, primary)", f"ece_{P}"),
        row(f"MCE ({P} bins, primary)", f"mce_{P}"),
        row(f"ECE ({S} bins, sensitivity)", f"ece_{S}"),
        row(f"MCE ({S} bins, sensitivity)", f"mce_{S}"),
        row("Multiclass Brier score", "brier"),
        "",
        f"Reliability diagram: [`reliability_diagram_test.png`](reliability_diagram_test.png) ({P} equal-width bins on the top-one probability, test set).",
        "",
        "**Reading.** "
        + ("The NLL fell" if a["nll"] < b["nll"] else "The NLL rose")
        + f" by {abs(a['nll'] - b['nll']):.4f}. The {P}-bin ECE {'fell' if a[f'ece_{P}'] < b[f'ece_{P}'] else 'rose'} by {abs(a[f'ece_{P}'] - b[f'ece_{P}']):.4f} and the {S}-bin ECE {'fell' if a[f'ece_{S}'] < b[f'ece_{S}'] else 'rose'} by {abs(a[f'ece_{S}'] - b[f'ece_{S}']):.4f}; the Brier score {'fell' if a['brier'] < b['brier'] else 'rose'} by {abs(a['brier'] - b['brier']):.4f}. "
        + ("Every metric moved in the same direction." if len({a['nll'] < b['nll'], a[f'ece_{P}'] < b[f'ece_{P}'], a[f'ece_{S}'] < b[f'ece_{S}'], a['brier'] < b['brier']}) == 1
           else "The metrics do not move together, so the test set supports only the changes stated above, metric by metric; it does not show an overall improvement in calibration error."),
        "",
        "## 4. Invariance: top-one predictions, accuracy, confusion matrix, QWK",
        "",
        f"- Top-one predictions changed: **{inv['top_one_predictions_changed']}** of {metrics['n']}.",
        f"- Accuracy identical: **{inv['accuracy_identical']}** ({a['correct']} / {a['n']} = {a['accuracy']:.4f}).",
        f"- Confusion matrix identical: **{inv['confusion_matrix_identical']}**.",
        f"- QWK identical: **{inv['qwk_identical']}** ({a['qwk']:.6f}).",
    ]
    if "confusion_matrix_matches_committed" in inv:
        L.append(f"- Against the committed `evaluation_summary.json`: confusion matrix matches **{inv['confusion_matrix_matches_committed']}**, accuracy matches **{inv['accuracy_matches_committed']}**, QWK matches **{inv['qwk_matches_committed']}**.")
    L += [
        "",
        "Confusion matrix (rows = true grade 0..4, columns = predicted grade 0..4), identical before and after:",
        "",
        "```",
        *[" ".join(f"{v:4d}" for v in r) for r in a["confusion_matrix"]],
        "```",
        "",
        "## 5. Runtime integration",
        "",
        f"- The inference engine loads `{runtime['runtime_artifact']}` (SHA-256 `{runtime['runtime_artifact_sha256']}`) beside the checkpoint and refuses to serve if it is missing, malformed, fitted on a split other than `val`, or fitted for another checkpoint.",
        f"- `predict()` returns `class_scores` = $\\mathrm{{softmax}}(z/T)$ - the vector the API's `classScores` and the interface's score card receive - with the raw softmax kept per class as `uncalibratedScore` for audit. The persisted model version reads `{runtime['model_version_reported']}`.",
        f"- {runtime['images']} test images graded through the engine: largest absolute difference between the served vector and $\\mathrm{{softmax}}(z/T)$ from `test_predictions.csv` = {runtime['max_abs_difference_served_vs_softmax_logits_over_T']:.2e} (tolerance {runtime['tolerance']}). Evidence: `runtime_integration_evidence.json`; automated tests: `backend/tests/test_probability_calibration.py`.",
        "",
        "## 6. What shows the test labels were not used to select or optimise $T$",
        "",
        "These are checkable properties of the script and artefacts; the claim extends no further than they do.",
        "",
        "- `fit_temperature()` receives only validation logits and labels, and refuses any record whose split tag is not `val` (tested).",
        "- Phase 1 reads the manifest filtered to `split == 'val'`; the test rows are first read in phase 2, after the artifact was written. The order of operations is recorded in `leakage_audit.json` and the log.",
        f"- The artifact's SHA-256 at freeze time equals its SHA-256 after evaluation: **{audit_doc['artifact_unchanged_after_evaluation']}**.",
        f"- The artifact carries only the keys {sorted(ARTIFACT_KEYS)}; none is a test statistic (tested). Its `validation_ids_sha256` names the exact 526 validation images.",
        f"- Validation and test image ids are disjoint (overlap {audit_doc['val_test_id_overlap']}), and `validation_predictions.csv` contains only `val` rows (tested).",
        "- Refitting from `validation_predictions.csv` alone reproduces the artifact's $T$ (tested), so $T$ is a function of the validation file and nothing else.",
        "- No test-metric gate chooses between $T=1$ and the fitted $T$.",
        "",
        "One qualification: the validation partition was also used during training to select the checkpoint epoch (best validation QWK, `training_protocol.md`). Fitting $T$ on it is the standard procedure (Guo et al., 2017), but it means the validation NLL is not an independent estimate; the test-set figures in section 3 are.",
        "",
        "## 7. Limitations",
        "",
        "- One scalar is fitted; class-wise miscalibration that a single temperature cannot correct is not corrected.",
        "- ECE and MCE depend on the binning; MCE in particular is driven by sparsely populated bins, so both binnings are reported.",
        "- Calibration changes how probabilities are reported. It does not change which grade is predicted and is not a diagnosis.",
        "",
    ]
    with open(os.path.join(out_dir, "calibration_evaluation_report.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images-dir", help="directory holding <image_id>.png for the val and test splits")
    parser.add_argument("--plot", nargs=2, metavar=("TEST_METRICS_JSON", "PNG"),
                        help="only draw the reliability diagram from an existing test_metrics.json")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--held-out", default=DEFAULT_HELD_OUT)
    parser.add_argument("--eval-summary", default=DEFAULT_EVAL_SUMMARY)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--runtime-artifact", default=RUNTIME_ARTIFACT)
    args = parser.parse_args(argv)

    if args.plot:
        with open(args.plot[0], encoding="utf-8") as fh:
            m = json.load(fh)
        plot_reliability(m["uncalibrated"], m["calibrated"], m["temperature"], args.plot[1])
        return 0
    if not args.images_dir:
        parser.error("--images-dir is required")
    os.makedirs(args.out, exist_ok=True)
    tee = Tee(os.path.join(args.out, "calibration_execution.log"))
    sys.stdout = tee
    started = time.time()
    try:
        run(args)
        print(f"\ndone in {time.time() - started:.1f}s")
    finally:
        sys.stdout = tee._stdout
        tee.close()
    return 0


def run(args):
    import torch
    from app.core.config import settings

    args.checkpoint_path = args.checkpoint or settings.MODEL_CHECKPOINT_PATH
    banner("PROBABILITY CALIBRATION (TEMPERATURE SCALING) - EfficientNet-B0 DR classifier")
    print(f"script   : {SCRIPT_NAME}  sha256 {sha256_file(os.path.abspath(__file__))}")
    print(f"torch    : {torch.__version__} (cuda available: {torch.cuda.is_available()})  numpy {np.__version__}")
    print(f"manifest : {rel(args.manifest)}  sha256 {sha256_file(args.manifest)}")
    print(f"images   : {args.images_dir}")

    audit = AuditTrail()
    ids = read_ids_by_split(args.manifest)
    overlap = len(ids.get("val", set()) & ids.get("test", set()))
    print(f"split sizes (ids only): { {k: len(v) for k, v in sorted(ids.items())} }; val/test overlap {overlap}")
    if overlap:
        raise SystemExit(f"REFUSED: val and test share {overlap} image ids")
    audit("setup", "read image ids per split (no grades) for the overlap check", val_test_overlap=overlap)

    service, model, device = load_weights(args.checkpoint_path)
    ckpt_sha = service.verified_sha256
    audit("setup", "loaded checkpoint through the runtime loader", sha256=ckpt_sha)
    print(f"checkpoint: {rel(args.checkpoint_path)}  sha256 {ckpt_sha}  device {device}")

    fitted = phase_fit(args, audit, model, device, ckpt_sha)
    evaluated = phase_evaluate(args, audit, model, device, ckpt_sha, fitted)
    runtime = phase_runtime(args, audit, args.checkpoint_path, evaluated)

    sha_after = sha256_file(fitted["artifact_path"])
    unchanged = sha_after == fitted["artifact_sha256"] == sha256_file(args.runtime_artifact)
    if not unchanged:
        raise SystemExit("REFUSED: the frozen artifact changed after evaluation")
    first_test_step = min(e["step"] for e in audit.events if "test" in e["event"] and e["phase"] == "evaluate")
    freeze_step = max(e["step"] for e in audit.events if e["event"].startswith("froze T"))
    audit_doc = {
        "val_test_id_overlap": overlap,
        "validation_ids_sha256": ids_digest(ids["val"]),
        "test_ids_sha256": ids_digest(ids["test"]),
        "artifact_sha256_at_freeze": fitted["artifact_sha256"],
        "artifact_sha256_after_evaluation": sha_after,
        "artifact_unchanged_after_evaluation": unchanged,
        "freeze_step": freeze_step,
        "first_test_partition_access_step": first_test_step,
        "temperature_frozen_before_test_partition_was_read": freeze_step < first_test_step,
        "test_metric_gate_on_temperature": False,
        "events": audit.events,
    }
    write_json(os.path.join(args.out, "leakage_audit.json"), audit_doc)
    with open(fitted["artifact_path"], encoding="utf-8") as fh:
        artifact = json.load(fh)
    generate_report(args.out, artifact, evaluated["metrics"], audit_doc, runtime)
    print("report written: calibration_evaluation_report.md")


if __name__ == "__main__":
    sys.exit(main())
