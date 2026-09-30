"""
EVIDENCE INTEGRITY GATE — every change must pass this before it lands.

This file encodes the requirements from the Chapter Four QA review as
executable assertions. It exists because the project has already failed those
requirements once: an evidence package was produced by scripts that trained on
`torch.randn()` tensors and wrote metrics from formulas, passed an internal
"unconditional pass", and was only caught by an external reviewer.

Documentation cannot prevent that recurring. A test can.

The rules below are deliberately blunt and occasionally over-strict. That is the
point: a gate that is easy to satisfy by accident protects nothing. If a rule
fires on legitimate work, widen it explicitly and say why in the same commit —
do not delete it.

Grouped by the reviewer's own findings:

  A. Evidence must be produced, never synthesised
  B. Training and evaluation must touch real data
  C. Reported numbers must recompute from raw artefacts
  D. The served model must be the evaluated model
  E. No fabricated clinical identity or authority
  F. One specification, consistently quoted
  G. Objectives traced to the right evidence
"""

import ast
import csv
import hashlib
import json
import math
import os
import re
from collections import Counter

import pytest

from app.core.config import settings

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHAPTER4 = os.path.join(REPO_ROOT, "docs", "chapter4")
SCRIPTS = os.path.join(REPO_ROOT, "backend", "scripts")
NOTEBOOKS = os.path.join(REPO_ROOT, "notebooks")

EXPECTED_CHECKPOINT_SHA256 = "8ee14d7591a8e6a1b86c15416a77375a198bd49399b3977a3de79a00e3dd14fa"


def read(path):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def python_sources(*dirs):
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if name.endswith(".py"):
                yield os.path.join(d, name)


# =====================================================================
# A. Evidence must be produced, never synthesised
# =====================================================================

# Calls that manufacture numbers. Legitimate uses exist (augmentation,
# initialisation during training) but not in scripts that emit evidence.
SYNTHESIS_CALLS = {
    "randn", "rand", "normal", "uniform", "lognormal", "poisson",
    "kaiming_normal_", "kaiming_uniform_", "xavier_normal_", "xavier_uniform_",
}

# Scripts whose output is presented as measurement. These may not fabricate.
EVIDENCE_SCRIPTS = [
    "analyze_clinical_metrics.py",
    "benchmark_resources.py",
    "benchmark_cpu_end_to_end.py",
    "evaluate_model.py",
    "verify_gate_downsampling.py",
    "assemble_submission_package.py",
]


# Narrow, justified exemptions. Format: (script, enclosing function).
# A blanket allowlist would defeat the rule, so each entry names the single
# function permitted to generate, and why generating there is not fabrication.
SYNTHESIS_EXEMPTIONS = {
    # Generates fundus-LIKE IMAGES to benchmark against when the 9.51 GB dataset
    # is unavailable. It fabricates the input, never the measurement: timings are
    # measured on whatever it is given, and the output is labelled
    # image_source="synthetic" with a caveat naming which stages stay valid.
    ("benchmark_cpu_end_to_end.py", "generate_synthetic_fundus"),
}


def _synthesis_calls_in(path):
    """Return synthesis calls made in real code, ignoring strings and comments."""
    script = os.path.basename(path)
    tree = ast.parse(read(path), filename=path)

    # Map each node to its enclosing function so exemptions can be scoped.
    enclosing = {}
    for parent in ast.walk(tree):
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(parent):
                enclosing.setdefault(child, parent.name)

    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
        if name not in SYNTHESIS_CALLS:
            continue
        if (script, enclosing.get(node)) in SYNTHESIS_EXEMPTIONS:
            continue
        hits.append(f"{script}:{node.lineno} {name}() in {enclosing.get(node, '<module>')}()")
    return hits


def test_synthetic_benchmark_output_is_labelled_as_such():
    """
    The exemption above is only safe while synthetic runs announce themselves.
    If that labelling is removed, a generated result could be mistaken for a
    measured one downstream.
    """
    path = os.path.join(SCRIPTS, "benchmark_cpu_end_to_end.py")
    if not os.path.exists(path):
        pytest.skip("benchmark not present")
    text = read(path)
    for marker in ('"image_source"', '"synthetic_caveat"', "compute_only_mean_ms"):
        assert marker in text, (
            f"benchmark_cpu_end_to_end.py no longer records {marker}; a synthetic "
            "result could be mistaken for a measured one")


@pytest.mark.parametrize("script", EVIDENCE_SCRIPTS)
def test_evidence_scripts_do_not_synthesise_numbers(script):
    """
    A script that reports a measurement must not be able to invent one.

    The deleted `prepare_submission_package.py` generated 100 benchmark timings
    with np.random.normal, added fake OS spikes at hardcoded indices, and
    rescaled them so the mean landed on exactly 95.76 ms.
    """
    path = os.path.join(SCRIPTS, script)
    if not os.path.exists(path):
        pytest.skip(f"{script} not present")

    hits = _synthesis_calls_in(path)
    assert not hits, (
        f"{script} calls a random/initialiser function. Evidence scripts must "
        f"measure, not generate:\n  " + "\n  ".join(hits)
    )


def test_the_deleted_fabrication_scripts_are_not_reintroduced():
    """
    Three scripts generated the rejected evidence. Two of them wrote to paths
    holding genuine artefacts, so restoring either would silently destroy it.
    """
    banned = [
        "create_evaluated_checkpoint.py",   # random Kaiming init -> production weights path
        "generate_dataset_manifest.py",     # invented the N=8,000 manifest
        "prepare_submission_package.py",    # synthesised timings and reports
    ]
    present = [b for b in banned
               if os.path.exists(os.path.join(SCRIPTS, b))]
    assert not present, (
        "Fabrication tooling has reappeared: " + ", ".join(present) +
        ". These generate evidence rather than measuring it."
    )


def test_checkpoint_is_never_written_by_anything_but_training():
    """Only the training pipeline may write to the production weights path."""
    offenders = []
    for path in python_sources(SCRIPTS):
        name = os.path.basename(path)
        if name.startswith("train_"):
            continue
        text = read(path)
        if re.search(r"torch\.save\s*\(", text) and "models/weights" in text.replace("\\", "/"):
            offenders.append(name)
    assert not offenders, (
        "Non-training script writes to the production weights path: " + ", ".join(offenders)
    )


# =====================================================================
# B. Training and evaluation must touch real data
# =====================================================================

@pytest.mark.parametrize("rel", [
    "backend/scripts/train_efficientnet_b0.py",
    "backend/scripts/evaluate_model.py",
    "notebooks/colab_train_and_evaluate.py",
])
def test_pipeline_scripts_open_real_images(rel):
    """
    The rejected training script never opened a retinal image. Any script that
    claims to train or evaluate must read image files from disk.
    """
    path = os.path.join(REPO_ROOT, rel)
    if not os.path.exists(path):
        pytest.skip(f"{rel} not present")
    text = read(path)
    assert "Image.open" in text, f"{rel} never opens an image file"


def test_evaluation_loads_the_model_and_runs_inference():
    """
    The rejected evaluation script did not import PyTorch. It began with a
    manually specified confusion matrix.
    """
    text = read(os.path.join(SCRIPTS, "evaluate_model.py"))
    for required in ("import torch", "torch.load", "model.eval()", "inference_mode"):
        assert required in text, f"evaluate_model.py is missing {required!r}"


def test_no_simulated_metric_variables_anywhere():
    """
    The rejected script built `sim_val_loss`, `sim_val_acc`, `sim_val_qwk` and
    `sim_val_f1` from formulas and random numbers.
    """
    offenders = []
    for path in list(python_sources(SCRIPTS, NOTEBOOKS)):
        for m in re.finditer(r"\bsim_(?:val_|train_)?(?:loss|acc|accuracy|qwk|f1|kappa)\w*", read(path)):
            offenders.append(f"{os.path.basename(path)}: {m.group(0)}")
    assert not offenders, (
        "Simulated metric variables found:\n  " + "\n  ".join(offenders)
    )


# =====================================================================
# C. Reported numbers must recompute from raw artefacts
# =====================================================================

def _load_predictions():
    path = os.path.join(CHAPTER4, "held_out_predictions.csv")
    if not os.path.exists(path):
        pytest.skip("held_out_predictions.csv not present")
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_confusion_matrix_recomputes_from_the_predictions():
    """
    The reviewer found the report's matrix disagreeing with the prediction CSV
    and with its own QWK and macro-F1 claims.
    """
    rows = _load_predictions()
    computed = [[0] * 5 for _ in range(5)]
    for r in rows:
        computed[int(r["true_grade"])][int(r["predicted_grade"])] += 1

    summary_path = os.path.join(CHAPTER4, "evaluation_summary.json")
    with open(summary_path, encoding="utf-8") as fh:
        declared = json.load(fh)["confusion_matrix"]

    assert computed == declared, (
        f"evaluation_summary.json disagrees with held_out_predictions.csv\n"
        f"  from CSV : {computed}\n  declared : {declared}"
    )


def test_reported_accuracy_and_qwk_recompute_from_the_predictions():
    rows = _load_predictions()
    yt = [int(r["true_grade"]) for r in rows]
    yp = [int(r["predicted_grade"]) for r in rows]

    acc = sum(t == p for t, p in zip(yt, yp)) / len(yt)

    # QWK from first principles, so this does not depend on the analysis script.
    n, k = len(yt), 5
    obs = [[0] * k for _ in range(k)]
    for t, p in zip(yt, yp):
        obs[t][p] += 1
    tc, pc = Counter(yt), Counter(yp)
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            w = ((i - j) ** 2) / ((k - 1) ** 2)
            num += w * obs[i][j]
            den += w * tc[i] * pc[j] / n
    qwk = 1 - num / den

    with open(os.path.join(CHAPTER4, "evaluation_summary.json"), encoding="utf-8") as fh:
        declared = json.load(fh)

    assert abs(acc - declared["accuracy"]) < 5e-6, (
        f"accuracy {acc:.6f} != declared {declared['accuracy']}")
    assert abs(qwk - declared["qwk"]) < 5e-6, (
        f"QWK {qwk:.6f} != declared {declared['qwk']}")


def test_every_prediction_row_agrees_with_its_own_scores():
    """`predicted_grade` must be the argmax of the recorded distribution."""
    violations = []
    for r in _load_predictions():
        scores = [float(r[f"score_grade_{k}"]) for k in range(5)]
        if scores.index(max(scores)) != int(r["predicted_grade"]):
            violations.append(r["image_id"])
    assert not violations, f"{len(violations)} argmax violations: {violations[:5]}"


@pytest.mark.parametrize("summary_name,csv_name,value_col", [
    ("benchmark_summary.json", "benchmark_timings.csv", "inference_time_ms"),
])
def test_benchmark_summary_recomputes_from_its_raw_timings(summary_name, csv_name, value_col):
    """
    The reviewer found only the mean matching: median, P95, min and max in the
    report all disagreed with the CSV, because the CSV had been generated to
    hit a target mean.
    """
    sp = os.path.join(CHAPTER4, summary_name)
    cp = os.path.join(CHAPTER4, csv_name)
    if not (os.path.exists(sp) and os.path.exists(cp)):
        pytest.skip("benchmark artefacts not present")

    with open(cp, newline="", encoding="utf-8") as fh:
        vals = sorted(float(r[value_col]) for r in csv.DictReader(fh))
    with open(sp, encoding="utf-8") as fh:
        s = json.load(fh)

    mean = sum(vals) / len(vals)
    checks = {
        "mean_ms": mean,
        "median_ms": vals[len(vals) // 2],
        "p95_ms": vals[int(0.95 * len(vals))],
        "min_ms": min(vals),
        "max_ms": max(vals),
    }
    bad = [f"{k}: csv {v:.4f} vs summary {s[k]}"
           for k, v in checks.items() if k in s and abs(v - s[k]) > 1e-3]
    assert not bad, f"{summary_name} disagrees with {csv_name}:\n  " + "\n  ".join(bad)


def test_cpu_benchmark_csv_and_json_agree():
    jp = os.path.join(CHAPTER4, "cpu_end_to_end_benchmark.json")
    cp = os.path.join(CHAPTER4, "cpu_end_to_end_benchmark.csv")
    if not (os.path.exists(jp) and os.path.exists(cp)):
        pytest.skip("CPU benchmark artefacts not present")

    with open(jp, encoding="utf-8") as fh:
        stages = {s["stage"]: s for s in json.load(fh)["stages"]}
    with open(cp, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            st = stages[row["stage"]]
            for col in ("mean_ms", "median_ms", "p95_ms", "min_ms", "max_ms"):
                assert abs(float(row[col]) - st[col]) < 1e-9, (
                    f"{row['stage']}.{col}: csv {row[col]} vs json {st[col]}")


# =====================================================================
# D. The served model must be the evaluated model
# =====================================================================

def test_checkpoint_digest_matches_config_and_manifest():
    weights = os.path.join(REPO_ROOT, "backend", "models", "weights", "efficientnet_b0_dr.pth")
    assert settings.MODEL_CHECKPOINT_SHA256 == EXPECTED_CHECKPOINT_SHA256

    manifest = read(os.path.join(CHAPTER4, "checkpoint_manifest.md"))
    assert EXPECTED_CHECKPOINT_SHA256 in manifest, (
        "checkpoint_manifest.md does not record the configured digest")

    if not os.path.exists(weights):
        pytest.skip("checkpoint binary not in this checkout")
    d = hashlib.sha256()
    with open(weights, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            d.update(chunk)
    assert d.hexdigest() == EXPECTED_CHECKPOINT_SHA256, (
        "The committed weights are not the evaluated weights.")


def test_manifest_records_real_hashes_and_no_invented_patient_id():
    """
    The rejected manifest hashed constructed metadata strings and carried a
    `patient_id` column. APTOS 2019 publishes no patient identifier, so
    patient-level partitioning was never possible.
    """
    path = os.path.join(CHAPTER4, "dataset_split_manifest.csv")
    if not os.path.exists(path):
        pytest.skip("manifest not present")
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    assert "patient_id" not in rows[0], (
        "manifest carries a patient_id column; APTOS publishes no patient identifier")
    assert "sha256_hash" in rows[0]

    bad = [r["image_id"] for r in rows[:200]
           if not re.fullmatch(r"[0-9a-f]{64}", r["sha256_hash"])]
    assert not bad, f"malformed SHA-256 values: {bad[:5]}"

    # Distinct hashes: metadata-derived hashes were previously ~100% unique in a
    # way real duplicated images are not, but the decisive check is that they
    # are verifiable against the files, which verify_manifest_hashes.py does.
    assert len({r["sha256_hash"] for r in rows}) > 1


# =====================================================================
# E. No fabricated clinical identity or authority
# =====================================================================

FABRICATED_IDENTITY = [
    "GMC-7492104", "Adaeze Okonjo", "St. Jude Retinal",
    "a.okonjo@retina-clinic.nhs.uk", "dr.adaeze",
]

OVERCLAIM_PHRASES = [
    "Certified Diagnosis", "Certified ICDR", "Official Clinical Evaluation",
    "legally immutable", "Authoritative Human-in-the-Loop",
    "FDA SaMD", "NHS DTAC",
]


def _code_files():
    for base, _dirs, files in os.walk(REPO_ROOT):
        parts = base.replace("\\", "/").split("/")
        if any(p in {".git", "node_modules", ".venv", "dist", "__pycache__", "build"} for p in parts):
            continue
        for name in files:
            if name.endswith((".py", ".ts", ".tsx")):
                yield os.path.relpath(os.path.join(base, name), REPO_ROOT)


def test_no_fabricated_clinician_identity_in_code():
    """
    A fabricated professional registration number, or an affiliation with a real
    institution, must never appear. The seeded account previously used an
    nhs.uk email address.
    """
    offenders = []
    for rel in _code_files():
        if "test_editor_integrity_gate" in rel or "test_spec_doc_consistency" in rel:
            continue
        for lineno, line in enumerate(read(os.path.join(REPO_ROOT, rel)).split("\n"), 1):
            # Comments may record what was removed and why. Code may not.
            # This is scoped per LINE on purpose: an earlier version exempted
            # the whole file if it contained the explanatory comment, which
            # meant auth.py could have reintroduced the fabricated number
            # without the gate noticing. It did, and the gate stayed green.
            if line.lstrip().startswith(("#", "//", "*", "/*")):
                continue
            for term in FABRICATED_IDENTITY:
                if term in line:
                    offenders.append(f"{rel}:{lineno}: {term}")
    assert not offenders, "Fabricated clinical identity:\n  " + "\n  ".join(offenders)


def test_no_clinical_authority_overclaims_in_code():
    """
    Model output is decision support, not diagnosis. The generated PDF report
    previously printed "Certified ICDR Grade" and "Authoritative
    Human-in-the-Loop".
    """
    offenders = []
    for rel in _code_files():
        if "test_editor_integrity_gate" in rel:
            continue
        for lineno, line in enumerate(read(os.path.join(REPO_ROOT, rel)).splitlines(), 1):
            if line.lstrip().startswith(("#", "//", "*", "/*")):
                continue
            for phrase in OVERCLAIM_PHRASES:
                if phrase in line:
                    offenders.append(f"{rel}:{lineno}: {phrase!r}")
    assert not offenders, (
        "Clinical authority overclaim:\n  " + "\n  ".join(offenders))


def test_demo_identity_is_labelled_simulated():
    """A demonstration account must announce that it is one."""
    text = read(os.path.join(REPO_ROOT, "backend", "app", "routers", "auth.py"))
    assert "Simulated" in text or "SIM-" in text, (
        "The seeded account does not label itself simulated")


# =====================================================================
# F. One specification, consistently quoted
#    (numeric threshold checks live in test_spec_doc_consistency.py)
# =====================================================================

def test_inference_engine_still_fails_closed():
    """
    The engine must refuse to grade without verified weights. This was once
    inverted: a missing checkpoint left a randomly-initialised graph in place.
    """
    text = read(os.path.join(REPO_ROOT, "backend", "app", "services", "ai_service.py"))
    assert "ModelCheckpointError" in text
    assert "_verify_checkpoint_digest" in text
    # The mock must never be reachable as a silent fallback from predict().
    predict_body = text.split("def predict(", 1)[1]
    first_branch = predict_body[:800]
    assert "MockInferenceService().predict" not in first_branch, (
        "predict() delegates to the simulated engine instead of failing closed")


# =====================================================================
# G. Objectives traced to the right evidence
# =====================================================================

def test_objective_h_is_model_evaluation_not_gradcam():
    """
    The reviewer found objective (h) reassigned to Grad-CAM while the model
    evaluation report was itself titled "Objective h". Grad-CAM belongs to (g).
    """
    matrix = read(os.path.join(CHAPTER4, "objective_traceability_matrix.md"))
    row = next((l for l in matrix.split("\n") if l.startswith("| **h** |")), None)
    assert row, "objective (h) row missing from the traceability matrix"
    assert "Grad-CAM" not in row, (
        "objective (h) is assigned to Grad-CAM; it must be classification-"
        "performance evaluation, with Grad-CAM under (g)")
    assert re.search(r"[Ee]valuation|[Pp]erformance", row), (
        "objective (h) does not describe performance evaluation")
