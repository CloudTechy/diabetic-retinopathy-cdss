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

EXPECTED_CHECKPOINT_SHA256 = "67d0b89641f08057126dd411e380b25575ef29f71ae37ee5796d472d9203dbf7"


def read(path):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


CORRECTION_RECORDS = {
    "REVIEWER_RESPONSE.md",
    "independent_thesis_qa_gate_audit.md",
    "screenshot_evidence_manifest.md",
    "evidence_provenance.md",
    "model_evaluation_report.md",
}


def iter_markdown(include_correction_records=True):
    """Yield (relative path, text) for every committed markdown document."""
    docs_root = os.path.join(REPO_ROOT, "docs")
    for dirpath, dirnames, filenames in os.walk(docs_root):
        dirnames[:] = [d for d in dirnames if d not in ("__pycache__",)]
        for name in sorted(filenames):
            if not name.endswith(".md"):
                continue
            if not include_correction_records and name in CORRECTION_RECORDS:
                continue
            path = os.path.join(dirpath, name)
            yield os.path.relpath(path, REPO_ROOT), read(path)


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

# NOTE: this list must contain the BANNED strings verbatim. A scripted
# scope-rename once rewrote it in place, so the gate ended up banning the
# correct replacement term and permitting the term it was meant to catch.
# Any bulk rename must exclude this file.
OVERCLAIM_PHRASES = [
    "Certified Diagnosis", "Certified ICDR", "Certified Review",
    "Certified Clinical Report", "certified clinical grade",
    "Official Clinical Evaluation", "legally immutable",
    "Authoritative Human-in-the-Loop",
    "FDA SaMD", "NHS DTAC", "FDA/NHS",
    "referralPlan", "referral_plan",
    "certifiedGrade", "certified_grade",
]


def _walk_repo(suffixes):
    for base, _dirs, files in os.walk(REPO_ROOT):
        parts = base.replace("\\", "/").split("/")
        if any(p in {".git", "node_modules", ".venv", "dist", "__pycache__", "build"} for p in parts):
            continue
        for name in files:
            if name.endswith(suffixes):
                yield os.path.relpath(os.path.join(base, name), REPO_ROOT)


def _code_files():
    return _walk_repo((".py", ".ts", ".tsx"))


def _prose_files():
    """
    Documents were outside the overclaim rule entirely until an independent
    review found "FDA Class II SaMD" and "In compliance with FDA ... NHS England
    ... NICE" sitting in docs/ - a regulatory classification and a conformity
    claim that this research prototype does not hold. The rule had only ever
    read .py/.ts/.tsx, so the prose was never examined.
    """
    return _walk_repo((".md",))


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
    previously printed "Reviewer's Assessed Grade" and "Authoritative
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


# =====================================================================
# H. Every artefact must have a named producer, and cite only real data
#
# The fabricated validation_test_results.csv survived every rule above
# because nothing asserted that an evidence file must come from somewhere.
# =====================================================================

PROVENANCE_DOC = os.path.join(CHAPTER4, "evidence_provenance.md")


def test_provenance_document_exists_and_names_the_producing_run():
    assert os.path.exists(PROVENANCE_DOC), (
        "evidence_provenance.md is missing; every artefact must name its producer")
    text = read(PROVENANCE_DOC)
    assert "colab_train_and_evaluate.py" in text
    assert "produced **none**" in text or "produced none" in text, (
        "provenance doc must state which scripts produced no evidence")


# Documents describe the evidence; they are not themselves artefacts needing a
# producer. Everything else in docs/chapter4/ is.
PROVENANCE_EXEMPT_SUFFIXES = (".md",)


def test_every_committed_artefact_is_listed_in_the_provenance_doc():
    """
    An evidence file nobody claims to have produced is a liability.

    This walked a hardcoded list of twelve filenames until a QA pass found three
    artefacts shipping undeclared - blur_threshold_calibration.json,
    dataset_split_audit.json and test_execution.log - every one of them added
    after the list was written, and therefore outside the rule by construction.
    It now enumerates the directory, so a new artefact must be declared before
    it can ship.
    """
    if not os.path.isdir(CHAPTER4):
        pytest.skip("chapter4 directory absent")

    provenance = read(PROVENANCE_DOC)
    undeclared = []
    for name in sorted(os.listdir(CHAPTER4)):
        path = os.path.join(CHAPTER4, name)
        if os.path.isdir(path) or name.startswith("."):
            continue
        if name.endswith(PROVENANCE_EXEMPT_SUFFIXES):
            continue
        if name not in provenance:
            undeclared.append(name)

    assert not undeclared, (
        "Evidence files are committed that evidence_provenance.md does not "
        "name:\n  " + "\n  ".join(undeclared)
        + "\nName the script that produced each one, or remove the file. An "
          "artefact with no producer is indistinguishable from a fabricated one.")


def test_evidence_csvs_only_cite_image_ids_that_exist_in_the_manifest():
    """
    The fabricated validation_test_results.csv cited NONRET-XRAY-01,
    CORRUPT-BYTE-01 and four other identifiers absent from APTOS, plus two
    training images presented as held-out. Any CSV with an image_id column must
    reference the real dataset.
    """
    manifest_path = os.path.join(CHAPTER4, "dataset_split_manifest.csv")
    if not os.path.exists(manifest_path):
        pytest.skip("manifest not present")
    with open(manifest_path, newline="", encoding="utf-8") as fh:
        known = {r["image_id"] for r in csv.DictReader(fh)}

    offenders = []
    for name in sorted(os.listdir(CHAPTER4)):
        if not name.endswith(".csv") or name == "dataset_split_manifest.csv":
            continue
        with open(os.path.join(CHAPTER4, name), newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames or "image_id" not in reader.fieldnames:
                continue
            for row in reader:
                if row["image_id"] not in known:
                    offenders.append(f"{name}: {row['image_id']!r} is not in the manifest")

    assert not offenders, (
        "Evidence cites image identifiers that do not exist in the dataset:\n  "
        + "\n  ".join(offenders[:10]))


# =====================================================================
# I. Senior-review findings, encoded so they cannot regress
# =====================================================================

def test_split_builder_deduplicates_by_image_bytes():
    """
    The contaminated manifest arose from keying de-duplication on APTOS's
    duplicated_info.csv, which the competition download does not contain. The
    replacement must key on a hash we compute ourselves, exclude
    label-conflicting groups, and assert zero partition overlap.
    """
    path = os.path.join(SCRIPTS, "build_clean_split.py")
    assert os.path.exists(path), "build_clean_split.py is missing"
    text = read(path)
    for required in ("sha256", "conflicting", "LEAKAGE"):
        assert required in text, f"build_clean_split.py lacks {required!r}"


def test_pipeline_does_not_depend_on_duplicated_info_csv():
    """No code path may key de-duplication on a file the dataset does not ship."""
    offenders = []
    for path in list(python_sources(SCRIPTS, NOTEBOOKS)):
        tree = ast.parse(read(path), filename=path)
        # Collect docstring nodes so prose explaining the old behaviour is allowed.
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                body = getattr(node, "body", None)
                if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                    docstrings.add(id(body[0].value))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and "duplicated_info" in node.value and id(node) not in docstrings):
                offenders.append(f"{os.path.basename(path)}:{node.lineno}")
    assert not offenders, (
        "De-duplication still references duplicated_info.csv, which is absent "
        "from the Kaggle download: " + ", ".join(offenders))


def test_no_clinical_effectiveness_or_referral_claims_in_documents():
    """
    Retrospective image classification does not establish referral behaviour,
    error safety, or clinical usefulness. Those claims exceed the approved
    system scope.
    """
    banned = [
        "need referral", "needs referral", "sent home",
        "clinical cost is lowest", "errs toward over-referral",
        "safer direction", "recommends confirmatory",
        "demonstrated clinical", "clinically useful",
    ]
    offenders = []
    for rel, text in iter_markdown(include_correction_records=False):
        for lineno, line in enumerate(text.splitlines(), 1):
            low = line.lower()
            for phrase in banned:
                if phrase in low:
                    offenders.append(f"{rel}:{lineno}: {phrase!r}")
    assert not offenders, (
        "Clinical-effectiveness or referral claim beyond system scope:\n  "
        + "\n  ".join(offenders))


def test_binary_collapse_is_labelled_exploratory():
    """
    Referable-DR figures may be retained only as secondary exploratory
    analysis, never as system referral decisions or safety evidence.
    """
    report = os.path.join(CHAPTER4, "model_evaluation_report.md")
    if not os.path.exists(report):
        pytest.skip("evaluation report not present")
    text = read(report)
    if r"Grade $\ge 2$" not in text and "grade >= 2" not in text.lower():
        pytest.skip("no binary collapse reported")
    assert "xploratory" in text, (
        "the binary collapse section must be labelled exploratory")
    assert "not** system referral decisions" in text or "not system referral" in text, (
        "the binary collapse must state it is not a system referral decision")


def test_no_superseded_regulatory_or_identity_claims_in_screenshots_manifest():
    """
    The PDF and deployment screenshots displayed a fabricated clinician, a GMC
    number, "Reviewer's Assessed Grade", "research prototype" and "research prototype". They
    were removed rather than reshipped.
    """
    for name in ("10_tamper_evident_pdf_report.png", "live_vercel_verified.png"):
        for d in (os.path.join(CHAPTER4, "screenshots"),
                  os.path.join(REPO_ROOT, "docs", "chapter4_submission_package", "screenshots")):
            assert not os.path.exists(os.path.join(d, name)), (
                f"{name} is non-compliant and must not be shipped; "
                "regenerate it from the corrected build or leave it out")


def test_documents_use_relative_links_not_windows_file_urls():
    offenders = [rel for rel, text in iter_markdown() if "file:///" in text]
    assert not offenders, (
        "Windows file:/// links do not resolve for a reviewer:\n  " + "\n  ".join(offenders))


def test_programme_is_described_consistently():
    """The project is a PGD dissertation, not an MSc thesis."""
    offenders = []
    for rel, text in iter_markdown():
        for lineno, line in enumerate(text.splitlines(), 1):
            if "MSc" in line:
                offenders.append(f"{rel}:{lineno}")
    assert not offenders, "Incorrect programme description (MSc):\n  " + "\n  ".join(offenders)


def test_withdrawn_qa_audit_is_archived_not_presented_as_current_evidence():
    """
    The QA audit narrates withdrawn 544-image / 86.40% results. It belongs in
    archive/, not documentation/. A hand-move was silently undone by the next
    package assembly because the layout still pointed at documentation/, so
    this asserts the outcome rather than the intent.
    """
    pkg = os.path.join(REPO_ROOT, "docs", "chapter4_submission_package")
    if not os.path.isdir(pkg):
        pytest.skip("package not built")
    name = "independent_thesis_qa_gate_audit.md"
    assert not os.path.exists(os.path.join(pkg, "documentation", name)), (
        f"{name} is in documentation/, where it reads as current evidence")
    assert os.path.exists(os.path.join(pkg, "archive", name)), (
        f"{name} should be retained in archive/ as a correction record")


# =====================================================================
# J. Claims may not outrun the evidence that supports them
# =====================================================================

def test_objective_status_does_not_claim_completion_without_evidence():
    """
    implementation_status.md once marked all nine objectives "100% Completed"
    while the validation CSV was missing and three objectives rested on a
    contaminated partition. A status is a claim; it needs backing.
    """
    path = os.path.join(CHAPTER4, "implementation_status.md")
    if not os.path.exists(path):
        pytest.skip("implementation_status.md not present")
    text = read(path)

    assert "100% Complete" not in text, (
        "implementation_status.md claims blanket completion")

    validation_csv = os.path.join(CHAPTER4, "validation_test_results.csv")
    if not os.path.exists(validation_csv):
        assert "Partial" in text or "Superseded" in text, (
            "the validation-results CSV is absent, so at least one objective "
            "must be marked Partial or Superseded")


def test_screenshot_manifest_only_lists_figures_that_exist():
    """
    The manifest listed Figure 4.11 after its file had been deleted. A figure
    index that points at nothing is a defect a reader finds before you do.
    """
    manifest = os.path.join(CHAPTER4, "screenshot_evidence_manifest.md")
    shots = os.path.join(CHAPTER4, "screenshots")
    if not (os.path.exists(manifest) and os.path.isdir(shots)):
        pytest.skip("screenshot manifest or folder not present")

    text = read(manifest)
    # Only inspect table rows; prose may explain a removal.
    offenders = []
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        for name in re.findall(r"([0-9A-Za-z_]+\.png)", line):
            # Figures live either in screenshots/ or beside the other artefacts
            # (confusion_matrix.png and learning_curves.png are plots, not captures).
            if not any(os.path.exists(os.path.join(d, name)) for d in (shots, CHAPTER4)):
                offenders.append(name)
    assert not offenders, (
        "screenshot manifest table lists missing figures: " + ", ".join(sorted(set(offenders))))


def test_no_superseded_latency_value_in_documents():
    """The 298.31 ms figure came from the fabricated benchmark."""
    offenders = []
    for rel, text in iter_markdown(include_correction_records=False):
        for lineno, line in enumerate(text.splitlines(), 1):
            if "298.3" in line:
                offenders.append(f"{rel}:{lineno}")
    assert not offenders, (
        "Superseded latency value 298.3 ms still quoted:\n  " + "\n  ".join(offenders))


def test_contaminated_results_are_labelled_superseded():
    """
    While the active manifest is the contaminated one, any document reporting
    the 549-image figures must say they are provisional.
    """
    manifest = os.path.join(CHAPTER4, "dataset_split_manifest.csv")
    if not os.path.exists(manifest):
        pytest.skip("manifest not present")
    with open(manifest, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if len({r["sha256_hash"] for r in rows}) == len(rows):
        pytest.skip("manifest is already de-duplicated - clean rerun has happened")

    report = read(os.path.join(CHAPTER4, "model_evaluation_report.md"))
    assert "superseded" in report.lower(), (
        "the evaluation report quotes results from a contaminated partition "
        "without marking them superseded")


def test_validation_gate_failures_are_declared_as_a_defect():
    """
    The first genuine run of the validation pipeline rejected 10 of 10
    unmodified held-out APTOS images: the Laplacian threshold is calibrated for
    a corpus this one is not. The evidence may say so; what it may not do is sit
    beside a document claiming objective b is met.

    This rule ties the claim to the measurement. It does not require the gate to
    pass - it requires the documents to agree with it.
    """
    csv_path = os.path.join(CHAPTER4, "validation_test_results.csv")
    status_path = os.path.join(CHAPTER4, "implementation_status.md")
    if not (os.path.exists(csv_path) and os.path.exists(status_path)):
        pytest.skip("validation evidence or status document not present")

    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    wrong = [r for r in rows
             if (r["overall_status"] == "ACCEPTED") != (r["expected"] == "ACCEPT")]
    if not wrong:
        return  # the gate behaves; nothing to declare

    status = read(status_path)
    objective_b = next((l for l in status.split("\n")
                        if l.startswith("| **Objective b**")), "")
    assert "Defect" in objective_b or "Partial" in objective_b, (
        f"{len(wrong)} validation case(s) behaved unexpectedly "
        f"(e.g. {wrong[0]['test_case']} expected {wrong[0]['expected']}, "
        f"got {wrong[0]['overall_status']}), but implementation_status.md does "
        f"not mark objective b as a defect:\n  {objective_b.strip()}")
    assert "BLOCKING" in read(os.path.join(CHAPTER4, "known_limitations.md")), (
        "known_limitations.md must carry the blocking defect section")


# =====================================================================
# K. The served system must not fabricate, and must not be steerable
#
# Three defects found in release QA, all in the LIVE application rather
# than the evidence: a client could choose the displayed grade, a failed
# Grad-CAM was replaced by a synthetic overlay with hard-coded scores,
# and clinical routes served anonymous callers an authenticated session.
# =====================================================================

AI_SERVICE = os.path.join(REPO_ROOT, "backend", "app", "services", "ai_service.py")


def _real_inference_source():
    """The body of EfficientNetB0InferenceService only, excluding the mock."""
    text = read(AI_SERVICE)
    start = text.index("class EfficientNetB0InferenceService")
    end = text.find("\ndef get_ai_inference_service", start)
    return text[start:end if end != -1 else len(text)]


def test_real_engine_never_substitutes_a_synthetic_attribution():
    """
    The real service called create_mock_gradcam_heatmap on an empty CAM and in
    its exception handler, so a failed attribution was shown as a real one.
    """
    body = _real_inference_source()
    offenders = [f"line {i}" for i, line in enumerate(body.split("\n"), 1)
                 if "create_mock_gradcam_heatmap" in line
                 and not line.lstrip().startswith("#")]
    assert not offenders, (
        "EfficientNetB0InferenceService renders a synthetic Grad-CAM: "
        + ", ".join(offenders) +
        ". A failed explanation must be reported as unavailable, not invented.")


def test_real_engine_never_substitutes_demonstration_scores():
    """
    Its exception handler assigned scores = [0.04, 0.12, 0.78, 0.05, 0.01] and
    returned them as the model's class distribution.
    """
    import re as _re
    body = _real_inference_source()
    offenders = []
    for i, line in enumerate(body.split("\n"), 1):
        if line.lstrip().startswith("#"):
            continue
        # A literal 5-element float list assigned to a scores-like name.
        if _re.search(r"\b(scores|probs|class_scores)\b\s*=\s*\[\s*0?\.\d", line):
            offenders.append(f"line {i}: {line.strip()[:70]}")
    assert not offenders, (
        "EfficientNetB0InferenceService assigns literal class scores:\n  "
        + "\n  ".join(offenders) +
        "\nModel scores must come from the model.")


def test_a_request_cannot_choose_the_grade_the_real_engine_reports():
    """
    `grade = candidate_grade` let the caller pick the displayed class. The
    parameter may survive for interface compatibility, but must not reach the
    grade.
    """
    import re as _re
    body = _real_inference_source()
    offenders = [f"line {i}: {line.strip()[:70]}"
                 for i, line in enumerate(body.split("\n"), 1)
                 if not line.lstrip().startswith("#")
                 and _re.search(r"^\s*grade\s*=\s*candidate_grade", line)]
    assert not offenders, (
        "The real engine assigns the grade from the request:\n  "
        + "\n  ".join(offenders))
    assert "torch.argmax" in body, (
        "The real engine must derive the grade from the model's argmax")


def test_clinical_routes_use_strict_authentication():
    """
    assessments.py imported get_optional_current_user AS get_current_user, and
    that dependency returned the seeded demonstration account when no token was
    supplied - so every route served an authenticated session to anyone.
    """
    path = os.path.join(REPO_ROOT, "backend", "app", "routers", "assessments.py")
    for line in read(path).split("\n"):
        if line.lstrip().startswith("#"):
            continue
        assert "get_optional_current_user" not in line, (
            f"assessments.py uses the optional auth dependency: {line.strip()[:80]}")


def test_optional_auth_does_not_fabricate_a_session():
    """It returned the seed user for anonymous callers; it must return None."""
    path = os.path.join(REPO_ROOT, "backend", "app", "routers", "auth.py")
    text = read(path)
    body = text[text.index("async def get_optional_current_user"):]
    body = body[:body.index("\n@router")]
    assert "return await get_or_create_seed_user" not in body, (
        "get_optional_current_user materialises an account for anonymous "
        "callers; it must return None")


def test_login_verifies_the_password_unconditionally():
    """
    The verify_password call sat inside `if not user:`, so an account that
    already existed authenticated with any password.
    """
    path = os.path.join(REPO_ROOT, "backend", "app", "routers", "auth.py")
    text = read(path)
    body = text[text.index("async def login("):]
    body = body[:body.index("\n@router")]

    verify_lines = [(i, l) for i, l in enumerate(body.split("\n"), 1)
                    if "verify_password(" in l and not l.lstrip().startswith("#")]
    assert verify_lines, "login() never calls verify_password"

    # The check must be at function-top-level indentation (4 spaces), i.e. not
    # nested inside a conditional that some accounts can skip.
    for _i, line in verify_lines:
        indent = len(line) - len(line.lstrip())
        assert indent <= 8, (
            f"verify_password is nested {indent} spaces deep, so some accounts "
            f"can bypass it: {line.strip()[:70]}")

    assert 'credentials.password != "' not in body, (
        "login() compares the password to a literal; use verify_password only")


def test_any_committed_calibration_is_from_a_real_development_corpus():
    """
    A calibration in the evidence folder sets the thresholds the served system
    admits images by. A fixture run once landed there during a smoke test, so
    assert what it must be: measured, sizeable, and free of the test partition.
    """
    path = os.path.join(CHAPTER4, "blur_threshold_calibration.json")
    if not os.path.exists(path):
        pytest.skip("no calibration committed yet")

    with open(path, encoding="utf-8") as fh:
        cal = json.load(fh)

    subset = str(cal.get("subset", ""))
    assert "HELD-OUT" not in subset.upper(), (
        "the calibration included the held-out test partition, which leaks it")
    assert "train" in subset.lower(), (
        f"calibration subset is {subset!r}; it must be the development partitions")

    n = cal.get("images_measured", 0)
    assert n >= 500, (
        f"calibration measured only {n} images; a percentile over that few is "
        f"not an estimate of the corpus")

    corpus = str(cal.get("corpus", ""))
    for marker in ("scratchpad", "smoke", "fixture", "tmp", "temp"):
        assert marker not in corpus.lower(), (
            f"calibration corpus path looks synthetic: {corpus}")


def test_calibrated_thresholds_are_documented_where_they_are_claimed():
    """
    If the configured thresholds no longer match the uncalibrated defaults, the
    spec must carry the derivation. A changed number with no recorded percentile
    is how the original 60.0 came to exist.
    """
    spec_path = os.path.join(CHAPTER4, "validation_module_spec.md")
    if not os.path.exists(spec_path):
        pytest.skip("spec not present")
    spec = read(spec_path)

    uncalibrated = (settings.LAPLACIAN_BLUR_THRESHOLD == 60.0
                    and settings.MIN_IMAGE_DIMENSION == 512)
    if uncalibrated:
        assert "not calibrated" in spec, (
            "the thresholds are still the a priori defaults, so the spec must "
            "say they are uncalibrated")
        return

    assert "### Calibrated admission thresholds" in spec, (
        "the thresholds were changed from the defaults but the spec records no "
        "derivation; run apply_validation_thresholds.py rather than editing "
        "config.py by hand")
    assert "Declared percentile:" in spec, (
        "the spec records no declared percentile for the chosen thresholds")
    assert str(settings.LAPLACIAN_BLUR_THRESHOLD) in spec, (
        f"spec does not quote the configured blur threshold "
        f"({settings.LAPLACIAN_BLUR_THRESHOLD})")
    assert str(settings.MIN_IMAGE_DIMENSION) in spec, (
        f"spec does not quote the configured minimum dimension "
        f"({settings.MIN_IMAGE_DIMENSION})")


def test_gate_decisions_never_compare_a_display_rounded_metric():
    """
    Gate 3 compared round(x, 1) against its thresholds, quantising the decision
    boundary onto a 0.1 grid. With round thresholds like 18.0 and 60.0, images
    landed exactly on the boundary: eight flipped 18.0 -> 17.9 under analysis
    subsampling, failing the decision-preservation check.

    Round for display; decide on full precision.
    """
    gate_dir = os.path.join(REPO_ROOT, "backend", "app", "services", "validation")
    offenders = []

    for name in sorted(os.listdir(gate_dir)):
        if not name.endswith(".py"):
            continue
        path = os.path.join(gate_dir, name)
        tree = ast.parse(read(path), filename=path)

        # Names bound to the result of round(...)
        rounded = set()
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign)
                    and isinstance(node.value, ast.Call)
                    and getattr(node.value.func, "id", None) == "round"):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        rounded.add(tgt.id)
        if not rounded:
            continue

        # Comparisons against a settings.* threshold
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            uses_setting = any(
                isinstance(c, ast.Attribute)
                and getattr(c.value, "id", None) == "settings"
                for c in [node.left, *node.comparators])
            if not uses_setting:
                continue
            for operand in [node.left, *node.comparators]:
                if isinstance(operand, ast.Name) and operand.id in rounded:
                    offenders.append(f"{name}:{node.lineno} compares "
                                     f"rounded `{operand.id}`")

    assert not offenders, (
        "A gate decision compares a display-rounded metric:\n  "
        + "\n  ".join(offenders)
        + "\nRound for the report; compare the full-precision value.")


# ======================================================================
# RULE GROUP N - the gate must be able to run where it is enforced
#
# Every rule above is worthless if the job that runs them cannot collect.
# That is exactly what happened: tests/conftest.py imports the application
# stack, the integrity-gate job installs only what static analysis needs, so
# pytest died at conftest import with exit code 4 and main stayed red through
# three merges without a single rule executing.
#
# Parsed as text, not YAML: pyyaml is not among the job's dependencies, and a
# rule that cannot run in that job is the failure this group exists to stop.
# ======================================================================

def test_the_ci_job_that_enforces_these_rules_can_collect_them():
    """
    The workflow step running this file must either pass --noconftest or install
    the application dependencies that tests/conftest.py imports. Without one of
    the two, pytest fails at conftest import and reports a missing module in
    place of every rule it never reached.
    """
    workflow = os.path.join(REPO_ROOT, ".github", "workflows",
                            "evidence-integrity-gate.yml")
    if not os.path.exists(workflow):
        pytest.skip("evidence-integrity-gate.yml absent")

    raw = read(workflow)

    # Comment lines do not configure anything. An earlier version of this rule
    # checked the whole file, so deleting the --noconftest flag still passed:
    # the comment explaining the flag mentioned it. A rule satisfied by prose
    # about the thing is not checking the thing.
    text = "\n".join(line for line in raw.split("\n")
                     if not line.lstrip().startswith("#"))

    assert "test_editor_integrity_gate.py" in text, (
        "The evidence integrity workflow no longer runs this file. If it moved, "
        "point the workflow at the new path - do not leave the rules unenforced.")

    # conftest.py's third-party imports, and the distributions that supply them.
    conftest = read(os.path.join(REPO_ROOT, "backend", "tests", "conftest.py"))
    needed = {
        "pytest_asyncio": "pytest-asyncio",
        "sqlalchemy": "sqlalchemy",
        "fastapi": "fastapi",
    }
    imported = [dist for module, dist in needed.items()
                if re.search(rf"^\s*(?:import|from)\s+{module}\b", conftest,
                             flags=re.MULTILINE)]

    bypasses_conftest = "--noconftest" in text
    installs_app = ("requirements.txt" in text
                    and all(dist in text for dist in imported))

    assert bypasses_conftest or installs_app, (
        "The integrity-gate job cannot collect these rules.\n"
        f"tests/conftest.py imports {', '.join(imported) or 'application modules'}, "
        "which the job does not install, and the step does not pass --noconftest."
        "\nPytest then exits 4 at conftest import and no rule in this file runs - "
        "the job reports a missing module instead of a finding, which is how main "
        "stayed red through three merges with the gate never executing.")
# RULE GROUP L - an instrument must measure the system that exists
#
# The decision-preservation checker carried both defects it was built to
# find. It restated the thresholds as literals, so after calibration moved
# CONTRAST_THRESHOLD from 18.0 to 8.8 it went on reporting against 18.0;
# and it took its deviations from the display-rounded metrics, so every
# margin it printed was quantised onto the rounding grid.
#
# Both produce a report that LOOKS like evidence and measures nothing the
# running system does.
# ======================================================================

THRESHOLD_SETTINGS = (
    "RETINAL_MIN_COVERAGE",
    "RETINAL_MAX_COVERAGE",
    "RETINAL_RED_RATIO_MIN",
    "CONTRAST_THRESHOLD",
    "ILLUMINATION_EXTREME_RATIO_MAX",
    "LAPLACIAN_BLUR_THRESHOLD",
)

# Metrics the gates publish twice: rounded for the report, exact for the
# decision. Analysis code must read the exact one.
DUAL_PRECISION_METRICS = (
    "mask_coverage",
    "red_to_blue_ratio",
    "contrast_dynamic_range",
    "extreme_pixel_ratio",
    "laplacian_variance",
)


def test_verifier_reads_thresholds_from_the_live_configuration():
    """
    The table of thresholds the checker reports against must be built from
    `settings`, not from literals. When it was written in by hand, calibration
    moved CONTRAST_THRESHOLD from 18.0 to 8.8 and the report went on printing
    "threshold 18.0", counting 2,502 images against a cut-point the system no
    longer had.

    An earlier version of THIS rule only asked whether `settings.X` appeared
    somewhere in the file. It appears in the JSON record and in the print
    statements, so hardcoding the comparison table back satisfied it. The rule
    now reads the mapping itself.
    """
    path = os.path.join(REPO_ROOT, "backend", "scripts",
                        "verify_gate_downsampling.py")
    tree = ast.parse(read(path), filename=path)

    mapping = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and any(getattr(t, "id", None) == "NEAR" for t in node.targets)
                and isinstance(node.value, ast.Dict)):
            mapping = node.value

    assert mapping is not None, (
        "verify_gate_downsampling.py no longer defines a `NEAR` threshold "
        "mapping. If it was renamed, point this rule at the new name - do not "
        "delete the rule.")

    literals = []
    settings_used = set()
    for key, value in zip(mapping.keys, mapping.values):
        label = getattr(key, "value", "?")
        if not isinstance(value, (ast.Tuple, ast.List)) or not value.elts:
            literals.append(f"`{label}` is not a (threshold, band) pair")
            continue
        threshold = value.elts[0]
        if (isinstance(threshold, ast.Attribute)
                and getattr(threshold.value, "id", None) == "settings"):
            settings_used.add(threshold.attr)
        else:
            shown = getattr(threshold, "value", ast.dump(threshold))
            literals.append(f"`{label}` compares against the literal {shown}")

    assert not literals, (
        "The decision-preservation check restates thresholds instead of "
        "reading them:\n  " + "\n  ".join(literals)
        + "\nRead each one from `settings`, so the report cannot outlive the "
          "value it describes.")

    # And the report must still cover every threshold the gates enforce.
    reported = settings_used | {
        name for name in THRESHOLD_SETTINGS
        if f"settings.{name}" in read(path)}
    missing = [name for name in THRESHOLD_SETTINGS if name not in reported]
    assert not missing, (
        "Thresholds enforced by the gates but absent from the check:\n  "
        + "\n  ".join(missing))


def test_deviation_analysis_uses_the_values_the_gates_decide_on():
    """
    Deviations and margins must come from the full-precision metrics. Taking
    them from the rounded fields quantises every margin onto the rounding grid
    - which is how one run reported a closest margin of exactly 0.0000 for
    2,502 separate images and made a crowded cut-point look like a tie.
    """
    path = os.path.join(REPO_ROOT, "backend", "scripts",
                        "verify_gate_downsampling.py")
    source = read(path)

    offenders = []
    for lineno, line in enumerate(source.split("\n"), 1):
        code = line.split("#", 1)[0]
        for metric in DUAL_PRECISION_METRICS:
            if re.search(rf"\.{metric}\b(?!_exact)", code):
                offenders.append(f"line {lineno}: reads rounded `.{metric}` "
                                 f"- use `.{metric}_exact`")

    assert not offenders, (
        "The decision-preservation check measures display-rounded metrics:\n  "
        + "\n  ".join(offenders)
        + "\nThe gates publish both; analysis reads the exact one.")


def test_gates_publish_the_exact_value_behind_every_rounded_one():
    """
    The rounded fields are the report. The exact fields are what the gate
    compared against its threshold, so they are what an audit needs - without
    them, no one downstream can reconstruct why an image was admitted.
    """
    gate_dir = os.path.join(REPO_ROOT, "backend", "app", "services", "validation")
    missing = []

    for name in ("gate2_relevance.py", "gate3_quality.py"):
        source = read(os.path.join(gate_dir, name))
        for metric in DUAL_PRECISION_METRICS:
            if f"self.{metric} = " not in source:
                continue
            if f"self.{metric}_exact = " not in source:
                missing.append(f"{name}: publishes `{metric}` rounded "
                               f"but not `{metric}_exact`")

    assert not missing, (
        "A gate reports a rounded metric with no full-precision counterpart:"
        "\n  " + "\n  ".join(missing)
        + "\nPublish both: the rounded value for the report, the exact value "
          "for the decision record.")
# RULE GROUP M - a quoted statistic must be internally coherent
#
# reproducibility_runbook.md quoted a referable sensitivity of 91.2% with a
# 95% CI of (81.5, 90.5) - an interval that does not contain its own point
# estimate, so at least one of the three numbers was wrong and no reader
# could tell which. The same block carried two lines about "the 27 affected
# images" that the script stopped printing when the clean split left zero.
#
# Neither error needs the corpus to catch. Both are arithmetic.
# ======================================================================

CI_PATTERN = re.compile(
    r"(\d+\.\d+)\s*%?\s*(?:,|\s)?\s*95%\s*CI\s*[\(\[]?\s*"
    r"(\d+\.\d+)\s*[-,]\s*(\d+\.\d+)")


def _documentation_files():
    """Every prose artefact, including the assembled submission package."""
    for base in (os.path.join(REPO_ROOT, "docs"),):
        for root, _dirs, names in os.walk(base):
            for name in sorted(names):
                if name.endswith(".md"):
                    yield os.path.join(root, name)
    tracker = os.path.join(REPO_ROOT, "PROGRESS_TRACKER.md")
    if os.path.exists(tracker):
        yield tracker


def test_every_confidence_interval_contains_its_point_estimate():
    """
    A 95% CI that excludes the estimate it qualifies is not a tight interval or
    a rounding artefact - it is a transcription error, and it discredits every
    other figure beside it. This is pure arithmetic on the text, so there is no
    excuse for it reaching a reader.
    """
    offenders = []
    for path in _documentation_files():
        rel = os.path.relpath(path, REPO_ROOT)
        for lineno, line in enumerate(read(path).split("\n"), 1):
            for match in CI_PATTERN.finditer(line):
                estimate, low, high = (float(g) for g in match.groups())
                if low > high:
                    offenders.append(f"{rel}:{lineno} CI ({low}, {high}) is inverted")
                elif not (low <= estimate <= high):
                    offenders.append(
                        f"{rel}:{lineno} estimate {estimate} lies outside "
                        f"its own 95% CI ({low}, {high})")

    assert not offenders, (
        "A confidence interval does not contain the estimate it qualifies:\n  "
        + "\n  ".join(offenders)
        + "\nRecompute from clinical_metrics.json; do not adjust the interval "
          "to fit the estimate.")


def test_runbook_expected_output_matches_what_the_script_reports():
    """
    The reproducibility runbook says "Any divergence is a defect. Please report
    it." It diverged from itself: the expected-output block quoted a
    clean-subset comparison over "the 27 affected images" long after the clean
    split had reduced that to zero, which the script therefore never prints.

    Check the figures the block states against the recomputed metrics.
    """
    runbook = os.path.join(REPO_ROOT, "docs", "chapter4",
                           "reproducibility_runbook.md")
    metrics_path = os.path.join(REPO_ROOT, "docs", "chapter4",
                               "clinical_metrics.json")
    if not (os.path.exists(runbook) and os.path.exists(metrics_path)):
        pytest.skip("runbook or recomputed metrics absent")

    with open(metrics_path, encoding="utf-8") as fh:
        metrics = json.load(fh)
    leakage = metrics.get("leakage_audit") or {}
    leaked = leakage.get("n_test_leaked")

    text = read(runbook)
    blocks = re.findall(r"```text\n(.*?)```", text, flags=re.DOTALL)
    expected = "\n".join(b for b in blocks if "LEAKAGE AUDIT" in b)
    assert expected, (
        "reproducibility_runbook.md no longer shows an expected-output block "
        "containing the leakage audit. It is the only place a reader can check "
        "their own run against; restore it rather than removing it.")

    problems = []

    # A clean-subset comparison is printed only when something actually leaked.
    quotes_clean_subset = ("affected images" in expected
                           or "clean images" in expected)
    if leaked == 0 and quotes_clean_subset:
        problems.append(
            "quotes a clean-subset comparison, but the audit reports 0 leaked "
            "held-out images, so the script prints no such lines")
    if leaked and not quotes_clean_subset:
        problems.append(
            f"omits the clean-subset comparison, but the audit reports "
            f"{leaked} leaked held-out image(s), which the script does print")

    # The cohort size and leak count must be the recomputed ones.
    n_test = leakage.get("n_test")
    if n_test and f"/{n_test} " not in expected and f"/{n_test}(" not in expected:
        problems.append(f"does not show the audited cohort size of {n_test}")
    if leaked is not None and f"{leaked}/{n_test}" not in expected:
        problems.append(f"does not show the audited leak count {leaked}/{n_test}")

    assert not problems, (
        "The runbook's expected output does not match what the script reports:"
        "\n  " + "\n  ".join(problems)
        + "\nPaste the current output; a runbook promising output the script "
          "never produces is worse than no runbook.")


# ======================================================================
# RULE GROUP O - a handover may only carry what the run measured
#
# run_calibration_pipeline.py gathers artefacts into a zip for handover. Its
# first draft decided a step was "already done" by asking whether the artefact
# file existed - and every one of those artefacts is committed to this
# repository, so on a fresh clone it skipped four of five steps and would have
# shipped the previous run's evidence as freshly measured.
#
# That is the fabricated-artefact failure arriving by the back door. The rule
# below keeps the two properties that close it.
# ======================================================================

def test_the_pipeline_resumes_on_recorded_work_not_on_file_existence():
    """
    Resume must consult a state file recording what this run completed. Asking
    `os.path.exists(artefact)` cannot distinguish work done now from a file that
    arrived with `git clone`.
    """
    path = os.path.join(REPO_ROOT, "backend", "scripts",
                        "run_calibration_pipeline.py")
    if not os.path.exists(path):
        pytest.skip("run_calibration_pipeline.py absent")

    tree = ast.parse(read(path), filename=path)

    planners = [node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == "main"]
    assert planners, "run_calibration_pipeline.py has no main()"

    # The plan must branch on recorded completion.
    plan_source = ast.get_source_segment(read(path), planners[0]) or ""
    assert "completed" in plan_source and "load_state" in plan_source, (
        "The pipeline's plan does not consult recorded completion state. Resume "
        "must be based on what this run measured, not on whether a committed "
        "artefact happens to be present.")

    # And it must not decide skipping from the artefact's presence.
    offenders = []
    for node in ast.walk(planners[0]):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "exists"):
            for arg in node.args:
                src = ast.get_source_segment(read(path), arg) or ""
                if "proves" in src or "artefact" in src:
                    offenders.append(f"line {node.lineno}: skips on "
                                     f"os.path.exists({src})")

    assert not offenders, (
        "The pipeline decides a step is done from the artefact's presence:\n  "
        + "\n  ".join(offenders)
        + "\nThose artefacts are committed, so on a fresh clone this ships the "
          "previous run's evidence as though it had just been measured.")


def test_the_handover_refuses_artefacts_older_than_the_run():
    """
    `collect()` must compare each artefact's mtime against the run's start and
    leave the older ones out. Without it the zip silently mixes measured output
    with the committed copies, and nobody downstream can tell which is which.
    """
    path = os.path.join(REPO_ROOT, "backend", "scripts",
                        "run_calibration_pipeline.py")
    if not os.path.exists(path):
        pytest.skip("run_calibration_pipeline.py absent")

    source = read(path)
    tree = ast.parse(source, filename=path)

    collectors = [node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef) and node.name == "collect"]
    assert collectors, "run_calibration_pipeline.py has no collect()"

    body = ast.get_source_segment(source, collectors[0]) or ""

    assert "getmtime" in body, (
        "collect() does not check artefact modification times. An artefact older "
        "than the run start was not produced by it; shipping it puts unmeasured "
        "evidence in the handover.")
    assert "stale" in body.lower(), (
        "collect() has no stale category. An artefact that predates the run must "
        "be reported and withheld, not copied in beside the measured ones.")


# ======================================================================
# RULE GROUP P - an escape sequence that got written out as a control code
#
# A patch written with a non-raw Python string turned "\times" into TAB +
# "imes" and "\text{" into TAB + "ext{", so two documents rendered as
#
#     operating on a fixed $224     imes 224$
#     the metric is $r_{    ext{mean}} / (b_{    ext{mean}} + 10^{-6})$
#
# and sat there through a merge. Markdown prose in this project has no
# legitimate use for a literal tab, so one is a reliable signature of exactly
# this mistake - cheap to detect, and invisible to a reader skimming rendered
# output where the tab collapses to a space.
# ======================================================================

CONTROL_CHARACTERS = {
    "\t": r"a tab, usually a flattened \t from a non-raw string (e.g. \times)",
    "\x08": r"a backspace, usually a flattened \b",
    "\x0c": r"a form feed, usually a flattened \f",
    "\x0b": r"a vertical tab, usually a flattened \v",
    "\r": "a carriage return inside a line",
}


def test_no_document_contains_a_flattened_escape_sequence():
    """
    LaTeX in these documents is written with backslash escapes. Patching them
    through a non-raw string silently converts the escape into the control
    character it names, and the damage survives review because a tab renders as
    whitespace.
    """
    offenders = []
    for root, _dirs, names in os.walk(os.path.join(REPO_ROOT, "docs")):
        for name in sorted(names):
            if not name.endswith(".md"):
                continue
            path = os.path.join(root, name)
            rel = os.path.relpath(path, REPO_ROOT)
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
            # Normalise real line endings first; only in-line CRs are suspect.
            text = text.replace("\r\n", "\n")
            for lineno, line in enumerate(text.split("\n"), 1):
                for char, why in CONTROL_CHARACTERS.items():
                    if char in line:
                        context = line.replace(char, "<<HERE>>").strip()[:90]
                        offenders.append(f"{rel}:{lineno} contains {why}\n"
                                         f"      {context}")

    assert not offenders, (
        "A document contains a control character where an escape sequence was "
        "intended:\n  " + "\n  ".join(offenders)
        + "\n\nWrite the patch with a RAW string (r\"\\times\") or a written .py "
          "file - a heredoc and a plain Python string both flatten these.")


# ======================================================================
# RULE GROUP Q - a file must be the image its name claims
#
# storage/datasets/aptos2019/train_images/ on the development machine held 15
# files named after real APTOS images - every name an image_id present in
# dataset_split_manifest.csv. None was that image: all 640x480, 4-5 KB, and
# EIGHT byte-identical to each other, against real photographs of ~1.8 MB at up
# to 2848 px.
#
# Nothing caught it. Rule group H already required that a CSV may only cite
# image_ids from the manifest, and these files satisfied that perfectly - the
# ids were real. No check asked whether the FILE was the image the id names,
# and the manifest has carried the answer all along in its SHA-256 column.
#
# A script pointed at that directory produces output indistinguishable from
# evidence: real ids, plausible gate metrics, a grade per image.
# ======================================================================

CORPUS_CONSUMING_SCRIPTS = (
    "calibrate_blur_threshold.py",
    "generate_validation_evidence.py",
    "verify_gate_downsampling.py",
    "benchmark_cpu_end_to_end.py",
)


def test_every_script_that_measures_a_corpus_verifies_it_first():
    """
    Any script handed a directory of images must check those images against the
    manifest's recorded SHA-256 before measuring them.
    """
    missing = []
    for name in CORPUS_CONSUMING_SCRIPTS:
        path = os.path.join(REPO_ROOT, "backend", "scripts", name)
        if not os.path.exists(path):
            missing.append(f"{name}: absent - if it was renamed, point this "
                           f"rule at the new name rather than dropping it")
            continue
        tree = ast.parse(read(path), filename=path)
        calls = {node.func.id for node in ast.walk(tree)
                 if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name)}
        if "assert_corpus_is_authentic" not in calls:
            missing.append(f"{name}: measures a corpus without verifying it")

    assert not missing, (
        "A script measures images without checking they are the images they "
        "claim to be:\n  " + "\n  ".join(missing)
        + "\nCall assert_corpus_is_authentic(images_dir) before any "
          "measurement - see backend/scripts/corpus_guard.py.")


def test_the_corpus_guard_compares_bytes_against_the_manifest():
    """
    The guard's whole value is that it hashes the file and compares against the
    manifest. A version that only checked filenames would pass every impostor,
    because the impostors had real image_ids for names.
    """
    path = os.path.join(REPO_ROOT, "backend", "scripts", "corpus_guard.py")
    assert os.path.exists(path), (
        "corpus_guard.py is gone. It is the only thing standing between a "
        "directory of placeholders and a published evidence table.")

    source = read(path)
    tree = ast.parse(source, filename=path)

    assert "sha256" in source, "corpus_guard.py no longer hashes anything"
    assert "sha256_hash" in source, (
        "corpus_guard.py no longer reads the manifest's sha256_hash column, so "
        "it has nothing to compare a file against")

    checkers = [n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "check_corpus"]
    assert checkers, "corpus_guard.py has no check_corpus()"
    body = ast.get_source_segment(source, checkers[0]) or ""
    assert "_sha256(" in body, (
        "check_corpus() does not hash the files. Comparing names alone passes "
        "every impostor, since an impostor's name is a real image_id.")


# Claims of a regulatory STATUS the system does not hold. Being informed by a
# framework is a legitimate design statement and is not listed here; holding a
# class, or complying, is not.
REGULATORY_STATUS_CLAIMS = [
    "Class II Software as a Medical Device",
    "FDA Class II",
    "Classification**: SaMD",
    "SaMD Risk Categorization",
    "In compliance with FDA",
    "In accordance with international regulatory guidelines",
    "regulatory approval",
    "CE marked",
    "FDA cleared",
    "FDA approved",
]


def test_no_document_claims_a_regulatory_status_the_system_lacks():
    """
    This is a PGD research prototype. It has no device classification, has not
    been through a conformity assessment, and is not approved or cleared by any
    regulator. Documents may cite these frameworks as design references; they
    may not place the system inside one.
    """
    offenders = []
    for rel in _prose_files():
        if "chapter4_submission_package" in rel.replace("\\", "/"):
            continue        # a mirror of the sources checked above
        path = os.path.join(REPO_ROOT, rel)
        for lineno, line in enumerate(read(path).split("\n"), 1):
            for phrase in REGULATORY_STATUS_CLAIMS:
                if phrase.lower() in line.lower():
                    offenders.append(f"{rel}:{lineno} claims {phrase!r}")

    assert not offenders, (
        "A document places this prototype inside a regulatory framework it does "
        "not belong to:\n  " + "\n  ".join(offenders)
        + "\nWrite that the design is INFORMED BY the guidance, and state that "
          "no classification, conformity assessment or approval exists.")


# ======================================================================
# RULE GROUP R - one test result, and the documents must quote it
#
# An independent reviewer found no single authoritative passing run. The
# package carried FIVE different answers to "did the tests pass":
#
#     test_execution.log                 184 collected, 183 passed, 1 skipped
#     system_test_report.md              "183 collected - 182 passed"
#     independent_thesis_qa_gate_audit   "183 collected - 182 passed"
#     reproducibility_runbook.md         "183 collected - 182 passed"
#     PROGRESS_TRACKER.md                "175 passed, 1 skipped"
#     test_execution_output.txt (stale)  "167 passed, 3 skipped"
#
# The documents disagreed with the log they cite, and with each other. A
# reader cannot tell which run is the evidence, so none of them is.
#
# These rules do the arithmetic the reviewer did by hand.
# ======================================================================

TEST_LOG = os.path.join(REPO_ROOT, "docs", "chapter4", "test_execution.log")

# "183 passed, 1 skipped" / "183 passed, 1 skipped, 2 warnings in 68s"
_SUMMARY = re.compile(
    r"(?P<passed>\d+)\s+passed"
    r"(?:,\s*(?P<failed>\d+)\s+failed)?"
    r"(?:,\s*(?P<skipped>\d+)\s+skipped)?")
_COLLECTED = re.compile(r"collected\s+(\d+)\s+item")

# Any document statement of the form "N collected", "N passed", "N skipped".
_CLAIM = re.compile(
    r"\*{0,2}(\d+)\*{0,2}\s*(?:tests?\s+)?(collected|passed|skipped|failed)")


def _authoritative_counts():
    """(collected, passed, skipped, failed) from the committed log."""
    if not os.path.exists(TEST_LOG):
        return None
    text = read(TEST_LOG)
    summary = None
    for match in _SUMMARY.finditer(text):
        summary = match          # the last summary line is the run's verdict
    if summary is None:
        return None
    collected = None
    found = _COLLECTED.findall(text)
    if found:
        collected = int(found[-1])
    return (
        collected,
        int(summary.group("passed")),
        int(summary.group("skipped") or 0),
        int(summary.group("failed") or 0),
    )


def test_the_committed_test_log_records_a_passing_run():
    """
    The log is the evidence. If it records failures, no document may describe
    the suite as passing, and the right fix is to make the tests pass.
    """
    counts = _authoritative_counts()
    assert counts is not None, (
        f"No pytest summary found in {os.path.relpath(TEST_LOG, REPO_ROOT)}. "
        "The committed log is what every document cites; regenerate it.")
    collected, passed, skipped, failed = counts

    assert failed == 0, (
        f"The committed test log records {failed} failure(s). Fix the tests and "
        "regenerate the log - do not describe the suite as passing while its own "
        "evidence says otherwise.")

    if collected is not None:
        assert passed + skipped == collected, (
            f"The log's own arithmetic does not close: {collected} collected but "
            f"{passed} passed + {skipped} skipped = {passed + skipped}.")


def test_documents_quote_the_committed_test_log():
    """
    Every test count stated in a document must be one the log actually records.
    A document saying "183 collected - 182 passed" beside a log saying "184
    collected, 183 passed" leaves the reader to guess which is the run.
    """
    counts = _authoritative_counts()
    if counts is None:
        pytest.skip("no committed test log")
    collected, passed, skipped, failed = counts
    allowed = {"collected": collected, "passed": passed,
               "skipped": skipped, "failed": failed}

    targets = [
        os.path.join(REPO_ROOT, "docs", "chapter4", "system_test_report.md"),
        os.path.join(REPO_ROOT, "docs", "chapter4",
                     "independent_thesis_qa_gate_audit.md"),
        os.path.join(REPO_ROOT, "docs", "chapter4",
                     "reproducibility_runbook.md"),
        os.path.join(REPO_ROOT, "docs", "chapter4",
                     "objective_traceability_matrix.md"),
        os.path.join(REPO_ROOT, "PROGRESS_TRACKER.md"),
    ]

    offenders = []
    for path in targets:
        if not os.path.exists(path):
            continue
        rel = os.path.relpath(path, REPO_ROOT)
        for lineno, line in enumerate(read(path).split("\n"), 1):
            # An earlier version of this rule also required the word "test",
            # "suite" or "pytest" on the same line. The line it most needed to
            # catch - "**Overall Result:** **183 collected - 182 passed**" -
            # contains none of them, so the rule passed over the single
            # clearest contradiction in the package. The verbs below are
            # specific enough on their own in these documents.
            for value, kind in _CLAIM.findall(line):
                expected = allowed.get(kind)
                if expected is None:
                    continue
                if int(value) != expected:
                    offenders.append(
                        f"{rel}:{lineno} says {value} {kind}; the committed log "
                        f"records {expected}")

    assert not offenders, (
        "A document states a test count the committed log does not support:\n  "
        + "\n  ".join(offenders)
        + "\nRegenerate docs/chapter4/test_execution.log and quote it, or correct "
          "the document. One run is the evidence.")


def test_only_one_test_log_ships():
    """
    A second, older log beside the first is how the package came to carry two
    different answers. The package mirrors the canonical log; nothing else may
    sit beside it claiming to be a run.
    """
    package_logs = os.path.join(REPO_ROOT, "docs", "chapter4_submission_package",
                                "logs_and_metrics")
    if not os.path.isdir(package_logs):
        pytest.skip("submission package not assembled")

    found = sorted(
        name for name in os.listdir(package_logs)
        if re.search(r"test.*(execution|run|output)", name, re.I))

    assert len(found) <= 1, (
        "More than one test log ships in the submission package:\n  "
        + "\n  ".join(found)
        + "\nKeep exactly one. A superseded log beside the current one is "
          "indistinguishable from the current one to a reader.")


# ======================================================================
# RULE GROUP S - what the production path requires, the project must declare
#
# torch and torchvision were absent from backend/requirements.txt. The trained
# engine fails closed without them, correctly, so a deployment built from
# exactly what the project declares returned HTTP 503 on every assessment -
# and the 503 told the operator to "Install backend/requirements.txt", a file
# that did not contain the module it was complaining about.
#
# One omission, three symptoms: a dead local stack, a failing CI suite, and
# main red for three commits.
# ======================================================================

REQUIREMENTS = os.path.join(REPO_ROOT, "backend", "requirements.txt")

# Modules the serving path imports and cannot run without.
RUNTIME_CRITICAL_IMPORTS = ("torch", "torchvision")


def _declared_requirements():
    if not os.path.exists(REQUIREMENTS):
        return set()
    names = set()
    for line in read(REQUIREMENTS).split("\n"):
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        name = re.split(r"[<>=!\[;]", line, maxsplit=1)[0].strip().lower()
        if name:
            names.add(name)
    return names


def test_requirements_declares_what_the_engine_cannot_run_without():
    """
    The inference service raises ModelCheckpointError on ImportError for these
    modules. If they are not declared, the documented install produces a service
    that cannot serve, which is what happened.
    """
    declared = _declared_requirements()
    missing = [m for m in RUNTIME_CRITICAL_IMPORTS if m.lower() not in declared]

    assert not missing, (
        "The production inference path imports modules the project does not "
        "declare:\n  " + "\n  ".join(missing)
        + f"\nAdd them to {os.path.relpath(REQUIREMENTS, REPO_ROOT)}. Until then "
          "a deployment built from this project's own requirements returns HTTP "
          "503 on every assessment.")


def test_the_dependency_error_gives_an_instruction_that_works():
    """
    The 503 raised when PyTorch is absent must name something that actually
    installs it. It previously said "Install backend/requirements.txt" while
    that file declared no torch - the one piece of advice the system offers at
    its own failure point, and it did not work.
    """
    path = os.path.join(REPO_ROOT, "backend", "app", "services", "ai_service.py")
    source = read(path)

    tree = ast.parse(source, filename=path)
    message = ""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise):
            continue
        segment = ast.get_source_segment(source, node) or ""
        if "PyTorch is not installed" in segment:
            message = segment
            break

    assert message, (
        "No raise carrying the 'PyTorch is not installed' message was found. If "
        "it moved, point this rule at it rather than dropping the rule.")

    assert "pip install torch" in message, (
        "The dependency error does not tell the operator how to install torch. "
        "Naming a requirements file is only useful if that file declares it; "
        "name the install command.")

    declared = _declared_requirements()
    if "requirements.txt" in message:
        assert "torch" in declared, (
            "The dependency error points at backend/requirements.txt, which does "
            "not declare torch. Either declare it there or stop citing the file.")


# ======================================================================
# RULE GROUP T - a split must belong to the total it is quoted beside
#
# PROGRESS_TRACKER.md read "APTOS 2019 - 3,662 images, split 2,453 / 526 / 525".
# Those partitions sum to 3,504. 3,662 is what APTOS publishes; 3,504 is what
# survives de-duplication and the removal of the 30 conflicting-label groups.
# The line contradicted itself, and a reader has no way to tell which number
# describes the cohort the model was trained and evaluated on.
# ======================================================================

SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST = 2453, 526, 525
RETAINED_TOTAL = SPLIT_TRAIN + SPLIT_VAL + SPLIT_TEST      # 3504
PUBLISHED_TOTAL = 3662


def test_the_partition_counts_are_never_quoted_against_the_published_total():
    """
    Wherever the three partition counts appear together, the total on that line
    must be the retained one. Quoting them beside 3,662 asserts a split of a
    cohort that was not split.
    """
    offenders = []
    for rel in _prose_files():
        if "chapter4_submission_package" in rel.replace("\\", "/"):
            continue
        for lineno, line in enumerate(read(os.path.join(REPO_ROOT, rel)).split("\n"), 1):
            has_train = f"{SPLIT_TRAIN:,}" in line or str(SPLIT_TRAIN) in line
            has_val = f"{SPLIT_VAL:,}" in line or str(SPLIT_VAL) in line
            has_test = f"{SPLIT_TEST:,}" in line or str(SPLIT_TEST) in line
            if not (has_train and has_val and has_test):
                continue
            # A correction record describes the SUPERSEDED split, which really
            # was of 3,662 before de-duplication. Rewriting those lines would
            # destroy the evidence that the defect existed - the mistake a
            # blanket figure replacement has already made twice here.
            if "superseded" in line.lower() or "withdrawn" in line.lower():
                continue
            if f"{PUBLISHED_TOTAL:,}" in line or str(PUBLISHED_TOTAL) in line:
                # Allowed only when the line also names the retained total, i.e.
                # it is explicitly contrasting the two.
                if f"{RETAINED_TOTAL:,}" not in line and str(RETAINED_TOTAL) not in line:
                    offenders.append(
                        f"{rel}:{lineno} quotes the split "
                        f"{SPLIT_TRAIN}/{SPLIT_VAL}/{SPLIT_TEST} (sum "
                        f"{RETAINED_TOTAL}) against the published total "
                        f"{PUBLISHED_TOTAL}")

    assert not offenders, (
        "A partition split is quoted against a total it does not sum to:\n  "
        + "\n  ".join(offenders)
        + f"\n{PUBLISHED_TOTAL} is what APTOS publishes; {RETAINED_TOTAL} is what "
          "was split. Name both, or name the one the partitions belong to.")


# ======================================================================
# RULE GROUP U - the interface may not advertise a limit the system does not use
#
# The upload panel read "Accepted: JPEG, PNG - Max: 15MB - Min: 512x512" while
# MIN_IMAGE_DIMENSION had been calibrated to 480. A clinician sizing images to
# the advertised limit would be working to a rule the system does not enforce,
# and a 490px image - which the gates now accept - was being described as too
# small on the very screen that accepts it.
# ======================================================================

def test_the_upload_screen_quotes_the_configured_limits():
    """
    Numbers the interface states as limits must come from config.py. They are a
    promise to the user about what the system will do.
    """
    screen = os.path.join(REPO_ROOT, "frontend", "src", "screens",
                          "NewAssessmentScreen.tsx")
    if not os.path.exists(screen):
        pytest.skip("NewAssessmentScreen.tsx absent")

    config = read(os.path.join(REPO_ROOT, "backend", "app", "core", "config.py"))

    def setting(name, pattern):
        match = re.search(pattern, config)
        assert match, f"{name} not found in config.py"
        return match.group(1)

    min_dim = setting("MIN_IMAGE_DIMENSION",
                      r"MIN_IMAGE_DIMENSION:\s*int\s*=\s*(\d+)")
    max_mb = setting("MAX_UPLOAD_SIZE_MB",
                     r"MAX_UPLOAD_SIZE_MB:\s*int\s*=\s*(\d+)")

    text = read(screen)
    offenders = []

    advertised_min = re.search(r"Min:\s*(\d+)\s*[xX\u00d7]\s*(\d+)", text)
    if advertised_min:
        for shown in advertised_min.groups():
            if shown != min_dim:
                offenders.append(
                    f"upload panel advertises a {shown}px minimum; "
                    f"MIN_IMAGE_DIMENSION is {min_dim}")

    advertised_max = re.search(r"Max:\s*(\d+)\s*MB", text)
    if advertised_max and advertised_max.group(1) != max_mb:
        offenders.append(
            f"upload panel advertises a {advertised_max.group(1)}MB maximum; "
            f"MAX_UPLOAD_SIZE_MB is {max_mb}")

    assert not offenders, (
        "The upload screen states a limit the system does not enforce:\n  "
        + "\n  ".join(offenders)
        + "\nThese numbers are a promise to the clinician about what will be "
          "accepted. Quote the configured value.")


def test_the_pdf_report_quotes_the_configured_threshold():
    """
    The generated consultation PDF printed "Laplacian variance: 71.6
    (Threshold >= 60.0)" while LAPLACIAN_BLUR_THRESHOLD was 4.3. The number had
    been written into the report template by hand, so a document handed to a
    clinician stated a decision rule the system had stopped applying - and it
    stated it beside a real measurement, which makes it look checked.
    """
    path = os.path.join(REPO_ROOT, "backend", "app", "services",
                        "report_service.py")
    if not os.path.exists(path):
        pytest.skip("report_service.py absent")

    offenders = []
    for lineno, line in enumerate(read(path).split("\n"), 1):
        code = line.split("#", 1)[0]
        if re.search(r"Threshold\s*>=\s*[0-9]", code):
            offenders.append(f"line {lineno}: threshold written as a literal")

    assert not offenders, (
        "The PDF report states a threshold as a literal:\n  "
        + "\n  ".join(offenders)
        + "\nRead it from `settings`. A report quoting a rule the system does "
          "not apply is worse than one that omits the rule.")


# ======================================================================
# RULE GROUP V - the training table must BE epoch_history.csv
#
# training_protocol.md carries a fifteen-row epoch table under a heading that
# cites epoch_history.csv as its source. Every one of the fifteen rows
# disagreed with that file. Not rounding: epoch 06 read 0.8519 against a
# recorded 0.8978, and the selected epoch's validation accuracy read 80.55%
# against a recorded 83.46%. The table held the SUPERSEDED run's figures and
# was never regenerated after the clean retrain.
#
# It also marked SIX epochs "**BEST**" - the running "new best so far" of a
# checkpoint callback - leaving a reader unable to tell which was the claim.
#
# Nothing caught it because every existing rule checked that metrics recompute
# from held_out_predictions.csv. No rule read the training history at all.
# ======================================================================

EPOCH_HISTORY = os.path.join(CHAPTER4, "epoch_history.csv")
TRAINING_PROTOCOL = os.path.join(CHAPTER4, "training_protocol.md")

_EPOCH_ROW = re.compile(r"^\|\s*\*{0,2}(\d{1,2})\*{0,2}\s*\|")


def test_training_table_matches_epoch_history():
    """
    Every row of the epoch table must equal the recorded run. The table is a
    transcription of a committed CSV, so there is no reason for it ever to
    differ, and a difference means the document is describing a different run.
    """
    if not (os.path.exists(EPOCH_HISTORY) and os.path.exists(TRAINING_PROTOCOL)):
        pytest.skip("training artefacts absent")

    with open(EPOCH_HISTORY, newline="", encoding="utf-8") as fh:
        raw = {int(r["epoch"]): r for r in csv.DictReader(fh)}

    offenders = []
    seen = set()
    for line in read(TRAINING_PROTOCOL).split("\n"):
        match = _EPOCH_ROW.match(line)
        if not match:
            continue
        epoch = int(match.group(1))
        record = raw.get(epoch)
        if record is None:
            continue
        seen.add(epoch)

        cells = [c.strip().replace("**", "").replace("`", "")
                 for c in line.split("|")[1:-1]]
        if len(cells) < 6:
            continue

        stated_acc, stated_qwk = cells[3].replace("%", ""), cells[4]
        actual_acc = float(record["val_accuracy"]) * 100
        actual_qwk = float(record["val_qwk"])

        try:
            if abs(float(stated_acc) - actual_acc) > 0.011:
                offenders.append(
                    f"epoch {epoch:02d}: val accuracy {stated_acc}% stated, "
                    f"{actual_acc:.2f}% recorded")
            if abs(float(stated_qwk) - actual_qwk) > 0.00011:
                offenders.append(
                    f"epoch {epoch:02d}: val QWK {stated_qwk} stated, "
                    f"{actual_qwk:.4f} recorded")
        except ValueError:
            offenders.append(f"epoch {epoch:02d}: unparsable row")

    assert seen, (
        "No epoch rows were found in training_protocol.md. If the table moved, "
        "point this rule at it rather than deleting the rule.")

    assert not offenders, (
        "The training table disagrees with epoch_history.csv:\n  "
        + "\n  ".join(offenders)
        + "\nThe table is a transcription of that file. Regenerate it; do not "
          "edit the numbers by hand.")


def test_exactly_one_epoch_is_marked_selected():
    """
    Six epochs were marked "BEST" beside one "BEST (SELECTED)". Only one
    checkpoint was evaluated, and a reader must be able to see which.
    """
    if not os.path.exists(TRAINING_PROTOCOL):
        pytest.skip("training_protocol.md absent")

    # Count TABLE ROWS only. An earlier version of this rule counted the whole
    # document and tripped on the paragraph that explains what SELECTED means -
    # a rule measuring prose ABOUT the thing instead of the thing.
    selected = sum(
        1 for line in read(TRAINING_PROTOCOL).split("\n")
        if _EPOCH_ROW.match(line) and re.search(r"\bSELECTED\b", line))

    assert selected == 1, (
        f"{selected} rows are marked SELECTED in training_protocol.md; exactly "
        "one checkpoint was evaluated. Mark intermediate improvements as "
        "improvements.")


# ======================================================================
# RULE GROUP W - the package may not keep what the source dropped
#
# assemble_submission_package.py copies and updates; it does not remove. When
# figure 4b was renamed, the package kept BOTH files - and the orphan was the
# rejected mid-animation capture, deleted precisely because it did not show
# what its caption claimed. It would have travelled to the editor inside the
# zip, where it is the only version anyone reads.
#
# Same shape as the superseded second test log: invisible in the source tree,
# present in the artefact that goes out.
# ======================================================================

def test_the_package_screenshots_match_the_source_exactly():
    """
    A figure in the package that the source no longer has is an orphan, and an
    orphan is a figure nobody is maintaining.
    """
    src = os.path.join(CHAPTER4, "screenshots")
    pkg = os.path.join(REPO_ROOT, "docs", "chapter4_submission_package",
                       "screenshots")
    if not (os.path.isdir(src) and os.path.isdir(pkg)):
        pytest.skip("screenshot directories absent")

    source = {f for f in os.listdir(src) if not f.startswith(".")}
    packaged = {f for f in os.listdir(pkg) if not f.startswith(".")}

    orphans = sorted(packaged - source)
    missing = sorted(source - packaged)

    problems = []
    if orphans:
        problems.append("in the package but not in the source: "
                        + ", ".join(orphans))
    if missing:
        problems.append("in the source but not in the package: "
                        + ", ".join(missing))

    assert not problems, (
        "The submission package's figures disagree with the source:\n  "
        + "\n  ".join(problems)
        + "\nThe package is what the examiner reads. Re-run "
          "assemble_submission_package.py and delete anything it leaves behind.")


# ======================================================================
# RULE GROUP X - the schema document must describe the schema that exists
#
# database_schema.md listed `certified_grade`, `certified_grade_label` and
# `referral_plan` on professional_reviews. The model declares
# reviewer_assessed_grade and reviewer_assessed_grade_label, and has never had a
# referral_plan column at all.
#
# A reviewer asked for those fields to be RENAMED, which was impossible: they do
# not exist. The document was describing a table the system does not have, in
# the one vocabulary the scope rules exclude - and an earlier answer defended
# the names as "historical", treating a false document as a naming problem.
#
# A schema document that does not match the model is worse than no schema
# document: it is confidently wrong.
# ======================================================================

SCHEMA_DOC = os.path.join(CHAPTER4, "database_schema.md")
MODELS = os.path.join(REPO_ROOT, "backend", "app", "models", "models.py")

# Tables whose documented columns are checked against the model. Keyed by the
# SQLAlchemy class name; the value is the heading the document uses.
CHECKED_TABLES = {"ProfessionalReview": "professional_reviews"}


def _model_columns(class_name):
    """Column attribute names declared on a model class."""
    tree = ast.parse(read(MODELS), filename=MODELS)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ClassDef) and node.name == class_name):
            continue
        names = set()
        for stmt in node.body:
            if not isinstance(stmt, ast.Assign):
                continue
            if not (isinstance(stmt.value, ast.Call)
                    and getattr(stmt.value.func, "id", None) == "Column"):
                continue
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        return names
    return set()


def test_schema_document_only_describes_columns_that_exist():
    """
    Every column the schema document names for a checked table must be declared
    on the corresponding model.
    """
    if not (os.path.exists(SCHEMA_DOC) and os.path.exists(MODELS)):
        pytest.skip("schema document or models absent")

    text = read(SCHEMA_DOC)
    offenders = []

    for class_name, table in CHECKED_TABLES.items():
        declared = _model_columns(class_name)
        assert declared, (
            f"No Column assignments found on model class {class_name}. If it "
            "moved, point this rule at the new name rather than dropping it.")

        # Column rows look like:  | `column_name` | TYPE | ... |
        section = text
        heading = "`%s`" % table
        if heading in text:
            start = text.index(heading)
            nxt = text.find("\n### ", start)
            section = text[start:nxt if nxt != -1 else len(text)]

        for match in re.finditer(r"^\|\s*`([a-z_][a-z0-9_]*)`\s*\|", section,
                                 re.MULTILINE):
            column = match.group(1)
            if column not in declared:
                offenders.append(
                    "%s: documents column `%s`, which %s does not declare"
                    % (table, column, class_name))

    assert not offenders, (
        "database_schema.md describes columns the model does not have:\n  "
        + "\n  ".join(offenders)
        + "\nTranscribe the table from backend/app/models/models.py. A schema "
          "document that does not match the model is confidently wrong, which "
          "is worse than absent.")
