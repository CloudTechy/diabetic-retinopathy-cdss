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
import importlib.util
import json
import math
import os
import posixpath
import re
import sys
from collections import Counter

import pytest

from app.core.config import settings

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHAPTER4 = os.path.join(REPO_ROOT, "docs", "chapter4")
SCRIPTS = os.path.join(REPO_ROOT, "backend", "scripts")
CONFIG_PY = os.path.join(REPO_ROOT, "backend", "app", "core", "config.py")
NOTEBOOKS = os.path.join(REPO_ROOT, "notebooks")

EXPECTED_CHECKPOINT_SHA256 = "67d0b89641f08057126dd411e380b25575ef29f71ae37ee5796d472d9203dbf7"


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


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
    # Found by the sixth independent review, in prose, a stylesheet comment, a
    # TSX comment, a validation message and the PDF footer respectively.
    "human clinician certifications", "human clinical certifications",
    "authoritative clinical artefact", "Certified Clinician",
    "meets diagnostic quality", "CLINICAL CONSULTATION RECORD",
    "legal and medical responsibility",
    # Seventh round: the word survived in a UI header visible in figure 07,
    # the PDF subtitle, a route docstring and three documents; "cryptographically
    # bound" and "signed & immutable" dress a SHA-256 over fields as a signature;
    # a fabricated-identity variant lived in the API contract; the integration
    # guide cited medical-device standards as its "implementation standard".
    "Consultation", "consultation", "cryptographically bound", "Cryptographically Bound",
    "SIGNED & IMMUTABLE", "Finalized & Signed", "Ada Okonjo", "adjunct diagnostic",
    "21 CFR", "ISO 13485", "IEC 62304", "Non-Repudiation", "Top Saliency Zone",
    # Eighth round, same family: a truncated unkeyed SHA-256 is hash anchoring,
    # not tamper evidence; the API write-lock is not immutability; nothing here
    # is a legal or official instrument; Grad-CAM names no biomarker.
    "tamper-evident", "Tamper-Evident", "Tamper Evidence", "Immutable Clinical Audit",
    "immutable audit", "Immutable Audit", "immutable review", "legal compliance",
    "Legal Traceability", "Official Review", "biomarkers", "hash-sealed",
    # Ninth round: served route descriptions and code comments.
    "digital signature", "strictly immutable", "Immutable Locked", "signs off on",
    # Editor round on rev10: a browser check implied diagnostic adequacy; three
    # architectural statements overstated the design.
    "diagnostic resolution", "adheres strictly to third normal form",
    "absolute physical domain separation", "HTTPS / TLS 1.3 REST",
    "wired to run on every commit",
    # Editor round on rev13: a truncated unkeyed SHA-256 and an API refusal
    # are a hash anchor and a write-lock, not cryptography; the header has no
    # TLS badge; screenshots, system tests and the build are not "the
    # training run".
    "cryptographic signature", "cryptographic immutability", "cryptographic audit",
    "Cryptographic Audit", "cryptographically verified", "Secure Private Binary Storage",
    "TLS badge", "Every artefact in this package derives",
    # Reviewer round on rev14: signature vocabulary survived in the PDF, the
    # stored hash prefix and the audit text; two routes said "securely" while
    # taking no credential; a latency claim and an error-format standard
    # predated the benchmark and the code; ids are not UUIDs.
    "Signature Timestamp", "SIG-SHA256", "Clinician signed record", "Signature Strip",
    "Integrity ID", "Sub-100ms", "sub-100ms", "sub-100 ms", "RFC 7807",
    "assigned UUID", "unique UUID", "Serve uploaded retinal images securely",
    "attribution heatmaps securely", "private storage",
    # Reviewer round on rev15: two documents still called the three benchmark
    # runs "the same 30 images" on "identical code" (run A used the superseded
    # split; run B held the a-priori thresholds); the API example carried a
    # fixed lesion sentence removed in round seven; three signing phrases.
    "over the same 30 images exist", "identical code", "Inferotemporal quadrant",
    "Clinician Signs Review", "a signed review", "Proportion of extreme-luminance pixels",
    # Editor round on rev21: "secure" sign-in over plain HTTP with published
    # defaults; a digest "match" with nothing to match against; a heuristic
    # gate that "confirms" anatomy; a test suite that certifies production
    # readiness; a second demo persona that never existed.
    "Secure PIN", "Secure Practitioner Sign In", "Secure Clinician Authentication",
    "SHA-256 match", "SHA-256 Verified", "SHA-256 verified",
    "Anatomical Relevance", "aperture confirmed", "spectral balance verified",
    "retinal anatomy", "retinal structure only", "landmarks",
    "Quality Gate Satisfied", "Production & Thesis", "Optom. Demo", "optometrist.demo",
    # Reviewer round on rev22: UI text that claimed anatomy for a coverage
    # heuristic, a dashboard tile that counted every record as "today", an
    # "acquisition" date that is the row's creation time, invented file
    # fallbacks, and a camera model preset on every record.
    "anatomical field-of-view", "Retinal field-of-view", "Total Today", "Acquisition Date",
    "Acquired:", "3400000", "Standard Fundus Camera", "Offline Edge AI", "Edge AI Enabled",
    "Quality Ambiguity",
    # Editor round on rev25: clinical inferences in technical-validation
    # messages, and a claim that single-cohort metrics predict runtime.
    "vascular characteristics", "Vascular edge sharpness", "ocular media opacities",
    "pupil dilation", "spectral balance confirmed", "media haze",
    "ophthalmic vascular pigmentation", "predictor of runtime behaviour", "valid predictor",
    # Reviewer round on rev26: the upload screen and the browser check still
    # asserted retinal identity or its absence; a colormap implied lesions.
    "Non-Retinal Modality Detected", "Non-Retinal Content Detected", "Non-retinal content detected",
    "Non-retinal diagram/document detected", "non-retinal subject", "Retinal FoV confirmed", "Modality Detected",
    "High Lesion Contrast", "Strips proprietary EXIF", "random UUID filename",
    # Editor round on rev28: descriptions that say the heuristics establish
    # retinal identity or verify quality.
    "Prevents non-retinal images", "strictly prevent", "Quality verified for inference",   # synthetic_retinal_fundus is a labelled TEST fixture generator; the screen rule bans it in the UI
    "Authorized for credentialed healthcare practitioners",
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
    # .css joined the list after "authoritative clinical artefact" was found in
    # a stylesheet comment, outside the scan.
    return _walk_repo((".py", ".ts", ".tsx", ".css"))


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
        for d in (os.path.join(CHAPTER4, "screenshots"),):
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
    The QA audit narrates withdrawn 544-image / 86.40% results. In a
    repository-relative archive, "archived" means the directory it sits in:
    docs/chapter4/archive/, never docs/chapter4/ itself.
    """
    name = "independent_thesis_qa_gate_audit.md"
    assert not os.path.exists(os.path.join(CHAPTER4, name)), (
        f"{name} is in docs/chapter4/, where it reads as current evidence")
    archived = os.path.join(CHAPTER4, "archive", name)
    assert os.path.exists(archived), (
        f"{name} should be retained in docs/chapter4/archive/ as a correction record")
    assert "SUPERSEDED" in read(archived)[:600], (
        f"{name} must open by saying it is superseded")


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
    "RETINAL_RED_SHARE_MIN",
    "RETINAL_RED_RATIO_MIN",
    "CONTRAST_THRESHOLD",
    "ILLUMINATION_EXTREME_RATIO_MAX",
    "ILLUMINATION_UNDEREXPOSED_BELOW",
    "ILLUMINATION_OVEREXPOSED_ABOVE",
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
# The tracker wrote intervals as "**91.2%** (81.5–90.5)" - no "95% CI" label -
# and the pattern above never saw them. Four of them excluded their own estimate.
CI_PATTERN_PAREN = re.compile(
    r"(\d+\.\d+)%\**\s*\(\s*(\d+\.\d+)\s*[–-]\s*(\d+\.\d+)\s*\)")


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
            for match in list(CI_PATTERN.finditer(line)) + list(CI_PATTERN_PAREN.finditer(line)):
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
    different answers. Exactly one file matching a test-log name may be in the
    archive manifest.
    """
    asm = _assembler()
    found = sorted(dst for _, dst in asm.manifest()
                   if asm.TEST_LOG_PATTERN.search(posixpath.basename(dst)))
    assert len(found) == 1, (
        "Exactly one test log must ship; the manifest has %d:\n  %s"
        % (len(found), "\n  ".join(found) or "(none)"))


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

def test_every_chapter4_file_ships_and_the_staged_copy_is_retired():
    """
    The archive is built from the repository tree by assemble_submission_package
    .py. Its own consistency check must pass here: every listed source exists,
    nothing under docs/chapter4 is silently left out, exactly one test log
    ships, the checkpoint digest is the evaluated one, and the retired staging
    directory - the second copy that once reverted a corrected VERIFY.py - has
    not come back.
    """
    asm = _assembler()
    found = asm.problems()
    assert not found, (
        "The archive cannot be built from this tree:\n  " + "\n  ".join(found))


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

def _resolve(*candidates):
    """First path that exists, so the rule runs in BOTH layouts.

    The archive is repository-relative, so the first candidate resolves both
    in the repository and inside an extracted archive. The later candidates
    describe the superseded documentation/ + source/ layout and are kept so a
    reviewer holding that older archive still gets a verdict, not a skip.

    An earlier version looked only in the repo layout and skipped silently in
    the package - for the reviewer, who has no other way to check the claim.
    """
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


_PKG_ROOT = os.path.dirname(REPO_ROOT)  # <package>, when REPO_ROOT is source/

SCHEMA_DOC = _resolve(
    os.path.join(CHAPTER4, "database_schema.md"),
    os.path.join(_PKG_ROOT, "documentation", "database_schema.md"),
    os.path.join(REPO_ROOT, "documentation", "database_schema.md"),
)
MODELS = _resolve(
    os.path.join(REPO_ROOT, "backend", "app", "models", "models.py"),
    os.path.join(_PKG_ROOT, "source", "backend", "app", "models", "models.py"),
)

# Tables are discovered from the document's own headings rather than listed
# here. A hardcoded list is how six tables went unchecked.
TABLE_HEADING = re.compile(r"^### \d+\. `([a-z_][a-z0-9_]*)`\s*$", re.MULTILINE)
COLUMN_ROW = re.compile(
    r"^\|\s*`([a-z_][a-z0-9_]*)`\s*\|[^|]*\|\s*(YES|NO)\s*\|", re.MULTILINE)


def _model_tables():
    """{table_name: (ClassName, {column: nullable})} for every mapped class."""
    tree = ast.parse(read(MODELS), filename=MODELS)
    out = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        table, columns = None, {}
        for stmt in node.body:
            if not isinstance(stmt, ast.Assign):
                continue
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id == "__tablename__":
                    table = getattr(stmt.value, "value", None)
            if not (isinstance(stmt.value, ast.Call)
                    and getattr(stmt.value.func, "id", None) == "Column"):
                continue
            # SQLAlchemy defaults nullable to True unless it is a primary key.
            nullable = True
            for kw in stmt.value.keywords:
                if kw.arg == "nullable":
                    nullable = getattr(kw.value, "value", True)
                elif kw.arg == "primary_key" and getattr(kw.value, "value", False):
                    nullable = False
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    columns[target.id] = bool(nullable)
        if table:
            out[table] = (node.name, columns)
    return out


def _documented_tables():
    """{table_name: {column: nullable}} as the schema document states them."""
    text = read(SCHEMA_DOC)
    headings = list(TABLE_HEADING.finditer(text))
    out = {}
    for i, m in enumerate(headings):
        stop = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        section = text[m.start():stop]
        out[m.group(1)] = {
            c: (flag == "YES") for c, flag in COLUMN_ROW.findall(section)}
    return out


def test_schema_document_matches_the_model_in_both_directions():
    """
    For every table the schema document describes, the set of documented
    columns and the set the model declares must be EQUAL, and each column's
    stated nullability must match.

    Equality, not containment. An omitted column is as misleading as an
    invented one: a reader who sees eleven rows and assumes that is the table
    has been misled either way.
    """
    assert SCHEMA_DOC, (
        "database_schema.md not found in either the repo layout "
        "(docs/chapter4/) or the extracted-package layout (documentation/). "
        "This rule must not skip: add the layout rather than let it pass.")
    assert MODELS, (
        "models.py not found in either layout. This rule must not skip.")

    model = _model_tables()
    documented = _documented_tables()
    assert documented, (
        "No table sections parsed out of %s. If the heading format changed, "
        "update TABLE_HEADING rather than letting this rule pass vacuously."
        % os.path.basename(SCHEMA_DOC))

    offenders = []

    # Equality of the TABLE SET too. A loop over documented tables cannot fail
    # for a table the document never mentions - which is how model_executions
    # and explanation_artifacts went unchecked while this rule was described
    # as bidirectional.
    for table in sorted(set(model) - set(documented)):
        offenders.append(
            "`%s`: declared on %s, but the document has no section for it"
            % (table, model[table][0]))

    for table in sorted(documented):
        if table not in model:
            offenders.append(
                "`%s`: documented, but no model class declares that "
                "__tablename__" % table)
            continue
        class_name, declared = model[table]
        assert declared, (
            "No Column assignments found on %s. If it moved, point this rule "
            "at the new name rather than dropping it." % class_name)
        stated = documented[table]

        for column in sorted(set(stated) - set(declared)):
            offenders.append(
                "`%s`.`%s`: documented, but %s does not declare it"
                % (table, column, class_name))
        for column in sorted(set(declared) - set(stated)):
            offenders.append(
                "`%s`.`%s`: declared on %s, but the document omits it"
                % (table, column, class_name))
        for column in sorted(set(stated) & set(declared)):
            if stated[column] != declared[column]:
                offenders.append(
                    "`%s`.`%s`: document says nullable=%s, %s declares "
                    "nullable=%s"
                    % (table, column, "YES" if stated[column] else "NO",
                       class_name, "YES" if declared[column] else "NO"))

    assert not offenders, (
        "database_schema.md disagrees with backend/app/models/models.py:\n  "
        + "\n  ".join(offenders)
        + "\nTranscribe each table from the model. A schema document that does "
          "not match the model is confidently wrong, which is worse than "
          "absent — and an omitted column misleads a reader exactly as much as "
          "an invented one.")


# ======================================================================
# RULE GROUP Y - prose must not cite model attributes that do not exist
#
# Rule X guards the schema document's table dictionaries. It did not guard
# prose, and prose is where the last phantom survived:
# verification_and_traceability_matrix.md cited
# `ProfessionalReview.certified_grade` as test evidence for a governance
# requirement. That column does not exist. The document asserting what had been
# verified was itself unverified, and rule X could not see it because the matrix
# carries no table dictionary.
#
# The check is scoped to class names the model actually declares, so it cannot
# be satisfied by rephrasing and needs no editing when a model class is added.
# ======================================================================

DOC_ROOTS = [
    CHAPTER4,
    os.path.join(REPO_ROOT, "docs"),
    os.path.join(_PKG_ROOT, "documentation"),
]

ATTR_REFERENCE = re.compile(r"`([A-Z][A-Za-z0-9]*)\.([a-z_][a-z0-9_]*)`")


def _model_classes():
    """{ClassName: {attribute names declared on it}} for mapped classes."""
    tree = ast.parse(read(MODELS), filename=MODELS)
    out = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        names, is_model = set(), False
        for stmt in node.body:
            if not isinstance(stmt, ast.Assign):
                continue
            for target in stmt.targets:
                if not isinstance(target, ast.Name):
                    continue
                if target.id == "__tablename__":
                    is_model = True
                else:
                    names.add(target.id)
        if is_model:
            out[node.name] = names
    return out


def _prose_files():
    seen, files = set(), []
    for root in DOC_ROOTS:
        if not root or not os.path.isdir(root):
            continue
        for base, _dirs, names in os.walk(root):
            parts = base.replace("\\", "/").split("/")
            # archive/ records what WAS wrong, and must be allowed to quote it.
            if "archive" in parts:
                continue
            for name in sorted(names):
                if not name.endswith(".md"):
                    continue
                full = os.path.join(base, name)
                key = os.path.realpath(full)
                if key not in seen:
                    seen.add(key)
                    files.append(full)
    return files


def test_prose_cites_only_model_attributes_that_exist():
    """
    Any `ClassName.attribute` in the documentation, where ClassName is a model
    class, must name an attribute that class declares.
    """
    assert MODELS, "models.py not found in either layout; this rule must not skip."

    classes = _model_classes()
    assert classes, (
        "No mapped classes parsed from models.py. If the models moved, point "
        "this rule at them rather than letting it pass vacuously.")

    files = _prose_files()
    assert files, "No documentation found; this rule must not pass vacuously."

    offenders = []
    for path in files:
        rel = os.path.relpath(path, REPO_ROOT)
        for lineno, line in enumerate(read(path).split("\n"), 1):
            # A correction record must be able to name the wrong column.
            if "superseded" in line.lower() or "never existed" in line.lower():
                continue
            for class_name, attr in ATTR_REFERENCE.findall(line):
                if class_name not in classes:
                    continue
                if attr not in classes[class_name]:
                    offenders.append(
                        "%s:%d cites `%s.%s`, which %s does not declare"
                        % (rel, lineno, class_name, attr, class_name))

    assert not offenders, (
        "Documentation cites model attributes that do not exist:\n  "
        + "\n  ".join(offenders)
        + "\nA document that states what was verified must itself be verified. "
          "Correct the attribute name against backend/app/models/models.py, or "
          "if the line is recording a past error, say so explicitly in it.")


# ======================================================================
# RULE GROUP X (continued) - the ER diagram must draw the foreign keys that exist
#
# database_schema.md drew ASSESSMENTS ||--o{ MODEL_EXECUTIONS (many runs per
# assessment) where model_executions.assessment_id is UNIQUE, and drew
# ASSESSMENTS ||--o{ EXPLANATION_ARTIFACTS where explanation_artifacts has no
# assessment column at all - it keys to ai_results. A diagram is a claim about
# the schema and is checked like one.
# ======================================================================

ER_EDGE = re.compile(r"^\s*([A-Z_]+)\s+([|o}{]{2})--([|o}{]{2})\s+([A-Z_]+)\s*:", re.M)
ONE_MARKERS = {"||", "o|"}
MANY_MARKERS = {"o{", "|{", "}o", "}|"}


def _model_foreign_keys():
    """{(child_table, parent_table): (fk_column_is_unique, fk_column_is_nullable)}"""
    tree = ast.parse(read(MODELS), filename=MODELS)
    tables = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        table, fks = None, []
        for stmt in node.body:
            if not isinstance(stmt, ast.Assign):
                continue
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id == "__tablename__":
                    table = getattr(stmt.value, "value", None)
            if not (isinstance(stmt.value, ast.Call)
                    and getattr(stmt.value.func, "id", None) == "Column"):
                continue
            unique = any(kw.arg == "unique" and getattr(kw.value, "value", False)
                         for kw in stmt.value.keywords)
            nullable = True
            for kw in stmt.value.keywords:
                if kw.arg == "nullable":
                    nullable = bool(getattr(kw.value, "value", True))
            for arg in stmt.value.args:
                if (isinstance(arg, ast.Call)
                        and getattr(arg.func, "id", None) == "ForeignKey"
                        and arg.args and isinstance(arg.args[0], ast.Constant)):
                    parent = str(arg.args[0].value).split(".")[0]
                    fks.append((parent, (unique, nullable)))
        if table:
            tables[table] = fks
    out = {}
    for child, fks in tables.items():
        for parent, flags in fks:
            out[(child, parent)] = flags
    return out


def test_er_diagram_edges_are_the_declared_foreign_keys():
    assert SCHEMA_DOC and MODELS, "schema document or models unresolved"
    fks = _model_foreign_keys()
    offenders = []
    edges = ER_EDGE.findall(read(SCHEMA_DOC))
    assert edges, "no erDiagram edges parsed from database_schema.md"
    for left, lm, rm, right in edges:
        a, b = left.lower(), right.lower()
        if (b, a) in fks:
            child, child_marker, parent_marker, (unique, nullable) = b, rm, lm, fks[(b, a)]
        elif (a, b) in fks:
            child, child_marker, parent_marker, (unique, nullable) = a, lm, rm, fks[(a, b)]
        else:
            offenders.append("%s -- %s: no ForeignKey between these tables" % (left, right))
            continue
        # The parent side: a nullable foreign key means the parent is optional.
        # The reviewer found four edges drawing a mandatory parent over a
        # nullable key.
        if nullable and parent_marker != "|o":
            offenders.append("%s -- %s: %s's foreign key is nullable, so the parent is optional "
                             "(|o), but the diagram draws %r" % (left, right, child, parent_marker))
        if not nullable and parent_marker != "||":
            offenders.append("%s -- %s: %s's foreign key is NOT NULL, so the parent is mandatory "
                             "(||), but the diagram draws %r" % (left, right, child, parent_marker))
        if unique and child_marker not in ONE_MARKERS:
            offenders.append("%s -- %s: %s's foreign key is UNIQUE, so the edge is "
                             "one-to-one, but the diagram draws %r" % (left, right, child, child_marker))
        if not unique and child_marker not in MANY_MARKERS:
            offenders.append("%s -- %s: %s's foreign key is not unique, so the edge is "
                             "one-to-many, but the diagram draws %r" % (left, right, child, child_marker))
    assert not offenders, (
        "The ER diagram disagrees with the declared foreign keys:\n  "
        + "\n  ".join(offenders))


# ======================================================================
# RULE GROUP Z - every link in every shipped document resolves inside the archive
#
# The sixth review counted 94 unresolved links. The documents were right; the
# archive had moved their targets. Links are therefore checked where they are
# READ: resolved from each document's archive location, against the archive
# manifest, so a document that links correctly in the repository but not in
# the archive fails here.
# ======================================================================

MARKDOWN_LINK = re.compile(r"\]\(([^)\s]+)\)")


def _assembler():
    path = os.path.join(SCRIPTS, "assemble_submission_package.py")
    spec = importlib.util.spec_from_file_location("assemble_submission_package", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_link_in_every_shipped_document_resolves_inside_the_archive():
    asm = _assembler()
    entries = asm.manifest()
    destinations = {dst for _, dst in entries}
    directories = set()
    for dst in destinations:
        parts = dst.split("/")
        for i in range(1, len(parts)):
            directories.add("/".join(parts[:i]))

    offenders = []
    for src, dst in entries:
        if not src.endswith(".md"):
            continue
        base = posixpath.dirname(dst)
        for lineno, line in enumerate(read(os.path.join(REPO_ROOT, src)).split("\n"), 1):
            for href in MARKDOWN_LINK.findall(line):
                href = href.split("#", 1)[0].strip("<>")
                if not href or re.match(r"^[a-z][a-z0-9+.-]*:", href):
                    continue
                if href.startswith("/"):
                    # Only this scanner would treat "/" as the archive root;
                    # GitHub and ordinary renderers treat it as the domain
                    # root. Thirty-nine such links shipped in rev10.
                    offenders.append("%s:%d -> %s  (root-absolute; must be relative)"
                                     % (src, lineno, href))
                    continue
                target = posixpath.normpath(posixpath.join(base, href))
                target = target.rstrip("/")
                if target.startswith("..") or (target not in destinations
                                                and target not in directories):
                    offenders.append("%s:%d -> %s  (resolves to %s, not in the archive)"
                                     % (src, lineno, href, target))

    assert not offenders, (
        "Links that do not resolve inside the shipped archive:\n  "
        + "\n  ".join(offenders[:40])
        + ("\n  ... and %d more" % (len(offenders) - 40) if len(offenders) > 40 else "")
        + "\nFix the link or ship the target. A reviewer reads the archive, "
          "not the repository.")


# ======================================================================
# RULE GROUP AA - the recorded environment is the one the documents describe,
# and it satisfies requirements.txt
#
# training_environment.md listed pillow 11.1.0 and numpy 2.1.3 against
# requirements bounds of pillow<11 and numpy<2 - an environment that
# requirements.txt could not have produced. system_test_report.md said
# "Pytest 9.1.1" while the log it cites recorded pytest-8.4.2. The deployment
# versions are now read from test_environment_freeze.txt, the `pip freeze` of
# the environment that produced the committed log, and both are checked.
# ======================================================================

FREEZE = os.path.join(CHAPTER4, "test_environment_freeze.txt")
REQUIREMENTS = os.path.join(REPO_ROOT, "backend", "requirements.txt")
TRAINING_ENV_DOC = os.path.join(CHAPTER4, "training_environment.md")
SYSTEM_TEST_REPORT = os.path.join(CHAPTER4, "system_test_report.md")


def _norm_name(name):
    return name.strip().lower().replace("_", "-")


def _freeze():
    out = {}
    for line in read(FREEZE).splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        name, version = line.split("==", 1)
        out[_norm_name(name)] = version.strip()
    return out


def _version_tuple(v):
    v = v.split("+", 1)[0]
    parts = []
    for piece in v.split("."):
        m = re.match(r"\d+", piece)
        parts.append(int(m.group()) if m else 0)
    return tuple(parts)


def _satisfies(version, spec):
    for clause in spec.split(","):
        clause = clause.strip()
        if not clause:
            continue
        m = re.match(r"(>=|<=|==|!=|~=|>|<)\s*([\w.+*-]+)", clause)
        assert m, "unparsed requirement clause %r" % clause
        op, want = m.groups()
        a, b = _version_tuple(version), _version_tuple(want)
        n = max(len(a), len(b))
        a, b = a + (0,) * (n - len(a)), b + (0,) * (n - len(b))
        ok = {">=": a >= b, "<=": a <= b, "==": a == b, "!=": a != b,
              ">": a > b, "<": a < b, "~=": a >= b}[op]
        if not ok:
            return False
    return True


def test_recorded_test_environment_satisfies_requirements_txt():
    """
    Every requirement in backend/requirements.txt must be present in the
    recorded environment at a version its specifier admits. Otherwise the
    committed log was produced by an environment requirements.txt cannot
    reproduce, and "reproducible" is not a checkable claim.
    """
    assert os.path.exists(FREEZE), (
        "docs/chapter4/test_environment_freeze.txt is missing: record `pip freeze` "
        "from the environment that produced test_execution.log")
    freeze = _freeze()
    offenders = []
    for line in read(REQUIREMENTS).splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        m = re.match(r"([A-Za-z0-9_.-]+)(\[[^\]]*\])?\s*(.*)$", line)
        name, spec = _norm_name(m.group(1)), m.group(3).strip()
        if name not in freeze:
            offenders.append("%s: required, but absent from the recorded environment" % name)
        elif spec and not _satisfies(freeze[name], spec):
            offenders.append("%s==%s does not satisfy %r" % (name, freeze[name], spec))
    assert not offenders, (
        "requirements.txt and the recorded test environment disagree:\n  "
        + "\n  ".join(offenders))


def test_documented_versions_are_the_recorded_ones():
    """
    system_test_report.md's framework line must quote the pytest and Python
    versions the committed log header records. training_environment.md's
    deployment table must quote, for every Python package it names, the
    version in the recorded freeze.
    """
    header = read(TEST_LOG).split("\n")[1] if os.path.exists(TEST_LOG) else ""
    log_pytest = re.search(r"pytest-(\d+\.\d+\.\d+)", header)
    log_python = re.search(r"Python (\d+\.\d+\.\d+)", header)
    assert log_pytest and log_python, "the test log header does not record pytest/Python versions"

    offenders = []
    report = read(SYSTEM_TEST_REPORT)
    doc_pytest = re.search(r"Pytest (\d+\.\d+\.\d+)", report)
    if not doc_pytest:
        offenders.append("system_test_report.md does not state the Pytest version")
    elif doc_pytest.group(1) != log_pytest.group(1):
        offenders.append("system_test_report.md says Pytest %s; the log records %s"
                         % (doc_pytest.group(1), log_pytest.group(1)))
    doc_python = re.search(r"Python (\d+\.\d+\.\d+)", report)
    if doc_python and doc_python.group(1) != log_python.group(1):
        offenders.append("system_test_report.md says Python %s; the log records %s"
                         % (doc_python.group(1), log_python.group(1)))

    freeze = _freeze()
    checked = 0
    for name, version in re.findall(
            r"^\|[^|]*\|\s*`([A-Za-z][A-Za-z0-9_-]*)`[^|]*\|\s*([^|]+?)\s*\|",
            read(TRAINING_ENV_DOC), re.M):
        key = _norm_name(name)
        if key not in freeze:
            continue
        checked += 1
        if version.strip("`* ") != freeze[key]:
            offenders.append("training_environment.md says %s %s; the recorded environment has %s"
                             % (name, version.strip(), freeze[key]))
    assert checked >= 4, (
        "training_environment.md's deployment table names fewer than four packages "
        "that appear in the recorded freeze; it should name at least torch, "
        "torchvision, pillow and numpy")
    assert not offenders, (
        "Documented versions are not the recorded ones:\n  " + "\n  ".join(offenders))


# ======================================================================
# RULE GROUP AB - a "mean ... ms" claim is the canonical CPU run
#
# architecture.md said "mean ~298 ms on standard CPU". No run recorded 298 ms.
# Outside the two documents that discuss the earlier runs explicitly
# (resource_benchmark.md, requirements_test_matrix.md), any latency stated as
# a mean must be a number in cpu_end_to_end_benchmark.json.
# ======================================================================

BENCHMARK_JSON = os.path.join(CHAPTER4, "cpu_end_to_end_benchmark.json")
MEAN_MS_CLAIMS = [
    re.compile(r"mean[^.\n]*?(\d+(?:\.\d+)?)\s*ms", re.I),
    # Only a number described AS the mean: "180.41 ms mean", "180.41 ms (mean)".
    # "350 ms budget passes on the mean" names a budget, not a measurement.
    re.compile(r"(\d+(?:\.\d+)?)\s*ms\**\s*\(?\s*mean\b", re.I),
]
LATENCY_DISCUSSION_DOCS = set()   # every run cited anywhere now ships as JSON
BENCHMARK_HISTORY = os.path.join(CHAPTER4, "benchmark_history")
T4_SUMMARY = os.path.join(CHAPTER4, "benchmark_summary.json")


def _benchmark_numbers():
    values = set()

    def walk(obj):
        if isinstance(obj, dict):
            # The sum of a run's stage means is a derived figure a document may
            # legitimately state ("the stages sum to 178.98 ms").
            stages = obj.get("stages")
            if isinstance(stages, list) and stages and isinstance(stages[0], dict):
                total = sum(float(st.get("mean_ms", 0)) for st in stages)
                values.add(round(total, 2)); values.add(round(total, 1))
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)
        elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
            values.add(round(float(obj), 2))
            values.add(round(float(obj), 1))
    walk(json.loads(read(BENCHMARK_JSON)))
    # Every earlier run of the harness that any document cites ships under
    # benchmark_history/, recovered from the commits they were recorded at.
    # A figure that is in none of these files was produced by no retained run.
    if os.path.isdir(BENCHMARK_HISTORY):
        for name in sorted(os.listdir(BENCHMARK_HISTORY)):
            if name.endswith(".json"):
                walk(json.loads(read(os.path.join(BENCHMARK_HISTORY, name))))
    if os.path.exists(T4_SUMMARY):
        walk(json.loads(read(T4_SUMMARY)))
    return values


def test_mean_latency_claims_are_the_canonical_run():
    allowed = _benchmark_numbers()
    offenders = []
    for base, dirs, files in os.walk(CHAPTER4):
        dirs[:] = [d for d in dirs if d != "archive"]
        for name in sorted(files):
            if not name.endswith(".md") or name in LATENCY_DISCUSSION_DOCS:
                continue
            rel = os.path.relpath(os.path.join(base, name), REPO_ROOT)
            for lineno, line in enumerate(read(os.path.join(base, name)).split("\n"), 1):
                for pattern in MEAN_MS_CLAIMS:
                    for value in pattern.findall(line):
                        if round(float(value), 2) not in allowed:
                            offenders.append("%s:%d states a mean of %s ms" % (rel, lineno, value))
    assert not offenders, (
        "Latency means that no recorded run produced:\n  " + "\n  ".join(sorted(set(offenders)))
        + "\nQuote cpu_end_to_end_benchmark.json, or move the discussion of other "
          "runs into resource_benchmark.md where they are labelled.")


# ======================================================================
# RULE GROUP AC - the README's headline table is the metrics files
#
# The package README carried "NPV 97.1%" for sight-threatening DR. The
# committed clinical_metrics.json says 95.39%. 97.1% was the superseded run's
# figure, surviving in the one document every reader opens first.
# ======================================================================

# Lifted to the archive root as README.md by the assembler, so the rule must
# read it there when run from an extracted archive - it failed to on the first
# extraction run, which is exactly the kind of layout blindness it polices.

PREDICTIONS_CSV = os.path.join(CHAPTER4, "held_out_predictions.csv")


def _operating_points_from_predictions():
    """
    {name: {sensitivity_pct, specificity_pct, ppv_pct, npv_pct}} computed from
    the raw predictions. clinical_metrics.json stores these at two decimals;
    rounding THAT to one decimal is a second rounding, and 97.647 -> 97.65 ->
    97.7 disagreed with VERIFY.py, which computes from the predictions.
    """
    import csv as _csv
    with open(PREDICTIONS_CSV, newline="", encoding="utf-8") as fh:
        pairs = [(int(r["true_grade"]), int(r["predicted_grade"])) for r in _csv.DictReader(fh)]
    out = {}
    for name, th in (("Referable DR", 2), ("Sight-threatening DR", 3), ("Any DR", 1)):
        tp = sum(1 for t, p in pairs if t >= th and p >= th)
        fn = sum(1 for t, p in pairs if t >= th and p < th)
        fp = sum(1 for t, p in pairs if t < th and p >= th)
        tn = sum(1 for t, p in pairs if t < th and p < th)
        out[name] = {"sensitivity_pct": 100.0 * tp / (tp + fn), "specificity_pct": 100.0 * tn / (tn + fp),
                     "ppv_pct": 100.0 * tp / (tp + fp), "npv_pct": 100.0 * tn / (tn + fn)}
    return out

SUBMISSION_README = _resolve(
    os.path.join(CHAPTER4, "SUBMISSION_README.md"),
    os.path.join(REPO_ROOT, "README.md"),
)
CLINICAL_METRICS = os.path.join(CHAPTER4, "clinical_metrics.json")


def _readme_row(text, label):
    m = re.search(r"^\|\s*\*\*%s\*\*[^|]*\|(.*)\|\s*$" % re.escape(label), text, re.M)
    return m.group(1) if m else None


def test_submission_readme_headline_table_matches_the_metrics_files():
    metrics = json.loads(read(CLINICAL_METRICS))
    bench = json.loads(read(BENCHMARK_JSON))
    text = read(SUBMISSION_README)
    ops = _operating_points_from_predictions()
    cm = metrics["confusion_matrix"]
    correct = sum(cm[i][i] for i in range(len(cm)))

    expected = {
        "Held-out cohort": ["N = %d" % metrics["n_test"]],
        "Quadratic Weighted Kappa": ["%.4f" % metrics["quadratic_weighted_kappa"]],
        "Exact accuracy": ["%.2f%%" % metrics["exact_accuracy_pct"],
                           "(%d / %d)" % (correct, metrics["n_test"])],
        "Within-one-grade agreement": ["%.2f%%" % metrics["within_one_grade_pct"]],
        "Referable DR": ["%.1f%%" % ops["Referable DR"]["sensitivity_pct"],
                         "%.1f%%" % ops["Referable DR"]["specificity_pct"]],
        "Sight-threatening DR": ["%.1f%%" % ops["Sight-threatening DR"]["sensitivity_pct"],
                                 "%.1f%%" % ops["Sight-threatening DR"]["npv_pct"]],
        "Any DR": ["%.1f%%" % ops["Any DR"]["sensitivity_pct"],
                   "%.1f%%" % ops["Any DR"]["specificity_pct"]],
        "End-to-end CPU latency": ["%.2f ms" % bench["total_mean_ms"],
                                   "%.2f ms" % bench["total_median_ms"],
                                   "%.2f ms" % bench["total_p95_ms"]],
    }
    offenders = []
    for label, tokens in expected.items():
        row = _readme_row(text, label)
        if row is None:
            offenders.append("row **%s** not found" % label)
            continue
        for token in tokens:
            if token not in row:
                offenders.append("row **%s** should contain %r; it reads: %s"
                                 % (label, token, row.strip()))
    assert not offenders, (
        "SUBMISSION_README.md's headline table disagrees with the metrics files:\n  "
        + "\n  ".join(offenders))


# ======================================================================
# The overclaim phrases are banned in prose too, not only in code
# ======================================================================

# "superseded" on its own no longer exempts a line: a current claim that
# merely mentions the superseded split was slipping through on that word.
CORRECTION_LINE_MARKERS = ("superseded run", "superseded contaminated", "superseded result",
                           "superseded set", "superseded sharpness", "never existed",
                           "earlier revision", "earlier version", "withdrawn", "replaced", "removed")


def test_no_clinical_authority_overclaims_in_documents():
    """
    The code rule never read the prose, and "human clinician certifications"
    sat in architecture.md through five reviews. A line that records a past
    error may name the phrase it corrects; any other occurrence fails.
    """
    offenders = []
    # The screenshot register was once excluded wholesale as a "correction
    # record", and "tamper-evident" sat in an ACTIVE figure caption through
    # two reviews. Then a LINE that mentioned the superseded run sheltered a
    # "cryptographic audit" claim at the other end of the same line, and the
    # independent reviewer showed that "replaced" or "removed" anywhere on
    # the line - or on the previous line - did the same. The unit is now the
    # SENTENCE (a table cell counts as one): a correction marker exempts only
    # the sentence it is in, plus phrases inside double quotes in a paragraph
    # that records a correction.
    for rel, text in iter_markdown(include_correction_records=True):
        if "/archive/" in rel.replace("\\", "/"):
            continue
        for lineno, phrase, sentence in _offending_phrases(text, OVERCLAIM_PHRASES):
            offenders.append("%s:%d: %r in: %s" % (rel, lineno, phrase, sentence[:90]))
    assert not offenders, (
        "Authority or certification language in documents:\n  " + "\n  ".join(offenders))


# A sentence records a correction when it says so with a correction PHRASE.
# "replaced", "removed", "withdrawn", "previously", "no longer" and "an
# earlier" alone do not: the independent reviewer built current claims
# around each of them ("...and no longer needs manual checks") and the rule
# let them through.
RECORD_MARKERS = ("earlier revision", "earlier version", "earlier set of", "earlier draft",
                  "superseded run", "superseded contaminated", "superseded result", "superseded set",
                  "superseded sharpness", "never existed", "removed, not regenerated",
                  "deleted rather than", "was withdrawn", "were withdrawn", "were removed, not",
                  "which contradicted", "was wrong", "were wrong", "had all four wrong")


def _units(text):
    """
    Yield (lineno, unit_lower, kind) for every sentence of prose and every
    cell of every table row, outside code fences. Soft-wrapped lines of one
    paragraph are joined before sentences are split, so a sentence that
    wraps is still one unit; a table row never joins its neighbours; a bare
    '>' or a blank line ends a paragraph; so does a list item (each item is
    its own paragraph - items without full stops were merging). Sentences
    split on . ! ? and ; (a clause after a semicolon is its own claim), with
    a closing ** tolerated before the space. kind is "heading", "cell" or
    "sentence"; a heading is never exempt.
    """
    lines = text.split("\n")
    in_code = False
    para = []   # [(lineno, text)]

    def flush():
        if not para:
            return
        first = para[0][0]
        joined = " ".join(t for _, t in para)
        # map each sentence start back to a line number
        starts, pos = [], 0
        for ln, t in para:
            starts.append((pos, ln))
            pos += len(t) + 1
        offset = 0
        # A correction that puts its claim after a colon ("An earlier revision
        # said: every PDF was tamper-evident.") is split from its marker and
        # FLAGGED. That is the safe direction: reword it, the gate fails closed.
        for sent in re.split(r"(?<=[.!?;:])(?:\*\*)?\s+|\s+[—–]\s+", joined):
            if not sent.strip():
                continue
            ln = max((l for p, l in starts if p <= offset), default=first)
            yield ln, sent.lower(), "sentence"
            offset += len(sent) + 1

    for i, raw in enumerate(lines, 1):
        line = re.sub(r"^(\s*>\s?)+", "", raw).strip()
        if line.startswith("```"):
            in_code = not in_code
            yield from flush(); para = []
            continue
        if in_code:
            continue
        if not line:
            yield from flush(); para = []
            continue
        if line.startswith("|"):
            yield from flush(); para = []
            for cell in _table_cells(line):
                for clause in re.split(r"(?<=[.!?;:])(?:\*\*)?\s+|\s+[—–]\s+", cell):
                    if clause.strip():
                        yield i, clause.lower(), "cell"
            continue
        if re.match(r"^#{1,6}\s", line):
            yield from flush(); para = []
            yield i, line.lower(), "heading"
            continue
        if re.match(r"^(?:[-*+]|\d+[.)])\s", line):
            yield from flush(); para = []
        para.append((i, line))
    yield from flush()


def _normalise(unit):
    """
    The reviewer matched "tamper‑evident" (non-breaking hyphen), "tamper-
evident"
    (hyphen at a soft wrap), "immutable **audit**" and "immutable  audit" past
    the rule. Emphasis markers and backticks are stripped, Unicode hyphens and
    spaces mapped to ASCII, a hyphen followed by whitespace rejoined, and
    whitespace collapsed, BEFORE matching.
    """
    # Reviewer's second pass: an en dash used as the hyphen inside a phrase, a
    # soft hyphen, a zero-width space and the &nbsp; entity each let a phrase
    # through. Invisible characters are deleted; dashes and entities mapped.
    unit = unit.replace("&nbsp;", " ").replace("­", "").replace("​", "").replace("‌", "").replace("﻿", "")
    for dash in ("‐", "‑", "‒", "–", "—"):
        unit = unit.replace(dash, "-")
    unit = unit.replace(" ", " ")
    unit = re.sub(r"[*_`]", "", unit)
    unit = re.sub(r"-\s+", "-", unit)
    return re.sub(r"\s+", " ", unit)


def _offending_phrases(text, phrases):
    """[(lineno, phrase, unit)] for every banned phrase outside a correction sentence."""
    out = []
    for lineno, unit, kind in _units(text):
        unit = _normalise(unit)
        # a heading is never a correction record; a sentence or cell is one
        # only when IT carries a correction phrase - and then only the words
        # it quotes are sheltered, plus the phrase named in the same sentence.
        if kind != "heading" and any(m in unit for m in RECORD_MARKERS):
            continue
        for phrase in phrases:
            if phrase.lower() in unit:
                out.append((lineno, phrase, unit))
    return out


# ======================================================================
# RULE GROUP AD - what the independent reviewer found in round seven
# ======================================================================

PROGRESS_TRACKER = os.path.join(REPO_ROOT, "PROGRESS_TRACKER.md")
AI_SERVICE = os.path.join(REPO_ROOT, "backend", "app", "services", "ai_service.py")
ASSESSMENT_SERVICE = os.path.join(REPO_ROOT, "backend", "app", "services", "assessment_service.py")


def test_a_file_declared_a_copy_of_another_is_byte_identical_to_it():
    """
    evidence_provenance.md said screenshots/09_confusion_matrix_empirical.png
    was "copy of docs/chapter4/confusion_matrix.png". It was the superseded
    N=549 run's matrix: same dimensions, different bytes, titled N=549. A
    provenance claim of "copy of" is now checked by hashing both files.
    """
    text = read(PROVENANCE_DOC)
    pairs = re.findall(r"^\|\s*`([^`]+)`\s*\|\s*copy of\s*`([^`]+)`", text, re.M)
    assert pairs, "no 'copy of' rows found; if the row was removed, remove this rule's reason too"
    offenders = []
    for target, source in pairs:
        t, src = os.path.join(REPO_ROOT, target), os.path.join(REPO_ROOT, source)
        if not (os.path.exists(t) and os.path.exists(src)):
            offenders.append("%s or %s is missing" % (target, source))
            continue
        if sha256_of(t) != sha256_of(src):
            offenders.append("%s is declared a copy of %s but the bytes differ" % (target, source))
    assert not offenders, "Provenance 'copy of' claims that are false:\n  " + "\n  ".join(offenders)


def test_progress_tracker_operating_points_and_precision_match_the_metrics():
    """
    PROGRESS_TRACKER.md ships at the archive root. Its operating-point table
    carried CIs that excluded their estimates, an NPV from the superseded run
    and per-class precisions matching nothing in clinical_metrics.json.
    """
    metrics = json.loads(read(CLINICAL_METRICS))
    text = read(PROGRESS_TRACKER)
    raw = _operating_points_from_predictions()
    ops = {o["name"]: dict(o, **raw[o["name"]]) for o in metrics["operating_points"]}
    offenders = []
    for name, label in (("Referable DR", "Referable DR"), ("Sight-threatening DR", "Sight-threatening DR"),
                        ("Any DR", "Any DR")):
        o = ops[name]
        m = re.search(r"^\|\s*\**%s\**[^|]*\|(.*)$" % re.escape(label), text, re.M)
        if not m:
            offenders.append("row for %s not found" % name)
            continue
        row = m.group(1)
        for token in ("%.1f%%" % o["sensitivity_pct"], "(%.1f–%.1f)" % tuple(o["sensitivity_ci95"]),
                      "%.1f%%" % o["specificity_pct"], "(%.1f–%.1f)" % tuple(o["specificity_ci95"]),
                      "%.1f%%" % o["ppv_pct"], "%.1f%%" % o["npv_pct"]):
            if token not in row:
                offenders.append("%s row lacks %r" % (name, token))
    for c in metrics["per_class"]:
        m = re.search(r"^\|\s*%d\s*\|[^|]*\|\s*(\d+)\s*\|\s*\**([\d.]+)%%\**\s*\|\s*([\d.]+)%%\s*\|" % c["grade"], text, re.M)
        if not m:
            offenders.append("per-class row for grade %d not found" % c["grade"])
            continue
        support, sens, prec = int(m.group(1)), float(m.group(2)), float(m.group(3))
        if support != c["support"] or abs(sens - round(c["sensitivity_pct"], 1)) > 0.05 \
                or abs(prec - round(c["precision_pct"], 1)) > 0.05:
            offenders.append("grade %d row says %d / %.1f%% / %.1f%%; the metrics say %d / %.1f%% / %.1f%%"
                             % (c["grade"], support, sens, prec, c["support"], c["sensitivity_pct"], c["precision_pct"]))
    assert not offenders, "PROGRESS_TRACKER.md disagrees with clinical_metrics.json:\n  " + "\n  ".join(offenders)


MATRIX = os.path.join(CHAPTER4, "objective_traceability_matrix.md")


def test_objective_letters_are_labelled_as_the_traceability_matrix_defines_them():
    """
    Five documents called the architecture "Objective g"; the matrix defines g
    as model integration and a as architecture. A reader following the letters
    was sent to the wrong objective.
    """
    titles = {k: t.strip() for k, t in re.findall(
        r"^\| \*\*([a-i])\*\* \| \*\*([^|]+?)\*\* \|", read(MATRIX), re.M)}
    assert len(titles) == 9, "the matrix should define objectives a-i"
    offenders = []
    for rel, text in iter_markdown():
        if "/archive/" in rel.replace("\\", "/"):
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if "Research Objective" not in line:
                continue
            for letter, stated in re.findall(r"Objective ([a-i]) \(([^)]*)\)", line):
                if stated.strip().lower() != titles[letter].lower():
                    offenders.append("%s:%d calls objective %s %r; the matrix defines it as %r"
                                     % (rel, lineno, letter, stated.strip(), titles[letter]))
    for rel in ("PROGRESS_TRACKER.md",):
        path = os.path.join(REPO_ROOT, rel)
        if os.path.exists(path):
            for letter, stated in re.findall(r"^\| \*\*([a-i])\*\* \| ([^|]+?) \|", read(path), re.M):
                if stated.strip().lower() != titles[letter].lower():
                    offenders.append("%s: objective %s is %r; the matrix defines it as %r"
                                     % (rel, letter, stated.strip(), titles[letter]))
    assert not offenders, "Objective labels drift from the traceability matrix:\n  " + "\n  ".join(offenders)


def test_peak_activation_region_is_derived_from_the_cam_not_a_constant():
    """
    ICDR_CLASS_METADATA carried a fixed "top_activation" sentence per grade -
    "Isolated parafoveal microaneurysm cluster" and so on - stored as
    top_activation_region and shown as "Top Saliency Zone". Nothing in it came
    from the image. The real engine must derive it from the CAM it computed.
    """
    src = read(AI_SERVICE)
    assert '"top_activation"' not in src, "a per-grade constant attribution string is back"
    tree = ast.parse(src, filename=AI_SERVICE)
    helper = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                   and n.name == "_peak_activation_region"), None)
    assert helper is not None, "_peak_activation_region is missing"
    assert any(isinstance(n, ast.Attribute) and n.attr == "argmax" for n in ast.walk(helper)), (
        "_peak_activation_region must locate the CAM's maximum (argmax); a region not "
        "read from the map is a constant with extra steps")
    # Every InferenceOutput built in the file: the region keyword must be a
    # NAME bound from the helper, except the simulated engine's, which must say so.
    seen_name = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg != "top_activation_region":
                continue
            if isinstance(kw.value, ast.Name):
                seen_name = True
            elif isinstance(kw.value, ast.Constant):
                assert "imulated" in str(kw.value.value), (
                    "a constant top_activation_region that does not declare itself simulated: %r"
                    % kw.value.value)
            else:
                raise AssertionError("top_activation_region built from %s" % type(kw.value).__name__)
    assert seen_name, "the real engine does not pass a derived region"


def test_review_signatory_is_the_authenticated_reviewer_only():
    """
    submit_professional_review took clinicianName / licenseNumber from the
    request body in preference to the signed-in reviewer, so a client could
    record a review under any name. Only the authenticated account may sign.
    """
    tree = ast.parse(read(ASSESSMENT_SERVICE), filename=ASSESSMENT_SERVICE)
    offenders = []
    seen_from_reviewer = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if not any(t in ("clinician_name", "license_num", "facility") for t in targets):
            continue
        for sub in ast.walk(node.value):
            if isinstance(sub, ast.Attribute) and sub.attr in ("clinicianName", "licenseNumber"):
                offenders.append("%s is assigned from the request body (%s) at line %d"
                                 % (targets[0], sub.attr, node.lineno))
            if isinstance(sub, ast.Attribute) and sub.attr == "full_name" and "clinician_name" in targets:
                seen_from_reviewer = True
    assert seen_from_reviewer, "clinician_name must come from the reviewer account"
    assert not offenders, "\n  ".join(offenders)


def test_audit_events_do_not_cascade_delete_with_their_assessment():
    """
    database_schema.md calls audit_events append-only and says removing an
    assessment blanks the reference but never deletes the event. The ORM
    relationship cascaded "all, delete-orphan". The mapping must not delete.
    """
    tree = ast.parse(read(MODELS), filename=MODELS)
    found = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "audit_events" for t in node.targets):
            continue
        if not (isinstance(node.value, ast.Call) and getattr(node.value.func, "id", None) == "relationship"):
            continue
        found = True
        for kw in node.value.keywords:
            if kw.arg == "cascade" and "delete" in str(getattr(kw.value, "value", "")):
                raise AssertionError("audit_events cascade=%r deletes events with the assessment"
                                     % kw.value.value)
    assert found, "audit_events relationship not found on the model"


def test_review_hash_covers_every_review_field():
    """
    The UI and schema say the hash is "over the review fields". It omitted the
    justification and the inconclusive reason. Found by walking the f-string
    that builds sig_payload: every review field must be referenced.
    """
    tree = ast.parse(read(ASSESSMENT_SERVICE), filename=ASSESSMENT_SERVICE)
    attrs = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "sig_payload" for t in node.targets):
            for sub in ast.walk(node.value):
                if isinstance(sub, ast.Attribute):
                    attrs.add(sub.attr)
                if isinstance(sub, ast.Name):
                    attrs.add(sub.id)
    assert attrs, "sig_payload assignment not found"
    required = {"agreement", "reviewerAssessedGrade", "justificationNotes", "inconclusiveReason",
                "clinician_name", "license_num", "facility", "sha256_hash"}
    missing = sorted(required - attrs)
    assert not missing, "sig_payload does not cover: %s" % ", ".join(missing)


def test_rejected_records_are_not_shown_as_pending_review():
    """
    The ledger rendered "Pending Human Review" for any record without a review,
    including rejected ones, which can never be reviewed. The branch for the
    rejected status must precede the pending badge.
    """
    path = os.path.join(REPO_ROOT, "frontend", "src", "screens", "RecordHistoryScreen.tsx")
    src = read(path)
    pending = src.find("Pending Human Review")
    assert pending != -1, "ledger no longer renders a pending badge; update this rule"
    guard = src.rfind("rec.status === 'rejected'", 0, pending)
    assert guard != -1 and pending - guard < 800, (
        "the pending badge is not guarded by a rejected-status branch")

EVALUATION_REPORT = os.path.join(CHAPTER4, "model_evaluation_report.md")


def _per_class_from_predictions():
    import csv as _csv
    with open(PREDICTIONS_CSV, newline="", encoding="utf-8") as fh:
        pairs = [(int(r["true_grade"]), int(r["predicted_grade"])) for r in _csv.DictReader(fh)]
    out = {}
    for g in range(5):
        tp = sum(1 for t, p in pairs if t == g and p == g)
        fn = sum(1 for t, p in pairs if t == g and p != g)
        fp = sum(1 for t, p in pairs if t != g and p == g)
        tn = sum(1 for t, p in pairs if t != g and p != g)
        out[g] = {"support": tp + fn, "sens": 100.0 * tp / (tp + fn), "spec": 100.0 * tn / (tn + fp),
                  "prec": 100.0 * tp / (tp + fp), "f1": 2.0 * tp / (2 * tp + fp + fn)}
    return out


PER_CLASS_ROW = re.compile(
    r"^\|\s*\*\*(\d)\*\*\s*\|[^|]*\|\s*(\d+)\s*\|\s*\**([\d.]+)%\**\s*\|[^|]*\|"
    r"\s*([\d.]+)%\s*\|\s*([\d.]+)%\s*\|\s*([\d.]+)\s*\|", re.M)


def test_evaluation_report_per_class_table_recomputes_from_the_predictions():
    """
    The per-class table carried Grade 0 specificity as 97.7%: the JSON stores
    97.65 and the table re-rounded it; 249/255 is 97.647%, i.e. 97.6%. Every
    one-decimal value in the table is now recomputed from the predictions,
    which is the only rounding there should be.
    """
    raw = _per_class_from_predictions()
    rows = {int(m.group(1)): m for m in PER_CLASS_ROW.finditer(read(EVALUATION_REPORT))}
    offenders = []
    for g in range(5):
        m = rows.get(g)
        if m is None:
            offenders.append("row for grade %d not found" % g)
            continue
        support, sens, spec, prec, f1 = (int(m.group(2)), float(m.group(3)), float(m.group(4)),
                                         float(m.group(5)), float(m.group(6)))
        want = raw[g]
        checks = (("support", support, want["support"], 0),
                  ("sensitivity", sens, round(want["sens"], 1), 1),
                  ("specificity", spec, round(want["spec"], 1), 1),
                  ("precision", prec, round(want["prec"], 1), 1),
                  ("F1", f1, round(want["f1"], 3), 3))
        for name, got, exp, _places in checks:
            if abs(got - exp) > 1e-9:
                offenders.append("grade %d %s: table says %s, the predictions give %s" % (g, name, got, exp))
    assert not offenders, (
        "model_evaluation_report.md per-class table disagrees with held_out_predictions.csv:\n  "
        + "\n  ".join(offenders))


# ======================================================================
# RULE GROUP AE - the editor's round on rev10: the delivered package must
# not be able to contradict the evidence
# ======================================================================

ENV_EXAMPLE = os.path.join(REPO_ROOT, ".env.example")
RETINAL_VALIDATOR = os.path.join(REPO_ROOT, "frontend", "src", "utils", "retinalValidator.ts")
ANALYZE_SCRIPT = os.path.join(SCRIPTS, "analyze_clinical_metrics.py")
COMPOSE_FILE = os.path.join(REPO_ROOT, "docker-compose.yml")
FRONTEND_PACKAGE = os.path.join(REPO_ROOT, "frontend", "package.json")


def _config_defaults():
    """{NAME: literal} for every annotated default on the Settings class."""
    tree = ast.parse(read(CONFIG_PY), filename=CONFIG_PY)
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            try:
                out[node.target.id] = ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                pass
    return out


def _env_example():
    out = {}
    for line in read(ENV_EXAMPLE).splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def test_env_example_cannot_restore_the_rejected_thresholds():
    """
    .env.example carried 512 / 60.0 / 18.0 - the a-priori thresholds that
    rejected 99.5% of genuine images - after config.py had been calibrated to
    480 / 4.3 / 8.8. Settings read the environment, so the shipped template
    could silently restore the defect. Every threshold and the checkpoint
    digest in the template must equal the config default.
    """
    cfg, env = _config_defaults(), _env_example()
    offenders = []
    for key in ("MIN_IMAGE_DIMENSION", "LAPLACIAN_BLUR_THRESHOLD", "CONTRAST_THRESHOLD",
                "ILLUMINATION_EXTREME_RATIO_MAX", "RETINAL_MIN_COVERAGE", "RETINAL_RED_SHARE_MIN",
                "RETINAL_RED_RATIO_MIN", "VALIDATION_ANALYSIS_MAX_DIM", "MODEL_CHECKPOINT_SHA256"):
        if key not in env:
            offenders.append("%s missing from .env.example" % key); continue
        if key not in cfg:
            offenders.append("%s has no literal default in config.py" % key); continue
        want, got = cfg[key], env[key]
        same = (str(want) == got) if isinstance(want, str) else (abs(float(got) - float(want)) < 1e-9)
        if not same:
            offenders.append("%s: .env.example says %s, config.py says %s" % (key, got, want))
    if "MODEL_CHECKPOINT_PATH" in env and "MODEL_CHECKPOINT_PATH" in cfg:
        if os.path.basename(env["MODEL_CHECKPOINT_PATH"]) != os.path.basename(str(cfg["MODEL_CHECKPOINT_PATH"])):
            offenders.append("MODEL_CHECKPOINT_PATH names a different file than config.py")
    assert not offenders, ".env.example disagrees with config.py:\n  " + "\n  ".join(offenders)


def test_browser_precheck_uses_the_backends_minimum_dimension():
    """
    The browser pre-check accepted 256x256 while the backend requires 480; a
    user could pass the browser and be refused by the server, and the message
    implied diagnostic adequacy. The constant must equal MIN_IMAGE_DIMENSION
    and the file must say the server is authoritative.
    """
    cfg = _config_defaults()
    src = read(RETINAL_VALIDATOR)
    # the dimension comes from the generated constants (rule group AQ checks
    # that file against config.py); a bare number here is a regression
    assert "width >= MIN_IMAGE_DIMENSION && height >= MIN_IMAGE_DIMENSION" in src, (
        "the browser dimension check does not use the generated MIN_IMAGE_DIMENSION")
    assert not re.search(r"width\s*>=\s*\d", src), "the browser dimension check uses a literal"
    assert cfg["MIN_IMAGE_DIMENSION"] > 0
    assert "re-checks every image" in src, "the browser check must say the server re-checks every image"

    # The aspect range must be the server's (it was 0.60-1.70 against 0.65-1.65).
    gate2 = read(os.path.join(REPO_ROOT, "backend", "app", "services", "validation", "gate2_relevance.py"))
    server = re.search(r"aspect_ratio\s*<\s*([\d.]+)\s*or\s*aspect_ratio\s*>\s*([\d.]+)", gate2)
    browser = re.search(r"aspectRatio\s*>=\s*([\d.]+)\s*&&\s*aspectRatio\s*<=\s*([\d.]+)", src)
    assert server and browser, "aspect-ratio checks not found"
    assert (float(browser.group(1)), float(browser.group(2))) == (float(server.group(1)), float(server.group(2))), (
        "browser aspect range %s-%s, server %s-%s" % (browser.group(1), browser.group(2), server.group(1), server.group(2)))

    # The browser's focus number is a gradient energy at 256 px. It must not be
    # labelled with the server's metric name or compared with its threshold.
    assert "Laplacian variance:" not in src, "the browser labels its own focus estimate as the server's Laplacian variance"
    assert not re.search(r">=\s*%s\b" % re.escape(str(cfg["LAPLACIAN_BLUR_THRESHOLD"])), src), (
        "the browser compares its own focus estimate with the server's threshold")
    assert "advisory" in src.lower(), "the browser focus estimate must be stated as advisory"


def test_every_checkpoint_filename_is_the_configured_one():
    """
    The integration guide told the reader seven times to place
    efficientnet_b0_dr_v1.pth; the configured, evaluated and packaged file is
    efficientnet_b0_dr.pth. Every .pth filename in shipped text must be it.
    """
    names = sorted(set(re.findall(r"[\w.-]+\.pth\b", read(CONFIG_PY))))
    assert len(names) == 1, "config.py names %s .pth files; expected exactly one" % names
    want = names[0]
    asm = _assembler()
    offenders = []
    for src, _dst in asm.manifest():
        if not src.endswith((".md", ".py", ".ts", ".tsx", ".yml", ".yaml", ".example")):
            continue
        if "/archive/" in src:
            continue
        # The test suite deliberately names checkpoints that do not exist
        # (does_not_exist.pth, impostor.pth) to prove the engine refuses them.
        if src.startswith("backend/tests/"):
            continue
        for lineno, line in enumerate(read(os.path.join(REPO_ROOT, src)).split("\n"), 1):
            low = line.lower()
            if any(marker in low for marker in CORRECTION_LINE_MARKERS):
                continue
            for name in re.findall(r"[\w.-]+\.pth\b", line):
                if name != want:
                    offenders.append("%s:%d names %s" % (src, lineno, name))
    assert not offenders, (
        "Checkpoint filenames that are not %s:\n  " % want + "\n  ".join(offenders))


def _dockerfile_copies(path):
    """Context-relative single files a Dockerfile COPYs (not '.' and not directories)."""
    files = []
    for line in read(path).splitlines():
        m = re.match(r"\s*COPY\s+(.+)", line)
        if not m:
            continue
        parts = m.group(1).split()
        for src in parts[:-1]:
            if src in (".", "./") or src.startswith("--"):
                continue
            files.append(src)
    return files


def test_archive_carries_everything_the_build_instructions_consume():
    """
    The runbook said `docker compose up --build`; the archive had neither
    Dockerfile, nor package.json, index.html, vite.config.ts or the tsconfigs.
    Every build input the compose file, the Dockerfiles and the frontend
    build script need must be in the manifest.
    """
    asm = _assembler()
    shipped = {src for src, _ in asm.manifest()}
    required = {"docker-compose.yml", "backend/requirements.txt", "backend/main.py",
                "frontend/package.json", "frontend/package-lock.json"}
    compose = read(COMPOSE_FILE)
    for ctx, df in re.findall(r"context:\s*\./(\w+)\s*\n\s*dockerfile:\s*(\S+)", compose):
        dockerfile = "%s/%s" % (ctx, df)
        required.add(dockerfile)
        if os.path.exists(os.path.join(REPO_ROOT, dockerfile)):
            for f in _dockerfile_copies(os.path.join(REPO_ROOT, dockerfile)):
                required.add("%s/%s" % (ctx, f))
    pkg = json.loads(read(FRONTEND_PACKAGE))
    build = pkg.get("scripts", {}).get("build", "")
    if "tsc" in build:
        required.add("frontend/tsconfig.json")
        tsconfig = re.sub(r"/\*.*?\*/", "", read(os.path.join(REPO_ROOT, "frontend", "tsconfig.json")), flags=re.S)
        tsconfig = re.sub(r"^\s*//[^\n]*$", "", tsconfig, flags=re.M)
        for ref in json.loads(tsconfig).get("references", []):
            required.add("frontend/" + ref["path"].lstrip("./"))
    if "vite" in build:
        required.update({"frontend/index.html", "frontend/vite.config.ts"})
    for cfg in ("frontend/postcss.config.js", "frontend/tailwind.config.js"):
        if os.path.exists(os.path.join(REPO_ROOT, cfg)):
            required.add(cfg)
    missing = sorted(r for r in required if r not in shipped)
    assert not missing, (
        "Build inputs the archive does not carry:\n  " + "\n  ".join(missing)
        + "\nAdd them to SINGLE_FILES in assemble_submission_package.py, or remove the "
          "instruction that needs them.")


def test_analysis_script_output_rounds_exactly_once():
    """
    analyze_clinical_metrics.py printed Any-DR sensitivity and Grade-0
    specificity as 97.7%: it stored 97.65 and rounded that again. This runs
    the script and checks every printed one-decimal figure against the
    counts, and checks the JSON holds the unrounded values.
    """
    import subprocess
    proc = subprocess.run([sys.executable, ANALYZE_SCRIPT], cwd=REPO_ROOT,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = proc.stdout
    raw_ops = _operating_points_from_predictions()
    raw_cls = _per_class_from_predictions()
    offenders = []
    for name, th in (("Referable DR", 2), ("Sight-threatening DR", 3), ("Any DR", 1)):
        after_header = out.split("--- %s" % name, 1)[1].split("\n", 1)[1]
        block = after_header.split("\n---", 1)[0]
        for label, key in (("Sensitivity", "sensitivity_pct"), ("Specificity", "specificity_pct"),
                           ("PPV", "ppv_pct"), ("NPV", "npv_pct")):
            m = re.search(r"%s ([\d.]+)%%" % label, block)
            want = "%.1f" % raw_ops[name][key]
            if not m or m.group(1) != want:
                offenders.append("%s %s printed %s; the counts give %s" % (name, label, m and m.group(1), want))
    for g in range(5):
        m = re.search(r"^%d\s+(\d+)\s+([\d.]+)\s+\S+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$" % g, out, re.M)
        if not m:
            offenders.append("per-class row %d not printed" % g); continue
        for got, key, fmt in ((m.group(2), "sens", "%.1f"), (m.group(3), "spec", "%.1f"),
                              (m.group(4), "prec", "%.1f"), (m.group(5), "f1", "%.3f")):
            if got != fmt % raw_cls[g][key]:
                offenders.append("grade %d %s printed %s; the counts give %s" % (g, key, got, fmt % raw_cls[g][key]))
    metrics = json.loads(read(CLINICAL_METRICS))
    for o in metrics["operating_points"]:
        for key in ("sensitivity_pct", "specificity_pct", "ppv_pct", "npv_pct"):
            if abs(o[key] - raw_ops[o["name"]][key]) > 1e-9:
                offenders.append("JSON %s %s is %r, not the unrounded %r" % (o["name"], key, o[key], raw_ops[o["name"]][key]))
    assert not offenders, "\n  ".join(["analyze_clinical_metrics.py rounds twice:"] + offenders)


def test_tls_is_stated_as_a_production_requirement():
    """
    The architecture diagram said HTTPS / TLS 1.3; the packaged stack serves
    plain HTTP. Any mention of TLS in the architecture must say it is a
    production deployment requirement.
    """
    text = read(os.path.join(CHAPTER4, "architecture.md"))
    for lineno, line in enumerate(text.splitlines(), 1):
        if "TLS" in line and "production" not in line.lower():
            raise AssertionError("architecture.md:%d mentions TLS without stating it is a production requirement" % lineno)


# ======================================================================
# RULE GROUP AF - the archive vouches for nothing outside itself
# ======================================================================

DEPLOYMENT_HOSTS = ("vercel.app", "spacehubtech.cloud", "drai-cdss")


def test_no_shipped_file_points_at_a_hosted_deployment():
    """
    The tracker linked a Vercel deployment and api.ts hard-coded a cloud
    backend for native builds. Nothing in the archive shows what those hosts
    serve, and the package's own screenshot manifest records that an earlier
    capture of that deployment showed withdrawn content. A reader who follows
    such a pointer may see what this package withdrew.
    """
    asm = _assembler()
    offenders = []
    for src, _dst in asm.manifest():
        if not src.endswith((".md", ".py", ".ts", ".tsx", ".yml", ".yaml", ".json", ".example", ".txt", ".log")):
            continue
        if "/archive/" in src or src.endswith("test_editor_integrity_gate.py"):
            continue  # the archive is superseded by definition; this file names the hosts it bans
        for lineno, line in enumerate(read(os.path.join(REPO_ROOT, src)).split("\n"), 1):
            low = line.lower()
            if any(marker in low for marker in CORRECTION_LINE_MARKERS):
                continue
            for host in DEPLOYMENT_HOSTS:
                if host in low:
                    offenders.append("%s:%d mentions %s" % (src, lineno, host))
    assert not offenders, "Pointers to hosted deployments the archive cannot vouch for:\n  " + "\n  ".join(offenders)


CONTAINER_FREEZE = os.path.join(CHAPTER4, "container_environment_freeze.txt")


def _container_freeze():
    out = {}
    for line in read(CONTAINER_FREEZE).splitlines():
        line = line.strip()
        if "==" in line and not line.startswith("#"):
            name, version = line.split("==", 1)
            out[_norm_name(name)] = version.strip()
    return out


def test_container_versions_are_recorded_and_documented():
    """
    training_environment.md described torch 2.14.1+cpu as the version used
    "in the backend container". That was the test venv's freeze; the container
    builds on python:3.11-slim and resolves its own. The container's pip
    freeze is now captured during the compose check and the document's
    container table must quote it.
    """
    assert os.path.exists(CONTAINER_FREEZE), (
        "docs/chapter4/container_environment_freeze.txt is missing: capture `pip freeze` "
        "inside the backend container during the compose check")
    freeze = _container_freeze()
    text = read(TRAINING_ENV_DOC)
    assert "CONTAINER_ENVIRONMENT" not in text, "the container table placeholder was never filled"
    section = text.split("### Backend container", 1)
    assert len(section) == 2, "training_environment.md has no 'Backend container' section"
    checked, offenders = 0, []
    for name, version in re.findall(r"^\|\s*`([A-Za-z][A-Za-z0-9_-]*)`\s*\|\s*([^|]+?)\s*\|", section[1], re.M):
        key = _norm_name(name)
        if key not in freeze:
            continue
        checked += 1
        if version.strip("`* ") != freeze[key]:
            offenders.append("%s: document says %s, container has %s" % (name, version, freeze[key]))
    assert checked >= 4, "the container table names fewer than four packages found in the container freeze"
    assert not offenders, "\n  ".join(["Container table disagrees with the container freeze:"] + offenders)


def test_health_endpoint_reports_the_verified_digest(test_client=None):
    """
    The build log's claim that /health shows the digest-verified checkpoint
    was an inference: the response carried only "checkpoint loaded". The
    response model now carries checkpoint_sha256, set from the digest the
    engine verified. Static check: the field exists and is wired.
    """
    health = read(os.path.join(REPO_ROOT, "backend", "app", "routers", "health.py"))
    assert "checkpoint_sha256" in health, "HealthResponse does not carry checkpoint_sha256"
    assert 'inference.get("checkpoint_sha256")' in health, "health does not pass the verified digest through"
    service = read(AI_SERVICE)
    assert "self.verified_sha256 = actual" in service, "the engine does not record the digest it verified"
    assert 'logger.warning(\n                "MODEL_CHECKPOINT_SHA256 is not set; serving' not in service, (
        "the blank-digest escape hatch is back")


# ======================================================================
# RULE GROUP AG - the editor's review of rev13
# ======================================================================

MODEL_EVAL_REPORT = os.path.join(CHAPTER4, "model_evaluation_report.md")
TEST_LOG = os.path.join(CHAPTER4, "test_execution.log")


def _table_cells(line):
    line = re.sub(r"`[^`]*`", "code", line)          # pipes inside code spans
    line = line.replace("\\|", "pipe")                # escaped pipes
    inner = line.strip()
    if inner.startswith("|"):
        inner = inner[1:]
    if inner.endswith("|"):
        inner = inner[:-1]
    return [c.strip() for c in inner.split("|")]


def test_every_markdown_table_row_has_the_header_width():
    """
    The operating-point table in model_evaluation_report.md lost half its
    Any-DR row to a note pasted into the middle of it: three cells under a
    six-column header, with the specificity and the missed count stranded in
    the blockquote below. A table row with fewer cells than its header renders
    as a broken table, and no rule looked at table structure.
    """
    # Second version. The first only looked at rows directly under a
    # header, so eight NFR rows stranded after a blockquote (no header above
    # them: literal pipe text in the rendered page) passed, and a table inside
    # a blockquote was never read. Now: blockquote markers are stripped, a
    # pipe row with no header above it is an orphan, and tables inside
    # blockquotes are checked like any other.
    asm = _assembler()
    offenders = []
    for src, _dst in asm.manifest():
        if not src.endswith(".md") or "/archive/" in src:
            continue
        raw = read(os.path.join(REPO_ROOT, src)).split("\n")
        lines = [re.sub(r"^(\s*>\s?)+", "", l) for l in raw]   # unwrap blockquotes
        i, in_code = 0, False
        while i < len(lines):
            line = lines[i]
            if line.strip().startswith("```"):
                in_code = not in_code
                i += 1
                continue
            if in_code or not line.lstrip().startswith("|") or len(_table_cells(line)) < 2:
                i += 1
                continue
            if not (i + 1 < len(lines) and re.match(r"^\s*\|?\s*:?-+:?\s*\|", lines[i + 1])):
                offenders.append("%s:%d is a table row with no header above it (renders as literal text)" % (src, i + 1))
                i += 1
                continue
            width = len(_table_cells(line))
            j = i + 2
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                cells = len(_table_cells(lines[j]))
                if cells != width:
                    offenders.append("%s:%d has %d cells under a %d-column header" % (src, j + 1, cells, width))
                j += 1
            i = j
    assert not offenders, "Markdown table structure:\n  " + "\n  ".join(offenders)


def test_binary_collapse_rows_match_the_predictions():
    """
    Each operating-point row of the evaluation report carries six cells -
    sensitivity, its CI, specificity, its CI, and the missed count - and every
    one of them is recomputed from held_out_predictions.csv (counts) and
    clinical_metrics.json (Wilson intervals).
    """
    import csv as _csv
    with open(PREDICTIONS_CSV, newline="", encoding="utf-8") as fh:
        pairs = [(int(r["true_grade"]), int(r["predicted_grade"])) for r in _csv.DictReader(fh)]
    metrics = {o["name"]: o for o in json.loads(read(CLINICAL_METRICS))["operating_points"]}
    raw = _operating_points_from_predictions()
    text = read(MODEL_EVAL_REPORT)
    offenders, seen = [], 0
    for th, name in ((1, "Any DR"), (2, "Referable DR"), (3, "Sight-threatening DR")):
        m = re.search(r"^\|\s*Grade \$\\ge %d\$\s*\|(.*)$" % th, text, re.M)
        if not m:
            offenders.append("row for grade >= %d not found" % th)
            continue
        cells = _table_cells("|" + m.group(1))
        if len(cells) != 5:
            offenders.append("grade >= %d row has %d value cells, expected 5" % (th, len(cells)))
            continue
        seen += 1
        fn = sum(1 for t, p in pairs if t >= th and p < th)
        expected = ["%.1f%%" % raw[name]["sensitivity_pct"],
                    "%.1f – %.1f" % tuple(metrics[name]["sensitivity_ci95"]),
                    "%.1f%%" % raw[name]["specificity_pct"],
                    "%.1f – %.1f" % tuple(metrics[name]["specificity_ci95"]),
                    str(fn)]
        for got, want in zip(cells, expected):
            if got.replace("**", "") != want:
                offenders.append("grade >= %d: table says %r, the predictions say %r" % (th, got, want))
    assert seen == 3 and not offenders, "Operating-point table disagrees with the predictions:\n  " + "\n  ".join(offenders)


def test_no_unqualified_append_only_or_cryptographic_control_claims():
    """
    What the implementation provides: an unkeyed SHA-256 over the review
    fields truncated to 24 hex characters, an API rule refusing a second
    review, ON DELETE SET NULL on the audit log's foreign keys, no trigger,
    no key, no signature. "Append-only" is therefore an application-level
    property and must say so on the same line; "cryptographic" as the name
    of a CONTROL (signature, immutability, audit, storage) is banned above.
    The limitation is stated where the audit log is described.
    """
    asm = _assembler()
    offenders = []
    for src, _dst in asm.manifest():
        if not src.endswith((".md", ".py", ".ts", ".tsx")) or "/archive/" in src:
            continue
        if src.endswith("test_editor_integrity_gate.py"):
            continue
        text = read(os.path.join(REPO_ROOT, src))
        if src.endswith(".md"):
            units = _units(text)
        else:   # code: a line is the unit; a correction comment exempts only itself
            units = ((n, l.lower(), l.lower()) for n, l in enumerate(text.split("\n"), 1))
        for lineno, unit, kind in units:
            unit = _normalise(unit)
            if kind != "heading" and any(m in unit for m in RECORD_MARKERS):
                continue
            if "append-only" in unit and "application-level" not in unit:
                offenders.append("%s:%d: 'append-only' without 'application-level' in the same sentence" % (src, lineno))
            if "cryptograph" in unit:
                offenders.append("%s:%d: %r" % (src, lineno, unit.strip()[:90]))
    assert not offenders, "Controls described more strongly than they operate:\n  " + "\n  ".join(offenders)
    for rel in ("docs/chapter4/database_schema.md", "docs/chapter4/api_contract.md"):
        text = read(os.path.join(REPO_ROOT, rel))
        assert "database-level immutability is not enforced" in text, (
            "%s describes the audit log without stating that the database enforces nothing" % rel)


def test_quoted_suite_durations_match_the_committed_log():
    """
    The runbook said "the committed log records 61 s" while the log's last
    line said 75.59s. Any document that quotes the log's duration must quote
    the number the log actually carries.
    """
    last = read(TEST_LOG).rstrip().splitlines()[-1]
    m = re.search(r"in ([\d.]+)s", last)
    assert m, "the committed log's last line carries no duration"
    actual = float(m.group(1))
    offenders = []
    for rel, text in iter_markdown(include_correction_records=True):
        if "/archive/" in rel.replace("\\", "/"):
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            for quoted in re.findall(r"committed log records ([\d.]+)\s*s\b", line):
                if abs(float(quoted) - actual) > 0.005:
                    offenders.append("%s:%d quotes %s s; the log records %.2f s" % (rel, lineno, quoted, actual))
    assert not offenders, "\n  ".join(["Quoted suite durations disagree with the log:"] + offenders)
    # The runbook's expected range must bracket the duration the shipped log
    # actually records; a 125 s run once shipped under "roughly 60-90 s".
    runbook = read(os.path.join(CHAPTER4, "reproducibility_runbook.md"))
    m = re.search(r"wall-clock between (\d+) and (\d+) s", runbook)
    assert m, "the runbook no longer states the expected wall-clock range"
    lo, hi = int(m.group(1)), int(m.group(2))
    assert lo <= actual <= hi, "the committed log records %.2f s; the runbook says between %d and %d s" % (actual, lo, hi)


# ======================================================================
# RULE GROUP AH - the reviewer's audit of rev14
# ======================================================================

REQUIREMENTS_MATRIX = os.path.join(CHAPTER4, "requirements_test_matrix.md")
IMPLEMENTATION_STATUS = os.path.join(CHAPTER4, "implementation_status.md")
API_CONTRACT = os.path.join(CHAPTER4, "api_contract.md")
KNOWN_LIMITATIONS = os.path.join(CHAPTER4, "known_limitations.md")
ASSESSMENTS_ROUTER = os.path.join(REPO_ROOT, "backend", "app", "routers", "assessments.py")


def _matrix_rows(text):
    """[(req_id, cells)] for every FR-/NFR- row of the requirements matrix."""
    rows = []
    for line in text.splitlines():
        m = re.match(r"^\|\s*\*\*((?:N)?FR-\d+)\*\*\s*\|", line)
        if m:
            # _table_cells blanks code spans, so the raw line travels too: the
            # citations live in backticks.
            rows.append((m.group(1), _table_cells(line), line))
    return rows


def test_every_pass_in_the_requirements_matrix_cites_something_that_exists():
    """
    NFR-02 said PASS on "Peak 328.8 MB" from benchmark_resources.py, a script
    that measures no memory; nothing shipped could have produced the number.
    A PASS must point at a file in the archive and, where it names a test,
    a test that exists in the suite. A row that measures nothing says so.
    """
    shipped = {dst for _src, dst in _assembler().manifest()}
    test_ids = set()
    for path in python_sources(os.path.join(REPO_ROOT, "backend", "tests")):
        test_ids.update(re.findall(r"^\s*(?:async\s+)?def (test_\w+)", read(path), re.M))
    offenders, checked, citations = [], 0, 0
    for req, cells, raw in _matrix_rows(read(REQUIREMENTS_MATRIX)):
        status, evidence = cells[-1], cells[-2]
        if "PASS" not in status:
            if re.search(r"\d+(\.\d+)?\s*MB", evidence) and "earlier revision" not in evidence:
                offenders.append("%s quotes a measurement while not PASS: %s" % (req, evidence[:60]))
            continue
        checked += 1
        for token in re.findall(r"`([^`]+)`", raw):
            if "/" in token:
                citations += 1
                if token not in shipped:
                    offenders.append("%s cites %s, which is not in the archive" % (req, token))
            elif "." in token and token.split(".")[-1].startswith("test_"):
                citations += 1
                if token.split(".")[-1] not in test_ids:
                    offenders.append("%s cites %s, which is not in the suite" % (req, token))
    assert checked >= 15, "fewer PASS rows than expected were checked (%d)" % checked
    assert citations >= 20, "fewer citations than expected were resolved (%d) - is the rule reading the backticks?" % citations
    nfr02 = {r: c for r, c, _ in _matrix_rows(read(REQUIREMENTS_MATRIX))}["NFR-02"]
    assert "NOT MEASURED" in nfr02[-1], "NFR-02 claims a status although nothing shipped measures memory"
    assert "328.8" not in nfr02[-1], "NFR-02 still carries the unmeasured figure as its status"
    assert not offenders, "Requirements matrix cites what does not exist:\n  " + "\n  ".join(offenders)


def test_requirements_matrix_thresholds_and_counts_match_code_and_itself():
    """
    FR-05 gave an "illumination check (0.20-0.85)"; the code rejects on an
    extreme-pixel ratio above 0.35 and nothing uses 0.20-0.85. The status
    page claimed FR-01..FR-17 and NFR-01..NFR-12; the matrix has 10 and 8.
    """
    cfg = _config_defaults()
    text = read(REQUIREMENTS_MATRIX)
    rows = {r: c for r, c, _ in _matrix_rows(text)}
    fr05 = rows["FR-05"][1]
    assert "0.20-0.85" not in fr05 and "0.20–0.85" not in fr05, "FR-05 still quotes a range the code does not use"
    nums = [float(x) for x in re.findall(r"(?<![\d.])(\d+\.\d+)(?![\d.])", fr05)]
    assert cfg["LAPLACIAN_BLUR_THRESHOLD"] in nums, "FR-05 does not quote LAPLACIAN_BLUR_THRESHOLD=%s" % cfg["LAPLACIAN_BLUR_THRESHOLD"]
    assert cfg["ILLUMINATION_EXTREME_RATIO_MAX"] in nums, "FR-05 does not quote ILLUMINATION_EXTREME_RATIO_MAX=%s" % cfg["ILLUMINATION_EXTREME_RATIO_MAX"]
    for n in nums:
        assert n in (cfg["LAPLACIAN_BLUR_THRESHOLD"], cfg["ILLUMINATION_EXTREME_RATIO_MAX"]), "FR-05 quotes %s, which is no Gate 3 threshold" % n
    fr = sorted(int(r[3:]) for r in rows if r.startswith("FR-"))
    nfr = sorted(int(r[4:]) for r in rows if r.startswith("NFR-"))
    assert fr == list(range(1, len(fr) + 1)) and nfr == list(range(1, len(nfr) + 1)), "requirement ids are not contiguous"
    status = read(IMPLEMENTATION_STATUS)
    m = re.search(r"FR-01–FR-(\d+)\) and non-functional requirements \(NFR-01–NFR-(\d+)\)", status)
    assert m, "implementation_status.md no longer states the requirement ranges"
    assert (int(m.group(1)), int(m.group(2))) == (len(fr), len(nfr)), (
        "implementation_status.md claims FR-01..FR-%s / NFR-01..NFR-%s; the matrix has %d / %d" % (m.group(1), m.group(2), len(fr), len(nfr)))


def test_unauthenticated_storage_routes_say_so_everywhere():
    """
    /storage/images and /storage/attributions take no credential (an <img>
    tag cannot send the bearer token). Their docstrings said "securely" and
    the comment said "private storage". Either the routes authenticate, or
    the code, the API contract and the limitations say plainly that they do
    not. Checked on the AST, not on prose about it.
    """
    import ast
    tree = ast.parse(read(ASSESSMENTS_ROUTER))
    open_routes = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        paths = [a.value for d in node.decorator_list if isinstance(d, ast.Call)
                 for a in d.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        if not any(str(pth).startswith("/storage/") for pth in paths):
            continue
        authenticated = any(
            isinstance(dflt, ast.Call) and getattr(dflt.func, "id", "") == "Depends"
            and any(getattr(a, "id", "") == "get_current_user" for a in dflt.args)
            for dflt in node.args.defaults + node.args.kw_defaults if dflt is not None)
        if not authenticated:
            open_routes.append(node.name)
            doc = ast.get_docstring(node) or ""
            assert "UNAUTHENTICATED" in doc, "%s takes no credential and its docstring does not say so" % node.name
            assert "securely" not in doc.lower(), "%s says 'securely' while taking no credential" % node.name
    assert len(open_routes) in (0, 2), "unexpected storage route set: %s" % open_routes
    if open_routes:
        assert "Authentication: none" in read(API_CONTRACT), "api_contract.md does not state that the storage routes are unauthenticated"
        assert "served without authentication" in read(KNOWN_LIMITATIONS), "known_limitations.md does not disclose the open storage routes"
        assert "without authentication" in read(os.path.join(CHAPTER4, "architecture.md")), "architecture.md still presents the storage as protected"


def test_api_contract_examples_are_labelled_and_respect_the_thresholds():
    """
    The contract's examples showed a PASSING Gate 3 with illuminationIndex
    0.54 (the index is 1 - extreme ratio; 0.54 means 46% extreme pixels,
    rejected above 35%) and a Laplacian variance above anything in the
    corpus, unlabelled. Examples are now labelled illustrative and must be
    consistent with the calibrated thresholds.
    """
    cfg = _config_defaults()
    text = read(API_CONTRACT)
    assert "illustrative example" in text, "api_contract.md does not label its JSON bodies as illustrative"
    blocks = re.findall(r"```json\n(.*?)```", text, re.S)
    assert blocks, "no JSON examples found"
    checked = 0
    for block in blocks:
        for m in re.finditer(r'"metric": "Laplacian: ([\d.]+) \(>= [\d.]+\), Illumination index: ([\d.]+)"', block):
            checked += 1
            lap, idx = float(m.group(1)), float(m.group(2))
            assert lap >= cfg["LAPLACIAN_BLUR_THRESHOLD"], "example Laplacian %s would be rejected" % lap
            assert (1.0 - idx) <= cfg["ILLUMINATION_EXTREME_RATIO_MAX"] + 1e-9, (
                "example illuminationIndex %s implies an extreme-pixel ratio of %.2f, above the %.2f limit"
                % (idx, 1.0 - idx, cfg["ILLUMINATION_EXTREME_RATIO_MAX"]))
    assert checked >= 1, "the Gate 3 metrics example was not found"


def test_no_paragraph_has_an_odd_number_of_bold_markers():
    """
    PROGRESS_TRACKER.md's objective-g row ended with a stray '**' that
    rendered as literal asterisks. Bold markers pair up within a paragraph.
    """
    asm = _assembler()
    offenders = []
    for src, _dst in asm.manifest():
        if not src.endswith(".md") or "/archive/" in src:
            continue
        text = read(os.path.join(REPO_ROOT, src))
        text = re.sub(r"```.*?```", "", text, flags=re.S)
        text = re.sub(r"`[^`\n]*`", "", text)
        for n, para in enumerate(re.split(r"\n\s*\n", text)):
            if para.count("**") % 2:
                offenders.append("%s: paragraph %d has an odd number of '**': %s" % (src, n + 1, para.strip()[:70]))
    assert not offenders, "Unbalanced bold markers:\n  " + "\n  ".join(offenders)


# ======================================================================
# RULE GROUP AI - the reviewer's audit of rev15
# ======================================================================

CAPTURE_LIVE = os.path.join(REPO_ROOT, "frontend", "scripts", "capture_live_screenshots.js")
PROVENANCE = os.path.join(CHAPTER4, "evidence_provenance.md")
DATABASE_SCHEMA = os.path.join(CHAPTER4, "database_schema.md")


def _json_example_after(text, heading):
    start = text.index(heading)
    m = re.search(r"```json\n(.*?)```", text[start:], re.S)
    assert m, "no JSON example under %s" % heading
    return json.loads(m.group(1))


def test_result_example_has_the_schema_fields_and_the_cam_derived_region():
    """
    The /result example listed executionTimeMs and disclaimer (not in the
    schema), omitted inferenceTimestamp (in it), and gave a fixed lesion
    sentence as topActivationRegion - the kind removed in round seven. The
    code returns a CAM-derived "cell of a 3x3 grid" string. The example's
    shape is compared with ModelObservationSchema itself.
    """
    from app.schemas.assessment import ModelObservationSchema
    example = _json_example_after(read(API_CONTRACT), "### `GET /api/v1/assessments/{id}/result`")
    expected = set(ModelObservationSchema.model_fields)
    assert set(example) == expected, "example keys %s != schema %s" % (sorted(example), sorted(expected))
    assert len(example["classScores"]) == 5
    assert "3x3 grid" in example["topActivationRegion"], "topActivationRegion is not the CAM-derived text the code returns"
    assert abs(sum(c["score"] for c in example["classScores"]) - 1.0) < 0.02, "class scores should sum to about 1"
    assert example["primaryScore"] == max(c["score"] for c in example["classScores"])
    source = read(AI_SERVICE)
    assert "3x3 grid" in source, "the code no longer describes the region on a 3x3 grid; update the rule and the example together"


def test_benchmark_run_notes_state_what_differs_between_the_runs():
    """
    requirements_test_matrix.md and PROGRESS_TRACKER.md described the three
    benchmark runs as the same images on identical code. resource_benchmark.md
    section 1b records that run A used the superseded split's images, run B
    was committed with the a-priori thresholds, and only the combined gates
    2+3 share (not per-stage shares) is durable. The two summaries must carry
    those three facts, not a cleaner story.
    """
    matrix = read(REQUIREMENTS_MATRIX)
    note = matrix[matrix.index("NFR-01 is specified on the mean"):]
    note = note[:note.index("\n\n|") if "\n\n|" in note else len(note)]
    for fact in ("superseded split", "a-priori", "2.4 pp", "1.41"):
        assert fact in note, "the NFR-01 note does not mention %r" % fact
    tracker = read(PROGRESS_TRACKER)
    line = next(l for l in tracker.splitlines() if "1.41" in l)
    assert "B and C" in line and "superseded split" in line and "2.4 pp" in line, (
        "the tracker's latency line does not say which runs are comparable, why run A is not, or that per-stage shares vary")


def test_storage_routes_are_documented_under_their_mount_prefix():
    """main.py mounts the assessments router under /api/v1; the contract said /storage/..."""
    text = read(API_CONTRACT)
    assert "`GET /api/v1/storage/images/{filename}`" in text and "`GET /api/v1/storage/attributions/{filename}`" in text, (
        "the storage routes are documented without the /api/v1 prefix they are mounted under")
    assert "`GET /storage/images" not in text


def test_capture_script_photographs_only_what_it_can_verify():
    """
    capture_live_screenshots.js's 04b step throws on the server-side
    rejection path (no ERR_ code is printed on that screen), so every run
    ended in "Capture failed" after figures 01-08 were written, the non-2xx
    guard never ran, and the manifest's "every API call returned 2xx" was
    unsupported. The step is gone; 04b has one producer, capture_rejection.js,
    and the provenance row says so.
    """
    script = read(CAPTURE_LIVE)
    assert "snap('04b" not in script and "_derived_blurred_negative" not in script, "the live capture script still carries the rejection step"
    assert "apiCalls.filter" in script, "the non-2xx guard is gone"
    prov = read(PROVENANCE)
    row = next(l for l in prov.splitlines() if l.startswith("| `docs/chapter4/screenshots/01`"))
    assert "capture_rejection.js" in row and "by that script and by" not in row, "the provenance row still names two producers for 04b"
    manifest = read(os.path.join(CHAPTER4, "screenshot_evidence_manifest.md"))
    assert "node scripts/capture_rejection.js" in manifest, "the regeneration instructions do not run the rejection capture"


def test_illumination_index_is_described_as_the_code_computes_it():
    """gate3 stores 1 - extreme_ratio; the schema said 'proportion of extreme pixels', the inverse."""
    row = next(l for l in read(DATABASE_SCHEMA).splitlines() if l.startswith("| `illumination_index`"))
    # "1 - ratio" or "one minus the share": the inverse of the extreme-pixel proportion
    assert ("1 −" in row or "1 -" in row or "one minus" in row.lower()) and "0.35" in row, (
        "illumination_index is not described as one minus the extreme-pixel proportion with the 0.35 limit")
    edge = next(l for l in read(DATABASE_SCHEMA).splitlines() if "PROFESSIONAL_REVIEWS :" in l and "USERS" in l)
    assert "signs" not in edge, "the ER edge still says a user 'signs' a review"


# ======================================================================
# RULE GROUP AJ - the reviewer's audit of rev16
# ======================================================================

def test_result_example_region_is_a_cell_the_code_can_name():
    """The example said "centre cell"; the code names the middle cell "central"."""
    source = read(AI_SERVICE)
    rows = re.search(r'row = \("(\w+)", "(\w+)", "(\w+)"\)', source).groups()
    cols = re.search(r'col = \("(\w+)", "(\w+)", "(\w+)"\)', source).groups()
    cells = {"central"} | {"%s-%s" % (r, c) for r in rows for c in cols if (r, c) != (rows[1], cols[1])}
    example = _json_example_after(read(API_CONTRACT), "### `GET /api/v1/assessments/{id}/result`")
    m = re.search(r"in the ([\w-]+) cell of a 3x3 grid", example["topActivationRegion"])
    assert m and m.group(1) in cells, "the example names a cell the code cannot emit: %s (code: %s)" % (
        m.group(1) if m else example["topActivationRegion"], sorted(cells))


def test_manifest_attributes_each_figure_to_its_producer():
    """
    The manifest's evidence paragraph attributed figures 4.1-4.9 and the
    "every API call returned 2xx" guarantee to capture_live_screenshots.js;
    figure 4.5 (04b) is produced by capture_rejection.js, which has no such
    guard. The paragraph must exclude 4.5 from both claims.
    """
    text = read(os.path.join(CHAPTER4, "screenshot_evidence_manifest.md"))
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if "**What these figures evidence.**" in l)
    block = []
    for l in lines[start:]:
        if not l.startswith(">"):
            break
        block.append(l)
    para = "\n".join(block)
    assert "Figures 4.1–4.9 were captured" not in para, "the paragraph still attributes all nine figures to the live script"
    assert "except 4.5" in para or "4.5 excepted" in para or "other than 4.5" in para, "the paragraph does not except figure 4.5"
    assert "capture_rejection.js" in para, "the paragraph does not name figure 4.5's producer"


# ======================================================================
# RULE GROUP AK - the editor's review of rev21
# ======================================================================

HEADER_TSX = os.path.join(REPO_ROOT, "frontend", "src", "components", "Header.tsx")
SIGNIN_TSX = os.path.join(REPO_ROOT, "frontend", "src", "screens", "SignInScreen.tsx")
AUTH_ROUTER = os.path.join(REPO_ROOT, "backend", "app", "routers", "auth.py")


def test_authenticated_session_badge_renders_only_for_a_signed_in_user():
    """
    Header.tsx rendered "Authenticated Session" unconditionally, so the
    sign-in screen showed it beside "Unauthenticated Clinical Workstation".
    The badge must sit inside a `{currentUser && (` block with no block
    close between the opener and the text.
    """
    src = re.sub(r"\{/\*.*?\*/\}", "", read(HEADER_TSX), flags=re.S)   # JSX comments may quote the text
    occurrences = [m.start() for m in re.finditer(r"Authenticated Session", src)]
    assert occurrences, "the badge text is gone; update the rule with the new wording"
    for at in occurrences:
        opener = src.rfind("{currentUser && (", 0, at)
        assert opener != -1, "the badge is not inside a currentUser conditional"
        between = src[opener:at]
        assert ")}" not in between, "a conditional block closes before the badge; it renders unconditionally"


def test_sign_in_offers_one_simulated_account_and_prefills_nothing():
    """
    Two buttons ("Consultant", "Optometrist") filled the same seeded account,
    and the form opened pre-filled with its credentials. One demonstration
    account exists, it is labelled simulated, and the form opens empty.
    """
    src = read(SIGNIN_TSX)
    rendered = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("//"))   # comments may record the old personas
    assert src.count("demo.clinician") >= 1
    assert "Optometrist" not in rendered and "Consultant" not in rendered, "a second persona is offered on the sign-in screen"
    for field in ("email", "password"):
        assert re.search(r"const \[%s, set\w+\] = useState<string>\(''\)" % field, src), (
            "the sign-in form pre-fills the %s field" % field)
    assert "Simulated" in src, "the demonstration account is not labelled simulated"
    auth = read(AUTH_ROUTER)
    aliases = re.search(r"DEMO_LOGIN_IDENTIFIERS = \{(.*?)\}", auth, re.S).group(1)
    assert "optometrist" not in aliases and "dr.demo" not in aliases, "the router still accepts aliases for personas that do not exist"
    assert '"full_name": "Dr. Demo Clinician (Simulated)"' in auth


def test_gate_titles_are_the_same_in_code_api_contract_and_screen():
    """
    The three gates are named in pipeline.py, assessment_service.py, the API
    contract's example and the stepper's initial state. They drifted
    ("Retinal Relevance", "Retinal Anatomical Relevance", "Retinal Relevance &
    Ophthalmic Geometry"). One vocabulary, read from pipeline.py.
    """
    pipeline = read(os.path.join(REPO_ROOT, "backend", "app", "services", "validation", "pipeline.py"))
    titles = re.findall(r'"title": "([^"]+)"', pipeline)
    assert len(titles) == 3, titles
    for rel in ("backend/app/services/assessment_service.py", "docs/chapter4/api_contract.md",
                "frontend/src/screens/ValidationStepperScreen.tsx"):
        text = read(os.path.join(REPO_ROOT, rel))
        for t in titles:
            assert t in text, "%s does not use the gate title %r from pipeline.py" % (rel, t)
    assert "anatom" not in " ".join(titles).lower(), "a gate title claims anatomy"


def test_gate2_details_state_the_boundary():
    """The passing Gate 2 text must say what passing does NOT confirm."""
    src = read(os.path.join(REPO_ROOT, "backend", "app", "services", "validation", "gate2_relevance.py"))
    assert "does not confirm retinal identity, anatomical correctness or clinical gradability" in src
    stepper = read(os.path.join(REPO_ROOT, "frontend", "src", "screens", "ValidationStepperScreen.tsx"))
    assert "does not confirm retinal identity, anatomical correctness or clinical gradability" in stepper


def test_system_test_report_states_a_result_not_a_certification():
    text = read(os.path.join(CHAPTER4, "system_test_report.md"))
    assert "Functional test result: 233 passed, 1 conditionally skipped" in text or re.search(
        r"Functional test result: \d+ passed, \d+ conditionally skipped", text), (
        "the summary block no longer states the functional result plainly")
    assert "Execution Status:" not in text, "the summary block still carries a status line"


def test_manifest_fixture_paragraphs_agree():
    """
    The manifest said the capture image is the held-out APTOS fixture
    d1f1ea894da1 (grade 2) and, four paragraphs later, that it is not from
    APTOS and has no ground truth. Both paragraphs must describe the same
    image.
    """
    text = read(os.path.join(CHAPTER4, "screenshot_evidence_manifest.md"))
    assert "d1f1ea894da1" in text
    for phrase in ("is **not** from APTOS", "appears nowhere in", "carries\n> no ground-truth grade", "no ground-truth grade"):
        assert phrase not in text, "the manifest still says the fixture is not from APTOS / has no ground truth"
    assert "one\n> genuine held-out APTOS image with a Grade 2 reference label" in text or \
           "one genuine held-out APTOS image with a Grade 2 reference label" in text.replace("\n> ", " ")


def test_no_gate_result_is_synthesised_as_passed():
    """
    assessment_service.py returned three PASSED gates with invented metrics
    ("Laplacian: 248.5", "Retinal FOV 92%") for any assessment that had no
    validation record. A gate that was not evaluated is "pending", and no
    GateResultSchema may be constructed with a literal "passed" status.
    """
    src = read(os.path.join(REPO_ROOT, "backend", "app", "services", "assessment_service.py"))
    assert not re.search(r'GateResultSchema\([^)]*status="passed"', src, re.S), "a gate result is constructed as passed with no evaluation behind it"
    assert not re.search(r'metric="(Laplacian: [\d.]+|Retinal FOV [\d.]+%|Valid JPEG)"', src), "invented gate metrics are back"
    for fallback in ("Sharpness confirmed", "Retinal aperture confirmed", "thresholds met\""):
        assert fallback not in src, "a fallback metric claims an outcome: %r" % fallback


# ======================================================================
# RULE GROUP AL - the reviewer's audit of rev22 (application data honesty)
# ======================================================================

STEPPER_TSX = os.path.join(REPO_ROOT, "frontend", "src", "screens", "ValidationStepperScreen.tsx")
APP_TSX = os.path.join(REPO_ROOT, "frontend", "src", "App.tsx")
NEW_ASSESSMENT_TSX = os.path.join(REPO_ROOT, "frontend", "src", "screens", "NewAssessmentScreen.tsx")
MODELS_PY = os.path.join(REPO_ROOT, "backend", "app", "models", "models.py")
REPORT_PY = os.path.join(REPO_ROOT, "backend", "app", "services", "report_service.py")


def test_stepper_never_rewrites_a_gate_status():
    """
    The stepper found the first 'failed' gate and rewrote every other gate
    to 'passed', so a 'pending' / "Not evaluated" gate got a green tick and
    the screen announced all three gates passed. The replay may set only
    'in_progress' and 'pending' as pacing states; 'passed' and 'failed' must
    come from the server object itself.
    """
    src = read(STEPPER_TSX)
    body = src[src.index("useEffect(() => {"):src.index("}, [assessment]);")]
    assert not re.search(r"status:\s*'passed'", body), "the replay writes 'passed' itself"
    assert not re.search(r"status:\s*'failed'", body), "the replay writes 'failed' itself"
    assert "'incomplete'" in src, "a gate without a server verdict has no outcome state of its own"
    assert "Validation not completed for this record" in src


def test_deep_links_substitute_no_record():
    """App.tsx fell back to records[0] / [1] / [2] when the requested kind of record did not exist."""
    src = "\n".join(l for l in read(APP_TSX).splitlines() if not l.lstrip().startswith("//"))   # comments record the old fallbacks
    assert not re.search(r"records\[\d+\]", src), "a deep link falls back to an arbitrary record"
    assert "was not found" in src, "a missing record must be reported, not substituted"


def test_nothing_about_acquisition_is_invented():
    """
    A camera model was preset on every record (screen default, service
    default, ORM default) and printed in figures and PDFs; file-size and
    file-name fallbacks named values of a file that was never selected; the
    PDF printed a fixed licence number when none existed.
    """
    nas = read(NEW_ASSESSMENT_TSX)
    assert re.search(r"const \[cameraModel, setCameraModel\] = useState<string>\(''\)", nas), "the camera model is preset"
    assert 'value="">Not recorded</option>' in nas, "the camera select has no 'Not recorded' option"
    assert "|| 3400000" not in nas and "synthetic_retinal_fundus" not in nas and "'3.4'" not in nas, "invented file fallbacks are back"
    svc = read(os.path.join(REPO_ROOT, "backend", "app", "services", "assessment_service.py"))
    assert "Topcon" not in svc, "the service invents a camera model"
    models = read(MODELS_PY)
    m = re.search(r"camera_model = Column\(([^\n]*)\)", models)
    assert m and "default=" not in m.group(1), "the ORM invents a camera model"
    report = read(REPORT_PY)
    assert 'or "SIM-' not in report and "Standard Fundus Camera" not in report, "the PDF invents a licence number or camera"
    assert "Record created" in report, "the PDF still labels the row's creation time as an acquisition date"


# ======================================================================
# RULE GROUP AM - the reviewer's audit of rev23: the fallbacks that survived
# ======================================================================

SCHEMAS_PY = os.path.join(REPO_ROOT, "backend", "app", "schemas", "assessment.py")
API_TS = os.path.join(REPO_ROOT, "frontend", "src", "services", "api.ts")
REVIEW_MODAL_TSX = os.path.join(REPO_ROOT, "frontend", "src", "screens", "ProfessionalReviewModal.tsx")


def test_request_schema_and_screens_invent_no_camera_or_dilation():
    """
    The camera rule covered the screen state, the service, the ORM and the
    PDF - and the request schema still defaulted cameraModel to a Topcon and
    isMydriatic to False, so a draft created without either came back with
    both; the stepper then fell back to 'Topcon TRC-NW400' on screen. Every
    layer must default to "not recorded".
    """
    schema = read(SCHEMAS_PY)
    block = schema[schema.index("class AssessmentCreateRequest"):]
    block = block[:block.index("\nclass ", 1)]
    assert re.search(r"cameraModel: Optional\[str\] = None", block), "AssessmentCreateRequest still defaults the camera"
    assert re.search(r"isMydriatic: Optional\[bool\] = None", block), "AssessmentCreateRequest still defaults dilation to False"
    resp = schema[schema.index("class AssessmentRecordResponse"):]
    nxt = resp.find("\nclass ", 1)
    resp = resp if nxt == -1 else resp[:nxt]   # it may be the last class in the file
    assert re.search(r"isMydriatic: Optional\[bool\] = None", resp), "AssessmentRecordResponse defaults dilation to False"
    models = read(MODELS_PY)
    m = re.search(r"is_mydriatic = Column\(([^\n]*)\)", models)
    assert m and "default=" not in m.group(1) and "nullable=True" in m.group(1), "the ORM still defaults dilation"
    svc = read(os.path.join(REPO_ROOT, "backend", "app", "services", "assessment_service.py"))
    assert "payload.isMydriatic or False" not in svc
    report = read(REPORT_PY)
    assert '"Not recorded" if assessment.is_mydriatic is None' in report, "the PDF prints a dilation protocol for an unrecorded one"
    for rel in _walk_repo((".ts", ".tsx")):
        src = read(os.path.join(REPO_ROOT, rel))
        for m in re.finditer(r"cameraModel\s*(\|\||\?\?)\s*'([^']*)'", src):
            assert m.group(2) == "Not recorded", "%s falls back to an invented camera: %r" % (rel, m.group(2))
    nas = read(NEW_ASSESSMENT_TSX)
    assert "useState<boolean | null>(null)" in nas, "the dilation control cannot say 'not recorded'"


def test_no_invented_review_reason_role_or_filename():
    modal = read(REVIEW_MODAL_TSX)
    assert re.search(r"const \[inconclusiveReason, setInconclusiveReason\] = useState<string>\(''\)", modal), (
        "the review modal pre-selects an inconclusive reason")
    assert "agreement !== 'inconclusive' || inconclusiveReason !== ''" in modal, "an inconclusive review can be submitted without a reason"
    api = read(API_TS)
    assert "|| 'Clinician'" not in api, "api.ts invents a role"
    router = read(ASSESSMENTS_ROUTER)
    assert 'or "fundus.jpg"' not in router, "the upload route invents a filename"
    stepper = read(STEPPER_TSX)
    assert "server.length === 0" in stepper, "an empty gate list would be announced as passed"
    svc = read(os.path.join(REPO_ROOT, "backend", "app", "services", "assessment_service.py"))
    assert 'agreement == "inconclusive" and not (review_input.inconclusiveReason' in svc, (
        "the API accepts an inconclusive review without a reason")
    dss = read(os.path.join(REPO_ROOT, "frontend", "src", "screens", "DecisionSupportScreen.tsx"))
    assert "technical image quality violation." not in dss, "the workspace invents a rejection reason"


# ======================================================================
# RULE GROUP AN - the editor's review of rev25: code and documents agree
# ======================================================================

VALIDATION_DIR = os.path.join(REPO_ROOT, "backend", "app", "services", "validation")
GATE2_PY = os.path.join(VALIDATION_DIR, "gate2_relevance.py")
GATE3_PY = os.path.join(VALIDATION_DIR, "gate3_quality.py")
VALIDATION_SPEC = os.path.join(CHAPTER4, "validation_module_spec.md")
TRAINING_SUMMARY = os.path.join(CHAPTER4, "training_summary.json")
RETINAL_VALIDATOR_TS = os.path.join(REPO_ROOT, "frontend", "src", "utils", "retinalValidator.ts")


def test_every_declared_validation_threshold_is_used_by_a_gate():
    """
    RETINAL_MAX_COVERAGE = 0.98 was declared in config.py, shipped in
    .env.example, and documented as a rejection - and no gate read it. A
    threshold the code does not execute is documentation of nothing.
    """
    names = re.findall(r"^\s+((?:RETINAL_|LAPLACIAN_|CONTRAST_|ILLUMINATION_|MIN_IMAGE_|VALIDATION_ANALYSIS_)[A-Z_]+):", read(CONFIG_PY), re.M)
    assert len(names) >= 9, names
    used = "\n".join(read(path) for path in python_sources(VALIDATION_DIR))
    unused = [n for n in names if "settings.%s" % n not in used]
    assert not unused, "declared in config.py but read by no gate: %s" % unused
    # and no threshold-like setting anywhere in config.py may be read by nothing
    # (MODEL_SCORE_THRESHOLD = 0.5 was declared, shipped and read nowhere)
    all_names = re.findall(r"^\s+([A-Z][A-Z0-9_]*(?:_THRESHOLD|_MIN|_MAX|_BELOW|_ABOVE|_RATIO_MAX))\s*:", read(CONFIG_PY), re.M)
    everywhere = "\n".join(read(os.path.join(dp, f)) for dp, _, fs in os.walk(os.path.join(REPO_ROOT, "backend", "app")) for f in fs if f.endswith(".py"))
    everywhere += "\n".join(read(path) for path in python_sources(os.path.join(REPO_ROOT, "backend", "scripts")))
    dead = [n for n in all_names if "settings.%s" % n not in everywhere]
    assert not dead, "declared in config.py but read nowhere in the application or its scripts: %s" % dead


def test_documented_gate2_thresholds_are_the_executed_ones():
    """
    The specification said red share >= 38% and coverage between 20% and
    98%; the code tested red share < 0.36 and only the 20% floor. Every
    Gate 2 number in the specification and in the browser pre-check is read
    back from config.py, and the code must use the settings, not literals.
    """
    cfg = _config_defaults()
    spec = read(VALIDATION_SPEC)
    sec = spec[spec.index("### Gate 2"):spec.index("### Gate 3")]
    m = re.search(r"red channel share \$\\ge ([\d.]+)\\%\$", sec)
    assert m and abs(float(m.group(1)) / 100 - cfg["RETINAL_RED_SHARE_MIN"]) < 1e-9, "the spec's red-share floor is not RETINAL_RED_SHARE_MIN"
    m = re.search(r"at least ([\d.]+)% of the frame", sec)
    assert m and abs(float(m.group(1)) / 100 - cfg["RETINAL_MIN_COVERAGE"]) < 1e-9, "the spec's coverage floor is not RETINAL_MIN_COVERAGE"
    assert "between 20.0% and 98.0%" not in sec and "No upper bound is enforced" in sec, "the spec still claims an upper coverage bound"
    m = re.search(r"ratio \$\\ge ([\d.]+)\$", sec)
    assert m and abs(float(m.group(1)) - cfg["RETINAL_RED_RATIO_MIN"]) < 1e-9
    gate2 = read(GATE2_PY)
    assert "settings.RETINAL_RED_SHARE_MIN" in gate2 and not re.search(r"red_share < 0\.\d+", gate2), "gate2 tests a literal red-share floor"
    assert "RETINAL_MAX_COVERAGE" not in gate2 and not re.search(r"^\s+RETINAL_MAX_COVERAGE:", read(CONFIG_PY), re.M), (
        "the unused upper-coverage setting is declared again")
    browser = read(RETINAL_VALIDATOR_TS)
    assert "redShare >= RETINAL_RED_SHARE_MIN" in browser and not re.search(r"redShare >= \d", browser), (
        "the browser's red-share floor is not the generated RETINAL_RED_SHARE_MIN")


def test_gate3_source_describes_its_calibrated_thresholds():
    """
    gate3_quality.py's docstring said >= 60.0 and >= 18.0 and a comment said
    the threshold was NOT calibrated and rejected every real image - the
    pre-calibration state, contradicting config.py (4.3 / 8.8) and the
    calibration evidence shipped beside it.
    """
    src = read(GATE3_PY)
    for stale in ("60.0", "18.0", "NOT calibrated", "every real", "not calibrated"):
        assert stale not in src, "gate3_quality.py still says %r" % stale
    for name in ("LAPLACIAN_BLUR_THRESHOLD", "CONTRAST_THRESHOLD", "ILLUMINATION_EXTREME_RATIO_MAX"):
        assert name in src.split("def evaluate_gate3")[1].split('"""')[1], "the docstring does not name %s" % name
    assert "does not confirm retinal identity, anatomical correctness or clinical gradability" in src


def test_validation_messages_make_no_clinical_inference():
    """
    Colour, contrast and Laplacian heuristics cannot find vascular
    characteristics, media opacities or inadequate dilation. Every
    rejection_reason / clinical_action / details string in the gates, the
    browser pre-check and the PDF is technical.
    """
    banned = ("vascular", "opacit", "cataract", "dilation", "haze", "pigmentation", "anatom")
    for path in (GATE2_PY, GATE3_PY):
        for m in re.finditer(r'(rejection_reason|clinical_action)="([^"]*)"', read(path)):
            low = m.group(2).lower()
            for b in banned:
                assert b not in low, "%s: %s=%r" % (os.path.basename(path), m.group(1), m.group(2))
    for m in re.finditer(r"gate2(?:Reason|Action) = [`'](.*?)[`'];", read(RETINAL_VALIDATOR_TS)):
        low = m.group(1).lower()
        for b in banned:
            assert b not in low, "retinalValidator.ts: %r" % m.group(1)
    report = read(REPORT_PY)
    assert "spectral balance confirmed" not in report and "no anatomical confirmation" in report


def test_training_time_arithmetic_follows_from_training_summary():
    """training_protocol.md said 52.5 min and ~210 s/epoch for 2,761.6 s over 15 epochs (46.0 min, 184.1 s/epoch)."""
    summary = json.loads(read(TRAINING_SUMMARY))
    total, epochs = float(summary["total_time_seconds"]), int(summary["epochs"])
    minutes, per_epoch = total / 60.0, total / epochs
    offenders = []
    for rel, text in iter_markdown(include_correction_records=True):
        if "/archive/" in rel.replace("\\", "/"):
            continue
        # per SENTENCE: a line that records the old arithmetic may also carry
        # the current figure, and only the recording sentence is exempt
        for lineno, unit, kind in _units(text):
            if kind != "heading" and any(m in unit for m in RECORD_MARKERS):
                continue
            for q in re.findall(r"~?([\d.]+) s/epoch", unit):
                if abs(float(q) - per_epoch) > 0.1:
                    offenders.append("%s:%d says %s s/epoch; the summary gives %.1f" % (rel, lineno, q, per_epoch))
            for q in re.findall(r"\(~?(?:approximately )?([\d.]+) min", unit):
                if abs(float(q) - minutes) > 0.5:
                    offenders.append("%s:%d says %s min; the summary gives %.1f" % (rel, lineno, q, minutes))
    assert not offenders, "\n  ".join(["Training-time arithmetic:"] + offenders)


def test_preprocessing_parity_claims_consistency_not_generalisation():
    spec = read(os.path.join(CHAPTER4, "preprocessing_and_augmentation_spec.md"))
    assert "does not establish performance on images from populations, cameras or clinical environments outside the evaluated APTOS cohort" in spec


def test_transcript_date_is_not_older_than_the_build_check():
    """VERIFICATION.md said "Produced on 2026-10-03" in an archive built on the 4th."""
    # lifted to the archive root by the assembler; in the repository it lives under docs/chapter4
    m = re.search(r"Produced on (\d{4}-\d{2}-\d{2})", read(_resolve(
        os.path.join(CHAPTER4, "VERIFICATION.md"), os.path.join(REPO_ROOT, "VERIFICATION.md"))))
    b = re.search(r"date \(UTC\): (\d{4}-\d{2}-\d{2})", read(os.path.join(CHAPTER4, "build_verification.log")))
    assert m and b, "dates not found"
    assert m.group(1) >= b.group(1), "the transcript (%s) predates the build check (%s)" % (m.group(1), b.group(1))


# ======================================================================
# RULE GROUP AO - the reviewer's audit of rev26
# ======================================================================

VERIFIER_PY = os.path.join(REPO_ROOT, "backend", "scripts", "verify_gate_downsampling.py")
FIXTURE_DIR = os.path.join(REPO_ROOT, "backend", "tests", "fixtures")


def test_gate3_cutoffs_are_settings_the_spec_quotes():
    """The spec said < 10 / > 245; the code used literals 25 / 235. Both are settings now, quoted by the spec."""
    cfg = _config_defaults()
    gate3 = read(GATE3_PY)
    assert "settings.ILLUMINATION_UNDEREXPOSED_BELOW" in gate3 and "settings.ILLUMINATION_OVEREXPOSED_ABOVE" in gate3
    assert not re.search(r"fg_pixels [<>] \d", gate3), "gate3 compares against a literal cut-off"
    spec = read(VALIDATION_SPEC)
    sec = spec[spec.index("### Gate 3"):]
    m = re.search(r"luminance < ([\d.]+) \(`ILLUMINATION_UNDEREXPOSED_BELOW`\) or > ([\d.]+) \(`ILLUMINATION_OVEREXPOSED_ABOVE`\)", sec)
    assert m, "the spec does not quote the cut-off settings"
    assert float(m.group(1)) == cfg["ILLUMINATION_UNDEREXPOSED_BELOW"] and float(m.group(2)) == cfg["ILLUMINATION_OVEREXPOSED_ABOVE"], (
        "the spec's cut-offs differ from config.py")
    for stale in ("(< 10)", "(> 245)"):
        assert stale not in sec.split("An earlier revision")[0]


def test_gate1_claims_only_what_it_does():
    """
    The spec claimed EXIF stripping and a random UUID filename; the service
    writes the bytes unchanged under <record id>_<8 hex>.<ext>. The Gate 1
    docstring said >= 512x512 (MIN_IMAGE_DIMENSION is 480).
    """
    spec = read(VALIDATION_SPEC)
    sec = spec[spec.index("### Gate 1"):spec.index("### Gate 2")]
    assert "No EXIF metadata is stripped" in sec, "the spec does not say that no metadata is stripped"
    doc = read(os.path.join(VALIDATION_DIR, "gate1_integrity.py"))
    assert "MIN_IMAGE_DIMENSION" in doc.split("def evaluate_gate1")[1].split('"""')[1], "the Gate 1 docstring does not name MIN_IMAGE_DIMENSION"
    assert not re.search(r">= \d+x\d+", doc.split("def evaluate_gate1")[1].split('"""')[1]), "the Gate 1 docstring hard-codes a resolution"
    svc = read(os.path.join(REPO_ROOT, "backend", "app", "services", "assessment_service.py"))
    assert 'ext = ".png" if image_bytes.startswith(b"\\x89PNG") else ".jpg"' in svc, "PNG bytes are stored as .jpg"
    assert "exif" not in svc.lower() or "no exif" in svc.lower()


def test_downsampling_verifier_runs_on_the_fixture():
    """
    Round twenty-six removed RETINAL_MAX_COVERAGE from config and left the
    verifier iterating a 'coverage_hi' key it no longer defined: the script
    that produced shipped evidence crashed on its first image. It must run.
    """
    import subprocess, sys, tempfile
    src = read(VERIFIER_PY)
    code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))   # a comment records the removal
    assert '"coverage_hi"' not in code and "settings.RETINAL_MAX_COVERAGE" not in code
    assert '"red_share"' in src.split("for key, value, deviation in")[1][:600], "the loop does not measure red_share"
    shipped = os.path.join(CHAPTER4, "gate_downsampling_verification.json")
    before = sha256_of(shipped) if os.path.exists(shipped) else None
    with tempfile.TemporaryDirectory() as out:
        # --out keeps the smoke run away from the shipped full-corpus JSON
        r = subprocess.run([sys.executable, VERIFIER_PY, FIXTURE_DIR, "--out", os.path.join(out, "v.json")],
                           cwd=os.path.join(REPO_ROOT, "backend"), capture_output=True, text=True, timeout=600)
        assert r.returncode == 0, "verify_gate_downsampling.py failed on the fixture:\n" + (r.stderr or r.stdout)[-1500:]
        assert os.path.exists(os.path.join(out, "v.json")), "the verifier wrote no JSON to --out"
    if before is not None:
        assert sha256_of(shipped) == before, "the smoke run overwrote the shipped gate_downsampling_verification.json"


def _schema_fields(module, cls):
    import importlib
    return set(getattr(importlib.import_module(module), cls).model_fields)


def test_api_contract_examples_match_their_schemas():
    """
    Round sixteen compared the /result example with its schema; the login,
    validation and review examples still carried fields the schemas do not
    have (expires_in, metrics objects, optionalObservation) and lacked ones
    they do. Response examples must have exactly the schema's keys; request
    examples may use only the schema's keys.
    """
    text = read(API_CONTRACT)
    login = _json_example_after(text, "### `POST /api/v1/auth/login`")
    assert set(login) == {"username", "password"} == _schema_fields("app.schemas.auth", "LoginRequest")
    token = json.loads(re.findall(r"```json\n(.*?)```", text[text.index("### `POST /api/v1/auth/login`"):], re.S)[1])
    assert set(token) == _schema_fields("app.schemas.auth", "TokenResponse"), sorted(token)
    assert set(token["user"]) == _schema_fields("app.schemas.auth", "ClinicianUserResponse"), sorted(token["user"])
    gates = _json_example_after(text, "### `GET /api/v1/assessments/{id}/validation`")
    want = _schema_fields("app.schemas.assessment", "GateResultSchema")
    assert len(gates) == 3 and all(set(g) == want for g in gates), [sorted(g) for g in gates]
    pipeline_titles = re.findall(r'"title": "([^"]+)"', read(os.path.join(VALIDATION_DIR, "pipeline.py")))
    assert [g["title"] for g in gates] == pipeline_titles
    review = _json_example_after(text, "### `POST /api/v1/assessments/{id}/review`")
    allowed = _schema_fields("app.schemas.assessment", "ClinicianReviewSubmitRequest")
    assert set(review) <= allowed, sorted(set(review) - allowed)
    if review["agreement"] == "inconclusive":
        assert review.get("inconclusiveReason"), "the example would be refused by the server"
    create = _json_example_after(text, "### `POST /api/v1/assessments`")
    assert set(create) <= _schema_fields("app.schemas.assessment", "AssessmentCreateRequest")


# ======================================================================
# RULE GROUP AP - the reviewer's minors on rev27
# ======================================================================

def test_validation_example_is_the_fixtures_real_gate_output():
    """
    The /validation example quoted the fixture's real numbers everywhere but
    one ("dynamic range 55.3" for a real 24.0) while the page said no value
    came from a recorded run. The page now says this example IS the fixture's
    output; every metric and details string in it is recomputed here.
    """
    from app.services.validation.gate1_integrity import evaluate_gate1
    from app.services.validation.gate2_relevance import evaluate_gate2
    from app.services.validation.gate3_quality import evaluate_gate3
    fixture = os.path.join(FIXTURE_DIR, "aptos_heldout_d1f1ea894da1.png")
    with open(fixture, "rb") as fh:
        raw = fh.read()
    g1, pil = evaluate_gate1(raw, "aptos_heldout_d1f1ea894da1.png")
    g2, g3 = evaluate_gate2(pil), evaluate_gate3(pil)
    assert g1.passed and g2.passed and g3.passed, "the fixture no longer passes the gates"
    text = read(API_CONTRACT)
    assert "actual output for the shipped held-out fixture" in text
    example = _json_example_after(text, "### `GET /api/v1/assessments/{id}/validation`")
    for shown, real in zip(example, (g1, g2, g3)):
        assert shown["metric"] == real.metric, "Gate %d metric: contract %r, fixture %r" % (shown["gateIndex"], shown["metric"], real.metric)
        assert shown["details"] == real.details, "Gate %d details: contract %r, fixture %r" % (shown["gateIndex"], shown["details"], real.details)


def test_every_served_route_is_named_in_the_contract():
    """The contract omitted /assessments/search, /assessments/{id}, /status, /reports/{id}/pdf and /health."""
    text = read(API_CONTRACT)
    missing = []
    for name in ("assessments.py", "auth.py", "health.py"):
        src = read(os.path.join(REPO_ROOT, "backend", "app", "routers", name))
        for path in re.findall(r'@router\.(?:get|post|put|delete|patch)\("([^"]+)"', src):
            shown = path.replace("{assessment_id}", "{id}")
            if name == "auth.py":
                shown = "/auth" + shown
            if shown not in text:
                missing.append(shown)
    assert not missing, "routes the application serves that the contract never names: %s" % sorted(set(missing))


def test_login_example_uses_the_seeded_accounts_identity():
    auth = read(AUTH_ROUTER)
    seeded = re.search(r'id="(USR-\d+)"', auth).group(1)
    token = json.loads(re.findall(r"```json\n(.*?)```", read(API_CONTRACT)[read(API_CONTRACT).index("### `POST /api/v1/auth/login`"):], re.S)[1])
    assert token["user"]["id"] == seeded, "the login example shows id %r; the seeded account is %r" % (token["user"]["id"], seeded)
    assert token["user"]["licenseNumber"] == "SIM-000001"


# ======================================================================
# RULE GROUP AQ - the editor's review of rev28: the browser pre-check
# performs what it reports and reports nothing it did not perform
# ======================================================================

THRESHOLDS_TS = os.path.join(REPO_ROOT, "frontend", "src", "utils", "validationThresholds.ts")
EXPORTER_PY = os.path.join(REPO_ROOT, "backend", "scripts", "export_frontend_thresholds.py")


def _validator_code():
    """retinalValidator.ts with // and /* */ comments removed."""
    src = read(RETINAL_VALIDATOR_TS)
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    return "\n".join(l.split("//")[0] if "://" not in l else l for l in src.splitlines())


def test_browser_thresholds_are_generated_from_the_server_configuration():
    """
    The browser's thresholds were literals typed beside the server's and
    drifted three times (256 vs 480 px, 0.60-1.70 vs 0.65-1.65, and a
    coverage floor never applied). validationThresholds.ts is generated from
    config.py; the committed file must equal the generator's output, and the
    validator must import every threshold from it.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("export_frontend_thresholds", EXPORTER_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert read(THRESHOLDS_TS).replace("\r\n", "\n") == mod.render(), (
        "validationThresholds.ts is stale: run backend/scripts/export_frontend_thresholds.py")
    cfg = _config_defaults()
    for name, _purpose in mod.EXPORTED:
        m = re.search(r"export const %s = ([\d.]+);" % name, read(THRESHOLDS_TS))
        assert m and float(m.group(1)) == float(cfg[name]), "%s in the generated file differs from config.py" % name
    code = _validator_code()
    for name in ("MIN_IMAGE_DIMENSION", "MAX_UPLOAD_SIZE_MB", "RETINAL_MIN_COVERAGE", "RETINAL_RED_RATIO_MIN", "RETINAL_RED_SHARE_MIN"):
        assert re.search(r"\b%s\b" % name, code.split("export interface")[0]), "the validator does not import %s" % name
    # no bare copy of a server threshold may remain in the validator's code
    for literal in (r">=\s*480\b", r"<=\s*15\s*\*\s*1024", r">=\s*1\.15\b", r"<\s*1\.15\b", r">=\s*0\.36\b", r">=\s*0\.20?\b"):
        assert not re.search(literal, code), "the validator still compares against a literal threshold: %s" % literal


def test_browser_gate2_computes_and_applies_foreground_coverage():
    """
    The browser counted foreground pixels, never divided, never compared, and
    reported "Aperture coverage and colour profile within thresholds". It must
    compute the coverage, compare it with the server's minimum, require it for
    Gate 2, carry the measured value in the result, and say "within
    thresholds" only where gate2Passed is true.
    """
    code = _validator_code()
    assert re.search(r"foregroundCoverage\s*=\s*foregroundCount\s*/\s*\(sampleSize\s*\*\s*sampleSize\)", code), (
        "foreground coverage is not computed")
    assert re.search(r"const coveragePassed\s*=\s*coverageEvaluated\s*&&\s*foregroundCoverage\s*>=\s*RETINAL_MIN_COVERAGE", code), (
        "coverage is not compared with the server's RETINAL_MIN_COVERAGE")
    m = re.search(r"const gate2Passed\s*=\s*([^;]+);", code)
    assert m, "gate2Passed not found"
    for term in ("gate1Passed", "aspectPassed", "coveragePassed", "colorPassed", "notDiagramPassed"):
        assert term in m.group(1), "gate2Passed does not require %s" % term
    assert re.search(r"foregroundCoverage,\s*\n\s*coverageEvaluated,", code), "the measured coverage is not carried in the result"
    # every "within thresholds" is guarded by gate2Passed on the same expression
    for hit in re.finditer(r"within thresholds", code):
        window = code[max(0, hit.start() - 60):hit.start()]
        assert "gate2Passed ?" in window, "'within thresholds' is asserted without gate2Passed: ...%s" % window[-50:]
    gate2 = read(GATE2_PY)
    assert "settings.RETINAL_MIN_COVERAGE" in gate2, "the server no longer applies RETINAL_MIN_COVERAGE"
    screen = read(NEW_ASSESSMENT_TSX)
    assert "clientValidation.gate2.metric" in screen, "the upload screen does not display the measured coverage"


def test_browser_never_reports_gate3_as_passed():
    """
    The validator said it issues no sharpness verdict and then set
    gate3Passed = gate2Passed, serialised Gate 3 as passed and folded it into
    allPassed. Gate 3 is 'notEvaluated' in the browser; the browser's result
    is gates 1 and 2; only the server's response carries a Gate 3 verdict.
    """
    code = _validator_code()
    assert "gate3Passed" not in code and "allPassed" not in code, "the browser still carries a Gate 3 / all-gates pass state"
    assert re.search(r"const preflightPassed\s*=\s*gate1Passed\s*&&\s*gate2Passed\s*;", code), (
        "the browser's result is not exactly gates 1 and 2")
    block = code[code.rindex("gate3: {"):]
    block = block[:block.index("}")]
    assert "status: 'notEvaluated'" in block and "passed" not in block, "the returned Gate 3 is not notEvaluated"
    iface = read(RETINAL_VALIDATOR_TS)
    assert re.search(r"gate3: \{[^}]*status: 'notEvaluated';", iface, re.S), "the type allows a Gate 3 verdict from the browser"
    assert re.search(r"failedGate: 1 \| 2 \| null;", iface), "the browser can still report Gate 3 as the failed gate"
    for rel in _walk_repo((".ts", ".tsx")):
        src = read(os.path.join(REPO_ROOT, rel))
        assert "allPassed" not in src and "failedGate === 3" not in src and "gate3.passed" not in src, (
            "%s still reads a browser Gate 3 verdict" % rel)
    screen = read(NEW_ASSESSMENT_TSX)
    assert "not evaluated in the browser" in screen, "the upload screen does not say Gate 3 is not evaluated in the browser"
    # the pass line needs a result: an earlier branch showed "passed" while clientValidation was still null
    assert ") : clientValidation ? (" in screen and "browser pre-check pending" in screen


def test_gate_descriptions_claim_no_retinal_identity():
    gate2 = read(GATE2_PY)
    assert "passing these checks does not establish retinal identity" in gate2
    svc = read(os.path.join(REPO_ROOT, "backend", "app", "services", "assessment_service.py"))
    assert "All three configured technical gates passed; the image is eligible for model inference." in svc
    head = read(RETINAL_VALIDATOR_TS).split("export function")[0]
    assert "does not establish retinal identity" in head and "issues\n * no Gate 3 verdict" in head


# ======================================================================
# RULE GROUP AR - the reviewer's audit of rev29: nothing uncomputed is
# returned, and the pre-check is EXECUTED, not only read
# ======================================================================

PREFLIGHT_CHECK = os.path.join(REPO_ROOT, "frontend", "scripts", "check_preflight.cjs")
BUILD_LOG = os.path.join(CHAPTER4, "build_verification.log")


def _preflight_source_hash():
    import hashlib
    v = read(RETINAL_VALIDATOR_TS).replace("\r\n", "\n")
    t = read(THRESHOLDS_TS).replace("\r\n", "\n")
    return hashlib.sha256((v + "\n--\n" + t).encode("utf-8")).hexdigest()


def test_browser_precheck_returns_no_placeholder_values():
    """
    Executed by the reviewer, the validator returned "Signature Valid" for a
    signature it never read, a focus estimate of 180.0 it never calculated,
    and R/B 1.00 / red share 33% it never measured. A value that may not have
    been computed is null, and no default stands in for a measurement.
    """
    code = _validator_code()
    assert "Signature Valid" not in code, "the browser claims a signature check it does not perform"
    assert "the file signature is checked by the server" in code
    assert re.search(r"let redToBlueRatio: number \| null = null;", code) and re.search(r"let redShare: number \| null = null;", code), (
        "colour figures start from a placeholder instead of null")
    assert re.search(r"let focusEstimate: number \| null = null;", code), "the focus estimate starts from a placeholder"
    assert "180.0" not in code and "= 0.33" not in code and "= 1.0;" not in code, "a placeholder measurement is back"
    iface = read(RETINAL_VALIDATOR_TS)
    assert "focusEstimate: number | null;" in iface and "redToBlueRatio: number | null;" in iface and "redShare: number | null;" in iface
    assert "mimeType || 'image/jpeg'" not in code, "an undeclared MIME type is reported as image/jpeg"


def test_browser_precheck_was_executed_against_this_source():
    """
    Rules that read TypeScript could not see a default value being returned
    as a measurement. frontend/scripts/check_preflight.cjs bundles the real
    validator, runs it in Node on eight synthetic inputs and asserts every
    field. The build check runs it in the clean extraction; its output is in
    build_verification.log together with the SHA-256 of the two source files
    it ran. That hash must be the hash of the validator shipped here, so the
    recorded run cannot be of an older validator.
    """
    assert os.path.exists(PREFLIGHT_CHECK), "frontend/scripts/check_preflight.cjs is missing"
    script = read(PREFLIGHT_CHECK)
    for case in ("reddish disc", "small reddish disc", "all-dark frame", "bright neutral frame", "300x200 image", "2400x800 panorama",
                 "grey disc", "no canvas available"):
        assert case in script, "the executed check no longer covers: %s" % case
    assert '"check:preflight": "node scripts/check_preflight.cjs"' in read(os.path.join(REPO_ROOT, "frontend", "package.json"))
    shipped = {dst for _src, dst in _assembler().manifest()}
    assert "frontend/scripts/check_preflight.cjs" in shipped, "the executed check is not in the archive"
    log = read(BUILD_LOG)
    m = re.search(r"PREFLIGHT CHECK source sha256 ([0-9a-f]{64})", log)
    assert m, "build_verification.log does not record an executed pre-check"
    assert m.group(1) == _preflight_source_hash(), (
        "the recorded pre-check ran a different retinalValidator.ts / validationThresholds.ts than the one shipped")
    assert re.search(r"PREFLIGHT CHECK: 8/8 cases as expected", log), "the recorded pre-check did not pass all eight cases"
    assert "  FAIL " not in log.split("PREFLIGHT CHECK source")[1].split("PREFLIGHT CHECK: ")[0]
