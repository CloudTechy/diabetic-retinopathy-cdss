#!/usr/bin/env python3
"""
Assemble docs/chapter4_submission_package/ by COPYING committed artefacts.

This replaces prepare_submission_package.py, which did not assemble a package so
much as manufacture one. That script invented its own benchmark timings with
np.random.normal, inserted fake "OS context switch spikes" at hardcoded indices
12, 47 and 88, then rescaled the series so its mean came out at exactly
95.76 ms — and wrote evaluation documents quoting 86.40% accuracy and
kappa = 0.9415, figures no training run ever produced. Running it would
silently overwrite the genuine evidence with fabricated numbers.

The rule here is simple and absolute: this script **copies files and computes
checksums**. It does not generate data, and it cannot. Anything it cannot find
is reported as missing rather than invented.

Artefacts are produced by, and only by:
    notebooks/colab_train_and_evaluate.py      training, evaluation, plots
    backend/scripts/analyze_clinical_metrics.py    clinical metrics, leakage audit
    backend/scripts/benchmark_cpu_end_to_end.py    CPU end-to-end latency
    backend/scripts/benchmark_resources.py         forward-pass latency
    backend/scripts/verify_gate_downsampling.py    gate decision preservation

Usage:
    python backend/scripts/assemble_submission_package.py
    python backend/scripts/assemble_submission_package.py --check   # verify only
"""

import argparse
import hashlib
import os
import shutil
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHAPTER4 = os.path.join(REPO_ROOT, "docs", "chapter4")
PACKAGE = os.path.join(REPO_ROOT, "docs", "chapter4_submission_package")

EXPECTED_CHECKPOINT_SHA256 = "8ee14d7591a8e6a1b86c15416a77375a198bd49399b3977a3de79a00e3dd14fa"

# (source relative to repo root, destination relative to the package)
LAYOUT = [
    # Trained weights
    ("backend/models/weights/efficientnet_b0_dr.pth", "checkpoint/efficientnet_b0_dr.pth"),

    # Raw run artefacts
    ("docs/chapter4/dataset_split_manifest.csv", "dataset_sample_and_manifest/dataset_split_manifest.csv"),
    ("docs/chapter4/held_out_predictions.csv", "logs_and_metrics/held_out_predictions.csv"),
    ("docs/chapter4/epoch_history.csv", "logs_and_metrics/epoch_history.csv"),
    ("docs/chapter4/training_execution.log", "logs_and_metrics/training_execution.log"),
    ("docs/chapter4/training_summary.json", "logs_and_metrics/training_summary.json"),
    ("docs/chapter4/evaluation_summary.json", "logs_and_metrics/evaluation_summary.json"),
    ("docs/chapter4/clinical_metrics.json", "logs_and_metrics/clinical_metrics.json"),
    ("docs/chapter4/benchmark_timings.csv", "logs_and_metrics/benchmark_timings.csv"),
    ("docs/chapter4/benchmark_summary.json", "logs_and_metrics/benchmark_summary.json"),
    ("docs/chapter4/cpu_end_to_end_benchmark.json", "logs_and_metrics/cpu_end_to_end_benchmark.json"),
    ("docs/chapter4/cpu_end_to_end_benchmark.csv", "logs_and_metrics/cpu_end_to_end_benchmark.csv"),
    ("docs/chapter4/gate_downsampling_verification.json", "logs_and_metrics/gate_downsampling_verification.json"),
    ("docs/chapter4/validation_test_results.csv", "logs_and_metrics/validation_test_results.csv"),

    # Figures
    ("docs/chapter4/confusion_matrix.png", "visualizations/confusion_matrix.png"),
    ("docs/chapter4/learning_curves.png", "visualizations/learning_curves.png"),

    # Analysis documents
    ("docs/chapter4/model_evaluation_report.md", "documentation/model_evaluation_report.md"),
    ("docs/chapter4/dataset_audit.md", "documentation/dataset_audit.md"),
    ("docs/chapter4/training_protocol.md", "documentation/training_protocol.md"),
    ("docs/chapter4/training_environment.md", "documentation/training_environment.md"),
    ("docs/chapter4/checkpoint_manifest.md", "documentation/checkpoint_manifest.md"),
    ("docs/chapter4/preprocessing_and_augmentation_spec.md", "documentation/preprocessing_and_augmentation_spec.md"),
    ("docs/chapter4/resource_benchmark.md", "documentation/resource_benchmark.md"),
    ("docs/chapter4/known_limitations.md", "documentation/known_limitations.md"),
    ("docs/chapter4/reproducibility_runbook.md", "documentation/reproducibility_runbook.md"),
    ("docs/chapter4/validation_module_spec.md", "documentation/validation_module_spec.md"),
    ("docs/chapter4/objective_traceability_matrix.md", "documentation/objective_traceability_matrix.md"),
    ("docs/chapter4/system_test_report.md", "documentation/system_test_report.md"),
    ("docs/chapter4/screenshot_evidence_manifest.md", "documentation/screenshot_evidence_manifest.md"),
    ("docs/chapter4/independent_thesis_qa_gate_audit.md", "documentation/independent_thesis_qa_gate_audit.md"),
    ("docs/chapter4/evidence_provenance.md", "PROVENANCE.md"),
    ("docs/chapter4/CLEAN_RERUN_RUNBOOK.md", "CLEAN_RERUN_RUNBOOK.md"),
    ("docs/chapter4/architecture.md", "documentation/architecture.md"),
    ("docs/chapter4/database_schema.md", "documentation/database_schema.md"),
    ("docs/chapter4/api_contract.md", "documentation/api_contract.md"),
    ("docs/chapter4/requirements_test_matrix.md", "documentation/requirements_test_matrix.md"),
    ("docs/chapter4/implementation_status.md", "documentation/implementation_status.md"),
    ("docs/chapter4/dataset_split_audit.json", "logs_and_metrics/dataset_split_audit.json"),
    ("docs/chapter4/test_execution_output.txt", "logs_and_metrics/test_execution_output.txt"),
    ("backend/scripts/build_clean_split.py", "scripts/build_clean_split.py"),

    # Scripts that produce the evidence
    ("notebooks/colab_train_and_evaluate.py", "scripts/colab_train_and_evaluate.py"),
    ("backend/scripts/analyze_clinical_metrics.py", "scripts/analyze_clinical_metrics.py"),
    ("backend/scripts/benchmark_cpu_end_to_end.py", "scripts/benchmark_cpu_end_to_end.py"),
    ("backend/scripts/benchmark_resources.py", "scripts/benchmark_resources.py"),
    ("backend/scripts/verify_gate_downsampling.py", "scripts/verify_gate_downsampling.py"),
    ("backend/scripts/generate_validation_evidence.py", "scripts/generate_validation_evidence.py"),
    ("backend/scripts/integrity_gate.py", "scripts/integrity_gate.py"),
    ("backend/tests/test_editor_integrity_gate.py", "scripts/test_editor_integrity_gate.py"),
    ("backend/tests/test_spec_doc_consistency.py", "scripts/test_spec_doc_consistency.py"),
    ("backend/scripts/evaluate_model.py", "scripts/evaluate_model.py"),
    ("backend/scripts/train_efficientnet_b0.py", "scripts/train_efficientnet_b0.py"),
]

# Artefacts that require the APTOS dataset to regenerate. Their absence is
# reported but does not fail assembly, so a package can be built on a machine
# without the 9.51 GB download. They are never substituted or invented.
PENDING_WITHOUT_DATASET = {
    "logs_and_metrics/validation_test_results.csv":
        "regenerate with: python scripts/generate_validation_evidence.py <aptos>/train_images",
    "logs_and_metrics/gate_downsampling_verification.json":
        "regenerate with: python scripts/verify_gate_downsampling.py <aptos>/train_images",
    "logs_and_metrics/dataset_split_audit.json":
        "regenerate with: python scripts/build_clean_split.py <aptos>/train_images",
}

SCREENSHOTS_SRC = os.path.join(CHAPTER4, "screenshots")
SCREENSHOTS_DST = os.path.join(PACKAGE, "screenshots")

# Source trees mirrored wholesale. The reviewer asked for the backend and
# frontend integration sources and the complete test suite, without which
# objectives a (architecture), b (validation), g (integration) and i
# (functional testing) cannot be reproduced from the package alone.
SOURCE_TREES = [
    ("backend/app", "source/backend/app", (".py",)),
    ("backend/tests", "source/backend/tests", (".py",)),
    ("frontend/src", "source/frontend/src", (".ts", ".tsx", ".css")),
]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="Report differences without writing anything")
    args = parser.parse_args()

    print("=" * 78)
    print("ASSEMBLE CHAPTER 4 SUBMISSION PACKAGE" + ("  (check only)" if args.check else ""))
    print("=" * 78)

    # The checkpoint is the one artefact worth verifying rather than trusting.
    ckpt = os.path.join(REPO_ROOT, "backend", "models", "weights", "efficientnet_b0_dr.pth")
    if os.path.exists(ckpt):
        actual = sha256_file(ckpt)
        if actual != EXPECTED_CHECKPOINT_SHA256:
            print(f"[ABORT] Checkpoint digest mismatch.\n"
                  f"        expected {EXPECTED_CHECKPOINT_SHA256}\n"
                  f"        found    {actual}\n"
                  f"        Refusing to package weights that are not the evaluated ones.")
            return 1
        print(f"Checkpoint verified: {actual[:16]}...")
    else:
        print(f"[ABORT] Checkpoint not found at {ckpt}")
        return 1

    copied = updated = unchanged = missing = pending = 0
    for rel_src, rel_dst in LAYOUT:
        src = os.path.join(REPO_ROOT, rel_src)
        dst = os.path.join(PACKAGE, rel_dst)

        if not os.path.exists(src):
            if rel_dst in PENDING_WITHOUT_DATASET:
                print(f"  [PENDING] {rel_dst}")
                print(f"            {PENDING_WITHOUT_DATASET[rel_dst]}")
                pending += 1
            else:
                print(f"  [MISSING] {rel_src}")
                missing += 1
            continue

        if os.path.exists(dst) and sha256_file(src) == sha256_file(dst):
            unchanged += 1
            continue

        verb = "would update" if args.check else "updated"
        if not os.path.exists(dst):
            verb = "would add" if args.check else "added"
            copied += 1
        else:
            updated += 1

        if not args.check:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
        print(f"  [{verb}] {rel_dst}")

    # Source trees.
    src_files = 0
    for rel_src, rel_dst, exts in SOURCE_TREES:
        root = os.path.join(REPO_ROOT, rel_src)
        if not os.path.isdir(root):
            print(f"  [MISSING] {rel_src}/")
            missing += 1
            continue
        for base, dirs, names in os.walk(root):
            dirs[:] = [d for d in dirs if d not in ("__pycache__", "node_modules", ".pytest_cache")]
            for name in sorted(names):
                if not name.endswith(exts):
                    continue
                src = os.path.join(base, name)
                dst = os.path.join(PACKAGE, rel_dst,
                                   os.path.relpath(src, root).replace("\\", "/"))
                if os.path.exists(dst) and sha256_file(src) == sha256_file(dst):
                    continue
                if not args.check:
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(src, dst)
                src_files += 1
    if src_files:
        print(f"  [{'would sync' if args.check else 'synced'}] {src_files} source file(s)")

    # Screenshots are a directory, mirrored wholesale.
    shots = 0
    if os.path.isdir(SCREENSHOTS_SRC):
        for name in sorted(os.listdir(SCREENSHOTS_SRC)):
            if not name.lower().endswith(".png"):
                continue
            src = os.path.join(SCREENSHOTS_SRC, name)
            dst = os.path.join(SCREENSHOTS_DST, name)
            if os.path.exists(dst) and sha256_file(src) == sha256_file(dst):
                continue
            if not args.check:
                os.makedirs(SCREENSHOTS_DST, exist_ok=True)
                shutil.copy2(src, dst)
            print(f"  [{'would sync' if args.check else 'synced'}] screenshots/{name}")
            shots += 1

    print("-" * 78)
    print(f"added {copied}   updated {updated}   unchanged {unchanged}   "
          f"source {src_files}   screenshots {shots}   pending {pending}   missing {missing}")

    if missing:
        print(f"\n[INCOMPLETE] {missing} artefact(s) absent. Produce them with the scripts named")
        print("             in this file's docstring. Nothing is generated here.")
        return 1

    if args.check and (copied or updated or shots or src_files):
        print("\n[STALE] The package differs from the source artefacts. "
              "Re-run without --check.")
        return 1

    print("\n[OK] Package matches the committed artefacts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
