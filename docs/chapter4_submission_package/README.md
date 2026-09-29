# Chapter Four Evidence & Reproduction Submission Package

**Programme:** Postgraduate Diploma (PGD) in Computer Science  
**Faculty:** Faculty of Physical Sciences  
**Project:** AI-Based Clinical Decision Support System for Early Detection of Diabetic Retinopathy  
**Candidate:** Onyekelu Chukwuebuka Elochukwu (Reg No: 2024516020FN)  
**Submission Date:** September 2026  

---

## 1. Executive Summary & Verification Notice

This self-contained submission folder provides empirical, reproducible evidence addressing all 10 critical review recommendations and 14 required artifacts for Chapter Four evaluation.

Every artifact has been computed directly from verifiable assets:
- **Trained Neural Network Weights:** EfficientNet-B0 (15.60 MB, SHA-256 verified).
- **Exact Held-Out Test Evaluation ($N = 544$):** Quadratic Weighted Kappa $\kappa = 0.9415$, Multi-Class Accuracy $86.40\%$, 0 argmax contradictions.
- **Dataset Traceability:** 3,662 APTOS 2019 records with on-disk sample test image SHA-256 byte verification.
- **Inference Latency Benchmark:** 100 raw consecutive passes averaging $95.76\text{ ms} = 0.09576\text{ s}$ per patient encounter (standard CPU, batch size 1).
- **Clinical Web UI:** 9 high-resolution full-screen captures with PGD attribution, strict non-diagnostic boundary notices, and active triage/review workflows.

---

## 2. Directory Layout & Artifact Map

```text
docs/chapter4_submission_package/
├── checkpoint/
│   └── efficientnet_b0_dr.pth            # Trained PyTorch weights (15.60 MB, SHA-256 verified)
├── dataset_sample_and_manifest/
│   ├── dataset_split_manifest.csv        # 3,662 records (2,567 train, 551 val, 544 test)
│   ├── sample_test_images/               # 15 sample held-out PNGs matching manifest byte hashes
│   ├── verify_manifest_hashes.py         # One-command SHA-256 byte integrity verification
│   └── hash_verification_output.txt      # 100% hash match log output
├── documentation/
│   ├── objective_traceability_matrix.md  # Bidirectional mapping across 9 approved objectives
│   ├── checkpoint_manifest.md            # Weights provenance, layer hooks (features.8), param count
│   ├── model_evaluation_report.md        # Single-source metrics, confusion matrix, discussion
│   ├── validation_module_spec.md         # 3-gate specs (0.65-1.65 aspect ratio, sharpness >= 100)
│   ├── resource_benchmark.md             # Benchmark summary with verified arithmetic conversion
│   ├── training_protocol.md              # 15-epoch hyperparameter protocol and loss curves
│   └── system_test_report.md             # 8-screen end-to-end integration and API test suite
├── logs_and_metrics/
│   ├── held_out_predictions.csv          # 544 itemized rows with strict argmax validation
│   ├── epoch_history.csv                 # 15-epoch training and validation loss, accuracy, and QWK
│   ├── training_execution.log            # Raw epoch-by-epoch terminal training logs
│   ├── benchmark_timings.csv             # 100 individual inference run timings
│   └── validation_test_results.csv       # Test cases across 3 gates (valid, corrupt, blurry)
├── visualizations/
│   ├── confusion_matrix.png              # 5-class confusion matrix matching 544 held-out set
│   └── learning_curves.png               # 15-epoch convergence and QWK trajectory
├── screenshots/
│   ├── 01_signin_screen.png              # PGD Computer Science attribution, credential entry
│   ├── 02_clinical_dashboard.png         # Prioritized triage worklist & patient encounter table
│   ├── 03_new_assessment_upload.png     # Pre-flight upload interface with boundary disclaimers
│   ├── 04_validation_stepper_passed.png  # Real-time 3-stage validation passing stepper
│   ├── 04b_validation_stepper_rejected.png# Fail-closed rejection preventing model inference
│   ├── 05_decision_support_workspace.png # Grad-CAM viewer (Viridis, 60% opacity) & score breakdown
│   ├── 06_professional_review_modal.png  # Mandatory clinician concordance sign-off modal
│   ├── 07_completed_assessment_record.png# Tamper-evident finalized clinical assessment record
│   └── 08_record_history_audit.png       # Audit trail and filterable clinical search history
└── scripts/
    ├── evaluate_model.py                 # Evaluates checkpoint and recomputes confusion matrix
    ├── train_efficientnet_b0.py          # Supervised training loop with cosine annealing
    ├── benchmark_resources.py            # Automated 100-run inference latency benchmarking
    └── generate_aptos_manifest.py        # Manifest generator with duplicate grouping
```

---

## 3. Quick Verification Instructions

### A. Verify Model Checkpoint SHA-256 Digest
```bash
python -c "import hashlib; print(hashlib.sha256(open('docs/chapter4_submission_package/checkpoint/efficientnet_b0_dr.pth', 'rb').read()).hexdigest())"
# Expected output:
# 0d443fa065528b2a1d24baea7bf8d8bf71a6203b22b817585374a7becd547c07
```

### B. Verify Sample Test Image Hashes against Manifest
```bash
python docs/chapter4_submission_package/dataset_sample_and_manifest/verify_manifest_hashes.py
# Output: 15/15 MATCHED (100%), 0 byte drift
```

### C. Verify Zero Argmax Violations in Predictions
```bash
python -c "import csv; rows = list(csv.DictReader(open('docs/chapter4_submission_package/logs_and_metrics/held_out_predictions.csv', encoding='utf-8'))); violations = [r['image_id'] for r in rows if int(r['predicted_grade']) != [float(r[f'score_grade_{k}']) for k in range(5)].index(max([float(r[f'score_grade_{k}']) for k in range(5)]))]; print(f'Violations: {len(violations)} / {len(rows)}')"
# Expected output:
# Violations: 0 / 544
```

### D. Verify 100-Run Benchmark Arithmetic
```bash
python -c "import csv; timings = [float(r['inference_time_ms']) for r in csv.DictReader(open('docs/chapter4_submission_package/logs_and_metrics/benchmark_timings.csv', encoding='utf-8'))]; mean_ms = sum(timings)/len(timings); print(f'Runs: {len(timings)}, Mean: {mean_ms:.2f} ms = {mean_ms/1000.0:.5f} s (< 0.25000 s threshold)')"
# Expected output:
# Runs: 100, Mean: 95.76 ms = 0.09576 s (< 0.25000 s threshold)
```

---

## 4. Academic Integrity & Honest Research Boundaries

1. **Non-Diagnostic Framing:** In adherence to clinical governance, outputs are designated as *“Model-Generated Class Scores”* for decision support.
2. **Mild NPDR Sensitivity (69.64%):** Transparently discussed as a clinical limitation due to microaneurysm subtlety (10–30 pixels).
3. **No Simulation Presets:** All demo buttons and mock simulation presets have been purged from the user interface.
