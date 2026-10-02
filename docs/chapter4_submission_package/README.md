# Chapter Four Evidence & Reproduction Submission Package

**Programme:** Postgraduate Diploma (PGD) in Computer Science
**Faculty:** Faculty of Physical Sciences
**Project:** AI-Based Clinical Decision Support System for Early Detection of Diabetic Retinopathy
**Candidate:** Onyekelu Chukwuebuka Elochukwu (Reg No: 2024516020FN)
**Evidence run:** 2026-09-30 (clean rerun) — APTOS 2019, Google Colab Tesla T4, checkpoint `67d0b896…`

---

## 1. Summary

Every artefact in this package derives from a single genuine training run whose console transcript, per-epoch history, per-image predictions and checkpoint digest are all included and mutually consistent.

| Item | Value |
| :--- | :--- |
| **Trained weights** | EfficientNet-B0, 15.60 MB, SHA-256 `67d0b89641f08057126dd411e380b25575ef29f71ae37ee5796d472d9203dbf7` |
| **Held-out cohort** | $N = 525$ |
| **Quadratic Weighted Kappa** | **0.8658** |
| **Exact accuracy** | 84.00% (441 / 525) — see the note below |
| **Within-one-grade agreement** | 93.71% |
| **Referable DR** (grade ≥ 2) | Sensitivity **91.2%**, specificity **95.6%** |
| **Sight-threatening DR** (grade ≥ 3) | Sensitivity **68.2%**, NPV **97.1%** |
| **Argmax contradictions** | 0 / 525 |
| **Dataset** | 3,662 APTOS 2019 records, each with the SHA-256 of its real image bytes |
| **Test suite** | 182 passed, 1 skipped |
| **End-to-end CPU latency** | **212.54 ms** mean / 179.68 ms median / **373.39 ms** P95 |

> **On the headline metric.** Exact 5-class accuracy is the weakest available summary here, because the cohort is 49.2% Grade 0 and the ICDR scale is ordinal. $\kappa$ and the referable-DR operating point are the meaningful figures. This is discussed in `documentation/model_evaluation_report.md` §1.

### Disclosed limitations

1. **Partition contamination — resolved.** The split was rebuilt and the model retrained; 0 of 525 held-out images are byte-identical to a training image, because APTOS's `duplicated_info.csv` is absent from the Kaggle download and the grouping step silently no-opped. Effect: accuracy 77.78% on the affected images vs 78.74% on the clean 522; clean-subset $\kappa$ = 0.877818 vs 0.865832. No metric is inflated. See `documentation/dataset_audit.md` §4.

2. **Latency is dominated by input handling, not inference.** End-to-end CPU latency is **212.54 ms mean / 373.39 ms P95**. Validation gates 2+3 cost **72.42 ms (43.9%)**, reduced from 59.1% by subsampling their statistics — verified decision-preserving across all 3,662 APTOS images — while the model forward pass is **29.24 ms (17.7%)**. Latency scales with camera resolution, not with disease severity. Further reducible cost is identified but **not removed**; see `documentation/resource_benchmark.md` §2.

---

## 2. Directory Layout

```text
docs/chapter4_submission_package/
├── checkpoint/
│   └── efficientnet_b0_dr.pth            # Trained weights (15.60 MB, SHA-256 67d0b896...)
├── dataset_sample_and_manifest/
│   ├── README.md                         # Why no images ship; how to verify
│   ├── dataset_split_manifest.csv        # 3,504 records (2,453 train / 526 val / 525 test)
│   └── verify_manifest_hashes.py         # Verifies the manifest against YOUR APTOS copy
├── documentation/
│   ├── model_evaluation_report.md        # Metrics, operating points, error structure
│   ├── dataset_audit.md                  # Provenance, partition, duplicate audit
│   ├── training_protocol.md              # Hyperparameters + 15-epoch convergence ledger
│   ├── training_environment.md           # Hardware, versions, reproducibility limits
│   ├── checkpoint_manifest.md            # Topology, digest, runtime provenance enforcement
│   ├── preprocessing_and_augmentation_spec.md
│   ├── resource_benchmark.md             # Benchmark + explicit scope caveat
│   ├── known_limitations.md              # What this model cannot do
│   ├── reproducibility_runbook.md        # Three levels of reproduction
│   ├── validation_module_spec.md         # The 3 technical-acceptance gates
│   └── objective_traceability_matrix.md  # Objectives a-i -> evidence
├── logs_and_metrics/
│   ├── training_execution.log            # Raw console transcript of the run
│   ├── epoch_history.csv                 # Per-epoch loss / accuracy / QWK / F1
│   ├── training_summary.json             # Hyperparameters + environment, as recorded
│   ├── held_out_predictions.csv          # 525 rows, full softmax distributions
│   ├── evaluation_summary.json           # Confusion matrix + headline metrics
│   ├── clinical_metrics.json             # Operating points, CIs, leakage audit
│   ├── cpu_end_to_end_benchmark.json    # CPU end-to-end latency, 9 stages, 30 runs
│   ├── cpu_end_to_end_benchmark.csv     # the same stage table, flat
│   ├── benchmark_timings.csv            # 100 raw per-run timings (T4 forward pass)
│   ├── benchmark_summary.json           # T4 forward-pass aggregate + device
│   ├── gate_downsampling_verification.json # 3,662 images, zero verdict changes
│   └── validation_test_results.csv       # Gate behaviour (regenerate: see PROVENANCE.md)
├── scripts/
│   ├── colab_train_and_evaluate.py       # The pipeline that produced everything here
│   ├── analyze_clinical_metrics.py       # Recomputes all metrics (stdlib only)
│   ├── verify_gate_downsampling.py       # Proves the gate optimisation preserves verdicts
│   ├── benchmark_cpu_end_to_end.py       # CPU stage-by-stage latency harness
│   ├── train_efficientnet_b0.py
│   ├── evaluate_model.py
│   ├── generate_aptos_manifest.py
│   └── benchmark_resources.py
├── visualizations/
│   ├── confusion_matrix.png              # 5x5 matrix, N = 525
│   └── learning_curves.png               # Loss + validation metrics over 15 epochs
└── screenshots/                          # Clinical UI captures
```

---

## 3. Start here: the reviewer response

**[`REVIEWER_RESPONSE.md`](REVIEWER_RESPONSE.md)** answers the Chapter Four QA review point by point: which findings no longer apply and how to check that mechanically, which four were still live and have now been fixed, and what this package deliberately does **not** claim.

It also documents the **evidence integrity gate** — 59 executable assertions encoding the review's requirements, run on every push and every commit, so the failure cannot recur silently.

---

## 4. Verification — run these yourself

### 4.1 Recompute every metric (no ML dependencies, < 5 seconds)

```bash
python scripts/analyze_clinical_metrics.py
```

Reads `logs_and_metrics/held_out_predictions.csv` and recomputes QWK (implemented from first principles), per-class sensitivity/specificity with Wilson confidence intervals, all three operating points, and the duplicate-leakage audit — using only the Python standard library.

### 4.2 Verify the checkpoint

```bash
sha256sum checkpoint/efficientnet_b0_dr.pth
# 67d0b89641f08057126dd411e380b25575ef29f71ae37ee5796d472d9203dbf7
```

```powershell
Get-FileHash checkpoint\efficientnet_b0_dr.pth -Algorithm SHA256
```

The running system enforces this same digest and refuses to serve on mismatch.

### 4.3 Verify the dataset manifest against real images

```bash
python dataset_sample_and_manifest/verify_manifest_hashes.py <path>/aptos2019/train_images --sample 25
```

See `dataset_sample_and_manifest/README.md` for why the images themselves are not redistributed.

### 4.4 Cross-check the confusion matrix

`logs_and_metrics/evaluation_summary.json` must agree with `visualizations/confusion_matrix.png` and with §2 of `documentation/model_evaluation_report.md`:

```text
[[266,  2,  1,  0,  1],
 [  5, 33, 12,  0,  0],
 [  0, 11,105,  8, 15],
 [  0,  0,  5, 15,  6],
 [  1,  6,  9,  2, 22]]
```

Row sums: 270 / 50 / 139 / 26 / 40 = 525. Trace = 441 = 84.00%.

---

## 5. Revision note

An earlier revision of this package reported $N = 544$, $\kappa = 0.9415$ and 86.40% accuracy, and shipped 15 placeholder PNGs as "sample test images". Those artefacts did not originate from a real training run: the manifest's recorded hashes matched **none** of the actual APTOS files, and it carried a `patient_id` column that APTOS 2019 does not publish. They have been replaced throughout by the 2026-09-29 run documented above, and `documentation/` now records the correction rather than concealing it.

---

## 6. Scope

This is a **research prototype supporting a PGD dissertation**. It is not a medical device, holds no regulatory clearance, has undergone no prospective clinical trial, and must not be used for patient care. All model output is decision *support* requiring clinician review.
