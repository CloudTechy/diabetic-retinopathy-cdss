# Chapter Four Evidence & Reproduction Submission Package

**Programme:** Postgraduate Diploma (PGD) in Computer Science
**Faculty:** Faculty of Physical Sciences
**Project:** AI-Based Clinical Decision Support System for Early Detection of Diabetic Retinopathy
**Candidate:** Onyekelu Chukwuebuka Elochukwu (Reg No: 2024516020FN)
**Evidence run:** 2026-09-29 — APTOS 2019, Google Colab Tesla T4, checkpoint `8ee14d75…`

---

## 1. Summary

Every artefact in this package derives from a single genuine training run whose console transcript, per-epoch history, per-image predictions and checkpoint digest are all included and mutually consistent.

| Item | Value |
| :--- | :--- |
| **Trained weights** | EfficientNet-B0, 15.60 MB, SHA-256 `8ee14d7591a8e6a1b86c15416a77375a198bd49399b3977a3de79a00e3dd14fa` |
| **Held-out cohort** | $N = 549$ |
| **Quadratic Weighted Kappa** | **0.8777** |
| **Exact accuracy** | 78.69% (432 / 549) — see the note below |
| **Within-one-grade agreement** | 92.71% |
| **Referable DR** (grade ≥ 2) | Sensitivity **86.6%**, specificity **96.3%** |
| **Sight-threatening DR** (grade ≥ 3) | Sensitivity **82.4%**, NPV **97.1%** |
| **Argmax contradictions** | 0 / 549 |
| **Dataset** | 3,662 APTOS 2019 records, each with the SHA-256 of its real image bytes |
| **Test suite** | 58 passed, 1 skipped |

> **On the headline metric.** Exact 5-class accuracy is the weakest available summary here, because the cohort is 49.2% Grade 0 and the ICDR scale is ordinal. $\kappa$ and the referable-DR operating point are the meaningful figures. This is discussed in `documentation/model_evaluation_report.md` §1.

### Two disclosed limitations

1. **Duplicate leakage (measured, immaterial).** 27 of 549 held-out images are byte-identical to a training image, because APTOS's `duplicated_info.csv` is absent from the Kaggle download and the grouping step silently no-opped. Effect: accuracy 77.78% on the affected images vs 78.74% on the clean 522; clean-subset $\kappa$ = 0.877818 vs 0.877747. No metric is inflated. See `documentation/dataset_audit.md` §4.

2. **Latency is not measured on the deployment target.** The 8.36 ms benchmark is a **Tesla T4 forward pass**. The system deploys on **CPU** and the figure excludes decode, validation, preprocessing and Grad-CAM. **No clinical-workstation latency claim is made.** See `documentation/resource_benchmark.md` §4.

---

## 2. Directory Layout

```text
docs/chapter4_submission_package/
├── checkpoint/
│   └── efficientnet_b0_dr.pth            # Trained weights (15.60 MB, SHA-256 8ee14d75...)
├── dataset_sample_and_manifest/
│   ├── README.md                         # Why no images ship; how to verify
│   ├── dataset_split_manifest.csv        # 3,662 records (2,563 train / 550 val / 549 test)
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
│   ├── held_out_predictions.csv          # 549 rows, full softmax distributions
│   ├── evaluation_summary.json           # Confusion matrix + headline metrics
│   ├── clinical_metrics.json             # Operating points, CIs, leakage audit
│   ├── benchmark_timings.csv             # 100 raw per-run timings
│   ├── benchmark_summary.json            # Benchmark aggregate + device
│   └── validation_test_results.csv       # Gate behaviour across image types
├── scripts/
│   ├── colab_train_and_evaluate.py       # The pipeline that produced everything here
│   ├── analyze_clinical_metrics.py       # Recomputes all metrics (stdlib only)
│   ├── benchmark_cpu_end_to_end.py       # CPU stage-by-stage latency harness
│   ├── train_efficientnet_b0.py
│   ├── evaluate_model.py
│   ├── generate_aptos_manifest.py
│   └── benchmark_resources.py
├── visualizations/
│   ├── confusion_matrix.png              # 5x5 matrix, N = 549
│   └── learning_curves.png               # Loss + validation metrics over 15 epochs
└── screenshots/                          # Clinical UI captures
```

---

## 3. Verification — start here

### 3.1 Recompute every metric (no ML dependencies, < 5 seconds)

```bash
python scripts/analyze_clinical_metrics.py
```

Reads `logs_and_metrics/held_out_predictions.csv` and recomputes QWK (implemented from first principles), per-class sensitivity/specificity with Wilson confidence intervals, all three operating points, and the duplicate-leakage audit — using only the Python standard library.

### 3.2 Verify the checkpoint

```bash
sha256sum checkpoint/efficientnet_b0_dr.pth
# 8ee14d7591a8e6a1b86c15416a77375a198bd49399b3977a3de79a00e3dd14fa
```

```powershell
Get-FileHash checkpoint\efficientnet_b0_dr.pth -Algorithm SHA256
```

The running system enforces this same digest and refuses to serve on mismatch.

### 3.3 Verify the dataset manifest against real images

```bash
python dataset_sample_and_manifest/verify_manifest_hashes.py <path>/aptos2019/train_images --sample 25
```

See `dataset_sample_and_manifest/README.md` for why the images themselves are not redistributed.

### 3.4 Cross-check the confusion matrix

`logs_and_metrics/evaluation_summary.json` must agree with `visualizations/confusion_matrix.png` and with §2 of `documentation/model_evaluation_report.md`:

```text
[[267,  3,  0,  0,  0],
 [  3, 40, 11,  0,  1],
 [  3, 25, 80, 15, 27],
 [  0,  1,  4, 17,  7],
 [  0,  1,  7,  9, 28]]
```

Row sums: 270 / 55 / 150 / 29 / 45 = 549. Trace = 432 = 78.69%.

---

## 4. Revision note

An earlier revision of this package reported $N = 544$, $\kappa = 0.9415$ and 86.40% accuracy, and shipped 15 placeholder PNGs as "sample test images". Those artefacts did not originate from a real training run: the manifest's recorded hashes matched **none** of the actual APTOS files, and it carried a `patient_id` column that APTOS 2019 does not publish. They have been replaced throughout by the 2026-09-29 run documented above, and `documentation/` now records the correction rather than concealing it.

---

## 5. Scope

This is a **research prototype supporting a PGD dissertation**. It is not a medical device, holds no regulatory clearance, has undergone no prospective clinical trial, and must not be used for patient care. All model output is decision *support* requiring clinician review.
