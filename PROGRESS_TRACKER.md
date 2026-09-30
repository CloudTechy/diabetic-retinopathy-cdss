# Diabetic Retinopathy CDSS — Engineering Progress Tracker

**Research Project**: AI-Based Clinical Decision Support System for Early Detection of Diabetic Retinopathy
**Researcher / Author**: Onyekelu Chukwuebuka Elochukwu (2024516020FN)
**Programme**: PGD Computer Science, Faculty of Physical Sciences
**Evidence Basis**: Genuine APTOS 2019 training run, 2026-09-29 (Colab Tesla T4, checkpoint `8ee14d75…`)
**Status**: Chapter 4 evidence complete and internally consistent. **One open item: CPU end-to-end benchmark.**

---

## 1. Research Objectives Traceability

| Objective | Scope & Focus | Evidence | Status |
| :---: | :--- | :--- | :---: |
| **c** | Collect & partition retinal fundus dataset | [`dataset_audit.md`](docs/chapter4/dataset_audit.md), [`dataset_split_manifest.csv`](docs/chapter4/dataset_split_manifest.csv) | ✅ Complete (with disclosure) |
| **d** | Design EfficientNet-B0 for 5-grade ICDR classification | [`preprocessing_and_augmentation_spec.md`](docs/chapter4/preprocessing_and_augmentation_spec.md), [`architecture.md`](docs/chapter4/architecture.md) | ✅ Complete |
| **e** | Implement CNN with compound scaling & frozen inference | [`efficientnet_b0_dr.pth`](backend/models/weights/efficientnet_b0_dr.pth), [`checkpoint_manifest.md`](docs/chapter4/checkpoint_manifest.md) | ✅ Complete |
| **f** | Implement 3-stage validation pipeline (Gates 1–3) | [`backend/app/services/validation/`](backend/app/services/validation/), [`validation_module_spec.md`](docs/chapter4/validation_module_spec.md) | ✅ Complete |
| **g** | CDSS architecture with clinician-in-the-loop governance | [`architecture.md`](docs/chapter4/architecture.md), [`database_schema.md`](docs/chapter4/database_schema.md), [`api_contract.md`](docs/chapter4/api_contract.md) | ✅ Complete |
| **h** | Evaluate on held-out cohort ($N = 525$) | [`model_evaluation_report.md`](docs/chapter4/model_evaluation_report.md), [`held_out_predictions.csv`](docs/chapter4/held_out_predictions.csv), [`confusion_matrix.png`](docs/chapter4/confusion_matrix.png) | ✅ Complete |
| **i** | Benchmark efficiency, system test & governance | [`system_test_report.md`](docs/chapter4/system_test_report.md), [`resource_benchmark.md`](docs/chapter4/resource_benchmark.md) | ✅ Complete |

---

## 2. Empirical Results ($N = 525$ held-out cohort)

### Primary metrics

| Metric | Value |
| :--- | :---: |
| **Quadratic Weighted Kappa ($\kappa$)** | **0.8658** |
| Exact 5-class accuracy | 84.00% (441 / 525) |
| Within-one-grade agreement | 93.71% |
| Macro F1 | 0.7031 |

### Clinical operating points — the numbers to lead with

| Endpoint | Sensitivity | Specificity | NPV |
| :--- | :---: | :---: | :---: |
| **Referable DR** (grade $\ge$ 2) | **91.2%** (81.5–90.5) | **96.3%** (93.7–97.9) | 91.2% |
| **Sight-threatening DR** (grade $\ge$ 3) | **68.2%** (72.2–89.4) | 91.0% (88.0–93.2) | **97.1%** |
| Any DR (grade $\ge$ 1) | 97.8% | 98.9% | 97.8% |

Of 224 referable cases, 30 were missed — 28 of them Grade 2 (the mildest referable grade). Exactly **two** sight-threatening cases were released as non-referable.

### Per-class sensitivity

| Grade | Class | Support | Sensitivity | 95% CI |
| :---: | :--- | :---: | :---: | :---: |
| 0 | No Apparent DR | 270 | **98.9%** | 96.8–99.6 |
| 1 | Mild NPDR | 55 | 72.7% | 59.8–82.7 |
| 2 | Moderate NPDR | 150 | **53.3%** | 45.4–61.1 |
| 3 | Severe NPDR | 29 | 58.6% | 40.7–74.5 |
| 4 | Proliferative DR | 45 | 62.2% | 47.6–74.9 |

> Recompute every figure above from committed artefacts, standard library only:
> ```bash
> python backend/scripts/analyze_clinical_metrics.py
> ```

---

## 3. Training Run

| Property | Value |
| :--- | :--- |
| Dataset | APTOS 2019 — 3,662 images, split 2,453 / 526 / 525 (grade-stratified) |
| Initialisation | EfficientNet-B0, ImageNet `IMAGENET1K_V1` |
| Epochs | 15; **best = epoch 11** (val $\kappa$ = 0.9130) |
| Optimiser | AdamW, lr $10^{-4}$, wd $10^{-4}$, CosineAnnealingLR |
| Loss | Class-weighted cross-entropy |
| Hardware | Google Colab Tesla T4, PyTorch 2.11.0+cu128 |
| Wall-clock | 2,762 s (~210 s/epoch) |
| Checkpoint SHA-256 | `8ee14d7591a8e6a1b86c15416a77375a198bd49399b3977a3de79a00e3dd14fa` |
| Checkpoint size | 15.60 MB (16,358,249 bytes) |
| Total parameters | 4,013,953 (0 trainable at inference) |

---

## 4. Model & Runtime

- **Grad-CAM target layer:** `features.8` (1,280-channel final conv).
- **Preprocessing:** `Resize((224,224))` → `ToTensor()` → ImageNet normalise. Identical in training and inference — **no train/serve skew**.
- **Fail-closed loading:** a missing, corrupt or digest-mismatched checkpoint raises `ModelCheckpointError`. The engine never serves untrained weights, and the simulated engine is opt-in by name only (`AI_INFERENCE_ENGINE=mock`).
- **End-to-end CPU latency:** mean **212.54 ms**, median 179.68 ms, P95 **373.39 ms** over 30 held-out images (4-thread x86_64, no GPU).
- **vs the 315.25 ms pre-optimisation baseline:** raw 1.91×, but the two runs were on different Colab CPUs — unchanged-code stages were themselves 1.14–1.40× faster. Attributable improvement is **~1.4–1.7×**. Immune to that caveat: validation fell from **59.1% to 43.9%** of the request.
- **Where it goes:** validation gates 2+3 = **72.42 ms (43.9%)**; PNG encode = 30.18 ms (18.3%); model forward pass = 29.24 ms (17.7%). Inference is not the bottleneck.
- **Grad-CAM composition:** vectorised, **307.87 → 28.86 ms** (10.7x), byte-identical output.
- **Validation gates:** statistics on a nearest-neighbour subsample. In the deployment benchmark gates 2+3 went **186.42 → 72.42 ms**. **Verified on all 3,662 real APTOS images: zero verdict changes, Laplacian deviation exactly 0.0.**
- **GPU reference:** 8.34 ms Tesla T4 forward pass — training-environment comparison only, not a deployment figure.

---

## 5. Test Suite

**157 passed, 2 skipped** (the skip requires PyTorch, absent from the local venv). Includes 8 tests guarding the fail-closed inference invariant, 13 asserting Grad-CAM render equivalence, and 38 asserting the validation gates reach the same verdict when subsampled.

```bash
cd backend && .venv/Scripts/python.exe -m pytest tests/ -q
```

---

## 6. Open Items

> **Resolved 2026-09-30:** the three scripts that generated the original fabricated evidence
> (`create_evaluated_checkpoint.py`, `generate_dataset_manifest.py`, `prepare_submission_package.py`)
> have been deleted. Two of them wrote to paths holding genuine evidence, so running either would have
> silently destroyed it. Package assembly is now `assemble_submission_package.py`, which only copies
> committed artefacts and verifies the checkpoint digest.


| # | Item | Why it matters |
| :---: | :--- | :--- |
| 1 | **Decode once, share one reduced copy across gates 1–3** | `gate2` is still the largest stage at 56.06 ms (34.0%). The residual is not the statistics — it is the reduction itself, plus each gate independently converting the full-resolution image. Measured, not speculative. |
| 2 | Re-split with byte-hash duplicate grouping, then re-train | 27/525 held-out images are byte-identical to a training image because `duplicated_info.csv` is absent from the Kaggle download. Measured effect: nil (clean-subset $\kappa$ = 0.877818 vs 0.865832 full). Disclosed in [`dataset_audit.md`](docs/chapter4/dataset_audit.md) §4; remediation deferred as it would invalidate the hash-verified checkpoint for no measurable gain. |
| 3 | ~~Refresh UI screenshots~~ **DONE** | Recaptured 2026-09-30 from the current build with sanitised demo identity, and the displayed sharpness threshold corrected to match the backend. The score values shown remain demonstration fixtures: these figures evidence interface behaviour, not model performance. |
| 4 | External-cohort validation | No evaluation on any dataset other than APTOS 2019. No generalisation claim is made. |

---

## 7. Evidence Register (`docs/chapter4/`)

**Raw artefacts from the training and benchmark runs** — [`cpu_end_to_end_benchmark.json`](docs/chapter4/cpu_end_to_end_benchmark.json), [`cpu_end_to_end_benchmark.csv`](docs/chapter4/cpu_end_to_end_benchmark.csv), [`training_execution.log`](docs/chapter4/training_execution.log), [`epoch_history.csv`](docs/chapter4/epoch_history.csv), [`training_summary.json`](docs/chapter4/training_summary.json), [`held_out_predictions.csv`](docs/chapter4/held_out_predictions.csv), [`evaluation_summary.json`](docs/chapter4/evaluation_summary.json), [`benchmark_timings.csv`](docs/chapter4/benchmark_timings.csv), [`benchmark_summary.json`](docs/chapter4/benchmark_summary.json), [`dataset_split_manifest.csv`](docs/chapter4/dataset_split_manifest.csv), [`clinical_metrics.json`](docs/chapter4/clinical_metrics.json), [`confusion_matrix.png`](docs/chapter4/confusion_matrix.png), [`learning_curves.png`](docs/chapter4/learning_curves.png)

**Analysis documents** — [`dataset_audit.md`](docs/chapter4/dataset_audit.md), [`preprocessing_and_augmentation_spec.md`](docs/chapter4/preprocessing_and_augmentation_spec.md), [`training_protocol.md`](docs/chapter4/training_protocol.md), [`training_environment.md`](docs/chapter4/training_environment.md), [`checkpoint_manifest.md`](docs/chapter4/checkpoint_manifest.md), [`model_evaluation_report.md`](docs/chapter4/model_evaluation_report.md), [`resource_benchmark.md`](docs/chapter4/resource_benchmark.md), [`known_limitations.md`](docs/chapter4/known_limitations.md), [`validation_module_spec.md`](docs/chapter4/validation_module_spec.md), [`system_test_report.md`](docs/chapter4/system_test_report.md), [`requirements_test_matrix.md`](docs/chapter4/requirements_test_matrix.md), [`architecture.md`](docs/chapter4/architecture.md), [`database_schema.md`](docs/chapter4/database_schema.md), [`api_contract.md`](docs/chapter4/api_contract.md), [`reproducibility_runbook.md`](docs/chapter4/reproducibility_runbook.md), [`objective_traceability_matrix.md`](docs/chapter4/objective_traceability_matrix.md), [`implementation_status.md`](docs/chapter4/implementation_status.md), [`independent_thesis_qa_gate_audit.md`](docs/chapter4/independent_thesis_qa_gate_audit.md)

---

## 8. Deployment

- **Frontend (Vercel):** `https://frontend-six-psi-77.vercel.app`
- **Scope:** research prototype supporting a PGD dissertation. Not a medical device; no regulatory clearance; not for patient care.
