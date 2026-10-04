# Chapter 4 Implementation Status & Evidence Ledger

## Metadata & Traceability
- **Research Project:** AI-Based Clinical Decision Support System for Early Detection of Diabetic Retinopathy Using Retinal Image Classification
- **Author / Researcher:** Onyekelu Chukwuebuka Elochukwu (2024516020FN)
- **Supervising Department:** Department of Computer Science / Engineering
- **Date Generated:** 2026-09-28
- **Evaluation Status:** Fixed Evaluated EfficientNet-B0 Model Pipeline
- **Hardware/Software Environment:** Windows 11 x64; the test environment is recorded in `test_environment_freeze.txt` (Python 3.13, torch 2.14.1+cpu, numpy 2.5.2, pillow 11.3.0); Node 24.9, Vite 5.4.21, PostgreSQL 16 (`postgres:16-alpine`)

---

## 1. Research Objectives vs. Chapter Four Implementation Status

> [!CAUTION]
> **Objective b once had a blocking defect, closed 2026-10-01.** The clean retrain
> is complete and objectives c, f and h are evidenced on a leakage-free partition.
>
> **What the defect was.** The first genuine run of the image-validation pipeline
> showed Gate 3 rejecting **10 of 10 unmodified held-out APTOS images** with
> `ERR_MOTION_OR_DEFOCUS_BLUR`: the sharpness threshold was 60.0 against real
> images scoring 5.7–22.0, so the system would have refused to grade every
> genuine fundus photograph. Two further thresholds proved to be a priori in the
> same way. All three are now derived from the 1st percentile of the development
> corpus, and all 16 declared validation cases behave as declared
> ([`validation_test_results.csv`](validation_test_results.csv)). The record is
> kept in `known_limitations.md` §1 because the defect was real and shipped.

| Research Objective | Specification & Scope | Chapter Four Evidence Document | Implementation Status |
| :--- | :--- | :--- | :---: |
| **Objective a**<br>Design architecture, workflow & database | Multilayer research prototype architecture, decision boundaries, relational schema, audit ledger, and OOADM diagrams. | [`docs/chapter4/architecture.md`](/docs/chapter4/architecture.md)<br>[`docs/chapter4/database_schema.md`](/docs/chapter4/database_schema.md) | ✅ **Complete** |
| **Objective b**<br>Implement technical input-validation pipeline | 3-stage fail-closed validation pipeline: Gate 1 (Integrity), Gate 2 (Aperture & Relevance), Gate 3 (Sharpness/Blur). | [`docs/chapter4/validation_module_spec.md`](/docs/chapter4/validation_module_spec.md)<br>[`docs/chapter4/validation_test_results.csv`](/docs/chapter4/validation_test_results.csv) | ✅ **Complete**<br><sub>All three admission thresholds calibrated from the development corpus; 16/16 declared cases behave as declared</sub> |
| **Objective c**<br>Preprocess & partition retinal dataset | Dataset provenance audit, single-cohort audit (APTOS 2019 only), grade-stratified split (70/15/15: 2,453 / 526 / 525), and preprocessing specification. | [`docs/chapter4/dataset_audit.md`](/docs/chapter4/dataset_audit.md)<br>[`docs/chapter4/dataset_split_manifest.csv`](/docs/chapter4/dataset_split_manifest.csv)<br>[`docs/chapter4/preprocessing_and_augmentation_spec.md`](/docs/chapter4/preprocessing_and_augmentation_spec.md) | ✅ **Complete**<br><sub>Leakage-free split applied: 3,504 images, zero partition overlap</sub> |
| **Objective d**<br>Design CNN classification architecture | EfficientNet-B0 backbone, custom 5-class classification head (`Dropout(0.2) + Linear(1280, 5)`), `features.8` Grad-CAM hook. | [`docs/chapter4/checkpoint_manifest.md`](/docs/chapter4/checkpoint_manifest.md)<br>[`docs/model_integration_guide.md`](/docs/model_integration_guide.md) | ✅ **Complete** |
| **Objective e**<br>Implement the CNN model | Executable PyTorch neural network module, parameter freezing (`eval()`, `requires_grad=False`), and verifiable checkpoint weights file. | `backend/models/weights/efficientnet_b0_dr.pth`<br>[`docs/chapter4/checkpoint_manifest.md`](/docs/chapter4/checkpoint_manifest.md) | ✅ **Complete** |
| **Objective f**<br>Train the CNN classifier | Supervised training protocol, weighted cross-entropy loss for class imbalance, AdamW optimizer, cosine annealing, and validation curves. | [`docs/chapter4/training_protocol.md`](/docs/chapter4/training_protocol.md)<br>[`docs/chapter4/training_environment.md`](/docs/chapter4/training_environment.md) | ✅ **Complete**<br><sub>Retrained from ImageNet init on the clean split; checkpoint 67d0b896…</sub> |
| **Objective g**<br>Integrate model into CDSS backend | Connection of fixed evaluated checkpoint to FastAPI `/api/v1/assessments`, Grad-CAM visual attribution generation, and UI delivery. | [`backend/app/services/ai_service.py`](/backend/app/services/ai_service.py)<br>[`docs/chapter4/api_contract.md`](/docs/chapter4/api_contract.md) | ✅ **Complete** |
| **Objective h**<br>Evaluate held-out test performance | Independent evaluation on the $N=525$ held-out test fundus set, confusion matrix, QWK ($\kappa$), class-wise sensitivity/specificity, Mild NPDR analysis. | [`docs/chapter4/model_evaluation_report.md`](/docs/chapter4/model_evaluation_report.md)<br>[`docs/chapter4/held_out_predictions.csv`](/docs/chapter4/held_out_predictions.csv)<br>[`docs/chapter4/confusion_matrix.png`](/docs/chapter4/confusion_matrix.png)<br>[`docs/chapter4/resource_benchmark.md`](/docs/chapter4/resource_benchmark.md) | ✅ **Complete**<br><sub>N = 525 leakage-free; accuracy 84.00%, κ 0.8658</sub> |
| **Objective i**<br>Perform end-to-end system testing | Verification of functional requirements (FR-01–FR-17) and non-functional requirements (NFR-01–NFR-12) using the real fixed model. | [`docs/chapter4/system_test_report.md`](/docs/chapter4/system_test_report.md)<br>[`docs/chapter4/requirements_test_matrix.md`](/docs/chapter4/requirements_test_matrix.md)<br>[`docs/chapter4/screenshot_evidence_manifest.md`](/docs/chapter4/screenshot_evidence_manifest.md) | ✅ **Complete**<br><sub>Functional and latency tests pass against admitted images; the upload-to-classification path completes end to end</sub> |

---

## 2. Evidence Artifact Directory Index

All Chapter 4 evidence documents are indexed and linked below:

```text
docs/chapter4/
├── implementation_status.md                <-- Master progress & status summary
├── objective_traceability_matrix.md        <-- Bidirectional research objective mapping
├── dataset_audit.md                        <-- Provenance, resolution, duplicate & class analysis
├── dataset_split_manifest.csv              <-- Patient-partitioned manifest (Train/Val/Test)
├── preprocessing_and_augmentation_spec.md  <-- Circular masking, ImageNet norm, augmentation
├── training_protocol.md                    <-- Loss weighting, hyperparameter schedule, convergence
├── training_environment.md                 <-- Host hardware, compute drivers, framework versions
├── checkpoint_manifest.md                  <-- Topology, layer parameters, SHA-256 hash
├── model_evaluation_report.md              <-- Held-out test performance, QWK, Mild NPDR analysis
├── held_out_predictions.csv                <-- 525 itemized predictions on test set
├── confusion_matrix.png                    <-- 5x5 normalized confusion matrix heatmap
├── resource_benchmark.md                   <-- Empirical CPU/GPU latency, RAM RSS, FLOPs
├── validation_module_spec.md               <-- Formal Gate 1, Gate 2, Gate 3 algorithms
├── evidence_provenance.md                  <-- Which script produced which artefact/fail gate test outputs
├── system_test_report.md                   <-- Automated regression & end-to-end test run
├── requirements_test_matrix.md             <-- FR/NFR traceability matrix
├── screenshot_evidence_manifest.md         <-- 11 required interface figures & captions
├── architecture.md                         <-- Layered architecture & component breakdown
├── database_schema.md                      <-- Relational schema & append-only audit ledger
├── api_contract.md                         <-- OpenAPI/REST endpoint schemas
├── reproducibility_runbook.md              <-- Step-by-step replication commands
└── known_limitations.md                    <-- Clinical boundaries, Mild NPDR, hardware scope
```
