# Research Objective Traceability Matrix

## Metadata & Academic Governance
- **Research Project:** AI-Based Clinical Decision Support System for Early Detection of Diabetic Retinopathy
- **Author / Researcher:** Onyekelu Chukwuebuka Elochukwu (2024516020FN)
- **Degree & Faculty:** Postgraduate Diploma (PGD) in Computer Science, Faculty of Physical Sciences
- **Primary Benchmark:** APTOS 2019 Blindness Detection (3,662 retinal fundus photographs)
- **Trained Checkpoint:** `backend/models/weights/efficientnet_b0_dr.pth` (SHA-256: `67d0b89641f08057126dd411e380b25575ef29f71ae37ee5796d472d9203dbf7`)
- **Held-Out Result:** $N = 525$; QWK **0.8658**; referable-DR sensitivity **91.2%** / specificity **96.3%**
- **Audit Verification:** Prepared for Chapter Four Academic Evaluation

---

## 1. Traceability Table: Nine Approved Research Objectives

This matrix establishes direct, bidirectional traceability between the 9 approved Chapter One research objectives and the verified empirical evidence.

| Obj. ID | Objective Description | Scope & Verification Invariants | Repository Artifact | Chapter 4 Evidence Document | Empirical Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| **a** | **System architecture, workflow & database design** | CDSS architecture with clinician-in-the-loop governance; assessment state machine; 9-entity relational schema with domain separation; append-only audit trail. | `backend/app/models/models.py`<br>`backend/app/services/assessment_service.py` | [`architecture.md`](architecture.md)<br>[`database_schema.md`](database_schema.md)<br>[`api_contract.md`](api_contract.md) | **DOCUMENTED** |
| **b** | **Input-image technical validation** | Implement 3-stage pre-inference gating: Gate 1 (File Integrity), Gate 2 (Retinal Relevance & Camera Proportions: 0.65 to 1.65), Gate 3 (Sharpness/Laplacian $\ge 60.0$). Strict fail-closed abort. | `backend/app/services/validation/`<br>`tests/test_validation_pipeline.py` | [`docs/chapter4/validation_module_spec.md`](/docs/chapter4/validation_module_spec.md)<br>`validation_test_results.csv` | **EVALUATED** |
| **c** | **Dataset acquisition, preprocessing & partitioning** | Grade-stratified partitioning (70% train: 2,453; 15% val: 550; 15% test: 549). Patient-level isolation is not possible — APTOS publishes no patient identifier. Preprocessing is `Resize((224,224))` (bilinear) then ImageNet normalization; **no** fundus cropping is performed. | `backend/scripts/generate_aptos_manifest.py` | [`docs/chapter4/preprocessing_and_augmentation_spec.md`](/docs/chapter4/preprocessing_and_augmentation_spec.md) | **DOCUMENTED** |
| **d** | **CNN classification architecture design** | EfficientNet-B0 backbone selection. Hook `features.8` (1,280-channel final convolutional feature layer) for Grad-CAM. Classification head: `Linear(1280, 5)`. Total parameters: 4,013,953. | `backend/app/services/ai_service.py` | [`docs/chapter4/checkpoint_manifest.md`](/docs/chapter4/checkpoint_manifest.md) | **VERIFIED** |
| **e** | **Baseline / Model Implementation** | Initialize PyTorch model architecture, verify weights state dict structure, and load with cryptographic SHA-256 verification. The engine **fails closed**: a missing, corrupt or digest-mismatched checkpoint raises rather than serving untrained weights. | `backend/models/weights/efficientnet_b0_dr.pth` | [`docs/chapter4/checkpoint_manifest.md`](/docs/chapter4/checkpoint_manifest.md) | **VERIFIED** |
| **f** | **CNN Classifier Training** | Supervised training over 15 epochs using AdamW ($10^{-4}$) with Cosine Annealing and class-weighted cross-entropy loss, on a Colab Tesla T4. Optimal checkpoint selected at **Epoch 11** (Val QWK: **0.9130**). | `notebooks/colab_train_and_evaluate.py` | [`docs/chapter4/training_protocol.md`](/docs/chapter4/training_protocol.md)<br>`epoch_history.csv`<br>`learning_curves.png` | **EVALUATED** |
| **g** | **Model integration & decision-support workflow** | Integrate model into FastAPI backend and clinical UI. Present model-generated class scores, persist evaluations in PostgreSQL, generate tamper-evident PDF reports, and record professional review sign-offs. | `backend/app/routers/assessments.py`<br>`frontend/src/` | [`docs/chapter4/api_contract.md`](/docs/chapter4/api_contract.md)<br>[`docs/chapter4/architecture.md`](/docs/chapter4/architecture.md) | **INTEGRATED** Grad-CAM visual explainability is delivered as part of this objective: forward activations and gradients are hooked at `features.8` (1,280 channels) and rendered as a Viridis saliency overlay in the fundus viewer. |
| **h** | **Classification-performance evaluation** | Held-out evaluation on $N = 525$ untouched images. QWK **0.8658**, accuracy 84.00%, macro F1 0.7031, zero argmax violations. Per-class sensitivity/specificity with Wilson CIs; referable-DR operating point 91.2% / 96.3%; byte-level leakage audit disclosed. | `backend/scripts/evaluate_model.py`<br>`backend/scripts/analyze_clinical_metrics.py` | [`model_evaluation_report.md`](model_evaluation_report.md)<br>`held_out_predictions.csv`<br>`confusion_matrix.png`<br>`clinical_metrics.json` | **EVALUATED** |
| **i** | **Functional testing and end-to-end evaluation** | Comprehensive functional testing across all clinical screens and API workflows (157 passed, 2 skipped). Computational efficiency measured **end-to-end on the CPU deployment target**: mean **212.54 ms**, P95 **373.39 ms** over 30 held-out images. Validation gates 2+3 are 43.9% of the request (down from 59.1%) against 17.7% for the model. The gate optimisation is verified decision-preserving on all 3,662 APTOS images. | `tests/`<br>`backend/scripts/benchmark_cpu_end_to_end.py` | [`docs/chapter4/system_test_report.md`](/docs/chapter4/system_test_report.md)<br>`resource_benchmark.md`<br>`benchmark_timings.csv` | **EVALUATED** |

---

## 2. Academic Traceability Notes
1. **Accreditation:** Postgraduate Diploma (PGD) Computer Science, Faculty of Physical Sciences.
2. **Methodological Boundaries:** In accordance with academic guidelines, automated outputs are presented strictly as *"Model-Generated Class Scores"* for decision support, with certified diagnosis reserved exclusively for credentialed clinicians.
