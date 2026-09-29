# Research Objective Traceability Matrix

## Metadata & Academic Governance
- **Research Project:** AI-Based Clinical Decision Support System for Early Detection of Diabetic Retinopathy
- **Author / Researcher:** Onyekelu Chukwuebuka Elochukwu (2024516020FN)
- **Degree & Faculty:** Postgraduate Diploma (PGD) in Computer Science, Faculty of Physical Sciences
- **Primary Benchmark:** APTOS 2019 Blindness Detection (3,662 retinal fundus photographs)
- **Trained Checkpoint:** `backend/models/weights/efficientnet_b0_dr.pth` (SHA-256: `0d443fa065528b2a1d24baea7bf8d8bf71a6203b22b817585374a7becd547c07`)
- **Audit Verification:** Prepared for Chapter Four Academic Evaluation

---

## 1. Traceability Table: Nine Approved Research Objectives

This matrix establishes direct, bidirectional traceability between the 9 approved Chapter One research objectives and the verified empirical evidence.

| Obj. ID | Objective Description | Scope & Verification Invariants | Repository Artifact | Chapter 4 Evidence Document | Empirical Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| **a** | **Problem formulation & dataset acquisition** | Source APTOS 2019 registry (3,662 fundus photographs across 5 ICDR severity levels). Identify duplicate groupings via perceptual hashes. | `storage/datasets/aptos2019/`<br>`backend/scripts/generate_aptos_manifest.py` | [`docs/chapter4/dataset_audit.md`](file:///c:/Users/USER/Documents/TECH4MATION/diabetic-retinopathy-cdss/docs/chapter4/dataset_audit.md)<br>`dataset_split_manifest.csv` | **DOCUMENTED** |
| **b** | **Input-image technical validation** | Implement 3-stage pre-inference gating: Gate 1 (File Integrity), Gate 2 (Retinal Relevance & Camera Proportions: 0.65 to 1.65), Gate 3 (Sharpness/Laplacian $\ge 100$). Strict fail-closed abort. | `backend/app/services/validation/`<br>`tests/test_validation_pipeline.py` | [`docs/chapter4/validation_module_spec.md`](file:///c:/Users/USER/Documents/TECH4MATION/diabetic-retinopathy-cdss/docs/chapter4/validation_module_spec.md)<br>`validation_test_results.csv` | **EVALUATED** |
| **c** | **Preprocessing and data preparation** | Duplicate-group-aware stratified partitioning (70% train: 2,567; 15% val: 551; 15% test: 544). Tight circular fundus crop, resizing to $224 \times 224$, and ImageNet normalization. | `backend/scripts/generate_aptos_manifest.py` | [`docs/chapter4/preprocessing_and_augmentation_spec.md`](file:///c:/Users/USER/Documents/TECH4MATION/diabetic-retinopathy-cdss/docs/chapter4/preprocessing_and_augmentation_spec.md) | **DOCUMENTED** |
| **d** | **CNN classification architecture design** | EfficientNet-B0 backbone selection. Hook `features.8` (1,280-channel final convolutional feature layer) for Grad-CAM. Classification head: `Linear(1280, 5)`. Total parameters: 4,013,953. | `backend/app/services/ai_service.py` | [`docs/chapter4/checkpoint_manifest.md`](file:///c:/Users/USER/Documents/TECH4MATION/diabetic-retinopathy-cdss/docs/chapter4/checkpoint_manifest.md) | **VERIFIED** |
| **e** | **Baseline / Model Implementation** | Initialize PyTorch model architecture, verify weights state dict structure, and implement model loading with cryptographic SHA-256 validation. | `backend/models/weights/efficientnet_b0_dr.pth` | [`docs/chapter4/checkpoint_manifest.md`](file:///c:/Users/USER/Documents/TECH4MATION/diabetic-retinopathy-cdss/docs/chapter4/checkpoint_manifest.md) | **VERIFIED** |
| **f** | **CNN Classifier Training** | Supervised training over 15 epochs using AdamW ($10^{-4}$) with Cosine Annealing and class-weighted cross-entropy loss. Optimal checkpoint selected at Epoch 14 (Val QWK: 0.9421). | `backend/scripts/train_efficientnet_b0.py` | [`docs/chapter4/training_protocol.md`](file:///c:/Users/USER/Documents/TECH4MATION/diabetic-retinopathy-cdss/docs/chapter4/training_protocol.md)<br>`epoch_history.csv`<br>`learning_curves.png` | **EVALUATED** |
| **g** | **Model integration & decision-support workflow** | Integrate model into FastAPI backend and clinical UI. Present model-generated class scores, persist evaluations in PostgreSQL, generate tamper-evident PDF reports, and record professional review sign-offs. | `backend/app/routers/assessments.py`<br>`frontend/src/` | [`docs/chapter4/api_contract.md`](file:///c:/Users/USER/Documents/TECH4MATION/diabetic-retinopathy-cdss/docs/chapter4/api_contract.md)<br>[`docs/chapter4/architecture.md`](file:///c:/Users/USER/Documents/TECH4MATION/diabetic-retinopathy-cdss/docs/chapter4/architecture.md) | **INTEGRATED** |
| **h** | **Grad-CAM visual explainability** | Hook forward activations and backward gradients at `features.8` (1,280 channels). Generate 2D saliency heatmaps with Viridis colormap blending (0% to 100% opacity) in the fundus viewer. | `backend/app/services/ai_service.py`<br>`frontend/src/components/FundusViewer.tsx` | [`docs/chapter4/screenshots/05_decision_support_workspace.png`](file:///c:/Users/USER/Documents/TECH4MATION/diabetic-retinopathy-cdss/docs/chapter4/screenshots/05_decision_support_workspace.png) | **EVALUATED** |
| **i** | **Functional testing and end-to-end evaluation** | Comprehensive functional testing across all 8 clinical screens and API workflows; supported by computational efficiency benchmarking (CPU mean latency: 95.76 ms = 0.09576 s). | `tests/`<br>`backend/scripts/benchmark_resources.py` | [`docs/chapter4/system_test_report.md`](file:///c:/Users/USER/Documents/TECH4MATION/diabetic-retinopathy-cdss/docs/chapter4/system_test_report.md)<br>`resource_benchmark.md`<br>`benchmark_timings.csv` | **EVALUATED** |

---

## 2. Academic Traceability Notes
1. **Accreditation:** Postgraduate Diploma (PGD) Computer Science, Faculty of Physical Sciences.
2. **Methodological Boundaries:** In accordance with academic guidelines, automated outputs are presented strictly as *"Model-Generated Class Scores"* for decision support, with certified diagnosis reserved exclusively for credentialed clinicians.
