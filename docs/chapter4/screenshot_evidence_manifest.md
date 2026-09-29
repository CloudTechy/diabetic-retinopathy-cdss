# Visual Evidence & Screenshot Manifest

## Metadata & Traceability
- **Research Project:** AI-Based Clinical Decision Support System for Early Detection of Diabetic Retinopathy
- **Author / Researcher:** Onyekelu Chukwuebuka Elochukwu (2024516020FN)
- **Related Research Objective:** Objective g (Design CDSS software architecture) & Objective i (Verification)
- **Git Commit:** `22cda2c` (Baseline)
- **Date Generated:** 2026-09-28
- **Total Evidence Screenshots:** 11 Figure Panels

---

## 1. Itemized Screenshot Evidence Register

| Figure ID | Screen / Component Name | Key UI Elements & Visual Invariants Demonstrated | High-Resolution Evidence File |
| :---: | :--- | :--- | :--- |
| **Figure 4.1** | **Sign-In & Practitioner Auth** | Demonstrates research prototype clinical login, practitioner role badge, session security notices, and academic scope boundary notice. | [`docs/chapter4/screenshots/01_signin_screen.png`](file:///c:/Users/USER/Documents/TECH4MATION/diabetic-retinopathy-cdss/docs/chapter4/screenshots/01_signin_screen.png) |
| **Figure 4.2** | **Assessment Dashboard / Worklist** | Shows active clinical triage queue with summary metric cards, eye laterality (OD/OS), technical quality status, and non-diagnostic model score indicators. | [`docs/chapter4/screenshots/02_clinical_dashboard.png`](file:///c:/Users/USER/Documents/TECH4MATION/diabetic-retinopathy-cdss/docs/chapter4/screenshots/02_clinical_dashboard.png) |
| **Figure 4.3** | **New Assessment Fundus Upload** | Displays authentic drag-and-drop ophthalmic upload interface, client pre-flight checks, camera specifications, and de-identified study ID metadata form without synthetic presets. | [`docs/chapter4/screenshots/03_new_assessment_upload.png`](file:///c:/Users/USER/Documents/TECH4MATION/diabetic-retinopathy-cdss/docs/chapter4/screenshots/03_new_assessment_upload.png) |
| **Figure 4.4** | **3-Stage Validation (Passed)** | Demonstrates sequential real-time validation progression with all 3 gates clearing: Gate 1 File Integrity, Gate 2 Retinal Relevance, Gate 3 Quality (Laplacian $\sigma_L^2 = 142.4$). | [`docs/chapter4/screenshots/04_validation_stepper_passed.png`](file:///c:/Users/USER/Documents/TECH4MATION/diabetic-retinopathy-cdss/docs/chapter4/screenshots/04_validation_stepper_passed.png) |
| **Figure 4.5** | **3-Stage Validation (Rejected)** | Demonstrates the fail-closed safety lock: image rejected at Gate 3 with specific non-diagnostic feedback; automated model inference strictly blocked. | [`docs/chapter4/screenshots/04b_validation_stepper_rejected.png`](file:///c:/Users/USER/Documents/TECH4MATION/diabetic-retinopathy-cdss/docs/chapter4/screenshots/04b_validation_stepper_rejected.png) |
| **Figure 4.6** | **Decision-Support Workspace** | Displays dual-layer fundus viewer with zoom/pan, smooth Grad-CAM opacity slider (0–100%), colormap selector (Viridis/Inferno), and 5-class score distribution card. | [`docs/chapter4/screenshots/05_decision_support_workspace.png`](file:///c:/Users/USER/Documents/TECH4MATION/diabetic-retinopathy-cdss/docs/chapter4/screenshots/05_decision_support_workspace.png) |
| **Figure 4.7** | **Professional Review Modal** | Demonstrates human-in-the-loop tri-state selector (Agree / Disagree / Unable to determine), optional clinical observations, mandatory confirmation checkbox, and scope attribution notice. | [`docs/chapter4/screenshots/06_professional_review_modal.png`](file:///c:/Users/USER/Documents/TECH4MATION/diabetic-retinopathy-cdss/docs/chapter4/screenshots/06_professional_review_modal.png) |
| **Figure 4.8** | **Completed Assessment Record** | Displays finalized consultation report with "Preliminary Model Observation", "Professional Review Response", clinical scope boundary notice, and tamper-evident SHA-256 hash. | [`docs/chapter4/screenshots/07_completed_assessment_record.png`](file:///c:/Users/USER/Documents/TECH4MATION/diabetic-retinopathy-cdss/docs/chapter4/screenshots/07_completed_assessment_record.png) |
| **Figure 4.9** | **Record History & Audit Drawer** | Shows search and filter controls across historical assessments with chronological slide-out audit trail demonstrating append-only integrity logs. | [`docs/chapter4/screenshots/08_record_history_audit.png`](file:///c:/Users/USER/Documents/TECH4MATION/diabetic-retinopathy-cdss/docs/chapter4/screenshots/08_record_history_audit.png) |
| **Figure 4.10**| **Empirical Confusion Matrix** | High-resolution empirical 5x5 confusion matrix heatmap from the untouched held-out evaluation cohort ($N = 544$) with Quadratic Weighted Kappa ($\kappa = 0.9415$). | [`docs/chapter4/confusion_matrix.png`](file:///c:/Users/USER/Documents/TECH4MATION/diabetic-retinopathy-cdss/docs/chapter4/confusion_matrix.png) |
| **Figure 4.11**| **Server-Rendered Report PDF** | Complete clinical assessment report rendered via ReportLab with patient metadata, validation telemetry, 5-class score table, and professional review response. | [`docs/chapter4/screenshots/10_tamper_evident_pdf_report.png`](file:///c:/Users/USER/Documents/TECH4MATION/diabetic-retinopathy-cdss/docs/chapter4/screenshots/10_tamper_evident_pdf_report.png) |
