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
| **Figure 4.1** | **Sign-In & Practitioner Auth** | Demonstrates research prototype clinical login, practitioner role badge, session security notices, and academic scope boundary notice. | [`docs/chapter4/screenshots/01_signin_screen.png`](/docs/chapter4/screenshots/01_signin_screen.png) |
| **Figure 4.2** | **Assessment Dashboard / Worklist** | Shows active clinical triage queue with summary metric cards, eye laterality (OD/OS), technical quality status, and non-diagnostic model score indicators. | [`docs/chapter4/screenshots/02_clinical_dashboard.png`](/docs/chapter4/screenshots/02_clinical_dashboard.png) |
| **Figure 4.3** | **New Assessment Fundus Upload** | Displays authentic drag-and-drop ophthalmic upload interface, client pre-flight checks, camera specifications, and de-identified study ID metadata form without synthetic presets. | [`docs/chapter4/screenshots/03_new_assessment_upload.png`](/docs/chapter4/screenshots/03_new_assessment_upload.png) |
| **Figure 4.4** | **3-Stage Validation (Passed)** | Demonstrates sequential real-time validation progression with all 3 gates clearing: Gate 1 File Integrity, Gate 2 Retinal Relevance, Gate 3 Quality (Laplacian $\sigma_L^2 = 142.4$). | [`docs/chapter4/screenshots/04_validation_stepper_passed.png`](/docs/chapter4/screenshots/04_validation_stepper_passed.png) |
| **Figure 4.5** | **3-Stage Validation (Rejected)** | Demonstrates the fail-closed safety lock: image rejected at Gate 3 with specific non-diagnostic feedback; automated model inference strictly blocked. | [`docs/chapter4/screenshots/04b_validation_stepper_rejected.png`](/docs/chapter4/screenshots/04b_validation_stepper_rejected.png) |
| **Figure 4.6** | **Decision-Support Workspace** | Displays dual-layer fundus viewer with zoom/pan, smooth Grad-CAM opacity slider (0–100%), colormap selector (Viridis/Inferno), and 5-class score distribution card. | [`docs/chapter4/screenshots/05_decision_support_workspace.png`](/docs/chapter4/screenshots/05_decision_support_workspace.png) |
| **Figure 4.7** | **Professional Review Modal** | Demonstrates human-in-the-loop tri-state selector (Agree / Disagree / Unable to determine), optional clinical observations, mandatory confirmation checkbox, and scope attribution notice. | [`docs/chapter4/screenshots/06_professional_review_modal.png`](/docs/chapter4/screenshots/06_professional_review_modal.png) |
| **Figure 4.8** | **Completed Assessment Record** | Displays finalized consultation report with "Preliminary Model Observation", "Professional Review Response", clinical scope boundary notice, and tamper-evident SHA-256 hash. | [`docs/chapter4/screenshots/07_completed_assessment_record.png`](/docs/chapter4/screenshots/07_completed_assessment_record.png) |
| **Figure 4.9** | **Record History & Audit Drawer** | Shows search and filter controls across historical assessments with chronological slide-out audit trail demonstrating append-only integrity logs. | [`docs/chapter4/screenshots/08_record_history_audit.png`](/docs/chapter4/screenshots/08_record_history_audit.png) |
| **Figure 4.10**| **Empirical Confusion Matrix** | High-resolution empirical 5x5 confusion matrix heatmap from the untouched held-out evaluation cohort ($N = 549$) with Quadratic Weighted Kappa ($\kappa = 0.8777$). | [`docs/chapter4/confusion_matrix.png`](/docs/chapter4/confusion_matrix.png) |
| **Figure 4.11**| **Server-Rendered Report PDF** | Complete clinical assessment report rendered via ReportLab with patient metadata, validation telemetry, 5-class score table, and professional review response. | [`docs/chapter4/screenshots/10_tamper_evident_pdf_report.png`](/docs/chapter4/screenshots/10_tamper_evident_pdf_report.png) |

---

> [!NOTE]
> **What these figures do and do not evidence.**
>
> Figures 4.1–4.9 were recaptured on 2026-09-30 from the current build, after the identity and
> scope remediation. They show `Dr. Demo Clinician (Simulated)` / `SIM-000001` /
> `Research Prototype Environment`. Earlier versions of these captures displayed a fabricated
> clinician identity, a fabricated GMC registration number and a fabricated hospital name; those
> are superseded and must not be reproduced in the dissertation.
>
> **The score values shown are demonstration data, not model output.** These captures run the
> frontend against its built-in demo fixtures so the interface can be photographed deterministically
> without a live backend. The 5-class distributions visible in Figures 4.6 and 4.8 are fixtures.
> They evidence **interface behaviour and clinical workflow** — validation stepper states, the
> fail-closed rejection path, Grad-CAM blending controls, the review modal, the audit trail, report
> rendering — and nothing about model performance. The authoritative metrics are in
> [`model_evaluation_report.md`](model_evaluation_report.md).
>
> **Figure 4.10 is different in kind.** `confusion_matrix.png` is a genuine plot from the evaluated
> run (N = 549, κ = 0.8777), not a screenshot, and may be cited as a result.
>
> Thresholds displayed in the interface are kept consistent with `backend/app/core/config.py`.
> The Laplacian sharpness threshold reads 60.0 in both, matching `LAPLACIAN_BLUR_THRESHOLD`.
>
> Regenerate with:
> ```bash
> cd frontend && npm run build && npx vite preview --port 4173 --host 127.0.0.1
> node scripts/capture_screenshots.js     # in a second shell
> ```
