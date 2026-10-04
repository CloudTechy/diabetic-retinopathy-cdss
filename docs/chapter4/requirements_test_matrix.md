# Requirements Traceability & Verification Test Matrix

## Metadata & Traceability
- **Research Project:** AI-Based Clinical Decision Support System for Early Detection of Diabetic Retinopathy
- **Author / Researcher:** Onyekelu Chukwuebuka Elochukwu (2024516020FN)
- **Related Research Objective:** Objective a (System architecture, workflow & database design) & Objective i (Functional testing and end-to-end evaluation)
- **Last Revised:** 2026-10-04
- **Total Requirements Tracked:** 18 (10 Functional, 8 Non-Functional)
- **Overall Verification Status:** **257 of 258 automated tests PASSED, 1 skipped** (the skip is conditional on a superseded artefact; see `system_test_report.md`)

---

## 1. Functional Requirements (FR) Traceability

| Req ID | Requirement Description | Implementation Module | Automated Test Case | Test Result |
| :--- | :--- | :--- | :--- | :---: |
| **FR-01** | Authenticated practitioner access using JWT session tokens and inactivity timeout. | `backend/app/routers/auth.py` | `TestAuthEndpoints.test_login_and_me` | **PASS** |
| **FR-02** | Ophthalmic Assessment Ingest with a unique record identifier (`REC-YYYY-XXXXXX`), patient ID, and eye laterality (OD/OS). | `backend/app/routers/assessments.py` | `TestAssessmentEndpoints.test_create_and_upload_assessment` | **PASS** |
| **FR-03** | Gate 1 File Integrity: Magic bytes validation for JPEG/PNG, file size limit <= 15MB. | `backend/app/services/validation/gate1_integrity.py` | `TestGate1FileIntegrity.test_valid_jpeg_passes_gate1` | **PASS** |
| **FR-04** | Gate 2 Retinal Field Relevance: Circular mask, aspect ratio ($0.65-1.65$), chromatic $R/B > 1.15$. | `backend/app/services/validation/gate2_relevance.py` | `TestGate2RetinalRelevance.test_authentic_fundus_passes_gate2` | **PASS** |
| **FR-05** | Gate 3 Technical Quality: Laplacian blur variance ($\ge 4.3$, `LAPLACIAN_BLUR_THRESHOLD`), extreme-pixel ratio ($\le 0.35$, `ILLUMINATION_EXTREME_RATIO_MAX`). | `backend/app/services/validation/gate3_quality.py` | `TestGate3TechnicalQuality.test_sharp_fundus_passes_gate3` | **PASS** |
| **FR-06** | Fail-Closed Invariant: Gate failure aborts model execution and sets status to `rejected`. | `backend/app/services/assessment_service.py` | `TestSequentialValidationPipelineFailClosed.test_blurry_fundus_halts_at_gate3` | **PASS** |
| **FR-07** | Automated EfficientNet-B0 inference outputting 5-class score distribution with frozen weights. | `backend/app/services/ai_service.py` | `TestClinicianInTheLoopGovernance.test_model_execution_mode_is_strictly_evaluation` | **PASS** |
| **FR-08** | Grad-CAM visual attribution generation hooked onto final bottleneck `features.8`. | `backend/app/services/ai_service.py` | `TestAssessmentEndpoints.test_create_and_upload_assessment` | **PASS** |
| **FR-09** | Clinician Review with Tri-State Agreement (Agree / Disagree / Unable to determine). | `backend/app/routers/assessments.py` | `TestAssessmentEndpoints.test_review_friction_justification_rule` | **PASS** |
| **FR-10** | Hash-anchored PDF assessment report generation and application-level append-only event log. | `backend/app/services/report_service.py` | `TestAssessmentEndpoints.test_create_and_upload_assessment` | **PASS** |

---

## 2. Non-Functional Requirements (NFR) Traceability

| Req ID | Requirement Description | Target Constraint | Verification Method | Status |
| :--- | :--- | :--- | :--- | :---: |
| **NFR-01** | **Inference Latency:** Single-image CPU execution time. | $< 350.0$ ms CPU | `backend/scripts/benchmark_cpu_end_to_end.py` (Mean **180.41 ms**, P95 **342.49 ms**) | **PASS as written; NOT robustly met at the tail** — see note |
| **NFR-02** | **Memory Footprint:** Peak backend memory usage under load. | $< 512.0$ MB RSS | **No shipped artefact measures memory.** An earlier revision quoted "Peak 328.8 MB" with a PASS; the figure traces to a commit that predates the genuine run and `benchmark_resources.py` records no memory at all, so nothing in this package could have produced it. | **NOT MEASURED** |
| **NFR-03** | **Weights Storage:** Compressed checkpoint disk footprint. | $< 50.0$ MB | `backend/models/weights/efficientnet_b0_dr.pth` (15.6 MB) | **PASS** |
| **NFR-04** | **Parameter Efficiency:** Total neural network parameter count. | $\approx 4.01$ M | Parameter inspection: exactly 4,013,953 parameters | **PASS** |
| **NFR-05** | **Hash anchoring:** SHA-256 digest of the original image; SHA-256 over the review fields; the API refuses a second review. | Hash anchoring + API write-lock (not enforced at the database) | `TestClinicianInTheLoopGovernance.test_completed_review_immutability` | **PASS** |
| **NFR-06** | **Separation of Domains:** AI output and Reviewer-assessed grade stored in separate tables. | Independent DB entities | `TestClinicianInTheLoopGovernance.test_separate_storage_of_ai_result_and_review` | **PASS** |
| **NFR-07** | **Terminology Compliance:** Model outputs strictly labeled "model-generated class score". | Zero diagnostic claims | `TestClinicalTerminologyCompliance.test_api_result_terminology_compliance` | **PASS** |
| **NFR-08** | **Session Inactivity:** Automatic 15-minute inactivity workstation auto-lock. | Clinical workstation safety | `frontend/src/components/Header.tsx` timer listener | **PASS** |

> [!CAUTION]
> **NFR-01 is specified on the mean, and the tail is not reproducible.**
>
> Three runs of this harness are cited ([`resource_benchmark.md`](resource_benchmark.md) §1b),
> and they are not three runs of one thing: run A used the superseded split's
> images (1,807 KB mean) and its checkpoint cannot be established from the
> artefacts; runs B and C used the clean checkpoint over the same 30 clean-split
> images (1,798 KB mean). Against the 350 ms budget:
>
> | Run | Mean | Verdict | P95 | Verdict |
> | :--- | ---: | :--- | ---: | :--- |
> | A (2026-09-30) | 212.54 ms | PASS | 373.39 ms | **breach** |
> | B (2026-10-01) | 254.31 ms | PASS | 447.95 ms | **breach** |
> | C (2026-10-01, cited) | 180.41 ms | PASS | 342.49 ms | pass |
>
> The **mean passes in every run**. The **P95 breaches in two of three**. B and
> C share the harness, checkpoint, images and request count (run B was committed
> while `config.py` still held the a-priori thresholds, which the harness does not
> act on) and otherwise differ in which machine Colab allocated — **1.41×** apart
> in absolute time. Their combined gates 2+3 share agreed to within 0.5 pp (41.9%
> and 41.4%); individual stage shares differed by up to 2.4 pp.
>
> So the requirement passes as written, and the tail it does not examine lands
> on either side of the budget depending on hardware. Quoting only run C would
> be choosing the flattering number. The tail is input resolution in `gate2`,
> not the model.
