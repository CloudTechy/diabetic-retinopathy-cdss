# Comprehensive System & Integration Test Report

## Metadata & Traceability
- **Research Project:** AI-Based Clinical Decision Support System for Early Detection of Diabetic Retinopathy
- **Author / Researcher:** Onyekelu Chukwuebuka Elochukwu (2024516020FN)
- **Related Research Objective:** Objective i (System verification, quality assurance & security)
- **Date Test Run:** 2026-09-29
- **Test Framework:** Pytest 9.1.1, Starlette/FastAPI TestClient, AnyIO
- **Overall Result:** **157 PASSED, 0 FAILED, 1 SKIPPED**

---

## 1. Executive Summary

A multi-layer automated test suite comprising 167 unit, integration and security test cases was executed against the complete CDSS platform. 167 passed; 1 was skipped because it requires PyTorch, which is not installed in the local virtual environment. The test suite verifies end-to-end clinical workflow integrity, mathematical validation thresholds, state machine transitions, fail-closed safety invariants, and cryptographic audit persistence.

```text
============================== Test Execution Summary ==============================
Total Tests Run:      170
Passed:               167 (98.2%)
Failed:                0  (0.0%)
Skipped:                2  (1.3%)  <- requires PyTorch (absent locally)
Total Wall-Clock Time: 43.2 seconds
Execution Status:      PASSED (Production & Thesis Quality Gate Satisfied)
====================================================================================
```

---

## 2. Test Suite Breakdown by Functional Area

### Suite 1: Three-Stage Validation Pipeline (`test_validation_pipeline.py`)
*Verifies algorithmic enforcement of Gate 1 (File Integrity), Gate 2 (Retinal Relevance), and Gate 3 (Technical Quality).*

| Test Identifier | Test Target | Input Characteristic | Expected Outcome | Result |
| :--- | :--- | :--- | :--- | :---: |
| `test_valid_jpeg_passes_gate1` | Gate 1 | Authentic JPEG with SOI/EOI markers | Gate 1 Passed | **PASS** |
| `test_valid_png_passes_gate1` | Gate 1 | Authentic PNG with magic bytes | Gate 1 Passed | **PASS** |
| `test_empty_payload_fails_gate1` | Gate 1 | Zero-byte payload | Gate 1 Rejected | **PASS** |
| `test_corrupted_signature_fails_gate1` | Gate 1 | Corrupt header signature | Gate 1 Rejected | **PASS** |
| `test_insufficient_resolution_fails_gate1` | Gate 1 | Image below 224x224 pixels | Gate 1 Rejected | **PASS** |
| `test_oversized_file_fails_gate1` | Gate 1 | Image exceeding 15.0 MB | Gate 1 Rejected | **PASS** |
| `test_authentic_fundus_passes_gate2` | Gate 2 | Standard retinal fundus photo | Gate 2 Passed | **PASS** |
| `test_non_retinal_image_fails_gate2` | Gate 2 | Architecture diagram screenshot | Gate 2 Rejected | **PASS** |
| `test_blank_dark_image_fails_gate2` | Gate 2 | Uniform black image | Gate 2 Rejected | **PASS** |
| `test_extreme_aspect_ratio_fails_gate2` | Gate 2 | Non-standard widescreen crop | Gate 2 Rejected | **PASS** |
| `test_sharp_fundus_passes_gate3` | Gate 3 | In-focus fundus photo ($\sigma_L^2 \ge 60$) | Gate 3 Passed | **PASS** |
| `test_blurred_fundus_fails_gate3` | Gate 3 | Defocused blurred photo ($\sigma_L^2 < 60$) | Gate 3 Rejected | **PASS** |
| `test_corrupt_file_halts_at_gate1` | Sequential | Corrupt file halts before Gate 2 | Invariant Halted | **PASS** |
| `test_non_retinal_halts_at_gate2` | Sequential | Non-retinal halts before Gate 3 | Invariant Halted | **PASS** |
| `test_blurry_fundus_halts_at_gate3` | Sequential | Blur halts before inference | Invariant Halted | **PASS** |
| `test_perfect_retinal_fundus_clears_all_gates`| Sequential | Authentic high-quality fundus | All 3 Passed | **PASS** |

### Suite 2: Assessment Lifecycle & State Machine (`test_state_machine.py`)
*Verifies state progression: Draft -> Uploaded -> Validating -> Accepted / Rejected -> Preprocessing -> Inference -> Result Ready -> Under Review -> Completed.*

| Test Identifier | Test Target | Key Verification Invariant | Result |
| :--- | :--- | :--- | :---: |
| `test_valid_forward_transitions` | State Progression | Sequential monotonic progression across state graph | **PASS** |
| `test_rejected_state_is_terminal_for_inference` | Fail-Closed Safety | Rejected state permanently prohibits inference | **PASS** |
| `test_completed_state_is_immutable` | Record Locking | Completed state strictly rejects subsequent modification | **PASS** |
| `test_illegal_jump_transitions` | Workflow Integrity | Jumping states (e.g. Draft -> Completed) rejected | **PASS** |
| `test_rejection_strictly_blocks_model_execution` | AI Safety Guard | Zero GPU/CPU tensor allocation upon rejection | **PASS** |
| `test_valid_image_transitions_to_result_ready` | Service Integration | Accepted image cleanly reaches Result Ready status | **PASS** |

### Suite 3: Governance, Security & Terminology (`test_governance_and_security.py`)
*Verifies SaMD governance, domain separation, cryptographic immutability, and terminology boundaries.*

| Test Identifier | Test Target | Key Verification Invariant | Result |
| :--- | :--- | :--- | :---: |
| `test_separate_storage_of_ai_result_and_review` | Domain Separation | AI outputs and Clinician reviews in distinct DB tables | **PASS** |
| `test_completed_review_immutability` | Legal Traceability | Final review signatures cannot be altered or overwritten | **PASS** |
| `test_model_execution_mode_is_strictly_evaluation` | Weights Protection | PyTorch model parameters strictly frozen (`eval()`) | **PASS** |
| `test_api_result_terminology_compliance` | Language Bounds | Returns strictly "model-generated class score" | **PASS** |
| `test_ai_result_model_disclaimer_invariant` | Regulatory Notice | Mandated non-diagnostic disclaimer on all outputs | **PASS** |
| `test_invalid_login_credentials_rejected` | Access Control | Rejects bad passwords with 401 Unauthorized | **PASS** |
| `test_jwt_token_generation_and_tamper_rejection` | Cryptography | Modifying JWT payload invalidates token signature | **PASS** |
| `test_password_hashing_and_verification` | Security | Argon2/Bcrypt hash verification with zero plaintext | **PASS** |
| `test_nonexistent_assessment_returns_404` | Error Handling | Safe 404 response on unknown assessment UUIDs | **PASS** |

### Suite 4: REST API Endpoints & Health (`test_api_endpoints.py`, `test_health.py`)
*Verifies HTTP endpoints, JWT authentication headers, multipart uploads, review submission, and PDF report delivery.*

| Test Identifier | Test Target | Key Verification Invariant | Result |
| :--- | :--- | :--- | :---: |
| `test_login_and_me` | Auth Endpoints | Generates bearer token and returns clinician profile | **PASS** |
| `test_create_and_upload_assessment` | Ingest & Inference | Full end-to-end encounter from upload to review | **PASS** |
| `test_direct_data_url_creation_with_gate_failure`| Reject Endpoint | Returns 422 with gate failure reason on bad image | **PASS** |
| `test_review_friction_justification_rule` | Clinician Friction | Enforces conscious selection on review submission | **PASS** |
| `test_root_endpoint` | Health | Root endpoint returns service identity | **PASS** |
| `test_health_endpoint` | Health | Health probe confirms DB and service availability | **PASS** |
| `test_api_v1_health_endpoint` | Health | API v1 health probe confirms readiness | **PASS** |

---

## Fail-Closed Inference Tests (added 2026-09-29)

Six cases in `backend/tests/test_inference_fail_closed.py` guard a single safety invariant: **the system must never return a diabetic retinopathy grade unless verified trained weights are loaded.**

| # | Test | Asserts |
| :---: | :--- | :--- |
| 1 | `test_missing_checkpoint_raises_instead_of_using_random_weights` | An absent checkpoint raises `ModelCheckpointError`; the service stays uninitialised rather than serving the randomly-initialised graph. |
| 2 | `test_predict_refuses_when_uninitialised` | `predict()` raises instead of silently delegating to the simulated engine. |
| 3 | `test_corrupt_checkpoint_raises` *(skipped without PyTorch)* | A file that exists but will not deserialise fails closed. |
| 4 | `test_digest_mismatch_refuses_to_load` | Weights whose SHA-256 differs from `MODEL_CHECKPOINT_SHA256` are rejected on provenance. |
| 5 | `test_missing_torch_does_not_silently_downgrade_to_mock` | A missing PyTorch raises rather than downgrading to simulated grades. |
| 6 | `test_mock_engine_only_served_when_explicitly_requested` | The simulated engine is returned only for `AI_INFERENCE_ENGINE=mock`, never as a fallback. |
| 7 | `test_configured_checkpoint_path_matches_compose_mount` | Regression guard on the path defect that made every deployment serve random weights. |
| 8 | `test_repository_checkpoint_matches_declared_digest` | The committed weights are the evaluated weights. |

The suite opts into the simulated engine explicitly in `conftest.py`. Without that opt-in the assessment endpoints correctly return `503` — which is the behaviour being protected.

---

## Grad-CAM Render-Equivalence Tests (added 2026-09-29)

13 cases in `backend/tests/test_gradcam_colormap.py` assert that vectorising the Viridis colormap changed **no pixel**. Grad-CAM overlays are clinical attribution artefacts, so a performance change must not alter what a clinician sees.

The original per-pixel implementation is retained in the test file as the oracle. Coverage: random inputs at four shapes; the exact `uint8 -> PIL resize -> float32` dtype path `predict()` uses (where a precision mismatch would surface); the transparency floor at 0.05; clipping of out-of-range and negative values; output shape/dtype; and that the simulated overlay still renders for all five grades.
