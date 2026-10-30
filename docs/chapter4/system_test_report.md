# Comprehensive System & Integration Test Report

## Metadata & Traceability
- **Research Project:** AI-Based Clinical Decision Support System for Early Detection of Diabetic Retinopathy
- **Author / Researcher:** Onyekelu Chukwuebuka Elochukwu (2024516020FN)
- **Related Research Objective:** Objective i (Functional testing and end-to-end evaluation)
- **Date Test Run:** 2026-10-13
- **Test Framework:** Pytest 8.4.2 on Python 3.13.5 (as recorded in the header of [`test_execution.log`](test_execution.log)), Starlette/FastAPI TestClient, AnyIO
- **Overall Result:** **312 collected — 312 passed, 0 failed, 0 skipped**
- **No test is skipped.** `test_real_model_end_to_end_pipeline` runs and passes against the digest-verified checkpoint using a genuine held-out APTOS image.

---

## 1. Executive Summary

A multi-layer automated test suite comprising 312 test cases was executed against the complete CDSS platform. All 312 passed; none was skipped. PyTorch is installed and `test_real_model_end_to_end_pipeline` runs and passes against the digest-verified checkpoint using a genuine held-out APTOS image. The test suite verifies end-to-end clinical workflow integrity, mathematical validation thresholds, state machine transitions, fail-closed safety invariants, and the application-level append-only event log.

```text
============================== Test Execution Summary ==============================
Tests Collected:      312
Passed:               312 (100.0%)
Failed:                 0  (0.0%)
Skipped:                0  (0.0%)
Total Wall-Clock Time: recorded on the last line of test_execution.log
Functional test result: 312 passed, 0 skipped
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
| `test_sharp_fundus_passes_gate3` | Gate 3 | In-focus fundus photo ($\sigma_L^2 \ge 4.3$) | Gate 3 Passed | **PASS** |
| `test_blurred_fundus_fails_gate3` | Gate 3 | Defocused blurred photo ($\sigma_L^2 < 4.3$) | Gate 3 Rejected | **PASS** |
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
*Verifies Prototype governance, domain separation, the API-level review write-lock and hash anchoring, and terminology boundaries.*

| Test Identifier | Test Target | Key Verification Invariant | Result |
| :--- | :--- | :--- | :---: |
| `test_separate_storage_of_ai_result_and_review` | Domain Separation | AI outputs and Clinician reviews in distinct DB tables | **PASS** |
| `test_completed_review_immutability` | Traceability | A recorded review cannot be altered or overwritten through the API | **PASS** |
| `test_model_execution_mode_is_strictly_evaluation` | Weights Protection | PyTorch model parameters strictly frozen (`eval()`) | **PASS** |
| `test_api_result_terminology_compliance` | Language Bounds | Returns strictly "model-generated class score" | **PASS** |
| `test_ai_result_model_disclaimer_invariant` | Regulatory Notice | Mandated non-diagnostic disclaimer on all outputs | **PASS** |
| `test_invalid_login_credentials_rejected` | Access Control | Rejects bad passwords with 401 Unauthorized | **PASS** |
| `test_jwt_token_generation_and_tamper_rejection` | Token signing (keyed HMAC) | Modifying JWT payload invalidates token signature | **PASS** |
| `test_password_hashing_and_verification` | Security | Argon2/Bcrypt hash verification with zero plaintext | **PASS** |
| `test_nonexistent_assessment_returns_404` | Error Handling | Safe 404 response on unknown assessment UUIDs | **PASS** |

### Suite 3b: Role-Based Account Administration (`test_account_management.py`)
*Verifies the two roles, the four account states, the transitions a Super Administrator controls, the separation of each physician's records, and the audit of every administrative event. No account is seeded: the suite bootstraps a Super Administrator the way a deployment does and creates physicians through the administration API.*

| Test Identifier | Test Target | Key Verification Invariant | Result |
| :--- | :--- | :--- | :---: |
| `test_01_superadmin_can_create_a_physician_account` | Account creation | A Super Administrator creates a physician and it appears in the account list | **PASS** |
| `test_02_admin_created_physician_is_active_and_must_change_password` | Temporary password | The account is ACTIVE, must change its password, is refused the clinical workflow until it does, and the pre-change token dies | **PASS** |
| `test_03_self_registration_creates_a_pending_account` | Registration | Self-registration yields PENDING_APPROVAL with role PHYSICIAN | **PASS** |
| `test_03b_public_registration_cannot_choose_a_role_or_status` | Privilege escalation | `role`/`status` fields in a public request are ignored | **PASS** |
| `test_04_pending_account_cannot_sign_in_or_reach_protected_endpoints` | Lock-out | A pending account receives 403 at sign-in; anonymous callers 401 | **PASS** |
| `test_05_approval_permits_sign_in_and_the_assessment_workflow` | Approval | After approval the physician signs in and creates an assessment | **PASS** |
| `test_06_rejection_prevents_access` | Rejection | A rejected account cannot sign in | **PASS** |
| `test_07_suspension_stops_an_existing_physician` | Suspension | The live session is invalidated and sign-in is refused | **PASS** |
| `test_08_reactivation_restores_access` | Reactivation | The physician signs in and uses the workflow again | **PASS** |
| `test_09_physician_cannot_use_superadmin_endpoints` | RBAC | Every `/admin` route answers 403 to a physician | **PASS** |
| `test_10_physician_cannot_change_their_role_through_a_request` | RBAC | No request payload alters a role | **PASS** |
| `test_10b_superadmin_is_not_granted_clinical_content` | Separation of duties | The Super Administrator receives 403 from the assessment routes | **PASS** |
| `test_10c_superadmin_cannot_approve_itself_and_registration_never_creates_a_superadmin` | Bootstrap boundary | Self-approval refused; registration never produces a SUPERADMIN | **PASS** |
| `test_11_physician_cannot_view_another_physicians_records` | Record isolation | Another physician's record is 404 on every route and absent from lists and search | **PASS** |
| `test_12_new_physician_starts_with_an_empty_history` | Clean workspace | A new account has no assessments; counts per account are correct | **PASS** |
| `test_13_no_password_hash_or_secret_is_returned` | Secrets | No hash, token version or temporary password in any listing; the audit log never records an attempted password | **PASS** |
| `test_14_administrative_events_are_audited` | Audit | Every required event type is recorded with actor, target, time and outcome | **PASS** |
| `test_14b_professional_review_is_audited` | Audit | A confirmed review is recorded with `PROFESSIONAL_REVIEW_CONFIRMED` | **PASS** |
| `test_15_last_active_superadmin_cannot_be_suspended` | Safeguard | The last active Super Administrator cannot be suspended | **PASS** |
| `test_bootstrap_is_idempotent_and_audited` | Bootstrap | A second bootstrap creates nothing; the first is audited | **PASS** |
| `test_batch_creation_for_a_supervised_demonstration` | Batch creation | Numbered accounts, distinct temporary passwords, limit of ten | **PASS** |
| `test_token_carries_only_subject_role_and_version` | Token claims | Claims are exactly `sub`, `role`, `tv`, `iat`, `exp` | **PASS** |
| `test_sign_out_invalidates_the_token` | Sign-out | The token stops being accepted | **PASS** |
| `test_expired_and_tampered_tokens_are_rejected` | Token validation | Expired and tampered tokens receive 401 | **PASS** |
| `test_registration_rejects_duplicates_and_weak_passwords` | Input rules | Duplicate username/email 409; weak or mismatched passwords 422 | **PASS** |

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

### Suite 5: Probability Calibration (`test_probability_calibration.py`, added 2026-10-13)
*Verifies that the temperature is fitted on the validation partition only, that the frozen temperature reproduces the committed test metrics without changing a prediction, and that the interface receives the calibrated probability vector (evidence: `docs/chapter4/probability_calibration/`).*

| Test Identifier | Key Verification Invariant | Result |
| :--- | :--- | :---: |
| `test_fit_refuses_test_tagged_records` | The fit refuses any record not tagged `val` | **PASS** |
| `test_artifact_was_fitted_on_validation_and_carries_no_test_statistic` | The fitted-parameter file names `val`, 526 images, and no test key | **PASS** |
| `test_prediction_manifests_are_the_two_disjoint_partitions` | 526 `val` and 525 `test` rows, equal to the split manifest, disjoint | **PASS** |
| `test_temperature_is_reproduced_from_the_validation_manifest_alone` | Refitting from the validation predictions reproduces $T$ | **PASS** |
| `test_leakage_audit_shows_temperature_frozen_before_test_partition_was_read` | $T$ frozen before the first test-partition read; artefact unchanged after | **PASS** |
| `test_test_metrics_are_reproduced_from_test_manifest_and_frozen_temperature` | NLL, ECE, MCE, Brier, accuracy, QWK recompute from the test predictions | **PASS** |
| `test_top_one_predictions_accuracy_confusion_matrix_and_qwk_unchanged` | Zero top-one changes; equal to `evaluation_summary.json` | **PASS** |
| `test_runtime_artifact_is_the_evidence_artifact` | The runtime copy is byte-identical to the evidence copy | **PASS** |
| `test_the_archive_ships_the_runtime_artifact` | The archive carries the file the engine needs to serve | **PASS** |
| `test_calibrated_probabilities_is_softmax_of_logits_over_t` | The runtime function is softmax(z / T) and keeps the argmax | **PASS** |
| `test_engine_refuses_an_invalid_artifact` (3 cases) | Wrong split, non-positive $T$, other checkpoint, missing file: refused | **PASS** |
| `test_engine_predict_returns_the_calibrated_vector` | `predict()` returns softmax(z / T), raw softmax kept for audit | **PASS** |
| `test_interface_receives_the_calibrated_vector` | The API's `classScores` are softmax(z / T); `calibrationTemperature` is $T$ | **PASS** |
| `test_real_checkpoint_serves_softmax_of_its_own_logits_over_t` | Real weights, real artefact, held-out image | **PASS** |

### Suite 6: Session Teardown (`test_session_teardown.py`, added 2026-10-13)
`test_a_session_using_the_test_client_exits` runs a session that uses the API client in a child process and asserts it returns: the in-memory database engine is disposed at session end, so a run no longer hangs after printing its summary. **PASS**

---

## Fail-Closed Inference Tests (added 2026-09-29)

Fourteen cases in `backend/tests/test_inference_fail_closed.py` guard a single safety invariant: **the system must never return a diabetic retinopathy grade unless verified trained weights are loaded.**

| # | Test | Asserts |
| :---: | :--- | :--- |
| 1 | `test_missing_checkpoint_raises_instead_of_using_random_weights` | An absent checkpoint raises `ModelCheckpointError`; the service stays uninitialised rather than serving the randomly-initialised graph. |
| 2 | `test_predict_refuses_when_uninitialised` | `predict()` raises instead of returning a prediction from any other source. |
| 3 | `test_corrupt_checkpoint_raises` *(skipped without PyTorch)* | A file that exists but will not deserialise fails closed. |
| 4 | `test_digest_mismatch_refuses_to_load` | Weights whose SHA-256 differs from `MODEL_CHECKPOINT_SHA256` are rejected on provenance. |
| 5 | `test_missing_torch_does_not_silently_downgrade` | A missing PyTorch raises rather than returning grades that did not come from the model. |
| 6–9 | `test_no_configuration_selects_another_engine` *(4 settings)* | Whatever `AI_INFERENCE_ENGINE` is set to (unset, `mock`, `simulated`, `pytorch`), a missing checkpoint raises: no setting selects another engine. |
| 10 | `test_missing_calibration_artifact_fails_closed` | A missing temperature-scaling artifact raises rather than serving uncalibrated output. |
| 11 | `test_simulated_engine_is_not_in_the_application` | No simulated engine or synthetic attribution generator is importable; EfficientNet-B0 is the only engine class. |
| 12 | `test_health_reports_not_ready_when_the_checkpoint_cannot_load` | With no loadable checkpoint, health reports not ready rather than another engine as ready. |
| 13 | `test_configured_checkpoint_path_matches_compose_mount` | Regression guard on the path defect that made every deployment serve random weights. |
| 14 | `test_repository_checkpoint_matches_declared_digest` | The committed weights are the evaluated weights. |

The API tests install an isolated test double (`backend/tests/inference_doubles.py`) as the inference singleton through an autouse fixture in `conftest.py`. It exists only in the test package; the application cannot select it, and without it the assessment endpoints return `503` when no checkpoint is loaded — which is the behaviour being protected. `test_real_model_end_to_end_pipeline` replaces it with the real engine.

---

## Grad-CAM Render-Equivalence Tests (added 2026-09-29)

11 cases in `backend/tests/test_gradcam_colormap.py` assert that vectorising the Viridis colormap changed **no pixel**. Grad-CAM overlays are clinical attribution artefacts, so a performance change must not alter what a clinician sees.

The original per-pixel implementation is retained in the test file as the oracle. Coverage: random inputs at four shapes; the exact `uint8 -> PIL resize -> float32` dtype path `predict()` uses (where a precision mismatch would surface); the transparency floor at 0.05; clipping of out-of-range and negative values; output shape/dtype; and that a CAM in the image's aspect ratio (square, landscape, portrait) renders as an RGBA overlay of that shape.
