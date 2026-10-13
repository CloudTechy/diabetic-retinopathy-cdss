# Clinical REST API Contract & Specification

## Metadata & Traceability
- **Research Project:** AI-Based Clinical Decision Support System for Early Detection of Diabetic Retinopathy
- **Author / Researcher:** Onyekelu Chukwuebuka Elochukwu (2024516020FN)
- **Related Research Objective:** Objective a (System architecture, workflow & database design) & Objective g (Model integration & decision-support workflow)
- **API Standard:** OpenAPI 3.1.0 / RESTful JSON; errors are FastAPI's `{"detail": ...}` bodies
- **Examples:** the JSON bodies below show the request and response shapes. The `/validation` example is the gates' actual output for the shipped held-out fixture (`backend/tests/fixtures/aptos_heldout_d1f1ea894da1.png`), and a rule in the suite recomputes it from that file; the other bodies are illustrative examples whose values are not taken from a recorded run.
- **Scope:** this contract covers the routes the assessment workflow uses and the account-administration routes. The application also serves `GET /api/v1/assessments/search`, `GET /api/v1/assessments/{id}`, `GET /api/v1/assessments/{id}/status`, `GET /api/v1/reports/{id}/pdf` (an alias of the report route), `GET /api/v1/admin/users/{user_id}`, `GET /api/v1/admin/summary` and `GET /health` (also under `/api/v1/health`); their shapes are in the OpenAPI schema the server generates.
- **Base Endpoint:** `/api/v1`
- **Roles:** every route below states who may call it. `SUPERADMIN` administers accounts and is not granted clinical content; `PHYSICIAN` runs the assessment workflow and sees only the records their own account created. A token whose account is not `ACTIVE`, or that was issued before the account's last password change, suspension or sign-out, is refused with 401; an account whose password is temporary receives 403 (`PASSWORD_CHANGE_REQUIRED`) from everything but `/auth/me`, `/auth/logout` and `/auth/change-password`.
- **Last Revised:** 2026-10-12

---

## 1. Authentication & Session Endpoints (`/api/v1/auth`)

### `POST /api/v1/auth/register`
- **Description:** A prospective physician requests an account. The account is created `PENDING_APPROVAL`; it cannot sign in until a Super Administrator approves it. The request has no role or status field, so a public request can never produce an active account or a Super Administrator.
- **Authentication:** none.
- **Request Body:**
  ```json
  {
    "fullName": "Ada Lovelace",
    "username": "ada.lovelace",
    "email": "ada.lovelace@example.org",
    "password": "Strong-Pass-2026",
    "passwordConfirmation": "Strong-Pass-2026",
    "licenseNumber": "REG-12345",
    "facility": "Eye Clinic A"
  }
  ```
- **Responses:** `201 Created` with `{ "id", "username", "status": "PENDING_APPROVAL", "message" }`; `409 Conflict` when the username or email is taken; `422` when the passwords differ or the password policy (at least 10 characters, letters and digits) is not met. `licenseNumber` is recorded as typed and is not verified against any register.

### `POST /api/v1/auth/login`
- **Description:** Authenticates an account and issues a signed session token (JWT, HS256). The password is verified for every account on every sign-in; the token carries only `sub` (account id), `role`, `tv` (session generation), `iat` and `exp`.
- **Request Body:**
  ```json
  {
    "username": "ada.lovelace",
    "password": "Strong-Pass-2026"
  }
  ```
- **Responses:**
  - `200 OK`:
    ```json
    {
      "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
      "token_type": "bearer",
      "user": {
        "id": "1b1f7a2e-5c7e-4a39-9a9e-3b4e6c5d2f10",
        "username": "ada.lovelace",
        "name": "Ada Lovelace",
        "email": "ada.lovelace@example.org",
        "role": "PHYSICIAN",
        "status": "ACTIVE",
        "mustChangePassword": false,
        "licenseNumber": "REG-12345",
        "facility": "Eye Clinic A",
        "sessionTimeoutMinutes": 15,
        "loginTime": "2026-10-12T09:15:02.118Z"
      }
    }
    ```
  - `401 Unauthorized`: wrong username or password (the attempt is audited; the password is not stored).
  - `403 Forbidden`: the account is `PENDING_APPROVAL`, `REJECTED` or `SUSPENDED`; `detail` says which.

### `POST /api/v1/auth/logout`
- **Description:** Invalidates every session of the signed-in account (its tokens stop being accepted).
- **Headers:** `Authorization: Bearer <token>`
- **Response:** `204 No Content`.

### `GET /api/v1/auth/me`
- **Description:** Retrieves the authenticated account's profile (the `user` object above). Available to an account that must still change its temporary password.
- **Headers:** `Authorization: Bearer <token>`
- **Response:** `200 OK` with user details.

### `POST /api/v1/auth/change-password`
- **Description:** Changes the signed-in account's password. Mandatory before anything else when the password is temporary. A successful change invalidates every session, including the current one.
- **Request Body:** `{ "currentPassword", "newPassword", "newPasswordConfirmation" }`
- **Responses:** `200 OK` with the user profile (`mustChangePassword: false`); `403` when the current password is wrong; `422` when the new password breaks the policy, differs from its confirmation or equals the current one.

---

## 1b. Account Administration Endpoints (`/api/v1/admin`) — `SUPERADMIN` only

Every route in this group depends on an active Super Administrator whose password is not temporary; a physician receives `403`. No response in this group carries a password hash or a token; a temporary password appears exactly once, in the response to the request that issued it.

| Method | Route | Purpose | Result |
| :--- | :--- | :--- | :--- |
| `GET` | `/admin/users` | All accounts; query `q` (username, full name or email), `role`, `status`, `limit` | array of `UserAccount` (with `assessmentCount`) |
| `GET` | `/admin/users/pending` | Registration requests awaiting approval | array of `UserAccount` |
| `POST` | `/admin/users` | Create a physician (`fullName`, `username`, `email`, optional `licenseNumber`, `facility`, `temporaryPassword`, `activateImmediately` default true) | `201` `{ "user", "temporaryPassword" }` |
| `POST` | `/admin/users/batch` | Up to ten numbered accounts for a supervised demonstration (`count`, `usernamePrefix`, optional `fullNamePrefix`, `emailDomain` default `example.org`, `facility`) | `201` `{ "created": [ { "user", "temporaryPassword" } ] }` |
| `PATCH` | `/admin/users/{user_id}/approve` | `PENDING_APPROVAL` → `ACTIVE` (not on one's own account) | `UserAccount` |
| `PATCH` | `/admin/users/{user_id}/reject` | `PENDING_APPROVAL` → `REJECTED`; body `{ "reason" }` optional | `UserAccount` |
| `PATCH` | `/admin/users/{user_id}/suspend` | `ACTIVE` → `SUSPENDED`; sessions invalidated; refused (`409`) for the last active Super Administrator | `UserAccount` |
| `PATCH` | `/admin/users/{user_id}/reactivate` | `SUSPENDED` → `ACTIVE` | `UserAccount` |
| `POST` | `/admin/users/{user_id}/reset-password` | New temporary password (shown once), sessions invalidated, password change required at next sign-in | `{ "user", "temporaryPassword" }` |
| `GET` | `/admin/audit-events` | Account-administration audit events, newest first; query `userId`, `eventType`, `limit` | array of audit events (`eventType`, `actor`, `actorUserId`, `targetUserId`, `assessmentId`, `outcome`, `details`, `timestamp`) |

A transition from any other state answers `409 Conflict`. The `UserAccount` object: `id`, `username`, `fullName`, `email`, `role`, `status`, `mustChangePassword`, `licenseNumber`, `facility`, `createdById`, `approvedById`, `approvedAt`, `rejectionReason`, `lastLoginAt`, `createdAt`, `updatedAt`, `assessmentCount`.

---

## 2. Assessment Lifecycle Endpoints (`/api/v1/assessments`)

All routes in this section require an active `PHYSICIAN` whose password is not temporary. The worklist and search return only the records the signed-in physician created, and every single-record route answers `404` for a record created by another account; a Super Administrator receives `403`.

### `GET /api/v1/assessments`
- **Description:** Fetches the active clinical assessment worklist.
- **Query Parameters:** `status` (optional), `laterality` (optional), `limit` (default: 50).
- **Response `200 OK`:** Array of `AssessmentRecord` objects.

### `POST /api/v1/assessments`
- **Description:** Initializes a new clinical assessment draft.
- **Request Body:**
  ```json
  {
    "patientId": "PT-2026-0814",
    "laterality": "OD",
    "cameraModel": "Topcon TRC-NW400",
    "isMydriatic": false,
    "clinicalNotes": "Screening for Type 2 Diabetes (HbA1c 8.4%)."
  }
  ```
- **Response `201 Created`:** Draft assessment object with its assigned record identifier (`REC-YYYY-XXXXXX`). `cameraModel` is optional: omitted or empty is stored as null and shown as "Not recorded"; nothing is preset.

### `POST /api/v1/assessments/{id}/upload`
- **Description:** Ingests fundus image binary (`multipart/form-data`) and initiates the sequential 3-gate validation pipeline. If passed, triggers real PyTorch EfficientNet-B0 inference and Grad-CAM generation.
- **Parameters:**
  - `file`: Binary image file (JPEG or PNG, $\le 15.0$ MB).
- **Responses:**
  - `200 OK`: Assessment updated to `needs_review` with `modelObservation` and `gradcamUrl`.
  - `422 Unprocessable Entity`: Validation failure (Gate 1, 2, or 3) with stage-specific rejection reason and recommended clinical action.

### `GET /api/v1/assessments/{id}/validation`
- **Description:** Returns detailed telemetry for each of the 3 validation gates.
- **Response `200 OK`:**
  ```json
  [
    {
      "name": "Gate 1",
      "gateIndex": 1,
      "status": "passed",
      "title": "File Integrity & Safe Decode",
      "metric": "MIME image/png, SHA-256 computed, 0.87 MB",
      "details": "Valid binary signature (0x89504E47), dimension 1844x1226 px.",
      "rejectionReason": null,
      "clinicalAction": null
    },
    {
      "name": "Gate 2",
      "gateIndex": 2,
      "status": "passed",
      "title": "Technical retinal-image relevance",
      "metric": "Aperture coverage: 71.5%, R/B ratio: 6.31",
      "details": "Input meets the configured geometry and colour-profile thresholds (foreground coverage 71.5%, R/B 6.31). This does not confirm retinal identity, anatomical correctness or clinical gradability.",
      "rejectionReason": null,
      "clinicalAction": null
    },
    {
      "name": "Gate 3",
      "gateIndex": 3,
      "status": "passed",
      "title": "Technical Quality & Sharpness",
      "metric": "Laplacian: 12.2 (>= 4.3), Illumination index: 1.00",
      "details": "The image met the configured technical thresholds (illumination index 1.00, dynamic range 24.0). This does not confirm retinal identity, anatomical correctness or clinical gradability.",
      "rejectionReason": null,
      "clinicalAction": null
    }
  ]
  ```

### `GET /api/v1/assessments/{id}/result`
- **Description:** Retrieves the model-generated class score distribution and explainability metadata. `classScores[].score` and `primaryScore` are the temperature-scaled probabilities, softmax(logits / T), with T fitted on the validation split only (`docs/chapter4/probability_calibration/`); `uncalibratedScore` is the raw softmax of the same logits, kept for audit. `calibrationTemperature` is the T applied, or `null` when none was (the simulated engine, or a result stored before calibration was introduced).
- **Response `200 OK`:**
  ```json
  {
    "primaryClassGrade": 2,
    "primaryClassLabel": "Moderate NPDR",
    "primaryScore": 0.7426,
    "classScores": [
      { "grade": 0, "label": "Grade 0: No Apparent DR", "score": 0.0491, "uncalibratedScore": 0.04 },
      { "grade": 1, "label": "Grade 1: Mild NPDR", "score": 0.1342, "uncalibratedScore": 0.12 },
      { "grade": 2, "label": "Grade 2: Moderate NPDR", "score": 0.7426, "uncalibratedScore": 0.78 },
      { "grade": 3, "label": "Grade 3: Severe NPDR", "score": 0.0603, "uncalibratedScore": 0.05 },
      { "grade": 4, "label": "Grade 4: Proliferative DR", "score": 0.0138, "uncalibratedScore": 0.01 }
    ],
    "targetLayer": "features.8 (Conv2d Bottleneck Residual)",
    "topActivationRegion": "Peak Grad-CAM activation in the central cell of a 3x3 grid over the frame (focal: 9% of the map is above half the peak).",
    "modelVersion": "EfficientNet-B0-DR-v1 (fixed weights; temperature-scaled T=1.093921, ts-v2)",
    "inferenceTimestamp": "2026-10-04T06:55:12.418Z",
    "calibrationTemperature": 1.093921
  }
  ```

### `POST /api/v1/assessments/{id}/review`
- **Description:** Records the reviewing professional's own independent response under the signed-in account and write-locks the assessment record. `clinicianName`, `licenseNumber` and `facility` in the body are ignored; the signatory is always the authenticated user. `justificationNotes` (at least 15 characters) is required for `disagree` and `inconclusive`; `inconclusiveReason` is required for `inconclusive` (422 otherwise). Unknown fields are ignored.
- **Request Body:**
  ```json
  {
    "agreement": "inconclusive",
    "reviewerAssessedGrade": 2,
    "reviewerAssessedGradeLabel": "Grade 2: Moderate NPDR",
    "justificationNotes": "Media haze over the posterior pole prevents a confident grade.",
    "inconclusiveReason": "Media Opacity / Cataract"
  }
  ```
- **Response `200 OK`:** Finalized assessment object with its record hash anchor (an unkeyed SHA-256 over the review fields, truncated to 24 hexadecimal characters) and state `completed`.

### `GET /api/v1/assessments/{id}/report`
- **Description:** Streams the server-rendered PDF assessment report, hash-anchored to the original image.
- **Response `200 OK`:** `Content-Type: application/pdf`.

### `GET /api/v1/storage/images/{filename}` and `GET /api/v1/storage/attributions/{filename}`
- **Description:** Stream the stored fundus photograph and the Grad-CAM heatmap for the `<img>` tags in the viewer.
- **Authentication: none.** These two routes take no credential; anyone holding a filename can fetch the file. Stored names are `REC-YYYY-XXXXXX_<8 hex>.jpg`, hard to guess but not secret. This is a disclosed limitation of the prototype ([`known_limitations.md`](known_limitations.md) §12), not a security property.

### `GET /api/v1/assessments/{id}/audit`
- **Description:** Returns the chronological application-level append-only event log for governance review. The application does not expose audit-event update or deletion operations, and assessment deletion does not cascade to audit events. However, database-level immutability is not enforced through triggers or restricted database privileges.
- **Response `200 OK`:** Array of timestamped audit events.
