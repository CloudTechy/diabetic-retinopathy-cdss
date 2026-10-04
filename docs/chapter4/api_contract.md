# Clinical REST API Contract & Specification

## Metadata & Traceability
- **Research Project:** AI-Based Clinical Decision Support System for Early Detection of Diabetic Retinopathy
- **Author / Researcher:** Onyekelu Chukwuebuka Elochukwu (2024516020FN)
- **Related Research Objective:** Objective a (System architecture, workflow & database design) & Objective g (Model integration & decision-support workflow)
- **API Standard:** OpenAPI 3.1.0 / RESTful JSON; errors are FastAPI's `{"detail": ...}` bodies
- **Examples:** every JSON body below is an illustrative example of the response shape. The values are not taken from a recorded run; the gate metrics shown are consistent with the calibrated thresholds in `backend/app/core/config.py`.
- **Base Endpoint:** `/api/v1`
- **Last Revised:** 2026-10-04

---

## 1. Authentication & Session Endpoints (`/api/v1/auth`)

### `POST /api/v1/auth/login`
- **Description:** Authenticates the user and issues an HMAC-SHA256 signed JWT bearer token. The token itself is valid for `ACCESS_TOKEN_EXPIRE_MINUTES` (24 hours); the 15-minute inactivity lock (`sessionTimeoutMinutes`) is enforced by the browser, not by the token.
- **Request Body:**
  ```json
  {
    "username": "clinician",
    "password": "ClinicalPassword2026!"
  }
  ```
- **Responses:**
  - `200 OK`:
    ```json
    {
      "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
      "token_type": "bearer",
      "user": {
        "id": "1",
        "name": "Dr. Demo Clinician (Simulated)",
        "email": "demo.clinician@research-prototype.invalid",
        "role": "Simulated Reviewer — Research Prototype",
        "licenseNumber": "SIM-000001",
        "facility": "Research Prototype Environment",
        "sessionTimeoutMinutes": 15,
        "loginTime": "2026-10-04T06:55:02.118Z"
      }
    }
    ```
  - `401 Unauthorized`: Invalid credentials.

### `GET /api/v1/auth/me`
- **Description:** Retrieves authenticated practitioner profile and facility metadata.
- **Headers:** `Authorization: Bearer <token>`
- **Response:** `200 OK` with user details.

---

## 2. Assessment Lifecycle Endpoints (`/api/v1/assessments`)

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
      "details": "The image met the configured technical thresholds (illumination index 1.00, dynamic range 55.3). This does not confirm retinal identity, anatomical correctness or clinical gradability.",
      "rejectionReason": null,
      "clinicalAction": null
    }
  ]
  ```

### `GET /api/v1/assessments/{id}/result`
- **Description:** Retrieves the model-generated class score distribution and explainability metadata.
- **Response `200 OK`:**
  ```json
  {
    "primaryClassGrade": 2,
    "primaryClassLabel": "Moderate NPDR",
    "primaryScore": 0.78,
    "classScores": [
      { "grade": 0, "label": "Grade 0: No Apparent DR", "score": 0.04 },
      { "grade": 1, "label": "Grade 1: Mild NPDR", "score": 0.12 },
      { "grade": 2, "label": "Grade 2: Moderate NPDR", "score": 0.78 },
      { "grade": 3, "label": "Grade 3: Severe NPDR", "score": 0.05 },
      { "grade": 4, "label": "Grade 4: Proliferative DR", "score": 0.01 }
    ],
    "targetLayer": "features.8 (Conv2d Bottleneck Residual)",
    "topActivationRegion": "Peak Grad-CAM activation in the central cell of a 3x3 grid over the frame (focal: 9% of the map is above half the peak).",
    "modelVersion": "EfficientNet-B0-DR-v1 (fixed weights)",
    "inferenceTimestamp": "2026-10-04T06:55:12.418Z"
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
