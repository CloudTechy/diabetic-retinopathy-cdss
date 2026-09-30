# Clinical REST API Contract & Specification

## Metadata & Traceability
- **Research Project:** AI-Based Clinical Decision Support System for Early Detection of Diabetic Retinopathy
- **Author / Researcher:** Onyekelu Chukwuebuka Elochukwu (2024516020FN)
- **Related Research Objective:** Objective g (Design CDSS software architecture)
- **API Standard:** OpenAPI 3.1.0 / RESTful JSON / RFC 7807 Problem Details
- **Base Endpoint:** `/api/v1`
- **Git Commit:** `22cda2c` (Baseline)
- **Date Approved:** 2026-09-28

---

## 1. Authentication & Session Endpoints (`/api/v1/auth`)

### `POST /api/v1/auth/login`
- **Description:** Authenticates clinical user and issues an HMAC-SHA256 signed JWT bearer token.
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
      "expires_in": 900,
      "user": {
        "id": "usr-01",
        "name": "Dr. Ada Okonjo",
        "role": "Consultant Medical Ophthalmologist",
        "licenseNumber": "SIM-000001"
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
- **Response `201 Created`:** Draft assessment object with assigned UUID.

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
      "gateIndex": 1,
      "title": "File Integrity & Safe Decode",
      "status": "passed",
      "metrics": { "mimeType": "image/jpeg", "fileSizeBytes": 1420950 }
    },
    {
      "gateIndex": 2,
      "title": "Retinal Relevance & Ophthalmic Geometry",
      "status": "passed",
      "metrics": { "aspectRatio": 1.0, "chromaticRatioRB": 1.48 }
    },
    {
      "gateIndex": 3,
      "title": "Technical Image Quality",
      "status": "passed",
      "metrics": { "laplacianVariance": 142.4, "illuminationIndex": 0.54 }
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
    "topActivationRegion": "Inferotemporal quadrant parafoveal hemorrhages and exudates",
    "modelVersion": "EfficientNet-B0-DR-v1 (Weights frozen)",
    "executionTimeMs": 284.5,
    "disclaimer": "NOTICE: CLINICAL DECISION SUPPORT ONLY — NOT FOR INDEPENDENT DIAGNOSIS..."
  }
  ```

### `POST /api/v1/assessments/{id}/review`
- **Description:** Submits authoritative human clinician certification and locks the consultation record.
- **Request Body:**
  ```json
  {
    "agreement": "agree",
    "certifiedGrade": 2,
    "certifiedGradeLabel": "Grade 2: Moderate NPDR",
    "justificationNotes": "Macular exudates corroborated on slit lamp exam.",
    "referralPlan": "Referral to secondary care / hospital medical retina clinic."
  }
  ```
- **Response `200 OK`:** Finalized assessment object with cryptographic signature and state `completed`.

### `GET /api/v1/assessments/{id}/report`
- **Description:** Streams server-rendered tamper-evident PDF assessment report.
- **Response `200 OK`:** `Content-Type: application/pdf`.

### `GET /api/v1/assessments/{id}/audit`
- **Description:** Returns chronological append-only audit trail for legal compliance.
- **Response `200 OK`:** Array of timestamped audit events.
