# System Architecture & Ophthalmic Data Flow Specification

## Metadata & Traceability
- **Research Project:** AI-Based Clinical Decision Support System for Early Detection of Diabetic Retinopathy
- **Author / Researcher:** Onyekelu Chukwuebuka Elochukwu (2024516020FN)
- **Related Research Objective:** Objective a (System architecture, workflow & database design)
- **Last Revised:** 2026-10-04

---

## 1. High-Level Architectural Tiers

The Diabetic Retinopathy Clinical Decision Support System (DR-CDSS) is designed as a decoupled three-tier clinical decision-support prototype for low-resource ophthalmic workstations and hospital local area networks. The packaged development configuration is not a secure deployment: `docker-compose.yml` defaults `SECRET_KEY` and the database password to published strings, runs with `DEBUG=true` and `--reload`, and serves plain HTTP; a production deployment must supply its own secrets through those same variables and terminate TLS in front of the API.

```mermaid
graph TD
    subgraph Tier1["Presentation Tier (Client Workstation)"]
        UI["React 18 + TypeScript SPA"]
        PreFlight["Client Pre-Flight CV Validator (retinalValidator.ts)"]
        Viewer["Dual-Layer WebGL / Canvas Viewer (FundusViewer.tsx)"]
    end

    subgraph Tier2["Application & Decision-Support Tier (FastAPI Engine)"]
        Router["RESTful API Gateways (/api/v1)"]
        ValPipe["3-Stage Validation Engine (Gates 1, 2, 3)"]
        StateMachine["Assessment State Machine Service"]
        PytorchEngine["EfficientNet-B0 Inference Harness (Frozen eval)"]
        GradCAM["Grad-CAM Saliency Engine (features.8)"]
        ReportEngine["Server-Side PDF Generator (ReportLab)"]
    end

    subgraph Tier3["Data & Persistence Tier"]
        DB[(PostgreSQL 16 Relational Store)]
        Vault[(Local application binary storage: /storage — image and heatmap files are served by filename without authentication)]
    end

    UI -->|HTTP REST — TLS termination is a production deployment requirement, not part of the packaged development stack| Router
    PreFlight -.->|Pre-screens files| UI
    Router --> StateMachine
    StateMachine --> ValPipe
    ValPipe -->|Fail-Closed Invariant| StateMachine
    StateMachine --> PytorchEngine
    PytorchEngine --> GradCAM
    StateMachine --> DB
    PytorchEngine --> Vault
    ReportEngine --> Vault
```

---

## 2. Data Flow Diagram (DFD Level 1) with Standardised Notation

> [!IMPORTANT]
> **Supervisor Notation Requirement:** In compliance with formal structured systems analysis conventions, external entities (sources and sinks of clinical data) are rendered with **two square-corner offset rectangles** representing the external boundary of the system.

### Visual Representation of External Entities & Processes:

```text
+-----------------------+
|  CLINICIAN / OPERATOR |-----+
+-----------------------+     | (Offset rectangle indicates External Entity)
  |   +-----------------------+
  |   |  CLINICIAN / OPERATOR |
  |   +-----------------------+
  |
  | 1. Uploads Fundus Image + Patient ID
  v
+-------------------------------------------------------+
| Process 1.0: Ingest & 3-Gate Technical Validation     |
+-------------------------------------------------------+
  |
  +---[Gate 1/2/3 Failure]---> State: Rejected (Inference Aborted)
  |                                 |
  |                                 v
  |                           +-----------------------+
  |                           |  AUDIT EVENT LOG      |-----+
  |                           +-----------------------+     |
  |                             +-----------------------+   |
  |                             |  AUDIT EVENT LOG      |---+
  |                             +-----------------------+
  |
  +---[All Gates Passed]
  |
  v
+-------------------------------------------------------+
| Process 2.0: Preprocessing & Frozen Neural Inference  |
| (EfficientNet-B0 backbone: features.8 hook)           |
+-------------------------------------------------------+
  |
  | 2. Computes 5-Class Score Distribution & Grad-CAM Heatmap
  v
+-------------------------------------------------------+
| Process 3.0: Presentation & Decision Support View     |
+-------------------------------------------------------+
  |
  | 3. Displays Interactive Dual-Layer Visual Attribution
  v
+-----------------------+
| REVIEWING OPHTHALMOL. |-----+
+-----------------------+     |
  |   +-----------------------+
  |   | REVIEWING OPHTHALMOL. |
  |   +-----------------------+
  |
  | 4. Submits Professional Review (Agree / Disagree / Unable to determine)
  v
+-------------------------------------------------------+
| Process 4.0: Review Sign-off, Report & Audit Storage |
+-------------------------------------------------------+
  |
  +---> Persists the write-locked review to PostgreSQL
  +---> Generates the hash-anchored Assessment Report PDF
```

---

## 3. The 9-Stage Assessment Lifecycle State Machine

```mermaid
stateDiagram-v2
    [*] --> Draft : Create Assessment
    Draft --> Uploaded : Image Uploaded
    Uploaded --> Validating : Launch 3-Gate Pipeline
    
    Validating --> Rejected : Gate 1, 2, or 3 Failure
    Validating --> Accepted : All 3 Gates Cleared
    
    Accepted --> Preprocessing : Image Resampling & Normalization
    Preprocessing --> Inference : EfficientNet-B0 Forward Pass
    Inference --> ResultReady : Softmax Scores + Grad-CAM Artifacts Generated
    
    ResultReady --> UnderReview : Clinician Opens Review Modal
    UnderReview --> Completed : Clinician records review
    
    Rejected --> [*] : Terminal (Inference Prohibited)
    Completed --> [*] : Write-Locked Record
```

---

## 3b. Accounts: roles, states and the use cases each role has

```mermaid
flowchart LR
    subgraph Actors
        A1(["Super Administrator"])
        A2(["Physician"])
        A3(["Prospective physician"])
    end
    A3 --> U0["Request an account (self-registration)"]
    A1 --> U1["Create physician account / batch"]
    A1 --> U2["Approve or reject a registration"]
    A1 --> U3["Suspend or reactivate an account"]
    A1 --> U4["Issue a temporary password"]
    A1 --> U5["View account status and audit history"]
    A2 --> U6["Sign in, change password, sign out"]
    A1 --> U6
    A2 --> U7["Create assessment, validate, infer, review, report (own records only)"]
```

```mermaid
stateDiagram-v2
    [*] --> PENDING_APPROVAL : self-registration
    [*] --> ACTIVE : created by Super Administrator (temporary password)
    PENDING_APPROVAL --> ACTIVE : approve
    PENDING_APPROVAL --> REJECTED : reject
    ACTIVE --> SUSPENDED : suspend (not the last active Super Administrator)
    SUSPENDED --> ACTIVE : reactivate
```

```mermaid
sequenceDiagram
    participant SA as Super Administrator
    participant API as FastAPI (/admin, /auth)
    participant DB as PostgreSQL
    participant P as Physician
    SA->>API: POST /admin/users (fullName, username, email)
    API->>DB: INSERT users (ACTIVE, must_change_password=true); INSERT audit_events x2
    API-->>SA: 201 { user, temporaryPassword }  (shown once)
    P->>API: POST /auth/login (temporary password)
    API->>DB: last_login_at; audit SIGN_IN_SUCCESS
    API-->>P: 200 token (mustChangePassword=true)
    P->>API: GET /assessments
    API-->>P: 403 PASSWORD_CHANGE_REQUIRED
    P->>API: POST /auth/change-password
    API->>DB: hashed_password; must_change_password=false; token_version+1; audit PASSWORD_CHANGED
    API-->>P: 200 (all earlier tokens now refused)
    P->>API: POST /auth/login (new password)
    API-->>P: 200 token
    P->>API: GET /assessments
    API-->>P: 200 [] (no record yet: the workspace is the account's own)
```

Enforcement is in the backend: `get_current_user` (any ACTIVE account, token version checked), `get_current_physician` (ACTIVE PHYSICIAN, password not temporary) on every clinical route, `get_current_superadmin` on every `/admin` route. The record queries take the caller's account id, so a physician's worklist, search and single-record routes return their own records only.

## 4. Key Architectural Safeguards & Design Decisions

1. **Physical & Logical Domain Separation:**
   - Model predictions (`ai_results`) and professional review responses (`professional_reviews`) reside in separate database entities. Nothing automated writes to a review.
2. **Deterministic Weights Preservation:**
   - The PyTorch inference harness loads weights in read-only mode (`torch.load(..., weights_only=True)`), explicitly sets `model.eval()`, and disables parameter autograd tracking (`requires_grad = False`).
3. **Fail-Closed Gate Architecture:**
   - Any failure across file integrity, spectral balance, or blur thresholds guarantees that no tensor forward pass is initiated, preventing invalid classification of corrupted data.
4. **Offline Local First Capability:**
   - The complete request path runs on the deployment CPU — **mean 180.41 ms, P95 342.49 ms** end-to-end over 30 held-out images on the canonical run ([`resource_benchmark.md`](resource_benchmark.md) §1), of which the forward pass itself is 29.22 ms — so the system needs no external cloud AI API and no image leaves the host.
