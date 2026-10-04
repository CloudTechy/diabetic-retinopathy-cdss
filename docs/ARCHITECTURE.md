# System Architecture & Infrastructure Overview

## 1. Overview
The **Diabetic Retinopathy Clinical Decision Support System (CDSS)** is an ophthalmic AI-assisted triage platform designed to evaluate retinal fundus photography, enforce rigorous technical quality and integrity gates, compute model-generated class scores, and provide interactive Grad-CAM visual attributions while preserving human-in-the-loop clinician authority.

## 2. Service Topology

```text
               +-------------------------------------------+
               |        Client Web Browser                 |
               | (Clinician / Technician / Administrator)   |
               +--------------------+----------------------+
                                    |
                                    v [HTTP :3000 / :5173]
                       +------------+------------+
                       |    Frontend Container   |
                       |    (React 18 + Vite)    |
                       +------------+------------+
                                    |
                                    v [HTTP API :8000]
                       +------------+------------+
                       |    Backend Container    |
                       |        (FastAPI)        |
                       +-----+-------------+-----+
                             |             |
           +-----------------+             +-----------------+
           |                                                 |
           v [:5432]                                         v [/app/storage]
+----------+----------+                           +----------+----------+
|  PostgreSQL Database|                           | Private Data Volume |
|   (PostgreSQL 16)   |                           | - /storage/images   |
|   Metadata & Audit  |                           | - /storage/reports  |
+---------------------+                           | - /attributions     |
+---------------------+
```

## 3. Storage Isolation & Security
- **Medical Images (`/storage/images`)**: Private local disk volume storing original uploaded images with SHA-256 integrity hashes.
- **Audit Logs & Metadata**: PostgreSQL relational database tracking validation pipeline status, model scores, clinician overrides, and timestamps.
- **Attributions (`/storage/attributions`)**: Grad-CAM visual saliency heatmaps generated upon inference.
- **Reports (`/storage/reports`)**: Generated PDF assessment reports, hash-anchored to the image.

## 4. Environment Execution
- **Containerization**: Managed via `docker-compose.yml` supporting hot reload development and isolated testing.
- **Fail-Closed Validation**: If an image fails format, resolution, blur, illumination, or field-of-view checks, AI inference is blocked.
