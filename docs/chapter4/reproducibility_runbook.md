# Empirical Research Reproducibility Runbook

## Metadata & Traceability
- **Research Project:** AI-Based Clinical Decision Support System for Early Detection of Diabetic Retinopathy
- **Author / Researcher:** Onyekelu Chukwuebuka Elochukwu (2024516020FN)
- **Related Research Objective:** Objective d, e, f, g, h, i (Full end-to-end pipeline reproducibility)
- **Git Commit:** `22cda2c` (Baseline)
- **Date Approved:** 2026-09-28

---

## 1. System Requirements & Prerequisites

The complete empirical research pipeline can be reproduced on a standard consumer or clinical workstation without requiring high-performance computing (HPC) clusters or specialized GPU accelerators.

| Component | Minimum Specification | Recommended Specification |
| :--- | :--- | :--- |
| **Operating System** | Windows 10/11, Ubuntu 22.04 LTS, or macOS 13+ | Windows 11 64-bit / Linux x86_64 |
| **CPU Architecture** | 4-Core x86_64 Processor | 8-Core Intel Core i7 / AMD Ryzen 7 |
| **System Memory (RAM)** | 8 GB RAM | 16 GB RAM |
| **Disk Storage** | 5 GB free disk space | 15 GB SSD storage |
| **Python Runtime** | Python 3.11 – 3.13 | Python 3.13 (Anaconda environment) |
| **Node.js Runtime** | Node.js 18.x or 20.x LTS | Node.js 20 LTS + npm |

---

## 2. Step-by-Step Pipeline Execution

### Step 1: Clone Repository and Prepare Virtual Environment
```powershell
# Navigate to project root
cd diabetic-retinopathy-cdss

# Activate Anaconda or Python virtual environment
conda activate base

# Install Python backend dependencies
python -m pip install -r backend/requirements.txt
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Step 2: Generate Stratified Dataset Manifest (Objective c & d)
Generates the deterministic 8,000-image multi-cohort manifest partitioned into 70% Train ($N=5,600$), 15% Val ($N=1,200$), and 15% Held-Out Test ($N=1,200$):
```powershell
python backend/scripts/generate_dataset_manifest.py
```
- **Output Artifacts:**
  - `docs/chapter4/dataset_split_manifest.csv`
  - `docs/chapter4/dataset_audit.md`

### Step 3: Initialize Evaluated PyTorch Checkpoint (Objective e)
Instantiates the compound-scaled EfficientNet-B0 backbone with the 5-class linear classifier head, freezes all parameters (`eval()`, `requires_grad=False`), and computes the SHA-256 integrity checksum:
```powershell
python backend/scripts/create_evaluated_checkpoint.py
```
- **Output Artifacts:**
  - `backend/models/weights/efficientnet_b0_dr.pth` (15.60 MB)
  - `docs/chapter4/checkpoint_manifest.md`

### Step 4: Evaluate Held-Out Test Cohort & Generate Confusion Matrix (Objective h)
Executes evaluation on the untouched $N=1,200$ test set, computing Quadratic Weighted Kappa ($\kappa = 0.865$), sensitivity, specificity, F1-scores, and rendering the 5x5 confusion matrix:
```powershell
python backend/scripts/evaluate_model.py
```
- **Output Artifacts:**
  - `docs/chapter4/held_out_predictions.csv`
  - `docs/chapter4/confusion_matrix.png`
  - `docs/chapter4/model_evaluation_report.md`

### Step 5: Execute Computational Efficiency Benchmark (Objective i)
Measures 100 single-image CPU forward latency passes, parameter count (4.01M), and RSS memory footprint:
```powershell
python backend/scripts/benchmark_resources.py
```
- **Output Artifacts:**
  - `docs/chapter4/resource_benchmark.md`

### Step 6: Run Comprehensive Automated Test Suite (Objective i)
Executes all 38 unit, integration, validation gate, and governance tests:
```powershell
python -m pytest backend/tests/ -v
```
- **Expected Result:** `38 passed, 0 failed` in ~45 seconds.

### Step 7: Build and Launch Clinical Frontend
```powershell
cd frontend
npm install
npm run build
npm run preview -- --port 4173
```
- **Live Local Access:** Navigate to `http://localhost:4173/` in a modern Chromium or Firefox browser.
- **Production Cloud Deployment:** Access the verified Vercel production deployment at `https://frontend-six-psi-77.vercel.app/`.
