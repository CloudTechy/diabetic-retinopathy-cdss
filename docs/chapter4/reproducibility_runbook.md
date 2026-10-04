# Empirical Research Reproducibility Runbook

## Metadata & Traceability
- **Research Project:** AI-Based Clinical Decision Support System for Early Detection of Diabetic Retinopathy
- **Author / Researcher:** Onyekelu Chukwuebuka Elochukwu (2024516020FN)
- **Related Research Objectives:** d, e, f, g, h, i
- **Last Revised:** 2026-10-04
- **Reference run:** Google Colab Tesla T4, PyTorch 2.11.0+cu128, 2026-09-30 (clean rerun), checkpoint `67d0b896…`

---

## 0. Three levels of reproduction

Be clear about which one you need — they have very different costs.

| Level | What it proves | Needs | Time |
| :--- | :--- | :--- | :---: |
| **A. Verify the reported metrics** | Every number in Chapter 4 follows from the committed predictions | Python 3.11+ only | < 5 s |
| **B. Verify artefact provenance** | The committed weights and dataset manifest are what they claim | Python 3.11+, APTOS download for the image hashes | minutes |
| **C. Re-train from scratch** | The pipeline end-to-end | Kaggle account, GPU runtime, 9.51 GB download | ~1.5 h |

**Level A requires no ML dependencies at all.** An examiner can confirm the statistical claims without installing PyTorch.

---

## 1. Level A — Verify the reported metrics (no ML dependencies)

```bash
python backend/scripts/analyze_clinical_metrics.py
```

Recomputes, from [`held_out_predictions.csv`](held_out_predictions.csv) alone, using only the standard library:

- Quadratic Weighted Kappa (implemented from first principles, not imported)
- Exact accuracy, within-one-grade agreement, over/under-call rates
- Per-class sensitivity, specificity, precision, F1, with Wilson confidence intervals
- Referable / sight-threatening / any-DR operating points
- The byte-level duplicate-leakage audit and the clean-subset re-scoring

### Expected output (abridged)

```text
HELD-OUT COHORT: N = 525
Exact accuracy       84.00%
Within-1-grade       93.71%
QWK                  0.865832

--- Referable DR (grade >= 2) ---
  Sensitivity 91.2%  95% CI (86.5, 94.4)
  Specificity 95.6%  95% CI (92.8, 97.4)

--- LEAKAGE AUDIT ---
  duplicate groups: 3504, each collapsed to one representative
    -> a group cannot span partitions by construction
  held-out images byte-identical to a training image: 0/525 (0.0%)
```

Any divergence from [`model_evaluation_report.md`](model_evaluation_report.md) is a defect. Please report it.

> [!NOTE]
> This block previously quoted a referable 95% CI of (81.5, 90.5) against a
> point estimate of 91.2% — an interval that excludes its own estimate — plus
> two lines about "the 27 affected images" and "the 522 clean images" held over
> from the superseded run. The clean split leaves **zero** byte-identical
> held-out images, so the script prints no clean-subset comparison at all. The
> block above is the current output verbatim. A CI that does not bracket its
> estimate is now caught by the evidence integrity gate.

---

## 2. Level B — Verify artefact provenance

### 2.1 Checkpoint identity

```bash
sha256sum backend/models/weights/efficientnet_b0_dr.pth
# Must print:
# 67d0b89641f08057126dd411e380b25575ef29f71ae37ee5796d472d9203dbf7
```

PowerShell:
```powershell
Get-FileHash backend\models\weights\efficientnet_b0_dr.pth -Algorithm SHA256
```

This same digest is asserted at runtime by the inference service, and by `test_repository_checkpoint_matches_declared_digest` in the test suite.

### 2.2 Dataset manifest integrity

[`dataset_split_manifest.csv`](dataset_split_manifest.csv) carries the SHA-256 of each image's **actual bytes**. With the APTOS `train_images/` directory available:

```bash
python docs/chapter4/verify_manifest_hashes.py
```

Or spot-check a single row by hand:

```bash
sha256sum aptos2019/train_images/005b95c28852.png
# Compare against the sha256_hash column for image_id 005b95c28852
```

> This check is the one that exposed the earlier fabricated manifest, in which **0 of 3,662** recorded hashes matched the real files. It is worth running.

### 2.3 Training/inference preprocessing parity

```bash
grep -n "Resize\|ToTensor\|Normalize" notebooks/colab_train_and_evaluate.py
grep -n "Resize\|ToTensor\|Normalize" backend/app/services/ai_service.py
```

Both must show `Resize((224, 224))` → `ToTensor()` → `Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])`. Divergence here would invalidate the held-out metrics as a predictor of runtime behaviour.

---

## 3. Level C — Re-train from scratch

### 3.1 Requirements

| Component | Requirement |
| :--- | :--- |
| **Accelerator** | NVIDIA GPU, ≥ 8 GB VRAM (reference run: Colab Tesla T4) |
| **Disk** | ~15 GB free (APTOS download is 9.51 GB) |
| **Kaggle** | Account with the APTOS 2019 competition rules accepted, plus an API token (`kaggle.json`) |
| **Python** | 3.11 – 3.13 |
| **Wall-clock** | ~40 min download + ~53 min training + ~2 min evaluation |

### 3.2 Execution (Google Colab)

**Cell 1** — supply the Kaggle token:
```python
from google.colab import files
import os
uploaded = files.upload()          # select kaggle.json
os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
with open(os.path.expanduser("~/.kaggle/kaggle.json"), "wb") as f:
    f.write(uploaded["kaggle.json"])
os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)
```

**Cell 2** — run the pipeline (set Runtime → Change runtime type → **T4 GPU** first):
```python
!git clone https://github.com/CloudTechy/diabetic-retinopathy-cdss.git
%cd diabetic-retinopathy-cdss
!python notebooks/colab_train_and_evaluate.py
```

**Cell 3** — retrieve artefacts:
```python
from google.colab import files
files.download('chapter4_evidence.zip')
```

The script downloads APTOS, builds the manifest with real byte hashes, trains 15 epochs, evaluates the held-out split, benchmarks, plots, and packages everything.

### 3.3 What will and will not reproduce exactly

**Will reproduce:** the dataset partition (seeded at 42 and independently verifiable against the committed manifest), the training procedure, and metrics in the same neighbourhood.

**Will not reproduce bit-exactly:**
- `torch.backends.cudnn.deterministic` is not set and `cudnn.benchmark` is not disabled, so cuDNN may select non-deterministic kernels.
- DataLoader workers are not seeded via `worker_init_fn`.
- Colab does not guarantee the same GPU model or driver between sessions.

Expect the checkpoint SHA-256 to differ and metrics to vary by a small margin. This is stated plainly rather than papered over; see [`training_environment.md`](training_environment.md) §3.

---

## 4. Run the application

```bash
cp .env.example .env
docker compose up --build
```

Then confirm the inference engine actually loaded real weights:

```bash
curl http://localhost:8000/api/v1/health
```

```json
{
  "status": "healthy",
  "inference_ready": true,
  "inference_engine": "pytorch",
  "inference_detail": "EfficientNet-B0 checkpoint loaded from /app/models/weights/efficientnet_b0_dr.pth"
}
```

- `"inference_engine": "mock"` means simulated grades are being served — check `AI_INFERENCE_ENGINE`.
- `"status": "degraded"` with `inference_ready: false` means the checkpoint is missing or failed digest verification. The system will return `503` on assessment submission rather than grade from untrained weights.

---

## 4b. Build and run the application from the archive

```bash
cd frontend && npm ci && npm run build && cd ..
docker compose config
docker compose up --build
```

On Windows, extract to a short path first (paths beyond 260 characters break `npm ci`'s nested binaries). The frontend build is `tsc && vite build`; the compose stack builds `backend/Dockerfile` and `frontend/Dockerfile` and needs no `.env` (every variable has a default). [`build_verification.log`](build_verification.log) is the transcript of exactly these commands from a clean extraction, ending with the backend's health response.

## 5. Run the test suite

From a fresh extraction of the archive (or a clean clone), in a new virtual environment built from `backend/requirements.txt`. This is exactly how the committed [`test_execution.log`](test_execution.log) was produced: its header records the interpreter and `rootdir`, and [`test_environment_freeze.txt`](test_environment_freeze.txt) is the `pip freeze` of that environment. A rule in the suite checks that freeze satisfies `requirements.txt`.

```bash
cd backend
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pytest tests/ -v
# Linux/macOS: .venv/bin/python in place of .venv/Scripts/python.exe
```

Expected: **214 collected — 213 passed, 1 skipped** (wall-clock ≈ 60–75 s with PyTorch installed; the committed log records 61 s). The skip is `test_contaminated_results_are_labelled_superseded`, conditional on an artefact of the superseded run being present. `test_real_model_end_to_end_pipeline` is **not** skipped: it sets `AI_INFERENCE_ENGINE=pytorch` for its own duration and runs against the digest-verified checkpoint using the genuine held-out fixture. It carries the `slow` marker so it *can* be deselected with `-m "not slow"`; nothing deselects it by default.
