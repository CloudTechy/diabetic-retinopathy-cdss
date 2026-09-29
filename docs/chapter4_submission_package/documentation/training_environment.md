# Model Training & Computational Environment Specification

## Metadata & Traceability
- **Research Project:** AI-Based Clinical Decision Support System for Early Detection of Diabetic Retinopathy
- **Author / Researcher:** Onyekelu Chukwuebuka Elochukwu (2024516020FN)
- **Related Research Objective:** Objective e (Implement the CNN) & Objective f (Train the CNN)
- **Date Generated:** 2026-09-29
- **Raw Evidence:** [`training_summary.json`](training_summary.json), [`training_execution.log`](training_execution.log)

---

## 1. Training Hardware — Google Colab (Tesla T4)

Training was executed on a hosted Google Colab GPU runtime. The environment values below are reported by PyTorch at run time and are recorded verbatim in [`training_summary.json`](training_summary.json).

| Component | Specification | Role |
| :--- | :--- | :--- |
| **Accelerator** | NVIDIA Tesla T4 (16 GB GDDR6) | Forward/backward passes, tensor operations |
| **Runtime** | Google Colab hosted GPU instance | Training and evaluation execution |
| **Framework** | PyTorch **2.11.0+cu128** | Autograd, optimisation, checkpointing |
| **CUDA Build** | 12.8 (per the `+cu128` wheel tag) | GPU kernel execution |
| **Training Duration** | 3,147 s total (~210 s/epoch × 15) | — |
| **DataLoader Workers** | 2 | Colab's standard allocation |

**Why a hosted runtime.** APTOS 2019 is a 9.51 GB download and the run required ~52 minutes of sustained GPU time. Colab was used for the training and evaluation stages only. The CDSS application itself is developed and deployed separately, as below.

---

## 2. Deployment & Development Environment

The trained checkpoint is served by the FastAPI backend on **CPU**, not on the training accelerator. `MODEL_DEVICE` defaults to `cpu` in [`.env.example`](../../.env.example) and `docker-compose.yml`.

| Category | Package / Tool | Version | Functional Role |
| :--- | :--- | :---: | :--- |
| **Deep Learning Framework** | `torch` | 2.6.0+ | Inference execution in the backend container |
| **Vision Library** | `torchvision` | 0.21.0+ | EfficientNet-B0 topology, ImageNet transforms |
| **Image Processing** | `pillow` (PIL) | 11.1.0 | Decoding, format conversion, resampling |
| **Scientific Computing** | `numpy` | 2.1.3 | Grad-CAM activation arrays, metric computation |
| **Backend Web Framework** | `fastapi` | 0.110.0 | RESTful API routing inference requests |
| **Database Engine** | `PostgreSQL` | 16.2 | Relational metadata and immutable audit logging |
| **Frontend Framework** | `React` / `TypeScript` | 18.2 / 5.4 | Clinical browser user interface |
| **Frontend Build Tool** | `Vite` | 5.4.21 | Development server and production build |
| **Container Runtime** | Docker Compose | — | Service orchestration |
| **Development OS** | Microsoft Windows 11 (64-bit) | — | Local development workstation |

**Training and serving environments differ, deliberately.** The model is trained once on a GPU and served many times on CPU. The consequence for reported latency is stated explicitly in [`resource_benchmark.md`](resource_benchmark.md): the committed benchmark was measured on the T4, so it is a *training-environment* figure and does not characterise the CPU deployment target.

---

## 3. Determinism & Reproducibility Controls

Applied in [`notebooks/colab_train_and_evaluate.py`](../../notebooks/colab_train_and_evaluate.py):

1. **Random Seeds:** `random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)`, `torch.cuda.manual_seed_all(42)`.
2. **Deterministic split:** the stratified partition is generated under the same seed before training begins, and is committed as [`dataset_split_manifest.csv`](dataset_split_manifest.csv) with a SHA-256 per image, so the exact partition can be reconstructed and verified independently of re-running the seed.
3. **Evaluation determinism:** held-out inference runs under `torch.inference_mode()` with `model.eval()` and no augmentation.

### Known limits on bit-exact reproduction

Honest scope: re-running this script will not necessarily reproduce the checkpoint byte-for-byte.

- `torch.backends.cudnn.deterministic` was **not** set, and `cudnn.benchmark` was **not** disabled, so cuDNN may select non-deterministic algorithms.
- DataLoader worker seeding was not pinned via `worker_init_fn`, so multi-worker sample ordering may vary.
- Colab does not guarantee the same GPU model or driver between sessions.

What *is* reproducible and verifiable without re-training: the dataset partition (via the committed per-image SHA-256 manifest), every reported metric (recomputable from [`held_out_predictions.csv`](held_out_predictions.csv) by [`analyze_clinical_metrics.py`](../../backend/scripts/analyze_clinical_metrics.py)), and the identity of the evaluated weights (via the SHA-256 in [`checkpoint_manifest.md`](checkpoint_manifest.md)).
