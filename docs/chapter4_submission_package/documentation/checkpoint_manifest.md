# Neural Network Checkpoint Manifest (Objective d & e)

## Metadata & Academic Provenance
- **Model Backbone:** EfficientNet-B0 (torchvision implementation)
- **Trained Weights Checkpoint:** `backend/models/weights/efficientnet_b0_dr.pth`
- **File Size:** 15.60 MB (16,358,249 bytes)
- **Cryptographic SHA-256 Digest:** `67d0b89641f08057126dd411e380b25575ef29f71ae37ee5796d472d9203dbf7`
- **Training Run:** 15 epochs on APTOS 2019, Google Colab Tesla T4, 2026-09-29
- **Selected Epoch:** **14** of 15 (peak validation $\kappa = 0.913045$, recorded in [`epoch_history.csv`](epoch_history.csv))
- **Degree Programme:** PGD Computer Science, Faculty of Physical Sciences

---

## 1. Architectural Details & Layer Hook Specification

- **Input Tensor Dimensions:** `(Batch, 3, 224, 224)` with ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.
- **Feature Extractor:**
  - `features.0` to `features.6`: Mobile Inverted Bottleneck (MBConv) stages.
  - `features.7`: Final MBConv stage producing 320 feature channels.
  - `features.8`: Final $1 \times 1$ pointwise convolutional expansion layer producing **1,280 feature channels** (`Conv2dNormActivation(320, 1280)`).
- **Explainability Target Layer:**
  - Grad-CAM hooks `features.8` (the 1,280-channel final convolutional feature layer) to extract high-level visual saliency patterns before global average pooling.
- **Classification Head:**
  - `nn.AdaptiveAvgPool2d(1)` $\to$ `nn.Dropout(p=0.2)` $\to$ `nn.Linear(in_features=1280, out_features=5)`.
- **Total Parameter Count:** 4,013,953 parameters (confirmed by the training log).
- **Trainable Parameters at Inference:** 0 — the served graph is `eval()` with `requires_grad = False` on every parameter.

---

## 2. Checkpoint Contents

The file is a plain `state_dict` produced by `torch.save(model.state_dict(), ...)`. It contains weights only: no optimizer state, no epoch counter, no pickled model class. It is loadable under `weights_only=True`, which is how the backend loads it.

---

## 3. Runtime Provenance Enforcement

This digest is not documentation alone. The backend verifies it before serving:

- `MODEL_CHECKPOINT_SHA256` in [`backend/app/core/config.py`](../../backend/app/core/config.py) carries the digest above.
- `EfficientNetB0InferenceService._verify_checkpoint_digest()` hashes the file on load and raises `ModelCheckpointError` on any mismatch.
- A missing, corrupt, or structurally incompatible checkpoint also raises. The engine never falls back to randomly-initialised weights or to the simulated engine.

This means a graded prediction served by the running system is demonstrably produced by the same bytes that produced [`model_evaluation_report.md`](model_evaluation_report.md).

---

## 4. Verification Command

```bash
sha256sum backend/models/weights/efficientnet_b0_dr.pth
# Output must match:
# 67d0b89641f08057126dd411e380b25575ef29f71ae37ee5796d472d9203dbf7  backend/models/weights/efficientnet_b0_dr.pth
```

On Windows PowerShell:

```powershell
Get-FileHash backend\models\weights\efficientnet_b0_dr.pth -Algorithm SHA256
```
