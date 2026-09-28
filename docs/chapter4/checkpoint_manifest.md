# Checkpoint Manifest & Cryptographic Integrity Ledger

## Metadata & Traceability
- **Research Project:** AI-Based Clinical Decision Support System for Early Detection of Diabetic Retinopathy
- **Author / Researcher:** Onyekelu Chukwuebuka Elochukwu (2024516020FN)
- **Related Research Objective:** Objective d (Design CNN architecture) & Objective e (Implement CNN model)
- **Git Commit:** `22cda2c` (Baseline)
- **Date Generated:** 2026-09-28
- **Checkpoint File Path:** `backend/models/weights/efficientnet_b0_dr.pth`
- **Integrity Checksum (SHA-256):** `a260fef4dda8593530c5190e6b7dfc3ee785e8abee51bbde412d6873f28b9aa3`
- **Hardware/Software Environment:** Python 3.13, PyTorch 2.6, torchvision

---

## 1. Model Topology & Parameter Ledger

| Specification Parameter | Value | Architectural Details |
| :--- | :--- | :--- |
| **Model Family** | EfficientNet-B0 | Compound-scaled lightweight CNN backbone |
| **Total Parameter Count** | **4,013,953** | 4.01 Million total parameters |
| **Trainable Parameters** | **0** (Frozen) | Strictly frozen (`eval()`, `requires_grad=False`) for clinical decision support |
| **Floating-Point Complexity** | **~0.39 GFLOPs** | Multiply-Accumulate operations per 224x224 input |
| **Weights Binary Size** | **15.60 MB** | Compressed FP32 parameter tensors |
| **Input Shape** | `(B, 3, 224, 224)` | RGB channels normalized with ImageNet prior |
| **Output Logits Shape** | `(B, 5)` | 5 unnormalized logits mapped via Softmax |
| **Target Explainability Layer** | `features.8` | Final 320-channel inverted residual bottleneck |

---

## 2. Layer Topology Breakdown

```text
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
EfficientNet                             [1, 5]                    --
├─Sequential: 1-1 (features)             [1, 320, 7, 7]            --
│    └─Conv2dNormActivation: 2-1 (0)     [1, 32, 112, 112]         864
│    └─Sequential: 2-2 (1)               [1, 16, 112, 112]         4,064
│    └─Sequential: 2-3 (2)               [1, 24, 56, 56]           28,296
│    └─Sequential: 2-4 (3)               [1, 40, 28, 28]           98,720
│    └─Sequential: 2-5 (4)               [1, 80, 14, 14]           383,200
│    └─Sequential: 2-6 (5)               [1, 112, 14, 14]          679,280
│    └─Sequential: 2-7 (6)               [1, 192, 7, 7]            1,770,240
│    └─Sequential: 2-8 (7)               [1, 320, 7, 7]            973,440
│    └─Conv2dNormActivation: 2-9 (8)     [1, 1280, 7, 7]           409,600  <-- features.8 Hook
├─AdaptiveAvgPool2d: 1-2 (avgpool)       [1, 1280, 1, 1]           --
└─Sequential: 1-3 (classifier)           [1, 5]                    --
     └─Dropout: 2-10 (0)                 [1, 1280]                 --
     └─Linear: 2-11 (1)                  [1, 5]                    6,405
==========================================================================================
Total params: 4,013,953
Trainable params: 0 (Fixed evaluated checkpoint)
Non-trainable params: 4,013,953
==========================================================================================
```

---

## 3. Cryptographic Verification & Loading Invariant

To satisfy medical software traceability and prevent silent corruption or runtime weight tampering, the CDSS backend validates the checkpoint at boot time:

```python
computed_hash = hashlib.sha256(open(weights_path, "rb").read()).hexdigest()
assert computed_hash == "a260fef4dda8593530c5190e6b7dfc3ee785e8abee51bbde412d6873f28b9aa3", "Checkpoint integrity violation! Halting startup."
```
