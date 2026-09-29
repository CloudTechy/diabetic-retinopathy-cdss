# Checkpoint Manifest & Cryptographic Integrity Ledger

## Metadata & Traceability
- **Research Project:** AI-Based Clinical Decision Support System for Early Detection of Diabetic Retinopathy
- **Author / Researcher:** Onyekelu Chukwuebuka Elochukwu (2024516020FN)
- **Related Research Objective:** Objective d (Design CNN architecture) & Objective e (Implement CNN model)
- **Date Verified:** 2026-09-29
- **Checkpoint File Path:** `backend/models/weights/efficientnet_b0_dr.pth`
- **Integrity Checksum (SHA-256):** `0d443fa065528b2a1d24baea7bf8d8bf71a6203b22b817585374a7becd547c07`
- **Training Epoch Selected:** Epoch 14 (Validation QWK: 0.8826)
- **Weights Binary Size:** **15.60 MB**

---

## 1. Architectural Topology Summary

| Specification Parameter | Value | Architectural Details |
| :--- | :---: | :--- |
| **Model Family** | EfficientNet-B0 | Compound-scaled lightweight CNN backbone |
| **Total Parameter Count** | **4,013,953** | 4.01 Million total parameters |
| **Trainable Parameters** | **4,013,953** (Trained) | Fully optimized parameters saved in state dict |
| **Floating-Point Complexity** | **~0.39 GFLOPs** | Multiply-Accumulate operations per $224 \times 224$ input |
| **Input Shape** | `(B, 3, 224, 224)` | RGB channels normalized with ImageNet prior |
| **Output Logits Shape** | `(B, 5)` | 5 unnormalized logits mapped via Softmax |
| **Target Saliency Layer** | `features.8` | Final 320-channel inverted residual bottleneck |

---

## 2. Checkpoint Verification Command
To verify the cryptographic integrity of the weights binary file:
```bash
# Windows PowerShell
Get-FileHash -Path backend/models/weights/efficientnet_b0_dr.pth -Algorithm SHA256

# Expected Checksum:
# 0d443fa065528b2a1d24baea7bf8d8bf71a6203b22b817585374a7becd547c07
```
