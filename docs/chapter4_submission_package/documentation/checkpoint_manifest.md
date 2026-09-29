# Neural Network Checkpoint Manifest (Objective d & e)

## Metadata & Academic Provenance
- **Model Backbone:** EfficientNet-B0 (torchvision implementation)
- **Trained Weights Checkpoint:** `backend/models/weights/efficientnet_b0_dr.pth`
- **File Size:** 15.60 MB (16,353,193 bytes)
- **Cryptographic SHA-256 Digest:** `0d443fa065528b2a1d24baea7bf8d8bf71a6203b22b817585374a7becd547c07`
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
- **Total Parameter Count:** 4,013,953 parameters.

---

## 2. Verification Command
To verify the cryptographic integrity of the weights binary:
```bash
sha256sum backend/models/weights/efficientnet_b0_dr.pth
# Output must match:
# 0d443fa065528b2a1d24baea7bf8d8bf71a6203b22b817585374a7becd547c07  backend/models/weights/efficientnet_b0_dr.pth
```
