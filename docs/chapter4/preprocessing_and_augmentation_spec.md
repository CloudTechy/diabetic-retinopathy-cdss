# Retinal Image Preprocessing & Augmentation Specification

## Metadata & Traceability
- **Research Project:** AI-Based Clinical Decision Support System for Early Detection of Diabetic Retinopathy
- **Author / Researcher:** Onyekelu Chukwuebuka Elochukwu (2024516020FN)
- **Related Research Objective:** Objective c (Preprocess and partition the retinal dataset) & Objective d
- **Git Commit:** `22cda2c` (Baseline)
- **Date Generated:** 2026-09-28
- **Evidence Files:** [`docs/chapter4/dataset_audit.md`](file:///c:/Users/USER/Documents/TECH4MATION/diabetic-retinopathy-cdss/docs/chapter4/dataset_audit.md), [`backend/app/services/ai_service.py`](file:///c:/Users/USER/Documents/TECH4MATION/diabetic-retinopathy-cdss/backend/app/services/ai_service.py)
- **Hardware/Software Environment:** Python 3.13, PyTorch 2.6 / torchvision, Pillow 11.1, NumPy 2.1.3

---

## 1. Clinical Motivation & Preprocessing Rationale

Raw digital fundus photography varies widely in dimension (from $1536 \times 1024$ to $3888 \times 2592$), aspect ratio, camera aperture masking, and dark peripheral border thickness. Preprocessing serves three essential clinical and computational objectives:
1. **Geometric Standardization:** Eliminates non-informative black margins while centering the posterior pole and macular region.
2. **Computational Tractability:** Reduces memory footprint to permit high batch-size convergence on standard workstations.
3. **Distributional Alignment:** Standardizes input channels to match the pre-trained feature statistics of EfficientNet-B0.

```mermaid
flowchart LR
    Raw["Raw Fundus Photo (Variable px)"] --> Crop["1. Circular Aperture Cropping (Bounding Box)"]
    Crop --> Resize["2. Bicubic Resizing (224x224 px / 512x512 px)"]
    Resize --> ToTensor["3. Float32 Tensor Conversion [0.0, 1.0]"]
    ToTensor --> Norm["4. Channel Normalization (ImageNet Statistics)"]
    Norm --> Model["5. EfficientNet-B0 Input Tensor"]
```

---

## 2. Deterministic Inference Pipeline (Validation & Inference)

During both offline held-out test evaluation and runtime CDSS inference, preprocessing is **strictly deterministic**: zero random alterations, zero test-time augmentation (TTA), ensuring 100% reproducible decision-support outputs.

### Step-by-Step Mathematical Transforms:

1. **Aperture Margin Auto-Cropping:**
   Finds non-zero luminance bounding box:
   $$\text{Mask}(x,y) = \mathbb{I}\left(0.299 R + 0.587 G + 0.114 B > 15.0\right)$$
   Crops image to $[\min(x_{\text{mask}}), \min(y_{\text{mask}}), \max(x_{\text{mask}}), \max(y_{\text{mask}})]$.

2. **Spatial Resampling:**
   - **Model Inference Tensor:** Resized to $224 \times 224$ pixels using bicubic interpolation with anti-aliasing (`Image.BICUBIC`).
   - **Visual Attribution Canvas:** Scaled to $512 \times 512$ pixels for high-fidelity WebGL viewer blending.

3. **Color Normalization (ImageNet Prior):**
   Input tensor $X \in [0.0, 1.0]^{3 \times 224 \times 224}$ is normalized per channel $c \in \{R, G, B\}$:
   $$\hat{X}_c = \frac{X_c - \mu_c}{\sigma_c}$$
   Where:
   $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$

---

## 3. Training-Only Data Augmentation Pipeline

To prevent overfitting on the majority classes and enhance invariant feature learning across minor camera variations, training-only data augmentation is applied dynamically on the training partition ($N = 5,600$):

| Augmentation Operator | Parameter Range | Probability ($p$) | Clinical Justification |
| :--- | :--- | :---: | :--- |
| **Random Horizontal Flip** | $p = 0.50$ | 0.50 | Retina structures (except laterality landmarks) are anatomically mirror-invariant for microaneurysm/lesion detection. |
| **Random Vertical Flip** | $p = 0.50$ | 0.50 | Invariant to slight camera angle orientation. |
| **Random Affine Rotation** | Degree: $[-15^\circ, +15^\circ]$ | 0.70 | Simulates subtle patient head tilt during chin-rest alignment. |
| **Random Perspective / Scale** | Scale: $[0.90, 1.10]$ | 0.40 | Simulates working-distance camera zoom differences (30° vs. 45° fields). |
| **Color Jitter (Brightness/Contrast)** | Brightness: $\pm 0.12$, Contrast: $\pm 0.12$ | 0.50 | Models inter-patient cataract optical haze and flash variations. |
| **Color Jitter (Hue)** | Hue: $\pm 0.04$ | 0.25 | Strictly bounded to prevent destroying retinal vascular orange/red spectral identity. |

> [!IMPORTANT]
> **SaMD Reproducibility Rule:** Data augmentation is strictly disabled during validation (`val`), held-out testing (`test`), and clinical inference runtime.
