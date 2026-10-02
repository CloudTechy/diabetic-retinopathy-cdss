# Input-Image Technical Validation Module Specification (Objective b)

## Metadata & Academic Context
- **Research Project:** AI-Based Clinical Decision Support System for Early Detection of Diabetic Retinopathy
- **Researcher:** Onyekelu Chukwuebuka Elochukwu (2024516020FN)
- **Programme:** PGD Computer Science, Faculty of Physical Sciences
- **Implementation Status:** Evaluated & Verified in Backend Validation Service

---

## 1. Fail-Closed 3-Stage Gating Architecture

Pre-inference validation prevents invalid, corrupted, or non-retinal photographs from reaching the neural network. Any gate failure triggers an immediate abort and generates non-diagnostic recapture feedback.

### Gate 1: File Integrity & Security
- **MIME Type & Magic Bytes:** Validates 0xFFD8FF (JPEG) and 0x89504E47 (PNG).
- **Payload Size Bound:** Rejects files > 15.0 MB to prevent denial-of-service memory exhaustion.
- **De-identification:** Strips proprietary EXIF metadata and generates a random UUID filename.

### Gate 2: Retinal Anatomical Relevance & Geometric Proportions
- **Calibrated Aspect Ratio Threshold:** Accepts $0.65 \leq \text{aspect ratio} \leq 1.65$. This range accommodates standard ophthalmic fundus cameras (4:3 = 1.333, 3:2 = 1.500, and square 1:1 apertures) while rejecting extreme panoramic strips (> 1.65) or elongated documents (< 0.65).
- **Circular Aperture Mask Coverage:** Analyzes foreground luminance (threshold > 15 intensity). Requires valid fundus circular aperture covering between 20.0% and 98.0% of total image area. Rejects non-retinal images (e.g. chest X-rays, faces).
- **Retinal Chromatic Signature:** Evaluates reddish/orange retinal reflection. Requires Red/Blue channel ratio $\ge 1.15$ and Red channel luminance share $\ge 38.0\%$.

### Gate 3: Technical Quality & Sharpness
- **Laplacian Variance Metric:** Applies discrete Laplacian operator $\nabla^2 I$. Rejects motion-blurred or defocussed photographs with variance $< 4.3$ (`LAPLACIAN_BLUR_THRESHOLD`), the 1st percentile of the development corpus - see the derivation recorded below.

> [!NOTE]
> **Resolved 2026-10-01. This threshold is now derived from the corpus.**
>
> It was previously 60.0, chosen a priori. Genuine held-out APTOS images measure
> **5.7 – 22.0** on the path this gate computes, so all ten unmodified images in
> [`validation_test_results.csv`](validation_test_results.csv) were rejected with
> `ERR_MOTION_OR_DEFOCUS_BLUR` — 99.5% of the development corpus would have been
> refused. Two further thresholds were found to be a priori in the same way:
> `CONTRAST_THRESHOLD` at 18.0 rejected 63.4% of genuine images, and
> `MIN_IMAGE_DIMENSION` at 512 rejected 1.3%.
>
> All three are now set from the 1st percentile of the measured distribution
> over the training and validation partitions. All 16 validation cases behave as
> declared. The reasoning below is kept because it explains why the metric has
> no corpus-independent value, which is why it had to be measured here.
>
> Laplacian variance is a spatial derivative: it scales with sensor resolution,
> optics, compression and any upstream resizing, so it has no corpus-independent
> value. The gate applies it to a bilinear-resized copy — 1024 px on the long
> edge when larger, full resolution otherwise — and the threshold must be
> calibrated against **that** path.
>
> Derive a defensible value with
> `python backend/scripts/calibrate_blur_threshold.py <aptos>/train_images`,
> then record the chosen percentile **and** the measured distribution here.
>
> The metric itself separates the two populations correctly: a deliberately
> blurred image scores 0.9 against 5.7 – 22.0 for unmodified ones. Only the
> cut-point is in the wrong place.

### Calibrated admission thresholds

| Setting | Value | Derivation |
| :--- | ---: | :--- |
| `LAPLACIAN_BLUR_THRESHOLD` | **4.3** | 1.0th percentile of Laplacian variance |
| `MIN_IMAGE_DIMENSION` | **480** px | 1.0th percentile of shortest edge |
| `CONTRAST_THRESHOLD` | **8.8** | 1.0th percentile of foreground std |

**Declared percentile:** 1.0% — Admit the sharpest and largest 99.0% of the development corpus; the remainder is treated as too degraded to grade.

**Measured over:** 2979 images, train+val (development), from
[`dataset_split_manifest.csv`](dataset_split_manifest.csv). The held-out test
partition is excluded: choosing an operating point on it would leak it.

**Distribution:** [`blur_threshold_calibration.json`](blur_threshold_calibration.json)
carries the full percentile table for both metrics, so this value can be
checked against the data it came from.

**Previous values:** `LAPLACIAN_BLUR_THRESHOLD` 60.0, `MIN_IMAGE_DIMENSION`
512, `CONTRAST_THRESHOLD` 18.0. All three were chosen a priori and
never measured against this corpus. The blur threshold rejected 10 of 10 genuine
held-out images; with it corrected, the contrast threshold still rejected 5 of 10.

*Applied 2026-10-01 by `apply_validation_thresholds.py`.*


- **Illumination Uniformity:** Analyzes extreme underexposed (< 10) and overexposed (> 245) pixel ratios, rejecting acquisitions with extreme ratio $> 0.35$.

---

## 2. Empirical Verification
**Note:** the previous `validation_test_results.csv` was fabricated — zero of its ten rows cited an actual held-out image, six used identifiers absent from APTOS entirely, and its metrics were demonstration constants. It has been removed. Regenerate genuine results with `backend/scripts/generate_validation_evidence.py <aptos>/train_images`, which runs the real gate functions over real held-out images plus negatives derived by a stated transformation. Until then, gate behaviour is evidenced by the automated suite (`backend/tests/test_validation_pipeline.py` and the 38 gate-downsampling tests).
