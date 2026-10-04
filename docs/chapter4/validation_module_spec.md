# Input-Image Technical Validation Module Specification (Objective b)

## Metadata & Academic Context
- **Research Project:** AI-Based Clinical Decision Support System for Early Detection of Diabetic Retinopathy
- **Researcher:** Onyekelu Chukwuebuka Elochukwu (2024516020FN)
- **Programme:** PGD Computer Science, Faculty of Physical Sciences
- **Implementation Status:** Evaluated & Verified in Backend Validation Service

---

## 1. Fail-Closed 3-Stage Gating Architecture

Pre-inference validation reduces the chance that invalid, corrupted, poor-quality or non-fundus-looking photographs reach the neural network. Any gate failure triggers an immediate abort and generates non-diagnostic recapture feedback.

### Gate 1: File Integrity & Security
- **MIME Type & Magic Bytes:** Validates 0xFFD8FF (JPEG) and 0x89504E47 (PNG).
- **Payload Size Bound:** Rejects files > 15.0 MB to prevent denial-of-service memory exhaustion.
- **Storage:** the uploaded bytes are stored unchanged under `<record id>_<8 hex>.<jpg|png>` (extension from the magic bytes). **No EXIF metadata is stripped**: an earlier revision claimed de-identification by EXIF stripping and a random UUID filename, neither of which the code performs.

### Gate 2: Technical retinal-image relevance (geometry and colour profile)

> Three heuristics, none of which identifies anatomy. Passing them means the input meets the configured geometry and colour-profile thresholds; it does not confirm retinal identity, anatomical correctness or clinical gradability. An earlier revision of this heading called the gate "Retinal Anatomical Relevance".
- **Aspect-ratio bounds (a-priori literals, not calibrated):** Accepts $0.65 \leq \text{aspect ratio} \leq 1.65$. This range accommodates standard ophthalmic fundus cameras (4:3 = 1.333, 3:2 = 1.500, and square 1:1 apertures) while rejecting extreme panoramic strips (> 1.65) or elongated documents (< 0.65).
- **Foreground (aperture) coverage:** foreground is luminance > 15; coverage must be at least 20.0% of the frame (`RETINAL_MIN_COVERAGE`). No upper bound is enforced: an earlier revision stated an upper bound of 98.0% from a setting (`RETINAL_MAX_COVERAGE`) that was declared but never used by the gate; the setting has been removed.
- **Colour profile:** Red/Blue channel ratio $\ge 1.15$ (`RETINAL_RED_RATIO_MIN`) and red channel share $\ge 36.0\%$ (`RETINAL_RED_SHARE_MIN`). An earlier revision stated 38.0% while the code tested 36%.

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


- **Illumination Uniformity:** counts foreground pixels with luminance < 25 (`ILLUMINATION_UNDEREXPOSED_BELOW`) or > 235 (`ILLUMINATION_OVEREXPOSED_ABOVE`) and rejects when their share exceeds 0.35 (`ILLUMINATION_EXTREME_RATIO_MAX`). An earlier revision stated < 10 and > 245, which the code never used.

---

## 2. Empirical Verification
**Current evidence:** [`validation_test_results.csv`](validation_test_results.csv) — **16 cases, all behaving as declared**: 10 unmodified held-out APTOS images ACCEPTED, and 6 negatives REJECTED, each derived from a held-out image by a transformation stated in the file's `derivation` column. It is produced by `backend/scripts/generate_validation_evidence.py <aptos>/train_images`, which runs the real gate functions rather than describing them.

> [!NOTE]
> **An earlier file of this name was fabricated** and has been replaced. Zero > of its ten rows cited an actual held-out image, six used identifiers absent > from APTOS entirely, and its metrics were frontend demonstration constants. > Rule group H now rejects any CSV whose `image_id` column cites an identifier > absent from `dataset_split_manifest.csv`.
