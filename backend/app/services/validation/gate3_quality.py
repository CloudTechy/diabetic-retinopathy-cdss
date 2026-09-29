from typing import Tuple, Optional
import numpy as np
from PIL import Image

from app.core.config import settings
from app.services.validation.downsample import downsample_for_analysis


class Gate3Result:
    def __init__(
        self,
        passed: bool,
        laplacian_variance: float = 0.0,
        illumination_index: float = 0.0,
        contrast_dynamic_range: float = 0.0,
        extreme_pixel_ratio: float = 0.0,
        metric: str = "",
        details: str = "",
        error_code: Optional[str] = None,
        rejection_reason: Optional[str] = None,
        clinical_action: Optional[str] = None,
    ):
        self.passed = passed
        self.laplacian_variance = laplacian_variance
        self.illumination_index = illumination_index
        self.contrast_dynamic_range = contrast_dynamic_range
        self.extreme_pixel_ratio = extreme_pixel_ratio
        self.metric = metric
        self.details = details
        self.error_code = error_code
        self.rejection_reason = rejection_reason
        self.clinical_action = clinical_action

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "laplacian_variance": self.laplacian_variance,
            "illumination_index": self.illumination_index,
            "contrast_dynamic_range": self.contrast_dynamic_range,
            "extreme_pixel_ratio": self.extreme_pixel_ratio,
            "metric": self.metric,
            "details": self.details,
            "error_code": self.error_code,
            "rejection_reason": self.rejection_reason,
            "clinical_action": self.clinical_action,
        }


def compute_laplacian_variance(gray_arr: np.ndarray) -> float:
    """Compute the variance of the 2D discrete Laplacian filter."""
    # Discrete 5-point Laplacian stencil:
    # L[y, x] = gray[y-1, x] + gray[y+1, x] + gray[y, x-1] + gray[y, x+1] - 4 * gray[y, x]
    laplacian = (
        gray_arr[:-2, 1:-1]
        + gray_arr[2:, 1:-1]
        + gray_arr[1:-1, :-2]
        + gray_arr[1:-1, 2:]
        - 4.0 * gray_arr[1:-1, 1:-1]
    )
    return float(np.var(laplacian))


def evaluate_gate3(pil_image: Image.Image) -> Gate3Result:
    """
    Gate 3: Technical Image Quality & Sharpness.
    Validates:
      1. Edge sharpness via Laplacian blur variance threshold (>= 60.0).
      2. Dynamic range and contrast adequacy (std >= 18.0).
      3. Illumination uniformity avoiding severe shadows, flash washout, or extreme underexposure.
    """
    rgb_image = pil_image.convert("RGB")
    width, height = rgb_image.size

    # Contrast and illumination are distribution statistics over the whole
    # image, so they are computed on a nearest-neighbour subsample. Nearest
    # SAMPLES pixels where bilinear would AVERAGE them; averaging would pull
    # extreme values toward the mean and bias the washout/underexposure checks
    # below in the unsafe direction. See downsample.py.
    analysis_image = downsample_for_analysis(rgb_image, settings.VALIDATION_ANALYSIS_MAX_DIM)
    analysis_arr = np.asarray(analysis_image, dtype=np.float32)
    gray = (0.299 * analysis_arr[:, :, 0]
            + 0.587 * analysis_arr[:, :, 1]
            + 0.114 * analysis_arr[:, :, 2])

    # Retinal mask (ignore camera black outer frame)
    foreground_mask = gray > 15.0
    fg_pixels = gray[foreground_mask] if np.any(foreground_mask) else gray.ravel()

    # 1. Laplacian blur variance
    # UNCHANGED. Laplacian variance is a spatial derivative: its value depends
    # on resolution by definition, and the 60.0 threshold is calibrated against
    # this specific path. It keeps its own resize and its own full-resolution
    # branch, and must not be folded into the analysis subsample above.
    if max(height, width) > 1024:
        scale = 1024.0 / max(height, width)
        scaled_img = pil_image.convert("L").resize(
            (int(width * scale), int(height * scale)), Image.Resampling.BILINEAR)
        scaled_gray = np.asarray(scaled_img, dtype=np.float32)
        laplacian_var = compute_laplacian_variance(scaled_gray)
    else:
        full_arr = np.asarray(rgb_image, dtype=np.float32)
        full_gray = (0.299 * full_arr[:, :, 0]
                     + 0.587 * full_arr[:, :, 1]
                     + 0.114 * full_arr[:, :, 2])
        laplacian_var = compute_laplacian_variance(full_gray)

    laplacian_var = round(laplacian_var, 1)

    # 2. Contrast dynamic range
    contrast_std = float(np.std(fg_pixels))
    contrast_dynamic_range = round(contrast_std, 1)

    # 3. Illumination uniformity & extreme pixel ratio
    underexposed_count = np.sum(fg_pixels < 25.0)
    overexposed_count = np.sum(fg_pixels > 235.0)
    total_fg = max(len(fg_pixels), 1)
    extreme_ratio = float((underexposed_count + overexposed_count) / total_fg)
    illumination_index = round(max(0.0, 1.0 - extreme_ratio), 2)

    # Gate 3 Evaluation Logic
    # 1. Blur check
    if laplacian_var < settings.LAPLACIAN_BLUR_THRESHOLD:
        return Gate3Result(
            passed=False,
            laplacian_variance=laplacian_var,
            illumination_index=illumination_index,
            contrast_dynamic_range=contrast_dynamic_range,
            extreme_pixel_ratio=round(extreme_ratio, 3),
            error_code="ERR_MOTION_OR_DEFOCUS_BLUR",
            metric=f"Laplacian variance: {laplacian_var:.1f} (Threshold >= {settings.LAPLACIAN_BLUR_THRESHOLD:.1f})",
            rejection_reason="Laplacian variance below acceptable sharpness threshold. Motion blur or optical defocus detected.",
            clinical_action="Please recapture the fundus photograph ensuring the patient maintains steady fixation and the camera focus is optimized.",
        )

    # 2. Contrast check
    if contrast_dynamic_range < settings.CONTRAST_THRESHOLD:
        return Gate3Result(
            passed=False,
            laplacian_variance=laplacian_var,
            illumination_index=illumination_index,
            contrast_dynamic_range=contrast_dynamic_range,
            extreme_pixel_ratio=round(extreme_ratio, 3),
            error_code="ERR_LOW_CONTRAST",
            metric=f"Dynamic range std: {contrast_dynamic_range:.1f} (Threshold >= {settings.CONTRAST_THRESHOLD:.1f})",
            rejection_reason="Retinal contrast dynamic range is too narrow. Image is flat, washed out, or obscured by media haze.",
            clinical_action="Check camera sensor exposure settings and evaluate for potential ocular media opacities (e.g. cataract).",
        )

    # 3. Illumination check
    if extreme_ratio > settings.ILLUMINATION_EXTREME_RATIO_MAX:
        return Gate3Result(
            passed=False,
            laplacian_variance=laplacian_var,
            illumination_index=illumination_index,
            contrast_dynamic_range=contrast_dynamic_range,
            extreme_pixel_ratio=round(extreme_ratio, 3),
            error_code="ERR_POOR_ILLUMINATION",
            metric=f"Extreme pixel ratio: {extreme_ratio * 100:.1f}% (Threshold <= {settings.ILLUMINATION_EXTREME_RATIO_MAX * 100:.0f}%)",
            rejection_reason="Severe illumination non-uniformity detected (excessive shadowing, flash washout, or extreme underexposure).",
            clinical_action="Re-align the camera flash unit and ensure adequate pupil dilation for uniform fundus illumination.",
        )

    metric = f"Laplacian: {laplacian_var:.1f} (>= {settings.LAPLACIAN_BLUR_THRESHOLD:.1f}), Illumination index: {illumination_index:.2f}"
    details = f"Vascular edge sharpness confirmed. Illumination homogeneity: {illumination_index * 100:.0f}%. Dynamic range: {contrast_dynamic_range:.1f}."

    return Gate3Result(
        passed=True,
        laplacian_variance=laplacian_var,
        illumination_index=illumination_index,
        contrast_dynamic_range=contrast_dynamic_range,
        extreme_pixel_ratio=round(extreme_ratio, 3),
        metric=metric,
        details=details,
    )
