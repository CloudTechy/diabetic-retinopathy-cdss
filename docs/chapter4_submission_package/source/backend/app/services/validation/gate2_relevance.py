from typing import Tuple, Optional
import numpy as np
from PIL import Image

from app.core.config import settings
from app.services.validation.downsample import downsample_for_analysis


class Gate2Result:
    def __init__(
        self,
        passed: bool,
        mask_coverage: float = 0.0,
        aspect_ratio: float = 1.0,
        red_channel_ratio: float = 0.0,
        red_to_blue_ratio: float = 0.0,
        metric: str = "",
        details: str = "",
        error_code: Optional[str] = None,
        rejection_reason: Optional[str] = None,
        clinical_action: Optional[str] = None,
    ):
        self.passed = passed
        self.mask_coverage = mask_coverage
        self.aspect_ratio = aspect_ratio
        self.red_channel_ratio = red_channel_ratio
        self.red_to_blue_ratio = red_to_blue_ratio
        self.metric = metric
        self.details = details
        self.error_code = error_code
        self.rejection_reason = rejection_reason
        self.clinical_action = clinical_action

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "mask_coverage": self.mask_coverage,
            "aspect_ratio": self.aspect_ratio,
            "red_channel_ratio": self.red_channel_ratio,
            "red_to_blue_ratio": self.red_to_blue_ratio,
            "metric": self.metric,
            "details": self.details,
            "error_code": self.error_code,
            "rejection_reason": self.rejection_reason,
            "clinical_action": self.clinical_action,
        }


def evaluate_gate2(pil_image: Image.Image) -> Gate2Result:
    """
    Gate 2: Retinal Anatomical Relevance & Geometric Gating.
    Validates:
      1. Image aspect ratio conforms to standard retinal camera fields (0.65 to 1.65).
      2. Circular fundus mask detection (aperture coverage between 20% and 98%).
      3. Reddish/orange retinal spectral signature (R/B ratio >= 1.15, R share >= 38%).
    Prevents non-retinal images (faces, chest X-rays, documents, anterior segment) from proceeding.
    """
    rgb_image = pil_image.convert("RGB")
    width, height = rgb_image.size
    aspect_ratio = round(width / float(height), 3)

    # 1. Aspect ratio check
    if aspect_ratio < 0.65 or aspect_ratio > 1.65:
        return Gate2Result(
            passed=False,
            aspect_ratio=aspect_ratio,
            error_code="ERR_INVALID_ASPECT_RATIO",
            metric=f"Aspect ratio: {aspect_ratio:.2f} (Standard: 0.65 - 1.65)",
            rejection_reason=f"Non-standard image aspect ratio ({aspect_ratio:.2f}). Retinal fundus photographs require standard camera proportions.",
            clinical_action="Please provide uncropped, standard fundus photographs from an ophthalmic fundus camera.",
        )

    # Every check below this point is a ratio or a mean over the whole image,
    # so it is computed on a nearest-neighbour subsample rather than the full
    # array. See downsample.py for why nearest and not bilinear. The aspect
    # ratio above deliberately uses the ORIGINAL dimensions.
    analysis_image = downsample_for_analysis(rgb_image, settings.VALIDATION_ANALYSIS_MAX_DIM)
    analysis_width, analysis_height = analysis_image.size

    img_arr = np.asarray(analysis_image, dtype=np.float32)
    # Luminance calculation (standard ITU-R BT.601)
    luminance = 0.299 * img_arr[:, :, 0] + 0.587 * img_arr[:, :, 1] + 0.114 * img_arr[:, :, 2]

    # 2. Circular Aperture / Foreground Mask Detection
    # Background in retinal cameras is dark (< 15 intensity)
    foreground_mask = luminance > 15.0
    total_pixels = analysis_width * analysis_height
    foreground_pixels = int(np.sum(foreground_mask))
    mask_coverage = round(foreground_pixels / float(total_pixels), 4)

    if mask_coverage < settings.RETINAL_MIN_COVERAGE:
        return Gate2Result(
            passed=False,
            mask_coverage=mask_coverage,
            aspect_ratio=aspect_ratio,
            error_code="ERR_RETINAL_MASK_ABSENT",
            metric=f"Aperture coverage: {mask_coverage * 100:.1f}% (Threshold >= {settings.RETINAL_MIN_COVERAGE * 100:.0f}%)",
            rejection_reason="Circular fundus aperture not detected. Image is predominantly blank, dark, or lacks retinal structure.",
            clinical_action="Ensure the camera lens cap is removed and alignment is centered on the posterior pole.",
        )

    # If image is almost entirely solid or has no mask at all, compute pixels in the foreground
    r_channel = img_arr[:, :, 0]
    g_channel = img_arr[:, :, 1]
    b_channel = img_arr[:, :, 2]

    if foreground_pixels > 0:
        r_mean = float(np.mean(r_channel[foreground_mask]))
        g_mean = float(np.mean(g_channel[foreground_mask]))
        b_mean = float(np.mean(b_channel[foreground_mask]))
    else:
        r_mean = float(np.mean(r_channel))
        g_mean = float(np.mean(g_channel))
        b_mean = float(np.mean(b_channel))

    total_rgb = r_mean + g_mean + b_mean + 1e-6
    red_share = r_mean / total_rgb
    red_to_blue = r_mean / (b_mean + 1e-6)

    # 3. Retinal Color Profile (Reddish/Orange vascular background)
    # Real fundus photographs have high red channel dominance over blue channel
    if red_to_blue < settings.RETINAL_RED_RATIO_MIN or red_share < 0.36:
        return Gate2Result(
            passed=False,
            mask_coverage=mask_coverage,
            aspect_ratio=aspect_ratio,
            red_channel_ratio=round(red_share, 3),
            red_to_blue_ratio=round(red_to_blue, 2),
            error_code="ERR_NON_RETINAL_SPECTRAL_PROFILE",
            metric=f"Red/Blue ratio: {red_to_blue:.2f} (Threshold >= {settings.RETINAL_RED_RATIO_MIN})",
            rejection_reason="Spectral profile does not exhibit retinal vascular characteristics. Image appears to be non-retinal (e.g., face, text, scenery, or anterior segment).",
            clinical_action="Ensure you are uploading posterior pole retinal fundus photography rather than external ocular or non-retinal images.",
        )

    metric = f"Retinal field-of-view: {mask_coverage * 100:.1f}%, R/B spectral ratio: {red_to_blue:.2f}"
    details = f"Retinal circular aperture confirmed ({mask_coverage * 100:.1f}% frame coverage). Vascular spectral balance verified."

    return Gate2Result(
        passed=True,
        mask_coverage=mask_coverage,
        aspect_ratio=aspect_ratio,
        red_channel_ratio=round(red_share, 3),
        red_to_blue_ratio=round(red_to_blue, 2),
        metric=metric,
        details=details,
    )
