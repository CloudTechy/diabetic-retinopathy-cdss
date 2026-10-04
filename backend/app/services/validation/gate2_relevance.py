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
        # The rounded fields above are for display. These are the values the
        # gate actually compared against its thresholds, and they are what any
        # audit or decision-preservation analysis must use: measuring a margin
        # from a rounded value quantises it onto the rounding grid.
        mask_coverage_exact: float = 0.0,
        red_to_blue_ratio_exact: float = 0.0,
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
        self.mask_coverage_exact = mask_coverage_exact
        self.red_to_blue_ratio_exact = red_to_blue_ratio_exact
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
            "mask_coverage_exact": self.mask_coverage_exact,
            "red_to_blue_ratio_exact": self.red_to_blue_ratio_exact,
            "metric": self.metric,
            "details": self.details,
            "error_code": self.error_code,
            "rejection_reason": self.rejection_reason,
            "clinical_action": self.clinical_action,
        }


def evaluate_gate2(pil_image: Image.Image) -> Gate2Result:
    """
    Gate 2: technical retinal-image relevance (geometry and colour profile).

    Three heuristics, none of which identifies anatomy: passing them means the
    input meets the configured geometry and colour-profile thresholds, not
    that it is a retina, that its anatomy is correct, or that it is gradable.
    Validates:
      1. Image aspect ratio conforms to standard retinal camera fields (0.65 to 1.65).
      2. Foreground (aperture) coverage of at least RETINAL_MIN_COVERAGE; no upper bound is enforced.
      3. Colour profile: R/B ratio >= RETINAL_RED_RATIO_MIN and red share >= RETINAL_RED_SHARE_MIN.
    Rejects inputs that fall outside the configured geometry and colour-profile
    heuristics; passing these checks does not establish retinal identity.
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
            clinical_action="Recapture or upload a technically clearer fundus photograph.",
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
    # Decide on full precision; round only for the report. Comparing the
    # rounded value puts the decision boundary on a fixed grid, so images
    # land exactly on a round threshold and flip under any tiny change.
    mask_coverage_exact = foreground_pixels / float(total_pixels)
    mask_coverage = round(mask_coverage_exact, 4)

    if mask_coverage_exact < settings.RETINAL_MIN_COVERAGE:
        return Gate2Result(
            passed=False,
            mask_coverage=mask_coverage,
            mask_coverage_exact=mask_coverage_exact,
            aspect_ratio=aspect_ratio,
            error_code="ERR_RETINAL_MASK_ABSENT",
            metric=f"Aperture coverage: {mask_coverage * 100:.1f}% (Threshold >= {settings.RETINAL_MIN_COVERAGE * 100:.0f}%)",
            rejection_reason="The image did not meet the configured foreground-coverage threshold.",
            clinical_action="Recapture or upload a technically clearer fundus photograph.",
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
    if red_to_blue < settings.RETINAL_RED_RATIO_MIN or red_share < settings.RETINAL_RED_SHARE_MIN:
        return Gate2Result(
            passed=False,
            mask_coverage=mask_coverage,
            aspect_ratio=aspect_ratio,
            red_channel_ratio=round(red_share, 3),
            red_to_blue_ratio=round(red_to_blue, 2),
            mask_coverage_exact=mask_coverage_exact,
            red_to_blue_ratio_exact=red_to_blue,
            error_code="ERR_NON_RETINAL_SPECTRAL_PROFILE",
            metric=f"R/B ratio: {red_to_blue:.2f} (threshold >= {settings.RETINAL_RED_RATIO_MIN}), red share: {red_share * 100:.1f}% (threshold >= {settings.RETINAL_RED_SHARE_MIN * 100:.0f}%)",
            rejection_reason="The image did not meet the configured colour-profile thresholds (red/blue ratio and red share).",
            clinical_action="Recapture or upload a technically clearer fundus photograph.",
        )

    metric = f"Aperture coverage: {mask_coverage * 100:.1f}%, R/B ratio: {red_to_blue:.2f}"
    details = (f"Input meets the configured geometry and colour-profile thresholds "
               f"(foreground coverage {mask_coverage * 100:.1f}%, R/B {red_to_blue:.2f}). "
               "This does not confirm retinal identity, anatomical correctness or clinical gradability.")

    return Gate2Result(
        passed=True,
        mask_coverage=mask_coverage,
        aspect_ratio=aspect_ratio,
        red_channel_ratio=round(red_share, 3),
        red_to_blue_ratio=round(red_to_blue, 2),
        mask_coverage_exact=mask_coverage_exact,
        red_to_blue_ratio_exact=red_to_blue,
        metric=metric,
        details=details,
    )
