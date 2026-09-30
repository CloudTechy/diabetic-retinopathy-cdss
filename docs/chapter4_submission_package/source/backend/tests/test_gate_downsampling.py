"""
Downsampling Gate 2 and Gate 3 statistics must not change a gate's verdict.

These gates decide whether an image is fit to be graded at all. Making them
faster is only acceptable if it is decision-preserving, so the property under
test is not "the metrics are close" but "the accept/reject outcome is the same
as at full resolution", across images deliberately placed near each threshold.

The one metric excluded from downsampling is Gate 3's Laplacian variance, which
is a spatial derivative and resolution-dependent by definition. There is an
explicit test that it is unaffected by this setting.
"""

import math

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

from app.core.config import settings
from app.services.validation.downsample import downsample_for_analysis
from app.services.validation.gate2_relevance import evaluate_gate2
from app.services.validation.gate3_quality import evaluate_gate3


def synthetic_fundus(width=1600, height=1200, seed=0, margin_frac=0.08,
                     brightness=1.0, blur=0.0):
    """Fundus-like image with knobs that move it across the gate thresholds."""
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (width, height), (5, 5, 5))
    draw = ImageDraw.Draw(img)

    margin = min(int(min(width, height) * margin_frac), min(width, height) // 2 - 2)
    base = tuple(int(c * brightness) for c in (185, 65, 25))
    draw.ellipse([margin, margin, width - margin, height - margin], fill=base)

    dx, dy, dr = int(width * 0.33), int(height * 0.5), int(width * 0.07)
    draw.ellipse([dx - dr, dy - dr, dx + dr, dy + dr], fill=(240, 210, 110))
    for i in range(14):
        angle = (i / 14) * 2 * math.pi
        draw.line([dx, dy,
                   int(dx + math.cos(angle) * width * 0.36),
                   int(dy + math.sin(angle) * height * 0.36)],
                  fill=(110, 20, 15), width=max(2, width // 700))

    arr = np.asarray(img).astype(np.float32)
    arr += rng.normal(0, 0.6, arr.shape).astype(np.float32)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    return img


def non_retinal(width=1600, height=1200):
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:, :, 0], arr[:, :, 1], arr[:, :, 2] = 50, 120, 220
    return Image.fromarray(arr, "RGB")


# Images chosen to straddle the thresholds: aperture coverage from 0.09 to 0.73
# against limits of 0.20/0.98, brightness sweeping the luminance mask cutoff,
# plus a non-retinal image and a heavily blurred one.
CASES = [
    ("standard", synthetic_fundus()),
    ("wide-aperture", synthetic_fundus(seed=1, margin_frac=0.02)),
    ("small-aperture", synthetic_fundus(seed=2, margin_frac=0.28)),
    ("tiny-aperture", synthetic_fundus(seed=3, margin_frac=0.38)),
    ("dim", synthetic_fundus(seed=4, brightness=0.18)),
    ("very-dim", synthetic_fundus(seed=5, brightness=0.10)),
    ("bright", synthetic_fundus(seed=6, brightness=1.0)),
    ("blurred", synthetic_fundus(seed=7, blur=8.0)),
    ("non-retinal", non_retinal()),
    ("tall", synthetic_fundus(width=1200, height=1600, seed=8)),
]


@pytest.fixture
def full_resolution_analysis(monkeypatch):
    """Restore pre-optimisation behaviour: analyse at full resolution."""
    monkeypatch.setattr(settings, "VALIDATION_ANALYSIS_MAX_DIM", 0)


@pytest.mark.parametrize("name,image", CASES, ids=[c[0] for c in CASES])
def test_gate2_verdict_unchanged_by_downsampling(name, image, monkeypatch):
    monkeypatch.setattr(settings, "VALIDATION_ANALYSIS_MAX_DIM", 0)
    full = evaluate_gate2(image)
    monkeypatch.setattr(settings, "VALIDATION_ANALYSIS_MAX_DIM", 512)
    reduced = evaluate_gate2(image)

    assert reduced.passed == full.passed, (
        f"{name}: verdict changed (full={full.passed} reduced={reduced.passed}); "
        f"coverage {full.mask_coverage} -> {reduced.mask_coverage}"
    )
    assert reduced.error_code == full.error_code
    assert abs(reduced.mask_coverage - full.mask_coverage) < 0.01
    assert abs(reduced.aspect_ratio - full.aspect_ratio) < 1e-9


@pytest.mark.parametrize("name,image", CASES, ids=[c[0] for c in CASES])
def test_gate3_verdict_unchanged_by_downsampling(name, image, monkeypatch):
    monkeypatch.setattr(settings, "VALIDATION_ANALYSIS_MAX_DIM", 0)
    full = evaluate_gate3(image)
    monkeypatch.setattr(settings, "VALIDATION_ANALYSIS_MAX_DIM", 512)
    reduced = evaluate_gate3(image)

    assert reduced.passed == full.passed, (
        f"{name}: verdict changed (full={full.passed} reduced={reduced.passed}); "
        f"contrast {full.contrast_dynamic_range} -> {reduced.contrast_dynamic_range}"
    )
    assert reduced.error_code == full.error_code


@pytest.mark.parametrize("name,image", CASES, ids=[c[0] for c in CASES])
def test_laplacian_variance_is_not_affected_by_the_analysis_setting(name, image, monkeypatch):
    """
    Sharpness is a spatial derivative, so it deliberately keeps its own resize
    path. Changing the analysis resolution must not move it at all.
    """
    monkeypatch.setattr(settings, "VALIDATION_ANALYSIS_MAX_DIM", 0)
    full = evaluate_gate3(image)
    monkeypatch.setattr(settings, "VALIDATION_ANALYSIS_MAX_DIM", 256)
    reduced = evaluate_gate3(image)

    assert reduced.laplacian_variance == full.laplacian_variance


def test_aspect_ratio_uses_original_dimensions(monkeypatch):
    """
    A regression guard: reading dimensions from the reduced copy would report a
    different aspect ratio and could reject a valid image, or admit an invalid
    one, purely as a side effect of the optimisation.
    """
    monkeypatch.setattr(settings, "VALIDATION_ANALYSIS_MAX_DIM", 64)
    image = synthetic_fundus(width=1600, height=1200, seed=11)
    result = evaluate_gate2(image)
    assert result.aspect_ratio == pytest.approx(1600 / 1200, abs=1e-3)


def test_non_retinal_image_is_still_rejected(monkeypatch):
    """The gate's actual job must survive the optimisation."""
    monkeypatch.setattr(settings, "VALIDATION_ANALYSIS_MAX_DIM", 512)
    result = evaluate_gate2(non_retinal())
    assert result.passed is False


def test_blurred_image_is_still_rejected(monkeypatch):
    monkeypatch.setattr(settings, "VALIDATION_ANALYSIS_MAX_DIM", 512)
    result = evaluate_gate3(synthetic_fundus(seed=12, blur=12.0))
    assert result.passed is False
    assert result.error_code == "ERR_MOTION_OR_DEFOCUS_BLUR"


class TestDownsampleHelper:
    def test_uses_nearest_neighbour_not_averaging(self):
        """
        Nearest must reproduce source pixels exactly. Bilinear or area
        resampling would blend them, which is what biases the extreme-pixel
        statistics Gate 3 relies on.
        """
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        arr[:50] = 255  # hard edge: any averaging creates intermediate values
        out = np.asarray(downsample_for_analysis(Image.fromarray(arr, "RGB"), 20))
        assert set(np.unique(out)).issubset({0, 255})

    def test_longest_side_is_capped(self):
        img = Image.new("RGB", (2000, 1000))
        assert max(downsample_for_analysis(img, 512).size) == 512

    def test_aspect_ratio_preserved(self):
        out = downsample_for_analysis(Image.new("RGB", (2000, 1000)), 512)
        assert out.size == (512, 256)

    def test_small_images_are_returned_untouched(self):
        img = Image.new("RGB", (300, 200))
        assert downsample_for_analysis(img, 512) is img

    def test_zero_disables_reduction(self):
        img = Image.new("RGB", (2000, 1000))
        assert downsample_for_analysis(img, 0) is img
