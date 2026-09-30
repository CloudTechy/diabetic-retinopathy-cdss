"""
Shared helper: produce a reduced copy of an image for distribution statistics.

Gates 2 and 3 both characterise an image through *aggregate pixel statistics* —
aperture coverage, channel means, contrast standard deviation, the proportion
of extreme pixels. None of these describe a specific pixel; they describe the
distribution the pixels are drawn from. Computing them over a 3-megapixel array
is wasted work: a large uniform sample estimates the same distribution.

Measured on the deployment CPU, running these gates at full resolution cost
186 ms — 59% of an entire clinical request, against 33 ms for the model itself.

**Nearest-neighbour is deliberate, not a shortcut.** It *samples* pixels;
bilinear and area resampling *average* them. Averaging is wrong here in two
distinct ways:

1. It smooths the aperture boundary, inventing intermediate luminances that
   cross the foreground threshold and shift measured coverage. Empirically,
   bilinear moved coverage by up to 0.011 where nearest moved it by 0.0007.
2. It pulls extreme pixel values toward the mean, which would bias precisely
   the statistics Gate 3 uses to detect washout and underexposure — and bias
   them in the unsafe direction, making a poor image look acceptable.

Nearest-neighbour subsampling is an unbiased estimator of the underlying pixel
distribution, so coverage ratios, channel means, standard deviation and extreme
proportions all survive it.

**What must NOT use this helper:** any metric that is a spatial derivative.
Gate 3's Laplacian variance measures sharpness between neighbouring pixels, so
its value depends on resolution by definition. That computation keeps its own
separate resize and its own threshold calibration, untouched.
"""

from PIL import Image


def downsample_for_analysis(image: Image.Image, max_dim: int) -> Image.Image:
    """
    Return a copy whose longest side is at most `max_dim`, sampled with
    nearest-neighbour. Images already within the limit are returned unchanged,
    so small inputs keep exact full-resolution statistics.

    `max_dim <= 0` disables reduction entirely, which restores the original
    full-resolution behaviour for anyone who wants it.
    """
    if max_dim is None or max_dim <= 0:
        return image

    width, height = image.size
    longest = max(width, height)
    if longest <= max_dim:
        return image

    scale = max_dim / float(longest)
    return image.resize(
        (max(1, int(width * scale)), max(1, int(height * scale))),
        Image.Resampling.NEAREST,
    )
