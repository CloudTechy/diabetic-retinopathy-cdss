"""
The vectorised Viridis colormap must render exactly what the scalar one did.

Grad-CAM overlays are clinical attribution artefacts: a clinician looks at the
heatmap to see where the model was responding. Speeding up how it is drawn must
not change a single pixel, or previously generated attributions stop matching
what the system produces today.
"""

import numpy as np
import pytest

from app.services.ai_service import (
    create_mock_gradcam_heatmap,
    generate_viridis_colormap,
    viridis_rgba_array,
)


def scalar_reference(values: np.ndarray) -> np.ndarray:
    """The original per-pixel implementation, kept here as the oracle."""
    h, w = values.shape
    out = np.zeros((h, w, 4), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            out[i, j] = generate_viridis_colormap(float(values[i, j]))
    return out


@pytest.mark.parametrize("shape", [(1, 1), (7, 7), (16, 32), (64, 64)])
def test_vectorised_matches_scalar_on_random_input(shape):
    rng = np.random.default_rng(1234)
    values = rng.random(shape).astype(np.float32)
    assert np.array_equal(viridis_rgba_array(values), scalar_reference(values))


def test_vectorised_matches_scalar_on_the_quantised_cam_path():
    """
    Reproduces the exact dtype path predict() uses: a CAM is written to uint8,
    upsampled by PIL, then divided by 255 into float32. That round trip is
    where a float32/float64 mismatch would show up as off-by-one pixels.
    """
    from PIL import Image

    rng = np.random.default_rng(7)
    cam = rng.random((7, 7)).astype(np.float32)
    cam /= cam.max()

    cam_pil = Image.fromarray((cam * 255).astype(np.uint8)).resize(
        (128, 128), Image.Resampling.BILINEAR)
    cam_arr = np.array(cam_pil, dtype=np.float32) / 255.0

    assert np.array_equal(viridis_rgba_array(cam_arr), scalar_reference(cam_arr))


def test_activation_floor_is_fully_transparent():
    """Values below 0.05 must render as RGBA (0,0,0,0), not merely dim."""
    values = np.array([[0.0, 0.01, 0.0499, 0.05, 0.9]], dtype=np.float32)
    out = viridis_rgba_array(values)

    assert out[0, 0].tolist() == [0, 0, 0, 0]
    assert out[0, 1].tolist() == [0, 0, 0, 0]
    assert out[0, 2].tolist() == [0, 0, 0, 0]
    assert out[0, 3].tolist() != [0, 0, 0, 0]
    assert out[0, 4][3] > 0


def test_values_outside_unit_range_are_clipped():
    """Guards against a NaN or out-of-range CAM producing garbage pixels."""
    values = np.array([[-5.0, 0.5, 5.0]], dtype=np.float32)
    out = viridis_rgba_array(values)

    assert out[0, 0].tolist() == [0, 0, 0, 0]          # clipped to 0 -> transparent
    assert np.array_equal(out[0, 2], viridis_rgba_array(
        np.array([[1.0]], dtype=np.float32))[0, 0])    # clipped to 1.0


def test_output_shape_and_dtype():
    out = viridis_rgba_array(np.zeros((9, 4), dtype=np.float32))
    assert out.shape == (9, 4, 4)
    assert out.dtype == np.uint8


@pytest.mark.parametrize("grade", [0, 1, 2, 3, 4])
def test_mock_heatmap_still_renders_for_every_grade(grade):
    """The simulated overlay uses the same vectorised path; it must still work."""
    img = create_mock_gradcam_heatmap(grade, width=64, height=64, laterality="OD")
    assert img.size == (64, 64)
    assert img.mode == "RGBA"
    assert np.array(img).max() > 0
