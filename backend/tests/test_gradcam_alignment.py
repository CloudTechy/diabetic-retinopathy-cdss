"""
The Grad-CAM overlay must have the shape of the photograph it explains.

The model sees the whole image resized (squashed) to 224x224, so each cell of
the activation map covers the same fraction of the photograph's width and
height. The overlay used to be resized to a fixed 512x512 square; the viewer
draws photograph and overlay with object-contain in the same box, so on a
non-square photograph (APTOS images are mostly 4:3 or 3:2) the photograph was
letterboxed while the overlay filled the box, and every hotspot sat over the
wrong part of the retina. The overlay now keeps the photograph's aspect ratio.
"""
import importlib.util

import pytest
from PIL import Image

from app.services.ai_service import EfficientNetB0InferenceService, attribution_size

torch_installed = importlib.util.find_spec("torch") is not None
requires_torch = pytest.mark.skipif(not torch_installed, reason="PyTorch not installed in this environment.")


@pytest.mark.parametrize("size, expected", [
    ((3216, 2136), (512, 340)),   # 3:2 landscape
    ((2048, 1536), (512, 384)),   # 4:3 landscape
    ((1050, 1050), (512, 512)),   # square
    ((1536, 2048), (384, 512)),   # portrait
])
def test_attribution_size_keeps_the_photographs_aspect_ratio(size, expected):
    assert attribution_size(*size) == expected


@requires_torch
def test_predict_writes_an_overlay_with_the_photographs_shape(tmp_path):
    import io

    import torch
    from app.core.config import settings

    class Stub(torch.nn.Module):
        """Positive weights everywhere, so the class score rises with every activation and the CAM is non-zero."""
        def __init__(self):
            super().__init__()
            self.features = torch.nn.Sequential(*[torch.nn.Identity() for _ in range(8)], torch.nn.Conv2d(3, 4, 1))
            self.head = torch.nn.Linear(4, 5)
            with torch.no_grad():
                self.features[8].weight.fill_(1.0)
                self.features[8].bias.fill_(1.0)
                self.head.weight.fill_(1.0)

        def forward(self, x):
            return self.head(self.features(x).mean(dim=(2, 3)))

    engine = EfficientNetB0InferenceService(checkpoint_path="unused")
    engine._model, engine._device, engine._initialized = Stub().eval(), torch.device("cpu"), True
    engine.calibration, engine.temperature = {"version": "test"}, 1.0

    old = settings.STORAGE_ATTRIBUTIONS_PATH
    settings.STORAGE_ATTRIBUTIONS_PATH = str(tmp_path)
    try:
        # 804x534 has the 3:2 shape of the 3216x2136 APTOS originals.
        out = engine.predict(Image.new("RGB", (804, 534), (180, 90, 40)))
    finally:
        settings.STORAGE_ATTRIBUTIONS_PATH = old
    assert out.gradcam_bytes, "no overlay was produced"
    overlay = Image.open(io.BytesIO(out.gradcam_bytes))
    assert overlay.size == (512, 340)
    assert abs(overlay.size[0] / overlay.size[1] - 804 / 534) < 0.01
