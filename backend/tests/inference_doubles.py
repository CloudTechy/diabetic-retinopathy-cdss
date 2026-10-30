"""
Test double for the inference engine. Test code only.

The application serves one engine, the fixed EfficientNet-B0 with its verified
checkpoint and temperature-scaling artifact; nothing in app/ can select this
class. The API tests install it as the process singleton (tests/conftest.py)
so the clinical journey runs without loading PyTorch weights on every test.
Tests that exercise the real engine replace the singleton themselves
(test_real_integration.py, test_probability_calibration.py).

Its output is a fixed, deterministic vector with no calibration record, which
is also what the uncalibrated-label tests rely on: a record without a
calibration temperature keeps the "model-generated class score" wording.
"""
import io
import os
import uuid

import numpy as np
from PIL import Image

from app.core.config import settings
from app.services.ai_service import (
    ICDR_CLASS_METADATA,
    BaseInferenceService,
    InferenceOutput,
    attribution_size,
    viridis_rgba_array,
)

FIXED_GRADE = 2
FIXED_SCORES = [0.04, 0.12, 0.78, 0.05, 0.01]
MODEL_VERSION = "test double (fixed vector, no model)"


class FixedOutputInferenceDouble(BaseInferenceService):
    """Returns FIXED_SCORES for every image and writes a small real PNG as its attribution map."""

    checkpoint_path = "(test double: no checkpoint)"
    verified_sha256 = None
    temperature = None

    def predict(self, pil_image: Image.Image, laterality: str = "OD") -> InferenceOutput:
        width, height = attribution_size(*pil_image.size)
        y, x = np.ogrid[:height, :width]
        cam = np.exp(-((x - width * 0.6) ** 2 + (y - height * 0.5) ** 2) / (2 * (width * 0.1) ** 2))
        attribution = Image.fromarray(viridis_rgba_array(cam / cam.max()))

        os.makedirs(settings.STORAGE_ATTRIBUTIONS_PATH, exist_ok=True)
        filename = f"gradcam_{uuid.uuid4().hex}.png"
        path = os.path.join(settings.STORAGE_ATTRIBUTIONS_PATH, filename)
        attribution.save(path, format="PNG")
        buf = io.BytesIO()
        attribution.save(buf, format="PNG")

        return InferenceOutput(
            primary_grade=FIXED_GRADE,
            primary_label=ICDR_CLASS_METADATA[FIXED_GRADE]["label"],
            primary_score=FIXED_SCORES[FIXED_GRADE],
            class_scores=[{"grade": i, "label": f"Grade {i}: {ICDR_CLASS_METADATA[i]['label']}",
                           "score": FIXED_SCORES[i]} for i in range(5)],
            target_layer="features.8 (Conv2d Bottleneck Residual)",
            top_activation_region="Test double: fixed attribution map.",
            model_version=MODEL_VERSION,
            execution_time_ms=1.0,
            disclaimer=("NOTICE: CLINICAL DECISION SUPPORT ONLY — NOT FOR INDEPENDENT DIAGNOSIS. "
                        "Test double output."),
            gradcam_bytes=buf.getvalue(),
            gradcam_filename=filename,
            gradcam_path=path,
            gradcam_url=f"/api/v1/storage/attributions/{filename}",
        )
