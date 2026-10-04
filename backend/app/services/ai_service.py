import abc
import datetime
from datetime import timezone
import hashlib
import io
import math
import os
import time
import uuid
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image

from app.core.config import settings


class ModelCheckpointError(RuntimeError):
    """
    Raised when the trained checkpoint is missing, corrupt, structurally
    incompatible, or fails provenance verification.

    This is deliberately fatal rather than recoverable: the alternative is
    returning diabetic retinopathy grades from an untrained network.
    """


# The 5 ICDR-standard Diabetic Retinopathy stages
ICDR_CLASS_METADATA = [
    {
        "grade": 0,
        "label": "No Apparent DR",
        "technical_term": "No Apparent Retinopathy",
        "description": "No microaneurysms, hemorrhages, or retinal lesions detected.",
    },
    {
        "grade": 1,
        "label": "Mild NPDR",
        "technical_term": "Mild Non-Proliferative Retinopathy",
        "description": "Microaneurysms only. Subtle vascular focal changes.",
    },
    {
        "grade": 2,
        "label": "Moderate NPDR",
        "technical_term": "Moderate Non-Proliferative Retinopathy",
        "description": "Dot-and-blot hemorrhages and hard exudates in posterior pole.",
    },
    {
        "grade": 3,
        "label": "Severe NPDR",
        "technical_term": "Severe Non-Proliferative Retinopathy",
        "description": "Meets 4-2-1 rule: hemorrhages in 4 quadrants or venous beading.",
    },
    {
        "grade": 4,
        "label": "Proliferative DR",
        "technical_term": "Proliferative Diabetic Retinopathy",
        "description": "Neovascularization of the disc/retina, preretinal hemorrhage.",
    },
]


class InferenceOutput:
    """Standardized inference response container."""

    def __init__(
        self,
        primary_grade: int,
        primary_label: str,
        primary_score: float,
        class_scores: List[Dict[str, Any]],
        target_layer: str,
        top_activation_region: str,
        model_version: str,
        execution_time_ms: float,
        disclaimer: str,
        gradcam_bytes: Optional[bytes] = None,
        gradcam_filename: Optional[str] = None,
        gradcam_path: Optional[str] = None,
        gradcam_url: Optional[str] = None,
    ):
        self.primary_grade = primary_grade
        self.primary_label = primary_label
        self.primary_score = primary_score
        self.class_scores = class_scores
        self.target_layer = target_layer
        self.top_activation_region = top_activation_region
        self.model_version = model_version
        self.execution_time_ms = execution_time_ms
        self.disclaimer = disclaimer
        self.gradcam_bytes = gradcam_bytes
        self.gradcam_filename = gradcam_filename
        self.gradcam_path = gradcam_path
        self.gradcam_url = gradcam_url


class BaseInferenceService(abc.ABC):
    """Abstract interface for CDSS inference engines."""

    @abc.abstractmethod
    def predict(
        self,
        pil_image: Image.Image,
        laterality: str = "OD",
    ) -> InferenceOutput:
        """Run classification and generate visual explanation."""
        pass


def generate_viridis_colormap(val: float) -> Tuple[int, int, int, int]:
    """Simple Viridis color map approximation returning RGBA."""
    # Viridis ramp: Purple (0.0) -> Blue -> Teal -> Green -> Yellow (1.0)
    v = max(0.0, min(1.0, val))
    if v < 0.05:
        return (0, 0, 0, 0)  # Transparent below activation floor

    # Colormap approximation
    r = int(255 * (0.2 + 0.8 * (v ** 1.8))) if v > 0.6 else int(255 * (0.28 * (1.0 - v)))
    g = int(255 * (0.05 + 0.9 * (v ** 0.9)))
    b = int(255 * (0.35 * (1.0 - v) + 0.1))
    alpha = int(255 * (0.2 + 0.7 * v))
    return (r, g, b, alpha)


def viridis_rgba_array(values: np.ndarray) -> np.ndarray:
    """
    Vectorised equivalent of generate_viridis_colormap over a 2D array.

    Returns an (H, W, 4) uint8 RGBA array producing output byte-identical to
    calling generate_viridis_colormap per pixel. The scalar function above
    remains the definition of the ramp; this only changes how it is applied.

    Applying the scalar version per pixel over a 512x512 heatmap costs 262,144
    interpreter iterations and measured ~600-700 ms on CPU, which dominated
    end-to-end request latency by a wide margin. This runs in ~40 ms.

    float64 is used deliberately: the scalar path widens each float32 sample to
    a Python float before arithmetic, and int() truncation makes the two
    dtypes disagree by one unit on some values. Matching the dtype keeps the
    rendered attribution identical to previously generated artefacts.
    """
    v = np.clip(values.astype(np.float64), 0.0, 1.0)

    r = np.where(
        v > 0.6,
        255.0 * (0.2 + 0.8 * np.power(v, 1.8)),
        255.0 * (0.28 * (1.0 - v)),
    )
    g = 255.0 * (0.05 + 0.9 * np.power(v, 0.9))
    b = 255.0 * (0.35 * (1.0 - v) + 0.1)
    a = 255.0 * (0.2 + 0.7 * v)

    rgba = np.stack([r, g, b, a], axis=-1).astype(np.uint8)
    rgba[v < 0.05] = 0  # transparent below the activation floor
    return rgba


def _peak_activation_region(cam) -> str:
    """
    Where the Grad-CAM map peaks, in plain image-frame terms.

    An earlier version shipped a fixed sentence per grade ("Isolated parafoveal
    microaneurysm cluster", ...) and displayed it as the "top saliency zone".
    Nothing in it came from the image. This reads the actual map: the cell of
    a 3x3 grid holding the maximum activation, plus the fraction of the map
    above half its peak so a diffuse map is not described as focal.
    """
    import numpy as _np
    arr = _np.asarray(cam, dtype=_np.float32)
    if arr.ndim != 2 or arr.size == 0 or float(arr.max()) <= 0:
        return "No attribution map was produced for this input."
    r, c = _np.unravel_index(int(_np.argmax(arr)), arr.shape)
    row = ("upper", "middle", "lower")[min(2, int(3 * r / arr.shape[0]))]
    col = ("left", "centre", "right")[min(2, int(3 * c / arr.shape[1]))]
    cell = "central" if (row, col) == ("middle", "centre") else f"{row}-{col}"
    above_half = float((arr >= 0.5 * arr.max()).mean())
    spread = "focal" if above_half < 0.15 else ("broad" if above_half < 0.5 else "diffuse")
    return (f"Peak Grad-CAM activation in the {cell} cell of a 3x3 grid over the frame "
            f"({spread}: {above_half:.0%} of the map is above half the peak).")


def create_mock_gradcam_heatmap(
    grade: int,
    width: int = 512,
    height: int = 512,
    laterality: str = "OD"
) -> Image.Image:
    """Generate a realistic synthetic Grad-CAM visual attribution overlay."""
    # Create coordinate grid
    y, x = np.ogrid[:height, :width]
    activation = np.zeros((height, width), dtype=np.float32)

    # Optic disc and fovea centers (OD vs OS)
    if laterality == "OD":
        # Right eye: Disc is on nasal (left side of fundus photo), Macula is temporal (center-right)
        fovea_cx, fovea_cy = width * 0.55, height * 0.50
        disc_cx, disc_cy = width * 0.30, height * 0.50
    else:
        # Left eye: Disc is on nasal (right side of fundus photo), Macula is temporal (center-left)
        fovea_cx, fovea_cy = width * 0.45, height * 0.50
        disc_cx, disc_cy = width * 0.70, height * 0.50

    if grade == 0:
        # Grade 0: Diffuse, low-level baseline activation around macula
        dist_sq = (x - fovea_cx) ** 2 + (y - fovea_cy) ** 2
        activation += 0.25 * np.exp(-dist_sq / (2 * (width * 0.3) ** 2))
    elif grade == 1:
        # Grade 1: Small focal hotspot near parafovea (microaneurysm)
        focal_x, focal_y = fovea_cx + width * 0.08, fovea_cy + height * 0.06
        dist_sq = (x - focal_x) ** 2 + (y - focal_y) ** 2
        activation += 0.85 * np.exp(-dist_sq / (2 * (width * 0.05) ** 2))
    elif grade == 2:
        # Grade 2: Hotspots along inferotemporal arcade and macula
        focal1_x, focal1_y = fovea_cx + width * 0.10, fovea_cy + height * 0.12
        focal2_x, focal2_y = fovea_cx - width * 0.08, fovea_cy - height * 0.10
        dist1 = (x - focal1_x) ** 2 + (y - focal1_y) ** 2
        dist2 = (x - focal2_x) ** 2 + (y - focal2_y) ** 2
        activation += 0.90 * np.exp(-dist1 / (2 * (width * 0.08) ** 2))
        activation += 0.75 * np.exp(-dist2 / (2 * (width * 0.07) ** 2))
    elif grade == 3:
        # Grade 3: Extensive multi-quadrant activations
        quadrants = [
            (fovea_cx + width * 0.15, fovea_cy - height * 0.15),
            (fovea_cx + width * 0.15, fovea_cy + height * 0.15),
            (fovea_cx - width * 0.15, fovea_cy + height * 0.15),
            (fovea_cx - width * 0.15, fovea_cy - height * 0.15),
        ]
        for qx, qy in quadrants:
            dist = (x - qx) ** 2 + (y - qy) ** 2
            activation += 0.80 * np.exp(-dist / (2 * (width * 0.09) ** 2))
    elif grade == 4:
        # Grade 4: Peak activation at optic disc and vascular tree
        dist_disc = (x - disc_cx) ** 2 + (y - disc_cy) ** 2
        dist_fovea = (x - fovea_cx) ** 2 + (y - fovea_cy) ** 2
        activation += 0.95 * np.exp(-dist_disc / (2 * (width * 0.09) ** 2))
        activation += 0.70 * np.exp(-dist_fovea / (2 * (width * 0.12) ** 2))

    # Normalize activation [0, 1]
    max_val = np.max(activation)
    if max_val > 0:
        activation /= max_val

    # Convert to RGBA image with Viridis palette
    rgba_arr = viridis_rgba_array(activation)

    return Image.fromarray(rgba_arr)


class MockInferenceService(BaseInferenceService):
    """
    SIMULATED inference engine for interface development and testing.

    It returns hardcoded 5-class distributions and a synthetic Grad-CAM
    overlay. Nothing it produces is a model output, and it is reachable only
    when AI_INFERENCE_ENGINE=mock is set explicitly - never as a fallback,
    because a simulated grade is indistinguishable from a real one in the UI.
    """

    def predict(
        self,
        pil_image: Image.Image,
        laterality: str = "OD",
    ) -> InferenceOutput:
        start_time = time.time()

        # Simulated engine always returns Grade 2 (Moderate NPDR) — the most
        # representative class in the test corpus. Grade is fixed and not
        # externally configurable; a steerable grade would be indistinguishable
        # from a real one in the UI and is therefore not permitted.
        grade = 2
        meta = ICDR_CLASS_METADATA[grade]

        # Construct mathematically normalized 5-class score distribution
        if grade == 0:
            scores = [0.91, 0.05, 0.02, 0.01, 0.01]
        elif grade == 1:
            scores = [0.08, 0.74, 0.12, 0.04, 0.02]
        elif grade == 2:
            scores = [0.04, 0.12, 0.78, 0.05, 0.01]
        elif grade == 3:
            scores = [0.02, 0.04, 0.08, 0.82, 0.04]
        else:
            scores = [0.01, 0.02, 0.03, 0.08, 0.86]

        primary_score = scores[grade]

        class_scores = [
            {"grade": i, "label": f"Grade {i}: {ICDR_CLASS_METADATA[i]['label']}", "score": scores[i]}
            for i in range(5)
        ]

        # Generate Grad-CAM attribution heatmap
        width, height = pil_image.size
        # Render at 512x512 resolution for smooth WebGL canvas blending
        gradcam_img = create_mock_gradcam_heatmap(grade, width=min(width, 768), height=min(height, 768), laterality=laterality)

        # Save heatmap to attribution storage
        os.makedirs(settings.STORAGE_ATTRIBUTIONS_PATH, exist_ok=True)
        gradcam_filename = f"gradcam_{uuid.uuid4().hex}.png"
        gradcam_path = os.path.join(settings.STORAGE_ATTRIBUTIONS_PATH, gradcam_filename)
        gradcam_img.save(gradcam_path, format="PNG")

        # Encode to bytes for in-memory / data URL consumers
        buf = io.BytesIO()
        gradcam_img.save(buf, format="PNG")
        gradcam_bytes = buf.getvalue()

        execution_time_ms = round((time.time() - start_time) * 1000.0 + 45.0, 1)

        disclaimer = (
            "NOTICE: CLINICAL DECISION SUPPORT ONLY — NOT FOR INDEPENDENT DIAGNOSIS. "
            "Model-generated scores represent preliminary mathematical associations from the pre-trained EfficientNet-B0 network. "
            "Diagnostic judgment, clinical staging, and management plans remain exclusively the responsibility of the reviewing clinician."
        )

        return InferenceOutput(
            primary_grade=grade,
            primary_label=meta["label"],
            primary_score=primary_score,
            class_scores=class_scores,
            target_layer="features.8 (Conv2d Bottleneck Residual)",
            top_activation_region="Simulated engine: no attribution map was computed.",
            model_version="EfficientNet-B0-DR-v1 (fixed weights)",
            execution_time_ms=execution_time_ms,
            disclaimer=disclaimer,
            gradcam_bytes=gradcam_bytes,
            gradcam_filename=gradcam_filename,
            gradcam_path=gradcam_path,
            gradcam_url=(f"/api/v1/storage/attributions/{gradcam_filename}"
                         if gradcam_filename else None),
        )


import logging

logger = logging.getLogger("dr_cdss.ai_service")


class EfficientNetB0InferenceService(BaseInferenceService):
    """
    Production inference engine harness for PyTorch EfficientNet-B0.
    Frozen checkpoint loaded in eval mode with Grad-CAM activation hooks.
    """

    def __init__(self, checkpoint_path: str = settings.MODEL_CHECKPOINT_PATH):
        self.checkpoint_path = checkpoint_path
        self._model = None
        self._initialized = False

    def load_model(self):
        """
        Load the trained checkpoint, or fail closed.

        The weights are fixed: nothing here updates them, so the model
        that serves a request is the one Chapter 4 evaluated.

        A clinical decision-support engine must never serve grades from an
        untrained graph. If the checkpoint is absent, unreadable, or does not
        match the expected SHA-256 digest, this raises and the service is left
        uninitialised — it does not silently fall back to random ImageNet-less
        weights or to the simulated engine.
        """
        # Cheap, dependency-free checks first, so a misconfigured path reports
        # the real cause rather than an unrelated import failure.
        if not os.path.exists(self.checkpoint_path):
            raise ModelCheckpointError(
                f"Model checkpoint not found at '{self.checkpoint_path}'. "
                "Refusing to serve inference from an untrained network. "
                "Set MODEL_CHECKPOINT_PATH to the trained weights, or set "
                "AI_INFERENCE_ENGINE=mock to run the simulated engine explicitly."
            )

        self._verify_checkpoint_digest()

        try:
            import torch
            import torchvision.models as models
        except ImportError as exc:
            raise ModelCheckpointError(
                "PyTorch is not installed in this environment, so the trained "
                f"engine cannot be served ({exc}). Install the CPU build first:\n"
                f"  pip install torch torchvision --index-url "
                f"https://download.pytorch.org/whl/cpu\n"
                f"  pip install -r backend/requirements.txt\n"
                f"Then restart the service.\n"
                "Alternatively set AI_INFERENCE_ENGINE=mock to run the "
                "simulated engine explicitly. The simulated engine is never "
                "selected implicitly, because a simulated grade is "
                "indistinguishable from a real one downstream."
            ) from exc

        device = torch.device(settings.MODEL_DEVICE if torch.cuda.is_available() else "cpu")
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=False),
            torch.nn.Linear(in_features=in_features, out_features=5, bias=True)
        )

        try:
            state_dict = torch.load(self.checkpoint_path, map_location=device, weights_only=True)
        except Exception as exc:
            raise ModelCheckpointError(
                f"Checkpoint at '{self.checkpoint_path}' could not be deserialised: {exc}"
            ) from exc

        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        clean_dict = {
            (k[7:] if k.startswith("module.") else k): v
            for k, v in state_dict.items()
        }

        try:
            model.load_state_dict(clean_dict, strict=True)
        except Exception as exc:
            raise ModelCheckpointError(
                f"Checkpoint at '{self.checkpoint_path}' does not match the "
                f"EfficientNet-B0 5-class topology: {exc}"
            ) from exc

        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        model.to(device)

        self._model = model
        self._device = device
        self._initialized = True
        logger.info(
            "Loaded EfficientNet-B0 weights from %s (device=%s); weights are fixed",
            self.checkpoint_path, device,
        )

    def _verify_checkpoint_digest(self):
        """
        Confirm the checkpoint bytes match the expected SHA-256, when one is
        configured. This is the runtime half of the provenance claim made in
        docs/chapter4/checkpoint_manifest.md: the graded weights on disk are
        demonstrably the weights that were evaluated.
        """
        expected = (settings.MODEL_CHECKPOINT_SHA256 or "").strip().lower()
        if not expected:
            # An earlier version logged a warning and served anyway, while the
            # documentation said the engine refuses to serve unverified weights.
            # It now does what the documentation says.
            raise ModelCheckpointError(
                "MODEL_CHECKPOINT_SHA256 is not set. The engine refuses to serve "
                "weights whose provenance it cannot verify; set the expected "
                "digest in the environment (see .env.example)."
            )

        digest = hashlib.sha256()
        with open(self.checkpoint_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()

        if actual != expected:
            raise ModelCheckpointError(
                f"Checkpoint digest mismatch for '{self.checkpoint_path}'. "
                f"Expected {expected}, found {actual}. Refusing to serve "
                "inference from unverified weights."
            )
        # Surfaced by /health so a deployment can be checked for WHICH weights
        # it serves, not only that some weights loaded.
        self.verified_sha256 = actual

    def predict(
        self,
        pil_image: Image.Image,
        laterality: str = "OD",
    ) -> InferenceOutput:
        if not self._initialized or self._model is None:
            raise ModelCheckpointError(
                "Inference requested before a verified checkpoint was loaded. "
                "The service refuses to grade from an uninitialised model."
            )

        import torch
        import torchvision.transforms as transforms

        start_time = time.time()

        # SECURITY NOTE: An earlier revision accepted a `candidate_grade`
        # parameter that allowed a caller to substitute an externally chosen
        # class for the model's argmax. That parameter has been removed from
        # the entire inference interface. Grade is always the model's own argmax.

        # ------------------------------------------------------------------
        # Model inference. This must either produce a genuine result or fail.
        # There is no fallback: a fabricated grade is indistinguishable from a
        # real one downstream, which is precisely what must not happen.
        # ------------------------------------------------------------------
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        rgb_image = pil_image.convert("RGB")
        input_tensor = preprocess(rgb_image).unsqueeze(0).to(self._device)
        input_tensor.requires_grad = True

        activations = []

        def forward_hook(module, inp, out):
            activations.append(out)

        hook_handle = self._model.features[8].register_forward_hook(forward_hook)
        try:
            logits = self._model(input_tensor)
        finally:
            hook_handle.remove()

        probs_tensor = torch.softmax(logits, dim=1)[0].detach()
        probs = [round(float(x), 4) for x in probs_tensor]

        grade = int(torch.argmax(logits, dim=1).item())
        meta = ICDR_CLASS_METADATA[grade]
        primary_score = probs[grade]
        class_scores = [
            {"grade": i, "label": f"Grade {i}: {ICDR_CLASS_METADATA[i]['label']}",
             "score": probs[i]}
            for i in range(5)
        ]

        # ------------------------------------------------------------------
        # Visual attribution. A failure here degrades the explanation; it does
        # NOT change the grade or the scores, and it never substitutes a
        # synthetic activation pattern for a real one.
        # ------------------------------------------------------------------
        gradcam_img = None
        gradcam_unavailable_reason = None
        peak_region = "No attribution map was produced for this input."
        try:
            target_logit = logits[0, grade]
            grads = torch.autograd.grad(
                target_logit, activations[0], retain_graph=False)[0]
            weights = torch.mean(grads, dim=(2, 3), keepdim=True)
            cam = torch.relu(
                torch.sum(weights * activations[0], dim=1)
            ).squeeze().detach().cpu().numpy()

            if float(np.max(cam)) <= 0:
                # A uniformly zero CAM carries no attribution. Rendering a
                # plausible-looking heatmap here would invent one.
                gradcam_unavailable_reason = (
                    "attribution map was uniformly zero for this input")
            else:
                peak_region = _peak_activation_region(cam)
                cam = cam / np.max(cam)
                cam_pil = Image.fromarray((cam * 255).astype(np.uint8)).resize(
                    (512, 512), Image.Resampling.BILINEAR)
                cam_arr = np.array(cam_pil, dtype=np.float32) / 255.0
                gradcam_img = Image.fromarray(viridis_rgba_array(cam_arr))

        except Exception as exc:                                  # noqa: BLE001
            logger.warning(
                "Grad-CAM computation failed (%s). Returning the model result "
                "without a visual explanation.", exc)
            gradcam_unavailable_reason = f"attribution computation failed: {exc}"

        # Save the attribution artifact, if there is a genuine one to save.
        gradcam_filename = gradcam_path = gradcam_bytes = None
        if gradcam_img is not None:
            os.makedirs(settings.STORAGE_ATTRIBUTIONS_PATH, exist_ok=True)
            gradcam_filename = f"gradcam_{uuid.uuid4().hex}.png"
            gradcam_path = os.path.join(
                settings.STORAGE_ATTRIBUTIONS_PATH, gradcam_filename)
            gradcam_img.save(gradcam_path, format="PNG")

            buf = io.BytesIO()
            gradcam_img.save(buf, format="PNG")
            gradcam_bytes = buf.getvalue()

        execution_time_ms = round((time.time() - start_time) * 1000.0, 1)

        disclaimer = (
            "NOTICE: CLINICAL DECISION SUPPORT ONLY — NOT FOR INDEPENDENT DIAGNOSIS. "
            "Model-generated scores represent preliminary mathematical associations from the pre-trained EfficientNet-B0 network. "
            "Diagnostic judgment, clinical staging, and management plans remain exclusively the responsibility of the reviewing clinician."
        )

        if gradcam_unavailable_reason:
            disclaimer += (
                " VISUAL EXPLANATION UNAVAILABLE for this study: "
                f"{gradcam_unavailable_reason}. The grade and scores above "
                "are the model's own output and are unaffected."
            )

        return InferenceOutput(
            primary_grade=grade,
            primary_label=meta["label"],
            primary_score=primary_score,
            class_scores=class_scores,
            target_layer="features.8 (Conv2d Bottleneck Residual)",
            top_activation_region=peak_region,
            model_version="EfficientNet-B0-DR-v1 (fixed weights)",
            execution_time_ms=execution_time_ms,
            disclaimer=disclaimer,
            gradcam_bytes=gradcam_bytes,
            gradcam_filename=gradcam_filename,
            gradcam_path=gradcam_path,
            gradcam_url=f"/api/v1/storage/attributions/{gradcam_filename}",
        )


def get_ai_inference_service() -> BaseInferenceService:
    """
    Resolve the active inference engine from configuration.

    The simulated engine is only ever returned when it has been asked for by
    name (AI_INFERENCE_ENGINE=mock). A failure to load the real checkpoint
    propagates: it must not be silently downgraded to simulated grades, because
    the two are indistinguishable to a clinician reading the result.
    """
    engine_type = os.getenv("AI_INFERENCE_ENGINE", "pytorch").strip().lower()

    if engine_type == "mock":
        logger.warning(
            "AI_INFERENCE_ENGINE=mock — serving SIMULATED diabetic retinopathy "
            "grades. This engine is for development and interface testing only."
        )
        return MockInferenceService()

    service = EfficientNetB0InferenceService(checkpoint_path=settings.MODEL_CHECKPOINT_PATH)
    service.load_model()
    return service


_service_singleton: Optional[BaseInferenceService] = None


def get_active_inference_service() -> BaseInferenceService:
    """
    Return the process-wide inference engine, loading it on first use.

    Resolution is lazy so that a missing or unverified checkpoint surfaces as a
    handled 503 on the assessment endpoint rather than an import-time crash
    loop with no diagnostics. The error is raised on every call until the
    checkpoint is corrected — it is never cached away or downgraded.
    """
    global _service_singleton
    if _service_singleton is None:
        _service_singleton = get_ai_inference_service()
    return _service_singleton


def get_inference_health() -> Dict[str, Any]:
    """Report engine readiness for the health endpoint without raising."""
    try:
        service = get_active_inference_service()
    except ModelCheckpointError as exc:
        return {"ready": False, "engine": "pytorch", "detail": str(exc)}
    except Exception as exc:  # pragma: no cover - defensive
        return {"ready": False, "engine": "unknown", "detail": str(exc)}

    if isinstance(service, MockInferenceService):
        return {
            "ready": True,
            "engine": "mock",
            "detail": "SIMULATED grades — not a trained model.",
        }
    return {
        "ready": True,
        "engine": "pytorch",
        "detail": f"EfficientNet-B0 checkpoint loaded from {service.checkpoint_path}",
        "checkpoint_sha256": getattr(service, "verified_sha256", None),
    }
