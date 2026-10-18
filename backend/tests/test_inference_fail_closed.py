"""
Fail-closed guarantees for the inference engine.

The property under test is narrow and safety-critical: the CDSS must never
return a diabetic retinopathy grade unless verified trained weights are loaded.
Silently degrading to a randomly-initialised network, or to any other source of
predictions, produces output a clinician cannot distinguish from a real one.
"""

import hashlib
import importlib.util
import os

import pytest

torch_installed = importlib.util.find_spec("torch") is not None
requires_torch = pytest.mark.skipif(
    not torch_installed, reason="PyTorch not installed in this environment."
)

from app.services.ai_service import (
    EfficientNetB0InferenceService,
    ModelCheckpointError,
    get_ai_inference_service,
)


@pytest.fixture
def blank_image():
    from PIL import Image
    return Image.new("RGB", (512, 512), (128, 64, 64))


def test_missing_checkpoint_raises_instead_of_using_random_weights(tmp_path):
    """A absent checkpoint must abort the load, not fall through to weights=None."""
    missing = tmp_path / "does_not_exist.pth"
    service = EfficientNetB0InferenceService(checkpoint_path=str(missing))

    with pytest.raises(ModelCheckpointError) as exc:
        service.load_model()

    assert "not found" in str(exc.value).lower()
    assert service._initialized is False
    assert service._model is None


def test_predict_refuses_when_uninitialised(blank_image, tmp_path):
    """predict() must raise rather than return a prediction from anywhere else."""
    service = EfficientNetB0InferenceService(checkpoint_path=str(tmp_path / "nope.pth"))

    with pytest.raises(ModelCheckpointError):
        service.predict(blank_image, laterality="OD")


@requires_torch
def test_corrupt_checkpoint_raises(tmp_path):
    """A file that exists but is not a loadable checkpoint must fail closed."""
    bogus = tmp_path / "corrupt.pth"
    bogus.write_bytes(b"this is not a torch checkpoint")

    service = EfficientNetB0InferenceService(checkpoint_path=str(bogus))

    from app.core.config import settings
    import hashlib
    original = settings.MODEL_CHECKPOINT_SHA256
    # The digest MATCHES the bogus bytes, so the failure below is the corrupt
    # file itself, not the digest check.
    settings.MODEL_CHECKPOINT_SHA256 = hashlib.sha256(b"this is not a torch checkpoint").hexdigest()
    try:
        with pytest.raises(ModelCheckpointError):
            service.load_model()
    finally:
        settings.MODEL_CHECKPOINT_SHA256 = original

    assert service._initialized is False


@requires_torch
def test_blank_digest_refuses_to_serve(tmp_path):
    """
    A blank MODEL_CHECKPOINT_SHA256 once meant "serve with a warning". The
    documentation says the engine refuses to serve unverified weights, so a
    blank digest must refuse before the file is even read.
    """
    bogus = tmp_path / "weights.pth"
    bogus.write_bytes(b"irrelevant")
    service = EfficientNetB0InferenceService(checkpoint_path=str(bogus))

    from app.core.config import settings
    original = settings.MODEL_CHECKPOINT_SHA256
    settings.MODEL_CHECKPOINT_SHA256 = ""
    try:
        with pytest.raises(ModelCheckpointError, match="not set"):
            service.load_model()
    finally:
        settings.MODEL_CHECKPOINT_SHA256 = original
    assert service._initialized is False


def test_digest_mismatch_refuses_to_load(tmp_path):
    """Weights that are not the graded artefact must be rejected on provenance."""
    from app.core.config import settings

    impostor = tmp_path / "impostor.pth"
    impostor.write_bytes(b"arbitrary bytes standing in for a different model")
    wrong_digest = hashlib.sha256(b"a completely different payload").hexdigest()

    service = EfficientNetB0InferenceService(checkpoint_path=str(impostor))
    original = settings.MODEL_CHECKPOINT_SHA256
    settings.MODEL_CHECKPOINT_SHA256 = wrong_digest
    try:
        with pytest.raises(ModelCheckpointError) as exc:
            service.load_model()
        assert "digest mismatch" in str(exc.value).lower()
    finally:
        settings.MODEL_CHECKPOINT_SHA256 = original


def test_missing_torch_does_not_silently_downgrade(tmp_path, monkeypatch):
    """
    Regression guard: the old loader caught ImportError and served grades that
    did not come from the model. A missing PyTorch must surface as a checkpoint
    error instead.
    """
    from app.core.config import settings

    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"placeholder")
    monkeypatch.setattr(settings, "MODEL_CHECKPOINT_SHA256", "")

    service = EfficientNetB0InferenceService(checkpoint_path=str(weights))
    if torch_installed:
        # With torch present the failure comes from deserialisation instead;
        # either way it must be a ModelCheckpointError, never a result.
        with pytest.raises(ModelCheckpointError):
            service.load_model()
    else:
        with pytest.raises(ModelCheckpointError) as exc:
            service.load_model()
        assert "pytorch is not installed" in str(exc.value).lower()


@pytest.mark.parametrize("engine_setting", [None, "mock", "simulated", "pytorch"])
def test_no_configuration_selects_another_engine(engine_setting, monkeypatch, tmp_path):
    """
    EfficientNet-B0 is the only engine. The former AI_INFERENCE_ENGINE switch
    to a simulated engine is gone: whatever it is set to, a missing checkpoint
    raises rather than returning predictions from another source.
    """
    from app.core.config import settings

    if engine_setting is None:
        monkeypatch.delenv("AI_INFERENCE_ENGINE", raising=False)
    else:
        monkeypatch.setenv("AI_INFERENCE_ENGINE", engine_setting)
    monkeypatch.setattr(settings, "MODEL_CHECKPOINT_PATH", str(tmp_path / "absent.pth"))
    with pytest.raises(ModelCheckpointError):
        get_ai_inference_service()


def test_missing_calibration_artifact_fails_closed(monkeypatch, tmp_path):
    """Verified weights without the fitted temperature must not serve uncalibrated output."""
    from app.services.ai_service import CalibrationArtifactError, load_calibration_artifact

    with pytest.raises(CalibrationArtifactError):
        load_calibration_artifact(str(tmp_path / "absent_artifact.json"), "0" * 64)


def test_simulated_engine_is_not_in_the_application():
    """No simulated engine or synthetic attribution generator remains importable from the app."""
    import app.services.ai_service as ai_svc

    for name in ("MockInferenceService", "create_mock_gradcam_heatmap"):
        assert not hasattr(ai_svc, name), name
    engines = [obj for obj in vars(ai_svc).values()
               if isinstance(obj, type) and issubclass(obj, ai_svc.BaseInferenceService)
               and obj is not ai_svc.BaseInferenceService]
    assert engines == [EfficientNetB0InferenceService]


def test_health_reports_not_ready_when_the_checkpoint_cannot_load(monkeypatch, tmp_path):
    """With no loadable checkpoint the health endpoint says so; it never reports another engine as ready."""
    import app.services.ai_service as ai_svc
    from app.core.config import settings

    monkeypatch.setattr(ai_svc, "_service_singleton", None)
    monkeypatch.setattr(settings, "MODEL_CHECKPOINT_PATH", str(tmp_path / "absent.pth"))
    health = ai_svc.get_inference_health()
    assert health["ready"] is False
    assert health["engine"] == "pytorch"


def test_configured_checkpoint_path_matches_compose_mount():
    """
    Regression guard: compose mounts ./backend at /app, so the weights land at
    /app/models/weights/. A path pointing anywhere else silently produced
    random-weight predictions before this was fixed.
    """
    from app.core.config import settings

    # Compared with separators normalised: the configured path is the container
    # path under compose and the repository path otherwise, and on Windows the
    # latter is backslash-separated. The invariant is WHERE it points, not which
    # separator the host happens to use.
    configured = settings.MODEL_CHECKPOINT_PATH.replace(chr(92), "/")
    assert configured.endswith("models/weights/efficientnet_b0_dr.pth"), (
        f"MODEL_CHECKPOINT_PATH is {settings.MODEL_CHECKPOINT_PATH!r}; compose "
        "mounts ./backend at /app, so the weights must resolve under "
        "models/weights/. A path pointing elsewhere silently produced "
        "random-weight predictions before this was fixed.")


def test_repository_checkpoint_matches_declared_digest():
    """The committed weights must be the evaluated weights."""
    from app.core.config import settings

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    weights = os.path.join(repo_root, "backend", "models", "weights", "efficientnet_b0_dr.pth")
    if not os.path.exists(weights):
        pytest.skip("Checkpoint binary not present in this checkout.")

    digest = hashlib.sha256()
    with open(weights, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)

    assert digest.hexdigest() == settings.MODEL_CHECKPOINT_SHA256
