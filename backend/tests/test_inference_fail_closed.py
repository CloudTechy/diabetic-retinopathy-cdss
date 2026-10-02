"""
Fail-closed guarantees for the inference engine.

The property under test is narrow and safety-critical: the CDSS must never
return a diabetic retinopathy grade unless verified trained weights are loaded.
Silently degrading to a randomly-initialised network or to the simulated engine
produces output a clinician cannot distinguish from a real prediction.
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
    MockInferenceService,
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
    """predict() must raise rather than quietly delegating to the mock engine."""
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
    original = settings.MODEL_CHECKPOINT_SHA256
    settings.MODEL_CHECKPOINT_SHA256 = ""
    try:
        with pytest.raises(ModelCheckpointError):
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


def test_missing_torch_does_not_silently_downgrade_to_mock(tmp_path, monkeypatch):
    """
    Regression guard: the old loader caught ImportError and served simulated
    grades. A missing PyTorch must now surface as a checkpoint error instead.
    """
    from app.core.config import settings

    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"placeholder")
    monkeypatch.setattr(settings, "MODEL_CHECKPOINT_SHA256", "")

    service = EfficientNetB0InferenceService(checkpoint_path=str(weights))
    if torch_installed:
        # With torch present the failure comes from deserialisation instead;
        # either way it must be a ModelCheckpointError, never a mock result.
        with pytest.raises(ModelCheckpointError):
            service.load_model()
    else:
        with pytest.raises(ModelCheckpointError) as exc:
            service.load_model()
        assert "pytorch is not installed" in str(exc.value).lower()


def test_mock_engine_only_served_when_explicitly_requested(monkeypatch, tmp_path):
    """The simulated engine is opt-in by name, never a fallback."""
    from app.core.config import settings

    monkeypatch.setenv("AI_INFERENCE_ENGINE", "mock")
    assert isinstance(get_ai_inference_service(), MockInferenceService)

    # Without the explicit opt-in, a broken checkpoint must raise rather than
    # silently returning the simulated engine.
    monkeypatch.setenv("AI_INFERENCE_ENGINE", "pytorch")
    monkeypatch.setattr(settings, "MODEL_CHECKPOINT_PATH", str(tmp_path / "absent.pth"))
    with pytest.raises(ModelCheckpointError):
        get_ai_inference_service()


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
