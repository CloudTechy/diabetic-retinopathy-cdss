"""
Probability calibration (temperature scaling): the fit uses the validation
partition only, the frozen temperature reproduces the committed test metrics
without changing a single prediction, and the interface receives the
calibrated probability vector.

Artefacts under test: docs/chapter4/probability_calibration/ and
backend/models/calibration/temperature_scaling_artifact.json, produced by
backend/scripts/calibrate_probabilities.py.
"""

import csv
import hashlib
import importlib.util
import json
import math
import os
import sys

import numpy as np
import pytest

from tests.conftest import create_synthetic_retinal_fundus, image_to_bytes

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
ROOT = os.path.dirname(BACKEND)
EVIDENCE = os.path.join(ROOT, "docs", "chapter4", "probability_calibration")
MANIFEST = os.path.join(ROOT, "docs", "chapter4", "dataset_split_manifest.csv")
EVAL_SUMMARY = os.path.join(ROOT, "docs", "chapter4", "evaluation_summary.json")
RUNTIME_ARTIFACT = os.path.join(BACKEND, "models", "calibration", "temperature_scaling_artifact.json")
CHECKPOINT = os.path.join(BACKEND, "models", "weights", "efficientnet_b0_dr.pth")
FIXTURE_IMAGE = os.path.join(HERE, "fixtures", "aptos_heldout_d1f1ea894da1.png")

torch_installed = importlib.util.find_spec("torch") is not None
requires_torch = pytest.mark.skipif(not torch_installed, reason="PyTorch not installed in this environment.")


def _script():
    path = os.path.join(BACKEND, "scripts", "calibrate_probabilities.py")
    sys.path.insert(0, os.path.dirname(path))
    spec = importlib.util.spec_from_file_location("calibrate_probabilities", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _json(name):
    with open(os.path.join(EVIDENCE, name), encoding="utf-8") as fh:
        return json.load(fh)


def _csv(name):
    with open(os.path.join(EVIDENCE, name), newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _manifest_ids(split):
    with open(MANIFEST, newline="", encoding="utf-8") as fh:
        return {r["image_id"] for r in csv.DictReader(fh) if r["split"] == split}


def _softmax(z, t=1.0):
    z = np.asarray(z, dtype=np.float64) / t
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


@pytest.fixture(scope="module")
def artifact():
    return _json("temperature_scaling_artifact.json")


@pytest.fixture(scope="module")
def temperature(artifact):
    return float(artifact["temperature"])


# ---------------------------------------------------------------------------
# 1. the fit: validation partition only
# ---------------------------------------------------------------------------

def test_fit_refuses_test_tagged_records():
    fit_temperature = _script().fit_temperature
    logits = np.array([[2.0, 0.1, 0.0, -1.0, 0.3], [0.0, 1.5, 0.2, 0.1, -0.4]])
    with pytest.raises(ValueError, match="test"):
        fit_temperature(logits, [0, 1], ["test", "test"])
    with pytest.raises(ValueError, match="test"):
        fit_temperature(logits, [0, 1], ["val", "test"])


def test_artifact_was_fitted_on_validation_and_carries_no_test_statistic(artifact):
    assert artifact["fit_split"] == "val"
    assert artifact["fit_counts"]["n"] == 526
    assert set(artifact) <= _script().ARTIFACT_KEYS

    def keys(node):
        if isinstance(node, dict):
            for k, v in node.items():
                yield k
                yield from keys(v)
    assert not [k for k in keys(artifact) if "test" in k.lower()]
    assert artifact["validation_ids_sha256"] == hashlib.sha256(
        "\n".join(sorted(_manifest_ids("val"))).encode()).hexdigest()


def test_prediction_manifests_are_the_two_disjoint_partitions():
    val, test = _csv("validation_predictions.csv"), _csv("test_predictions.csv")
    assert len(val) == 526 and {r["split"] for r in val} == {"val"}
    assert len(test) == 525 and {r["split"] for r in test} == {"test"}
    assert {r["image_id"] for r in val} == _manifest_ids("val")
    assert {r["image_id"] for r in test} == _manifest_ids("test")
    assert not ({r["image_id"] for r in val} & {r["image_id"] for r in test})


def test_temperature_is_reproduced_from_the_validation_manifest_alone(temperature):
    rows = _csv("validation_predictions.csv")
    logits = np.array([[float(r[f"logit_{k}"]) for k in range(5)] for r in rows])
    labels = np.array([int(r["true_grade"]) for r in rows])
    refit, _ = _script().fit_temperature(logits, labels, [r["split"] for r in rows])
    assert refit == pytest.approx(temperature, abs=1e-7)


def test_leakage_audit_shows_temperature_frozen_before_test_partition_was_read(artifact):
    audit = _json("leakage_audit.json")
    assert audit["temperature_frozen_before_test_partition_was_read"] is True
    assert audit["freeze_step"] < audit["first_test_partition_access_step"]
    assert audit["artifact_unchanged_after_evaluation"] is True
    assert audit["test_metric_gate_on_temperature"] is False
    assert audit["val_test_id_overlap"] == 0
    assert audit["artifact_sha256_at_freeze"] == _sha(os.path.join(EVIDENCE, "temperature_scaling_artifact.json"))
    fit_events = [e for e in audit["events"] if e["phase"] == "fit"]
    assert fit_events and all("test" not in e["event"] for e in fit_events)


# ---------------------------------------------------------------------------
# 2. the frozen temperature on the test partition
# ---------------------------------------------------------------------------

def test_test_metrics_are_reproduced_from_test_manifest_and_frozen_temperature(temperature):
    script = _script()
    rows = _csv("test_predictions.csv")
    logits = np.array([[float(r[f"logit_{k}"]) for k in range(5)] for r in rows])
    labels = np.array([int(r["true_grade"]) for r in rows])
    metrics = _json("test_metrics.json")
    assert metrics["temperature"] == temperature
    for key, probs in (("uncalibrated", _softmax(logits)), ("calibrated", _softmax(logits, temperature))):
        recomputed = script.evaluate(probs, labels)
        for m in ("nll", "brier", "ece_15", "mce_15", "ece_10", "mce_10", "accuracy", "qwk"):
            assert recomputed[m] == pytest.approx(metrics[key][m], abs=1e-12), (key, m)
        stored = np.array([[float(r[f"prob_{key}_{k}"]) for k in range(5)] for r in rows])
        assert np.max(np.abs(stored - probs)) < 1e-12


def test_top_one_predictions_accuracy_confusion_matrix_and_qwk_unchanged():
    metrics = _json("test_metrics.json")
    inv = metrics["invariance"]
    assert inv["top_one_predictions_changed"] == 0
    assert inv["accuracy_identical"] and inv["confusion_matrix_identical"] and inv["qwk_identical"]
    rows = _csv("test_predictions.csv")
    assert all(r["predicted_grade_uncalibrated"] == r["predicted_grade_calibrated"] for r in rows)
    with open(EVAL_SUMMARY, encoding="utf-8") as fh:
        committed = json.load(fh)
    assert metrics["calibrated"]["confusion_matrix"] == committed["confusion_matrix"]
    assert round(metrics["calibrated"]["accuracy"], 6) == committed["accuracy"]
    assert round(metrics["calibrated"]["qwk"], 6) == committed["qwk"]


# ---------------------------------------------------------------------------
# 3. runtime: the engine serves softmax(logits / T)
# ---------------------------------------------------------------------------

def test_runtime_artifact_is_the_evidence_artifact():
    assert _sha(RUNTIME_ARTIFACT) == _sha(os.path.join(EVIDENCE, "temperature_scaling_artifact.json"))


def test_the_archive_ships_the_runtime_artifact():
    """The engine fails closed without it, so an archive missing it cannot grade."""
    sys.path.insert(0, os.path.join(BACKEND, "scripts"))
    import assemble_submission_package as asm
    shipped = {dst for _, dst in asm.manifest()}
    assert "backend/models/calibration/temperature_scaling_artifact.json" in shipped


def test_calibrated_probabilities_is_softmax_of_logits_over_t(temperature):
    from app.services.ai_service import calibrated_probabilities
    z = [2.3, -0.4, 1.9, -1.2, 0.05]
    got = np.array(calibrated_probabilities(z, temperature))
    assert np.max(np.abs(got - _softmax(z, temperature))) < 1e-15
    assert math.isclose(sum(got), 1.0, abs_tol=1e-12)
    assert int(np.argmax(got)) == int(np.argmax(z))
    assert np.max(np.abs(np.array(calibrated_probabilities(z, 1.0)) - _softmax(z))) < 1e-15


@pytest.mark.parametrize("mutate, message", [
    (lambda a: a.update(fit_split="test"), "validation"),
    (lambda a: a.update(temperature=0.0), "temperature"),
    (lambda a: a["checkpoint"].update(sha256="0" * 64), "fitted for checkpoint"),
])
def test_engine_refuses_an_invalid_artifact(tmp_path, artifact, mutate, message):
    from app.services.ai_service import CalibrationArtifactError, load_calibration_artifact
    bad = json.loads(json.dumps(artifact))
    mutate(bad)
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(CalibrationArtifactError, match=message):
        load_calibration_artifact(str(path), artifact["checkpoint"]["sha256"])
    with pytest.raises(CalibrationArtifactError, match="not found"):
        load_calibration_artifact(str(tmp_path / "missing.json"), artifact["checkpoint"]["sha256"])


STUB_LOGITS = [0.4, 1.1, 2.6, -0.3, 0.9]


def _stub_engine(artifact):
    """The real engine class with a tiny model whose logits are fixed and known."""
    import torch
    from app.services.ai_service import EfficientNetB0InferenceService

    class Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.features = torch.nn.Sequential(*[torch.nn.Identity() for _ in range(8)],
                                                torch.nn.Conv2d(3, 4, 1))
            self.head = torch.nn.Linear(4, 5)
            torch.nn.init.zeros_(self.head.weight)
            with torch.no_grad():
                self.head.bias.copy_(torch.tensor(STUB_LOGITS))

        def forward(self, x):
            return self.head(self.features(x).mean(dim=(2, 3)))

    engine = EfficientNetB0InferenceService(checkpoint_path="unused")
    engine._model, engine._device, engine._initialized = Stub().eval(), torch.device("cpu"), True
    engine.calibration, engine.temperature = artifact, float(artifact["temperature"])
    return engine


@requires_torch
def test_engine_predict_returns_the_calibrated_vector(artifact, temperature):
    engine = _stub_engine(artifact)
    out = engine.predict(create_synthetic_retinal_fundus(256, 256))
    served = np.array([cs["score"] for cs in out.class_scores])
    expected = _softmax(STUB_LOGITS, temperature)
    assert np.max(np.abs(served - expected)) < 1e-6
    assert np.max(np.abs(np.array([cs["uncalibrated_score"] for cs in out.class_scores])
                         - _softmax(STUB_LOGITS))) < 1e-6
    assert np.max(np.abs(served - _softmax(STUB_LOGITS))) > 1e-4   # it is not the raw softmax
    assert out.primary_grade == 2 and out.primary_score == pytest.approx(expected[2], abs=1e-6)
    assert out.calibration_temperature == temperature
    assert f"T={temperature:.6f}" in out.model_version


@requires_torch
def test_interface_receives_the_calibrated_vector(authed_client, artifact, temperature):
    """Upload through the API with the stub engine; the response's classScores are softmax(z / T)."""
    import app.services.ai_service as ai_svc
    old = ai_svc._service_singleton
    ai_svc._service_singleton = _stub_engine(artifact)
    try:
        created = authed_client.post("/api/v1/assessments", json={
            "patientId": "PT-CALIBRATION", "laterality": "OD",
            "cameraModel": "Topcon TRC-NW400", "clinicalNotes": "calibration interface test"})
        assert created.status_code == 201
        aid = created.json()["id"]
        upload = authed_client.post(
            f"/api/v1/assessments/{aid}/upload",
            files={"file": ("fundus.jpg", image_to_bytes(create_synthetic_retinal_fundus(512, 512), "JPEG"),
                            "image/jpeg")})
        assert upload.status_code == 200, upload.text
        for obs in (upload.json()["modelObservation"],
                    authed_client.get(f"/api/v1/assessments/{aid}/result").json()):
            expected = _softmax(STUB_LOGITS, temperature)
            served = np.array([c["score"] for c in sorted(obs["classScores"], key=lambda c: c["grade"])])
            raw = np.array([c["uncalibratedScore"] for c in sorted(obs["classScores"], key=lambda c: c["grade"])])
            assert np.max(np.abs(served - expected)) < 1e-6
            assert np.max(np.abs(raw - _softmax(STUB_LOGITS))) < 1e-6
            assert obs["primaryScore"] == pytest.approx(expected[2], abs=1e-6)
            assert obs["calibrationTemperature"] == pytest.approx(temperature, abs=1e-6)
    finally:
        ai_svc._service_singleton = old


@pytest.mark.slow
@requires_torch
def test_real_checkpoint_serves_softmax_of_its_own_logits_over_t(tmp_path, temperature):
    """The real engine, real weights, real runtime artifact, a real held-out image."""
    import torch
    from PIL import Image
    from app.core.config import settings
    from app.services.ai_service import EfficientNetB0InferenceService, build_preprocess

    old = settings.STORAGE_ATTRIBUTIONS_PATH
    settings.STORAGE_ATTRIBUTIONS_PATH = str(tmp_path)
    try:
        engine = EfficientNetB0InferenceService(checkpoint_path=CHECKPOINT,
                                                calibration_artifact_path=RUNTIME_ARTIFACT)
        engine.load_model()
        assert engine.temperature == temperature
        with Image.open(FIXTURE_IMAGE) as im:
            image = im.convert("RGB")
        out = engine.predict(image)
        with torch.inference_mode():
            z = engine._model(build_preprocess()(image).unsqueeze(0)).double().numpy()[0]
    finally:
        settings.STORAGE_ATTRIBUTIONS_PATH = old
    served = np.array([cs["score"] for cs in out.class_scores])
    assert np.max(np.abs(served - _softmax(z, temperature))) < 1e-5
    assert out.primary_grade == int(np.argmax(z))
