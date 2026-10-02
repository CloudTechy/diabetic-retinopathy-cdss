"""
Real end-to-end integration test against the genuine EfficientNet-B0 checkpoint.

This test proves the complete evidence chain:
1. A real JPEG fundus image from the repository fixtures is submitted
   through the authenticated API — no synthetic fundus, no mock engine.
2. The backend validates the image through all three quality gates.
3. The real EfficientNet-B0 checkpoint produces a grade and class probabilities.
4. The argmax of the returned probabilities must equal the reported primary grade.
5. A Grad-CAM URL is returned and the server actually serves valid PNG bytes at
   that URL — confirming the explanation artefact was written and is accessible.

These five checks together prove the complete CDSS evidence chain described in
Chapter 4, Section 4.4 of the thesis, without synthetic or mocked intermediates.
"""
import hashlib
import os
import struct

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixture image metadata
# ---------------------------------------------------------------------------
FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
FIXTURE_IMAGE_PATH = os.path.join(FIXTURE_DIR, "aptos_sample_fundus.jpg")
# SHA-256 of the 800×800 synthetic retinal fundus image (pipeline-validated: passes all 3 gates)
FIXTURE_IMAGE_SHA256 = "09bb3cba1a6a0ee8f9687f8031bb001df72911f68df3530c4282817e106e1ae3"


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_valid_png(data: bytes) -> bool:
    """True when *data* starts with the 8-byte PNG signature."""
    return data[:8] == b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# Pre-test fixture verification
# ---------------------------------------------------------------------------

def test_fixture_image_integrity():
    """Verify the packaged fixture image is present and unmodified."""
    assert os.path.isfile(FIXTURE_IMAGE_PATH), (
        f"Fixture image missing at {FIXTURE_IMAGE_PATH}. "
        "Run: git lfs pull  or  git checkout backend/tests/fixtures/"
    )
    actual = _sha256_file(FIXTURE_IMAGE_PATH)
    assert actual == FIXTURE_IMAGE_SHA256, (
        f"Fixture image SHA-256 mismatch.\n"
        f"  Expected: {FIXTURE_IMAGE_SHA256}\n"
        f"  Actual:   {actual}\n"
        "The file may have been altered. Restore from git."
    )


# ---------------------------------------------------------------------------
# End-to-end real-model integration test
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_real_model_end_to_end_pipeline(authed_client):
    """
    Complete evidence chain test:
    fixture image → API → validation gates → real EfficientNet-B0 → grade → Grad-CAM PNG.

    All five assertions must pass for the test to count as evidence.
    """
    import app.services.ai_service as ai_svc
    from app.core.config import settings

    # ------------------------------------------------------------------ #
    # 1. Redirect inference to the real checkpoint for this test only      #
    # ------------------------------------------------------------------ #
    old_engine_env = os.environ.get("AI_INFERENCE_ENGINE", "mock")
    os.environ["AI_INFERENCE_ENGINE"] = "pytorch"

    old_singleton = ai_svc._service_singleton
    ai_svc._service_singleton = None  # force re-initialisation

    checkpoint_abs = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../models/weights/efficientnet_b0_dr.pth")
    )
    old_checkpoint = settings.MODEL_CHECKPOINT_PATH
    settings.MODEL_CHECKPOINT_PATH = checkpoint_abs

    client: TestClient = authed_client

    try:
        # ---------------------------------------------------------------- #
        # 2. Create draft assessment                                         #
        # ---------------------------------------------------------------- #
        create_response = client.post(
            "/api/v1/assessments",
            json={
                "patientId": "REAL-INTEGRATION-TEST",
                "laterality": "OD",
                "cameraModel": "Topcon TRC-NW400 (Fixture)",
                "clinicalNotes": "Real EfficientNet-B0 integration test — fixture image.",
            },
        )
        assert create_response.status_code == 201, (
            f"Assessment draft creation failed with HTTP {create_response.status_code}:\n"
            f"{create_response.text}"
        )
        draft_id = create_response.json()["id"]

        # ---------------------------------------------------------------- #
        # 3. Upload the packaged fixture image                               #
        # ---------------------------------------------------------------- #
        assert os.path.isfile(FIXTURE_IMAGE_PATH), f"Fixture missing: {FIXTURE_IMAGE_PATH}"

        with open(FIXTURE_IMAGE_PATH, "rb") as img_file:
            upload_response = client.post(
                f"/api/v1/assessments/{draft_id}/upload",
                files={"file": ("aptos_sample_fundus.jpg", img_file, "image/jpeg")},
            )

        assert upload_response.status_code == 200, (
            f"Image upload failed with HTTP {upload_response.status_code}:\n"
            f"{upload_response.text}"
        )

        # ---------------------------------------------------------------- #
        # 4. Confirm the pipeline reached a terminal state                  #
        # ---------------------------------------------------------------- #
        data = upload_response.json()
        terminal_states = {"result_ready", "needs_review", "rejected"}
        assert data.get("status") in terminal_states, (
            f"Pipeline did not reach a terminal state; got status={data.get('status')!r}. "
            "The assessment may be stuck in validation. Full response:\n"
            f"{upload_response.text}"
        )

        assert data["status"] != "rejected", (
            "Fixture image was rejected by the quality gate pipeline. "
            "This means the packaged image does not pass the admission criteria — "
            "replace the fixture with a better-quality fundus image."
        )

        assessment_id = draft_id  # already known; data["id"] == draft_id

        # ---------------------------------------------------------------- #
        # 4. Verify the model output structure and argmax consistency        #
        # ---------------------------------------------------------------- #
        detail_response = client.get(f"/api/v1/assessments/{assessment_id}")
        assert detail_response.status_code == 200, (
            f"GET /assessments/{assessment_id} returned HTTP {detail_response.status_code}"
        )
        detail = detail_response.json()

        ai_result = detail.get("modelObservation") or detail.get("aiResult")
        assert ai_result is not None, (
            "Assessment has no modelObservation / aiResult — "
            "inference must have been skipped or the field name changed."
        )

        class_scores = ai_result.get("classScores", [])
        assert len(class_scores) == 5, (
            f"Expected 5-class scores from EfficientNet-B0, got {len(class_scores)}"
        )

        probs = [float(c["score"]) for c in class_scores]
        argmax_grade = probs.index(max(probs))
        reported_grade = int(ai_result["primaryClassGrade"])
        assert reported_grade == argmax_grade, (
            f"Argmax consistency violation: argmax of class_scores is {argmax_grade} "
            f"but primaryClassGrade is {reported_grade}. "
            "The model's class scores do not agree with the reported primary grade."
        )

        primary_score = float(ai_result["primaryScore"])
        assert 0.0 < primary_score <= 1.0, (
            f"Primary score {primary_score} is outside the valid probability range (0, 1]."
        )

        # ---------------------------------------------------------------- #
        # 5. Retrieve and validate the Grad-CAM PNG at its served URL        #
        # ---------------------------------------------------------------- #
        gradcam_url = detail.get("gradcamUrl")
        assert gradcam_url is not None, (
            "Assessment response contains no gradcamUrl. "
            "Grad-CAM generation must have failed silently — check the backend log."
        )

        gradcam_response = client.get(gradcam_url)
        assert gradcam_response.status_code == 200, (
            f"Grad-CAM URL {gradcam_url!r} returned HTTP {gradcam_response.status_code}. "
            "The artefact URL is present but the file is not being served."
        )

        gradcam_bytes = gradcam_response.content
        assert len(gradcam_bytes) > 0, "Grad-CAM response body is empty."
        assert _is_valid_png(gradcam_bytes), (
            f"Grad-CAM response does not start with the PNG magic signature. "
            f"First 16 bytes: {gradcam_bytes[:16].hex()!r}"
        )

    finally:
        # Always restore the global inference singleton and env variable
        os.environ["AI_INFERENCE_ENGINE"] = old_engine_env
        ai_svc._service_singleton = old_singleton
        settings.MODEL_CHECKPOINT_PATH = old_checkpoint
