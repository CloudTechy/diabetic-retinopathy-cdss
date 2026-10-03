"""
End-to-end integration test against the genuine EfficientNet-B0 checkpoint,
using a GENUINE HELD-OUT APTOS IMAGE.

THE FIXTURE

    image_id   d1f1ea894da1
    split      test  (held out; the model never saw it in training)
    grade      2  (Moderate NPDR)
    sha256     d6eb606b07cdcd6045f2477f9fa3899df7f112f862897286a21f0eece06f2acf

All three are asserted below against
`docs/chapter4/dataset_split_manifest.csv`, so the fixture cannot be swapped
without the test saying so, and a reviewer can verify its provenance in one
command.

This replaces a synthetic 800x800 image that had been named
`aptos_sample_fundus.jpg` - a filename asserting a provenance it did not have,
and the same defect as the placeholder files once found in storage/datasets/.

WHAT THIS ESTABLISHES

A real held-out photograph travels the whole production path with no mocked
intermediate:

1. the three admission gates evaluate it at their calibrated thresholds;
2. the real checkpoint - digest-verified against MODEL_CHECKPOINT_SHA256 -
   produces a grade and five class probabilities;
3. the argmax of those probabilities equals the reported primary grade, so the
   displayed grade is the model's own and not a substituted one;
4. a Grad-CAM URL is returned, and fetching it yields non-empty bytes carrying
   the PNG signature.

WHAT IT STILL DOES NOT ESTABLISH

Accuracy. This is ONE image. The test does not assert which grade comes back,
because a single case proves nothing about performance and pinning it would
turn a wiring test into a brittle claim. The accuracy evidence is the held-out
cohort in `model_evaluation_report.md` (N = 525), of which this image is one
member.
"""
import csv
import hashlib
import os
import struct

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixture image metadata
# ---------------------------------------------------------------------------
FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
FIXTURE_IMAGE_PATH = os.path.join(FIXTURE_DIR, "aptos_heldout_d1f1ea894da1.png")
FIXTURE_IMAGE_ID = "d1f1ea894da1"
FIXTURE_SPLIT = "test"
# SHA-256 of the genuine held-out APTOS image, as recorded in
# docs/chapter4/dataset_split_manifest.csv. Pinned so the fixture cannot be
# swapped silently.
FIXTURE_IMAGE_SHA256 = "d6eb606b07cdcd6045f2477f9fa3899df7f112f862897286a21f0eece06f2acf"


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

def test_fixture_is_the_held_out_aptos_image_it_claims_to_be():
    """
    The fixture must BE the manifest row it names: same id, held-out split, and
    the SHA-256 recorded for that row.

    A pinned hash alone only proves the file has not changed since someone
    pinned it. It says nothing about whether the file is an APTOS image at all -
    which is exactly how a synthetic 800x800 picture came to be committed under
    the name `aptos_sample_fundus.jpg`, and how fifteen placeholders came to sit
    in storage/datasets/ wearing real held-out image ids.
    """
    # The archive is repository-relative, so one path serves both the
    # repository and an extracted archive. An earlier layout re-homed the
    # manifest under dataset_sample_and_manifest/ and this test SKIPPED when
    # run from the zip - and a skipped provenance check is indistinguishable
    # from an absent one. That layout is retired.
    here = os.path.dirname(os.path.abspath(__file__))
    # here is already .../backend/tests, so two levels reach the root.
    repo_root = os.path.dirname(os.path.dirname(here))
    manifest_path = os.path.abspath(
        os.path.join(repo_root, "docs", "chapter4", "dataset_split_manifest.csv"))

    assert os.path.exists(manifest_path), (
        "docs/chapter4/dataset_split_manifest.csv not found. It ships at that "
        "path in the archive; this test must not skip without it.")

    with open(manifest_path, newline="", encoding="utf-8") as fh:
        rows = {r["image_id"]: r for r in csv.DictReader(fh)}

    record = rows.get(FIXTURE_IMAGE_ID)
    assert record is not None, (
        f"{FIXTURE_IMAGE_ID} does not appear in dataset_split_manifest.csv. The "
        "integration fixture must be an image from the evaluated corpus.")

    assert record["split"] == FIXTURE_SPLIT, (
        f"{FIXTURE_IMAGE_ID} is split={record['split']!r}, not {FIXTURE_SPLIT!r}. "
        "Grading an image the model trained on would be meaningless as evidence.")

    assert record["sha256_hash"] == FIXTURE_IMAGE_SHA256, (
        "The pinned SHA-256 does not match the manifest row for "
        f"{FIXTURE_IMAGE_ID}.")

    assert _sha256_file(FIXTURE_IMAGE_PATH) == FIXTURE_IMAGE_SHA256, (
        f"The file at {FIXTURE_IMAGE_PATH} is not the image "
        f"{FIXTURE_IMAGE_ID} names. It has been altered or replaced.")


def test_fixture_image_integrity():
    """Verify the packaged fixture is present and unmodified."""
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

    # The checkpoint ships at its repository path, backend/models/weights/,
    # in the archive as well. A reviewer running this from the zip gets the
    # real checkpoint or a clear refusal, never a silent skip.
    here = os.path.dirname(os.path.abspath(__file__))
    checkpoint_abs = os.path.abspath(
        os.path.join(here, "..", "models", "weights", "efficientnet_b0_dr.pth"))

    assert os.path.exists(checkpoint_abs), (
        "backend/models/weights/efficientnet_b0_dr.pth not found. This test "
        "grades a real image with the real checkpoint and must not skip without it.")
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
                "clinicalNotes": ("Integration test against the real checkpoint using a "
                                  "genuine held-out APTOS image (d1f1ea894da1). "
                                  "Proves the path end to end; one image is "
                                  "not an accuracy claim."),
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
                files={"file": ("aptos_heldout_d1f1ea894da1.png", img_file,
                                "image/png")},
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
