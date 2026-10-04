import pytest
from PIL import Image

from app.services.validation.gate1_integrity import evaluate_gate1
from app.services.validation.gate2_relevance import evaluate_gate2
from app.services.validation.gate3_quality import evaluate_gate3
from app.services.validation.pipeline import ValidationPipeline
from tests.conftest import create_synthetic_retinal_fundus, image_to_bytes


class TestGate1FileIntegrity:
    """Verifies Stage 1 File Integrity checks."""

    def test_valid_jpeg_passes_gate1(self):
        img = create_synthetic_retinal_fundus(512, 512)
        raw_bytes = image_to_bytes(img, format="JPEG")

        res, pil_img = evaluate_gate1(raw_bytes, "retina.jpg")
        assert res.passed is True
        assert res.mime_type == "image/jpeg"
        assert res.width == 512
        assert res.height == 512
        assert len(res.sha256_hash) == 64
        assert pil_img is not None

    def test_valid_png_passes_gate1(self):
        img = create_synthetic_retinal_fundus(512, 512)
        raw_bytes = image_to_bytes(img, format="PNG")

        res, pil_img = evaluate_gate1(raw_bytes, "retina.png")
        assert res.passed is True
        assert res.mime_type == "image/png"
        assert pil_img is not None

    def test_empty_payload_fails_gate1(self):
        res, pil_img = evaluate_gate1(b"", "empty.jpg")
        assert res.passed is False
        assert res.error_code == "ERR_EMPTY_FILE"
        assert pil_img is None

    def test_corrupted_signature_fails_gate1(self):
        corrupt_bytes = b"NOT_A_REAL_IMAGE_HEADER_12345678"
        res, pil_img = evaluate_gate1(corrupt_bytes, "corrupt.jpg")
        assert res.passed is False
        assert res.error_code == "ERR_INVALID_FILE_SIGNATURE"
        assert pil_img is None

    def test_insufficient_resolution_fails_gate1(self):
        small_img = create_synthetic_retinal_fundus(256, 256)
        raw_bytes = image_to_bytes(small_img, format="JPEG")

        res, pil_img = evaluate_gate1(raw_bytes, "small.jpg")
        assert res.passed is False
        assert res.error_code == "ERR_INSUFFICIENT_RESOLUTION"
        assert pil_img is None

    def test_oversized_file_fails_gate1(self):
        # 16 MB mock payload
        oversized_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * (16 * 1024 * 1024)
        res, pil_img = evaluate_gate1(oversized_bytes, "huge.jpg")
        assert res.passed is False
        assert res.error_code == "ERR_FILE_SIZE_EXCEEDED"


class TestGate2RetinalRelevance:
    """Verifies Stage 2 Retinal Relevance & Geometry checks."""

    def test_authentic_fundus_passes_gate2(self):
        fundus_img = create_synthetic_retinal_fundus(512, 512, is_retinal=True)
        res = evaluate_gate2(fundus_img)
        assert res.passed is True
        assert res.mask_coverage >= 0.50
        assert res.red_to_blue_ratio >= 1.15

    def test_non_retinal_image_fails_gate2(self):
        non_retinal_img = create_synthetic_retinal_fundus(512, 512, is_retinal=False)
        res = evaluate_gate2(non_retinal_img)
        assert res.passed is False
        assert res.error_code == "ERR_NON_RETINAL_SPECTRAL_PROFILE"
        assert "colour-profile" in res.rejection_reason.lower() and res.error_code == "ERR_NON_RETINAL_SPECTRAL_PROFILE"

    def test_blank_dark_image_fails_gate2(self):
        blank_img = Image.new("RGB", (512, 512), (0, 0, 0))
        res = evaluate_gate2(blank_img)
        assert res.passed is False
        assert res.error_code == "ERR_RETINAL_MASK_ABSENT"

    def test_extreme_aspect_ratio_fails_gate2(self):
        narrow_img = Image.new("RGB", (1000, 400), (180, 70, 30))
        res = evaluate_gate2(narrow_img)
        assert res.passed is False
        assert res.error_code == "ERR_INVALID_ASPECT_RATIO"


class TestGate3TechnicalQuality:
    """Verifies Stage 3 Technical Quality & Sharpness checks."""

    def test_sharp_fundus_passes_gate3(self):
        sharp_fundus = create_synthetic_retinal_fundus(512, 512, blur=False)
        res = evaluate_gate3(sharp_fundus)
        assert res.passed is True
        assert res.laplacian_variance >= 60.0
        assert res.contrast_dynamic_range >= 18.0
        assert res.illumination_index >= 0.65

    def test_blurred_fundus_fails_gate3(self):
        blurred_fundus = create_synthetic_retinal_fundus(512, 512, blur=True)
        res = evaluate_gate3(blurred_fundus)
        assert res.passed is False
        assert res.error_code == "ERR_MOTION_OR_DEFOCUS_BLUR"
        assert res.laplacian_variance < 60.0
        assert "Laplacian" in res.metric


class TestSequentialValidationPipelineFailClosed:
    """Verifies Fail-Closed Sequential Execution Invariant."""

    def test_corrupt_file_halts_at_gate1(self):
        corrupt_bytes = b"INVALID_GARBAGE_PAYLOAD"
        result = ValidationPipeline.execute(corrupt_bytes, "corrupt.jpg")
        assert result.overall_status == "rejected"
        assert result.failed_gate == 1
        assert result.gate1_result.passed is False
        assert result.gate2_result is None
        assert result.gate3_result is None

        # Check gate records formatting for UI
        records = result.to_gate_records()
        assert records[0]["status"] == "failed"
        assert records[1]["status"] == "pending"
        assert records[2]["status"] == "pending"

    def test_non_retinal_halts_at_gate2(self):
        non_retinal = create_synthetic_retinal_fundus(512, 512, is_retinal=False)
        raw_bytes = image_to_bytes(non_retinal, "JPEG")

        result = ValidationPipeline.execute(raw_bytes, "non_retinal.jpg")
        assert result.overall_status == "rejected"
        assert result.failed_gate == 2
        assert result.gate1_result.passed is True
        assert result.gate2_result.passed is False
        assert result.gate3_result is None

        records = result.to_gate_records()
        assert records[0]["status"] == "passed"
        assert records[1]["status"] == "failed"
        assert records[2]["status"] == "pending"

    def test_blurry_fundus_halts_at_gate3(self):
        blurry = create_synthetic_retinal_fundus(512, 512, blur=True, is_retinal=True)
        raw_bytes = image_to_bytes(blurry, "JPEG")

        result = ValidationPipeline.execute(raw_bytes, "blurry.jpg")
        assert result.overall_status == "rejected"
        assert result.failed_gate == 3
        assert result.gate1_result.passed is True
        assert result.gate2_result.passed is True
        assert result.gate3_result.passed is False

        records = result.to_gate_records()
        assert records[0]["status"] == "passed"
        assert records[1]["status"] == "passed"
        assert records[2]["status"] == "failed"

    def test_perfect_retinal_fundus_clears_all_gates(self):
        sharp = create_synthetic_retinal_fundus(512, 512, blur=False, is_retinal=True)
        raw_bytes = image_to_bytes(sharp, "JPEG")

        result = ValidationPipeline.execute(raw_bytes, "perfect.jpg")
        assert result.overall_status == "passed"
        assert result.failed_gate is None
        assert result.gate1_result.passed is True
        assert result.gate2_result.passed is True
        assert result.gate3_result.passed is True

        records = result.to_gate_records()
        assert records[0]["status"] == "passed"
        assert records[1]["status"] == "passed"
        assert records[2]["status"] == "passed"
