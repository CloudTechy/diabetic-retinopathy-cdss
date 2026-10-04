import io
import hashlib
import os
import uuid
from typing import Tuple, Optional
from PIL import Image

from app.core.config import settings


class Gate1Result:
    def __init__(
        self,
        passed: bool,
        sha256_hash: str = "",
        width: int = 0,
        height: int = 0,
        file_size_bytes: int = 0,
        mime_type: str = "",
        stored_filename: str = "",
        metric: str = "",
        details: str = "",
        error_code: Optional[str] = None,
        rejection_reason: Optional[str] = None,
        clinical_action: Optional[str] = None,
    ):
        self.passed = passed
        self.sha256_hash = sha256_hash
        self.width = width
        self.height = height
        self.file_size_bytes = file_size_bytes
        self.mime_type = mime_type
        self.stored_filename = stored_filename
        self.metric = metric
        self.details = details
        self.error_code = error_code
        self.rejection_reason = rejection_reason
        self.clinical_action = clinical_action

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "sha256_hash": self.sha256_hash,
            "width": self.width,
            "height": self.height,
            "file_size_bytes": self.file_size_bytes,
            "mime_type": self.mime_type,
            "stored_filename": self.stored_filename,
            "metric": self.metric,
            "details": self.details,
            "error_code": self.error_code,
            "rejection_reason": self.rejection_reason,
            "clinical_action": self.clinical_action,
        }


def evaluate_gate1(image_bytes: bytes, original_filename: str = "upload.jpg") -> Tuple[Gate1Result, Optional[Image.Image]]:
    """
    Gate 1: File Integrity & Safe Decode.
    Validates:
      1. Non-empty payload.
      2. File size <= 15MB limit.
      3. Magic binary signatures for JPEG (\xFF\xD8\xFF) or PNG (\x89PNG\r\n\x1a\n).
      4. Valid decodable image structure via PIL.
      5. Minimum resolution: both sides >= MIN_IMAGE_DIMENSION.
      6. Computes the SHA-256 of the bytes. (The proposed stored_filename below is
         not what the service stores; it names the file <record id>_<8 hex>.<ext>.)
    No metadata is stripped; the bytes are returned as received.
    """
    file_size = len(image_bytes)
    sha256_hash = hashlib.sha256(image_bytes).hexdigest()

    # 1. Non-empty check
    if file_size == 0:
        return (
            Gate1Result(
                passed=False,
                sha256_hash=sha256_hash,
                file_size_bytes=0,
                error_code="ERR_EMPTY_FILE",
                metric="0 bytes",
                rejection_reason="The uploaded file contains 0 bytes. Empty payload received.",
                clinical_action="Please re-select a valid retinal fundus image file and upload again.",
            ),
            None,
        )

    # 2. File size limit
    if file_size > settings.MAX_UPLOAD_SIZE_BYTES:
        size_mb = file_size / (1024 * 1024)
        return (
            Gate1Result(
                passed=False,
                sha256_hash=sha256_hash,
                file_size_bytes=file_size,
                error_code="ERR_FILE_SIZE_EXCEEDED",
                metric=f"{size_mb:.2f} MB (> {settings.MAX_UPLOAD_SIZE_MB} MB limit)",
                rejection_reason=f"File size exceeds the 15 MB technical constraint ({size_mb:.2f} MB received).",
                clinical_action="Please compress or export the retinal photograph within the 15 MB limit.",
            ),
            None,
        )

    # 3. Magic byte signature check
    is_jpeg = image_bytes.startswith(b"\xff\xd8\xff")
    is_png = image_bytes.startswith(b"\x89PNG\r\n\x1a\n")

    if not (is_jpeg or is_png):
        detected_sig = image_bytes[:8].hex()
        return (
            Gate1Result(
                passed=False,
                sha256_hash=sha256_hash,
                file_size_bytes=file_size,
                error_code="ERR_INVALID_FILE_SIGNATURE",
                metric=f"Unrecognized header (0x{detected_sig})",
                rejection_reason="Binary magic number validation failed. Only standard JPEG and PNG formats are permitted.",
                clinical_action="Please export your retinal photograph in uncorrupted standard JPEG or PNG format.",
            ),
            None,
        )

    mime_type = "image/jpeg" if is_jpeg else "image/png"
    extension = ".jpg" if is_jpeg else ".png"
    stored_filename = f"{uuid.uuid4().hex}{extension}"

    # 4. Decodable image check via PIL
    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
        pil_image.verify()  # Verifies file integrity
        # Reopen because verify() consumes stream
        pil_image = Image.open(io.BytesIO(image_bytes))
        width, height = pil_image.size
    except Exception as exc:
        return (
            Gate1Result(
                passed=False,
                sha256_hash=sha256_hash,
                file_size_bytes=file_size,
                mime_type=mime_type,
                error_code="ERR_CORRUPT_IMAGE_STREAM",
                metric="Image decode failed",
                rejection_reason=f"Image stream is corrupted or truncated: {str(exc)}",
                clinical_action="The file appears structurally damaged. Please obtain a fresh export from the fundus camera.",
            ),
            None,
        )

    # 5. Dimension sanity check
    if width < settings.MIN_IMAGE_DIMENSION or height < settings.MIN_IMAGE_DIMENSION:
        return (
            Gate1Result(
                passed=False,
                sha256_hash=sha256_hash,
                width=width,
                height=height,
                file_size_bytes=file_size,
                mime_type=mime_type,
                error_code="ERR_INSUFFICIENT_RESOLUTION",
                metric=f"{width}x{height} px (< {settings.MIN_IMAGE_DIMENSION}x{settings.MIN_IMAGE_DIMENSION} px threshold)",
                rejection_reason=f"Image dimensions ({width}x{height} px) are below the minimum required resolution for technical analysis.",
                clinical_action=f"Ensure fundus camera exports at minimum {settings.MIN_IMAGE_DIMENSION}x{settings.MIN_IMAGE_DIMENSION} pixels.",
            ),
            None,
        )

    size_mb = file_size / (1024 * 1024)
    # The digest is COMPUTED and recorded; there is no reference digest to
    # match an upload against. An earlier revision said "SHA-256 match".
    metric = f"MIME {mime_type}, SHA-256 computed, {size_mb:.2f} MB"
    details = f"Valid binary signature ({'0xFFD8FF' if is_jpeg else '0x89504E47'}), dimension {width}x{height} px."

    return (
        Gate1Result(
            passed=True,
            sha256_hash=sha256_hash,
            width=width,
            height=height,
            file_size_bytes=file_size,
            mime_type=mime_type,
            stored_filename=stored_filename,
            metric=metric,
            details=details,
        ),
        pil_image,
    )
