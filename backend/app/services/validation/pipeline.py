from typing import Optional, List, Dict, Any, Tuple
from PIL import Image

from app.services.validation.gate1_integrity import evaluate_gate1, Gate1Result
from app.services.validation.gate2_relevance import evaluate_gate2, Gate2Result
from app.services.validation.gate3_quality import evaluate_gate3, Gate3Result


class ValidationPipelineResult:
    """Unified validation pipeline outcome representing the 3 sequential gates."""

    def __init__(
        self,
        overall_status: str,  # 'passed' or 'rejected'
        failed_gate: Optional[int] = None,
        failure_code: Optional[str] = None,
        failure_reason: Optional[str] = None,
        actionable_guidance: Optional[str] = None,
        gate1_result: Optional[Gate1Result] = None,
        gate2_result: Optional[Gate2Result] = None,
        gate3_result: Optional[Gate3Result] = None,
        pil_image: Optional[Image.Image] = None,
    ):
        self.overall_status = overall_status
        self.failed_gate = failed_gate
        self.failure_code = failure_code
        self.failure_reason = failure_reason
        self.actionable_guidance = actionable_guidance
        self.gate1_result = gate1_result
        self.gate2_result = gate2_result
        self.gate3_result = gate3_result
        self.pil_image = pil_image

    @property
    def is_passed(self) -> bool:
        return self.overall_status == "passed"

    def to_gate_records(self) -> List[Dict[str, Any]]:
        """Format the 3 gates for frontend UI visualization."""
        # Gate 1
        g1_status = "passed" if (self.gate1_result and self.gate1_result.passed) else "failed"
        g1_dict = {
            "gateIndex": 1,
            "name": "Gate 1",
            "title": "File Integrity & Safe Decode",
            "status": g1_status,
            "metric": self.gate1_result.metric if self.gate1_result else "Pending",
            "details": self.gate1_result.details if (self.gate1_result and self.gate1_result.passed) else None,
            "rejectionReason": self.gate1_result.rejection_reason if (self.gate1_result and not self.gate1_result.passed) else None,
            "clinicalAction": self.gate1_result.clinical_action if (self.gate1_result and not self.gate1_result.passed) else None,
        }

        # Gate 2
        if not self.gate1_result or not self.gate1_result.passed:
            g2_status = "pending"
            g2_metric = "Awaiting Gate 1 validation"
            g2_details = None
            g2_rejection = None
            g2_action = None
        else:
            g2_status = "passed" if (self.gate2_result and self.gate2_result.passed) else "failed"
            g2_metric = self.gate2_result.metric if self.gate2_result else "Evaluating"
            g2_details = self.gate2_result.details if (self.gate2_result and self.gate2_result.passed) else None
            g2_rejection = self.gate2_result.rejection_reason if (self.gate2_result and not self.gate2_result.passed) else None
            g2_action = self.gate2_result.clinical_action if (self.gate2_result and not self.gate2_result.passed) else None

        g2_dict = {
            "gateIndex": 2,
            "name": "Gate 2",
            "title": "Technical retinal-image relevance",
            "status": g2_status,
            "metric": g2_metric,
            "details": g2_details,
            "rejectionReason": g2_rejection,
            "clinicalAction": g2_action,
        }

        # Gate 3
        if not self.gate2_result or not self.gate2_result.passed:
            g3_status = "pending"
            g3_metric = "Awaiting Gate 2 validation"
            g3_details = None
            g3_rejection = None
            g3_action = None
        else:
            g3_status = "passed" if (self.gate3_result and self.gate3_result.passed) else "failed"
            g3_metric = self.gate3_result.metric if self.gate3_result else "Evaluating"
            g3_details = self.gate3_result.details if (self.gate3_result and self.gate3_result.passed) else None
            g3_rejection = self.gate3_result.rejection_reason if (self.gate3_result and not self.gate3_result.passed) else None
            g3_action = self.gate3_result.clinical_action if (self.gate3_result and not self.gate3_result.passed) else None

        g3_dict = {
            "gateIndex": 3,
            "name": "Gate 3",
            "title": "Technical Quality & Sharpness",
            "status": g3_status,
            "metric": g3_metric,
            "details": g3_details,
            "rejectionReason": g3_rejection,
            "clinicalAction": g3_action,
        }

        return [g1_dict, g2_dict, g3_dict]


class ValidationPipeline:
    """Executes the ordered 3-stage validation pipeline with strict fail-closed enforcement."""

    @staticmethod
    def execute(image_bytes: bytes, original_filename: str = "upload.jpg") -> ValidationPipelineResult:
        """
        Executes Gate 1 -> Gate 2 -> Gate 3 sequentially.
        If ANY gate fails, the pipeline aborts immediately, marking subsequent gates pending.
        """
        # --- Stage 1: File Integrity ---
        gate1_res, pil_img = evaluate_gate1(image_bytes, original_filename)
        if not gate1_res.passed:
            return ValidationPipelineResult(
                overall_status="rejected",
                failed_gate=1,
                failure_code=gate1_res.error_code,
                failure_reason=gate1_res.rejection_reason,
                actionable_guidance=gate1_res.clinical_action,
                gate1_result=gate1_res,
                gate2_result=None,
                gate3_result=None,
                pil_image=None,
            )

        # --- Stage 2: technical retinal-image relevance (geometry + colour profile) ---
        assert pil_img is not None
        gate2_res = evaluate_gate2(pil_img)
        if not gate2_res.passed:
            return ValidationPipelineResult(
                overall_status="rejected",
                failed_gate=2,
                failure_code=gate2_res.error_code,
                failure_reason=gate2_res.rejection_reason,
                actionable_guidance=gate2_res.clinical_action,
                gate1_result=gate1_res,
                gate2_result=gate2_res,
                gate3_result=None,
                pil_image=pil_img,
            )

        # --- Stage 3: Technical Quality & Sharpness ---
        gate3_res = evaluate_gate3(pil_img)
        if not gate3_res.passed:
            return ValidationPipelineResult(
                overall_status="rejected",
                failed_gate=3,
                failure_code=gate3_res.error_code,
                failure_reason=gate3_res.rejection_reason,
                actionable_guidance=gate3_res.clinical_action,
                gate1_result=gate1_res,
                gate2_result=gate2_res,
                gate3_result=gate3_res,
                pil_image=pil_img,
            )

        # All 3 gates passed!
        return ValidationPipelineResult(
            overall_status="passed",
            failed_gate=None,
            failure_code=None,
            failure_reason=None,
            actionable_guidance=None,
            gate1_result=gate1_res,
            gate2_result=gate2_res,
            gate3_result=gate3_res,
            pil_image=pil_img,
        )
