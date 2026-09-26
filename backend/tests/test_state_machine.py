import pytest
from app.services.assessment_service import (
    AssessmentService,
    InvalidStateTransitionError,
    VALID_TRANSITIONS,
)
from app.schemas.assessment import AssessmentCreateRequest
from tests.conftest import create_synthetic_retinal_fundus, image_to_bytes


class TestStateMachineTransitions:
    """Verifies Assessment lifecycle state machine integrity."""

    def test_valid_forward_transitions(self):
        """Verify each legal step in the clinical lifecycle."""
        # draft -> uploaded
        AssessmentService.validate_transition("draft", "uploaded")
        # uploaded -> validating
        AssessmentService.validate_transition("uploaded", "validating")
        # validating -> accepted
        AssessmentService.validate_transition("validating", "accepted")
        # validating -> rejected
        AssessmentService.validate_transition("validating", "rejected")
        # accepted -> preprocessing
        AssessmentService.validate_transition("accepted", "preprocessing")
        # preprocessing -> inference
        AssessmentService.validate_transition("preprocessing", "inference")
        # inference -> result_ready
        AssessmentService.validate_transition("inference", "result_ready")
        # result_ready -> under_review
        AssessmentService.validate_transition("result_ready", "under_review")
        # under_review -> completed
        AssessmentService.validate_transition("under_review", "completed")
        # result_ready -> completed (direct sign-off)
        AssessmentService.validate_transition("result_ready", "completed")

    def test_rejected_state_is_terminal_for_inference(self):
        """FAIL-CLOSED INVARIANT: A rejected assessment CANNOT proceed to inference."""
        assert "inference" not in VALID_TRANSITIONS["rejected"]
        assert "preprocessing" not in VALID_TRANSITIONS["rejected"]
        assert "result_ready" not in VALID_TRANSITIONS["rejected"]
        assert "completed" not in VALID_TRANSITIONS["rejected"]

        with pytest.raises(InvalidStateTransitionError):
            AssessmentService.validate_transition("rejected", "inference")

        with pytest.raises(InvalidStateTransitionError):
            AssessmentService.validate_transition("rejected", "preprocessing")

    def test_completed_state_is_immutable(self):
        """CLINICIAN AUTHORITY INVARIANT: A completed review record is immutable."""
        assert len(VALID_TRANSITIONS["completed"]) == 0

        with pytest.raises(InvalidStateTransitionError):
            AssessmentService.validate_transition("completed", "validating")

        with pytest.raises(InvalidStateTransitionError):
            AssessmentService.validate_transition("completed", "draft")

    def test_illegal_jump_transitions(self):
        """Ensure states cannot jump over mandatory validation gates."""
        with pytest.raises(InvalidStateTransitionError):
            AssessmentService.validate_transition("draft", "inference")

        with pytest.raises(InvalidStateTransitionError):
            AssessmentService.validate_transition("uploaded", "result_ready")

        with pytest.raises(InvalidStateTransitionError):
            AssessmentService.validate_transition("validating", "inference")


@pytest.mark.asyncio
class TestStateMachineServiceExecution:
    """Verifies state machine integration with database models and services."""

    async def test_rejection_strictly_blocks_model_execution(self, async_db):
        """Ensure failed validation sets 'rejected' and executes NO model inference."""
        create_req = AssessmentCreateRequest(
            patientId="PAT-TEST-REJECT",
            laterality="OD",
        )
        assessment = await AssessmentService.create_assessment(async_db, create_req)
        assert assessment.status == "draft"

        # Provide corrupted image bytes (fails Gate 1)
        corrupt_bytes = b"CORRUPTED_FILE_DATA_NOT_JPEG"
        result_assessment = await AssessmentService.process_and_validate_image(
            db=async_db,
            assessment_id=assessment.id,
            image_bytes=corrupt_bytes,
            original_filename="corrupt.jpg",
        )

        assert result_assessment.status == "rejected"
        assert result_assessment.validation_result is not None
        assert result_assessment.validation_result.status == "rejected"
        assert result_assessment.validation_result.failed_gate == 1
        # CRITICAL INVARIANT: Model execution and AI result are strictly None
        assert result_assessment.model_execution is None
        assert result_assessment.ai_result is None

    async def test_valid_image_transitions_to_result_ready(self, async_db):
        """Ensure valid image traverses all gates and reaches 'result_ready'."""
        create_req = AssessmentCreateRequest(
            patientId="PAT-TEST-PASS",
            laterality="OS",
            candidateGrade=2,
        )
        assessment = await AssessmentService.create_assessment(async_db, create_req)

        sharp_fundus = create_synthetic_retinal_fundus(512, 512, is_retinal=True, blur=False)
        raw_bytes = image_to_bytes(sharp_fundus, "JPEG")

        result_assessment = await AssessmentService.process_and_validate_image(
            db=async_db,
            assessment_id=assessment.id,
            image_bytes=raw_bytes,
            original_filename="sharp_retina.jpg",
            candidate_grade=2,
        )

        assert result_assessment.status == "result_ready"
        assert result_assessment.validation_result.status == "passed"
        assert result_assessment.model_execution is not None
        assert result_assessment.model_execution.execution_mode == "evaluation"
        assert result_assessment.ai_result is not None
        assert result_assessment.ai_result.primary_class_grade == 2
        assert len(result_assessment.ai_result.class_scores) == 5
