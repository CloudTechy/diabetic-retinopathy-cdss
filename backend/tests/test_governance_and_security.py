import base64
import pytest
from sqlalchemy import select
from app.models.models import Assessment, AIResult, ProfessionalReview, ModelExecution
from app.services.assessment_service import (
    AssessmentService,
    InvalidStateTransitionError,
    ReviewValidationError,
)
from app.schemas.assessment import AssessmentCreateRequest, ClinicianReviewSubmitRequest
from app.core.security import verify_password, get_password_hash, create_access_token, decode_access_token
from tests.conftest import create_synthetic_retinal_fundus, image_to_bytes


class TestClinicianInTheLoopGovernance:
    """Verifies clinician-in-the-loop governance invariants:
    - Review immutability
    - Separate storage of AI results vs clinician reviews
    - Friction-engineered override verification
    - Absence of online parameter updates (frozen weights)
    """

    @pytest.mark.asyncio
    async def test_separate_storage_of_ai_result_and_review(self, async_db):
        """Verifies AI results and clinician reviews are stored as distinct, independent records."""
        # 1. Create and process assessment through validation and AI inference
        create_req = AssessmentCreateRequest(
            patientId="PT-SEPARATE-STORAGE-01",
            laterality="OD",
        )
        assessment = await AssessmentService.create_assessment(async_db, create_req)
        fundus_img = create_synthetic_retinal_fundus(512, 512, is_retinal=True, blur=False)
        raw_bytes = image_to_bytes(fundus_img, "JPEG")

        validated = await AssessmentService.process_and_validate_image(
            db=async_db,
            assessment_id=assessment.id,
            image_bytes=raw_bytes,
            original_filename="retina_od.jpg",
            candidate_grade=2,
        )

        assert validated.ai_result is not None
        assert validated.professional_review is None
        ai_res_id = validated.ai_result.id

        # 2. Clinician records a review that disagrees with AI
        review_req = ClinicianReviewSubmitRequest(
            agreement="disagree",
            reviewerAssessedGrade=1,
            reviewerAssessedGradeLabel="Grade 1: Mild NPDR",
            justificationNotes="Isolated microaneurysms detected; lacks venous beading or IRMA required for Grade 2.",
        )

        reviewed = await AssessmentService.submit_professional_review(
            db=async_db,
            assessment_id=assessment.id,
            review_input=review_req,
            reviewer=None,
        )

        # 3. Invariant Verification: Both entities coexist independently in separate tables
        assert reviewed.ai_result is not None
        assert reviewed.professional_review is not None
        assert reviewed.ai_result.id == ai_res_id
        # AI Result remains unchanged (Grade 2)
        assert reviewed.ai_result.primary_class_grade == 2
        # Clinician Review holds certified grade (Grade 1)
        assert reviewed.professional_review.reviewer_assessed_grade == 1
        assert reviewed.professional_review.agreement == "disagree"
        assert reviewed.professional_review.is_immutable is True

        # Check DB directly with SQL selects
        ai_query = await async_db.execute(select(AIResult).where(AIResult.assessment_id == assessment.id))
        stored_ai = ai_query.scalar_one_or_none()
        assert stored_ai is not None
        assert stored_ai.primary_class_grade == 2

        review_query = await async_db.execute(select(ProfessionalReview).where(ProfessionalReview.assessment_id == assessment.id))
        stored_review = review_query.scalar_one_or_none()
        assert stored_review is not None
        assert stored_review.reviewer_assessed_grade == 1

    @pytest.mark.asyncio
    async def test_completed_review_immutability(self, async_db):
        """Verifies that once a review is completed, subsequent modification attempts fail."""
        create_req = AssessmentCreateRequest(patientId="PT-IMMUTABLE-01", laterality="OS")
        assessment = await AssessmentService.create_assessment(async_db, create_req)
        fundus_img = create_synthetic_retinal_fundus(512, 512, is_retinal=True, blur=False)
        raw_bytes = image_to_bytes(fundus_img, "JPEG")

        await AssessmentService.process_and_validate_image(
            db=async_db,
            assessment_id=assessment.id,
            image_bytes=raw_bytes,
            original_filename="retina_os.jpg",
            candidate_grade=0,
        )

        review_req = ClinicianReviewSubmitRequest(
            agreement="agree",
            reviewerAssessedGrade=0,
            reviewerAssessedGradeLabel="Grade 0: No Apparent DR",
            justificationNotes="Normal fundus appearance with sharp optic margins.",
        )

        reviewed = await AssessmentService.submit_professional_review(
            db=async_db,
            assessment_id=assessment.id,
            review_input=review_req,
            reviewer=None,
        )
        assert reviewed.status == "completed"

        # Attempt to submit a second review or alter the completed review
        second_review_req = ClinicianReviewSubmitRequest(
            agreement="disagree",
            reviewerAssessedGrade=3,
            reviewerAssessedGradeLabel="Grade 3: Severe NPDR",
            justificationNotes="Attempting post-completion modification.",
        )

        with pytest.raises((ReviewValidationError, InvalidStateTransitionError)):
            await AssessmentService.submit_professional_review(
                db=async_db,
                assessment_id=assessment.id,
                review_input=second_review_req,
                reviewer=None,
            )

    @pytest.mark.asyncio
    async def test_model_execution_mode_is_strictly_evaluation(self, async_db):
        """Verifies no online learning or training mode is enabled during clinical inference."""
        create_req = AssessmentCreateRequest(patientId="PT-FROZEN-01", laterality="OD")
        assessment = await AssessmentService.create_assessment(async_db, create_req)
        fundus_img = create_synthetic_retinal_fundus(512, 512, is_retinal=True, blur=False)
        raw_bytes = image_to_bytes(fundus_img, "JPEG")

        processed = await AssessmentService.process_and_validate_image(
            db=async_db,
            assessment_id=assessment.id,
            image_bytes=raw_bytes,
            original_filename="retina_frozen.jpg",
            candidate_grade=0,
        )

        assert processed.model_execution is not None
        assert processed.model_execution.execution_mode == "evaluation"
        assert "frozen" in processed.model_execution.model_version.lower()


class TestClinicalTerminologyCompliance:
    """Verifies non-diagnostic microcopy compliance:
    - Forbids misleading pseudo-certainty terms ('confidence', 'certainty', 'diagnostic accuracy')
    - Strictly mandates 'model-generated class score'
    """

    FORBIDDEN_TERMS = ["confidence score", "diagnostic certainty", "disease certainty", "ai diagnosis"]

    def test_api_result_terminology_compliance(self, test_client):
        # Create and process assessment
        fundus_img = create_synthetic_retinal_fundus(512, 512, is_retinal=True, blur=False)
        raw_bytes = image_to_bytes(fundus_img, "JPEG")
        data_url = f"data:image/jpeg;base64,{base64.b64encode(raw_bytes).decode('utf-8')}"

        res = test_client.post(
            "/api/v1/assessments",
            json={
                "patientId": "PT-TERMINOLOGY-01",
                "laterality": "OD",
                "imageDataUrl": data_url,
                "candidateGrade": 2,
            },
        )
        assert res.status_code == 201
        data = res.json()
        assessment_id = data["id"]

        result_res = test_client.get(f"/api/v1/assessments/{assessment_id}/result")
        assert result_res.status_code == 200
        obs_text = result_res.text.lower()

        # Verify no forbidden pseudo-certainty terms are returned in API payload
        for term in self.FORBIDDEN_TERMS:
            assert term not in obs_text, f"Forbidden clinical term '{term}' found in result payload"

        obs_json = result_res.json()
        assert "primaryScore" in obs_json
        assert "primaryClassGrade" in obs_json
        assert "classScores" in obs_json
        assert obs_json["modelVersion"] is not None

    @pytest.mark.asyncio
    async def test_ai_result_model_disclaimer_invariant(self, async_db):
        """Verifies the mandatory boundary disclaimer persisted on the AIResult database model."""
        create_req = AssessmentCreateRequest(patientId="PT-DISCLAIMER-01", laterality="OD")
        assessment = await AssessmentService.create_assessment(async_db, create_req)
        fundus_img = create_synthetic_retinal_fundus(512, 512, is_retinal=True, blur=False)
        raw_bytes = image_to_bytes(fundus_img, "JPEG")

        processed = await AssessmentService.process_and_validate_image(
            db=async_db,
            assessment_id=assessment.id,
            image_bytes=raw_bytes,
            original_filename="retina_disclaimer.jpg",
            candidate_grade=1,
        )

        assert processed.ai_result is not None
        assert processed.ai_result.disclaimer is not None
        disclaimer_upper = processed.ai_result.disclaimer.upper()
        assert "CLINICAL DECISION SUPPORT ONLY" in disclaimer_upper
        assert "NOT FOR INDEPENDENT DIAGNOSIS" in disclaimer_upper


class TestSecurityAndAccessControl:
    """Verifies security controls, token authentication, and rejection protections."""

    def test_invalid_login_credentials_rejected(self, test_client):
        res = test_client.post(
            "/api/v1/auth/login",
            json={"username": "unauthorized_user", "password": "wrong_password_xyz"},
        )
        assert res.status_code == 401
        assert "Invalid clinical credentials" in res.json()["detail"]

    def test_jwt_token_generation_and_tamper_rejection(self):
        token = create_access_token(subject="USR-TEST-123", extra_claims={"role": "clinician"})
        payload = decode_access_token(token)
        assert payload is not None
        assert payload["sub"] == "USR-TEST-123"
        assert payload["role"] == "clinician"

        # Tampered token fails validation
        tampered_token = token[:-5] + "XXXXX"
        tampered_payload = decode_access_token(tampered_token)
        assert tampered_payload is None

    def test_password_hashing_and_verification(self):
        pwd = "clinician_super_secret_2026"
        hashed = get_password_hash(pwd)
        assert hashed != pwd
        assert verify_password(pwd, hashed) is True
        assert verify_password("wrong_password", hashed) is False

    def test_nonexistent_assessment_returns_404(self, test_client):
        res = test_client.get("/api/v1/assessments/REC-NONEXISTENT-9999/status")
        assert res.status_code == 404
        assert "not found" in res.json()["detail"].lower()
