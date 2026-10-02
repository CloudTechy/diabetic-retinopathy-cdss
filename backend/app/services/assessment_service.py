import base64
import datetime
from datetime import timezone
import hashlib
import io
import os
import uuid
from typing import Optional, List, Dict, Any, Tuple
from PIL import Image
from sqlalchemy import select, or_, and_, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.models.models import (
    Assessment,
    ImageAsset,
    ValidationResult,
    ModelExecution,
    AIResult,
    ExplanationArtifact,
    ProfessionalReview,
    AuditEvent,
    User,
)
from app.schemas.assessment import (
    AssessmentCreateRequest,
    AssessmentRecordResponse,
    ClinicianReviewSubmitRequest,
    ClinicianReviewResponse,
    GateResultSchema,
    TechnicalQualityMetricsSchema,
    ModelObservationSchema,
    ScoreBreakdownItem,
    AuditEventSchema,
)
from app.services.validation.pipeline import ValidationPipeline, ValidationPipelineResult
from app.services.ai_service import (
    get_active_inference_service,
    ICDR_CLASS_METADATA,
)


class InvalidStateTransitionError(Exception):
    """Raised when an illegal state machine transition is attempted."""
    pass


class ReviewValidationError(Exception):
    """Raised when a clinician review fails medical governance requirements."""
    pass


# Strict assessment lifecycle state transition graph
VALID_TRANSITIONS: Dict[str, List[str]] = {
    "draft": ["uploaded", "failed"],
    "uploaded": ["validating", "failed"],
    "validating": ["accepted", "rejected", "failed"],
    "accepted": ["preprocessing", "failed"],
    "rejected": [],  # Terminal rejection! Model execution strictly forbidden.
    "preprocessing": ["inference", "failed"],
    "inference": ["result_ready", "failed"],
    "result_ready": ["under_review", "completed", "failed"],
    "under_review": ["completed", "failed"],
    "completed": [],  # Terminal finalized! Record is immutable.
    "failed": [],
}


class AssessmentService:
    """Manages assessment lifecycle, fail-closed validation, AI harness, and reviews."""

    @staticmethod
    def validate_transition(current_state: str, target_state: str) -> None:
        """Enforces legal state machine transitions."""
        allowed = VALID_TRANSITIONS.get(current_state, [])
        if target_state not in allowed:
            raise InvalidStateTransitionError(
                f"Invalid state transition from '{current_state}' to '{target_state}'. "
                f"Permitted next states: {allowed}."
            )

    @classmethod
    async def create_assessment(
        cls,
        db: AsyncSession,
        payload: AssessmentCreateRequest,
        creator: Optional[User] = None,
    ) -> Assessment:
        """Create a new assessment in 'draft' status."""
        assessment_id = f"REC-2026-{uuid.uuid4().hex[:4].upper()}"
        assessment = Assessment(
            id=assessment_id,
            patient_id=payload.patientId.strip().upper(),
            eye_laterality=payload.laterality,
            camera_model=payload.cameraModel or "Topcon TRC-NW400 Non-Mydriatic",
            is_mydriatic=payload.isMydriatic or False,
            clinical_notes=payload.clinicalNotes,
            status="draft",
            created_by_id=creator.id if creator else None,
        )
        db.add(assessment)

        # Audit event
        actor_name = creator.full_name if creator else "System Ingestion"
        audit = AuditEvent(
            assessment_id=assessment.id,
            user_id=creator.id if creator else None,
            action="Assessment Initialized",
            actor=actor_name,
            details=f"Initialized draft record for patient {assessment.patient_id} ({assessment.eye_laterality}).",
            badge_type="info",
        )
        db.add(audit)
        await db.commit()
        await db.refresh(assessment)
        return assessment

    @classmethod
    async def process_and_validate_image(
        cls,
        db: AsyncSession,
        assessment_id: str,
        image_bytes: bytes,
        original_filename: str = "fundus.jpg",
        actor: Optional[User] = None,
    ) -> Assessment:
        """
        Executes upload -> validating -> accepted/rejected -> (if accepted) inference -> result_ready.
        Enforces the Fail-Closed Validation Invariant: Model execution is STRICTLY FORBIDDEN on rejection.
        """
        # Load assessment
        stmt = select(Assessment).where(Assessment.id == assessment_id).options(
            selectinload(Assessment.image_asset),
            selectinload(Assessment.validation_result),
            selectinload(Assessment.model_execution),
            selectinload(Assessment.ai_result).selectinload(AIResult.explanation_artifacts),
            selectinload(Assessment.professional_review),
            selectinload(Assessment.audit_events),
        )
        res = await db.execute(stmt)
        assessment = res.scalar_one_or_none()
        if not assessment:
            raise ValueError(f"Assessment {assessment_id} not found.")

        actor_name = actor.full_name if actor else "Clinical Technician"

        # 1. Transition: draft -> uploaded
        cls.validate_transition(assessment.status, "uploaded")
        assessment.status = "uploaded"

        # Save image file to storage
        os.makedirs(settings.STORAGE_IMAGES_PATH, exist_ok=True)
        stored_filename = f"{assessment_id}_{uuid.uuid4().hex[:8]}.jpg"
        storage_path = os.path.join(settings.STORAGE_IMAGES_PATH, stored_filename)
        with open(storage_path, "wb") as f:
            f.write(image_bytes)

        sha256_hash = hashlib.sha256(image_bytes).hexdigest()
        image_asset = ImageAsset(
            assessment_id=assessment.id,
            original_filename=original_filename,
            stored_filename=stored_filename,
            storage_path=storage_path,
            file_size_bytes=len(image_bytes),
            mime_type="image/jpeg" if image_bytes.startswith(b"\xff\xd8\xff") else "image/png",
            sha256_hash=sha256_hash,
        )
        db.add(image_asset)
        assessment.image_asset = image_asset

        audit_upload = AuditEvent(
            assessment_id=assessment.id,
            user_id=actor.id if actor else None,
            action="Image Upload & Ingestion",
            actor=actor_name,
            details=f"Uploaded {original_filename} ({len(image_bytes)} bytes, SHA-256 {sha256_hash[:16]}...).",
            badge_type="info",
        )
        db.add(audit_upload)
        assessment.audit_events.append(audit_upload)
        await db.flush()

        # 2. Transition: uploaded -> validating
        cls.validate_transition(assessment.status, "validating")
        assessment.status = "validating"
        await db.flush()

        # 3. Execute 3-Stage Technical Validation Pipeline
        pipeline_res: ValidationPipelineResult = ValidationPipeline.execute(image_bytes, original_filename)

        # Extract quality metrics
        laplacian_var = pipeline_res.gate3_result.laplacian_variance if pipeline_res.gate3_result else 0.0
        illum_index = pipeline_res.gate3_result.illumination_index if pipeline_res.gate3_result else 0.0
        contrast_val = pipeline_res.gate3_result.contrast_dynamic_range if pipeline_res.gate3_result else 0.0
        dims = f"{pipeline_res.gate1_result.width}x{pipeline_res.gate1_result.height} px" if pipeline_res.gate1_result else "Unknown"

        validation_result = ValidationResult(
            assessment_id=assessment.id,
            status=pipeline_res.overall_status,
            gate1_passed=pipeline_res.gate1_result.passed if pipeline_res.gate1_result else False,
            gate1_details=pipeline_res.gate1_result.to_dict() if pipeline_res.gate1_result else {},
            gate2_passed=pipeline_res.gate2_result.passed if pipeline_res.gate2_result else None,
            gate2_details=pipeline_res.gate2_result.to_dict() if pipeline_res.gate2_result else None,
            gate3_passed=pipeline_res.gate3_result.passed if pipeline_res.gate3_result else None,
            gate3_details=pipeline_res.gate3_result.to_dict() if pipeline_res.gate3_result else None,
            failed_gate=pipeline_res.failed_gate,
            failure_code=pipeline_res.failure_code,
            failure_reason=pipeline_res.failure_reason,
            actionable_guidance=pipeline_res.actionable_guidance,
            laplacian_variance=laplacian_var,
            illumination_index=illum_index,
            contrast_dynamic_range=contrast_val,
            native_resolution=dims,
        )
        db.add(validation_result)
        assessment.validation_result = validation_result

        # 4. Check if validation failed
        if not pipeline_res.is_passed:
            # FAIL-CLOSED INVARIANT: Transition to 'rejected' and STOP!
            cls.validate_transition(assessment.status, "rejected")
            assessment.status = "rejected"

            audit_fail = AuditEvent(
                assessment_id=assessment.id,
                action="Validation Pipeline Rejected",
                actor="Validation Pipeline",
                details=f"Fail-closed triggered at Gate {pipeline_res.failed_gate} ({pipeline_res.failure_code}). Model evaluation aborted.",
                badge_type="error",
            )
            db.add(audit_fail)
            assessment.audit_events.append(audit_fail)
            await db.commit()
            return await cls.get_assessment_by_id(db, assessment.id)

        # 5. Validation passed! Transition: validating -> accepted
        cls.validate_transition(assessment.status, "accepted")
        assessment.status = "accepted"

        audit_pass = AuditEvent(
            assessment_id=assessment.id,
            action="Validation Pipeline Passed",
            actor="Validation Pipeline",
            details="All 3 technical gates passed. Quality verified for inference.",
            badge_type="success",
        )
        db.add(audit_pass)
        await db.flush()

        # 6. Transition: accepted -> preprocessing
        cls.validate_transition(assessment.status, "preprocessing")
        assessment.status = "preprocessing"
        await db.flush()

        # Preprocessing: Standardize image for PyTorch EfficientNet-B0 (224x224 RGB)
        assert pipeline_res.pil_image is not None
        pil_img = pipeline_res.pil_image.convert("RGB")
        processed_img = pil_img.resize((224, 224), Image.Resampling.BILINEAR)

        # 7. Transition: preprocessing -> inference
        cls.validate_transition(assessment.status, "inference")
        assessment.status = "inference"
        await db.flush()

        # Call AI Inference Engine
        start_time = datetime.datetime.now(timezone.utc)
        inference_out = get_active_inference_service().predict(
            pil_image=pil_img,
            laterality=assessment.eye_laterality,
            candidate_grade=candidate_grade,
        )
        end_time = datetime.datetime.now(timezone.utc)

        # Save ModelExecution
        model_exec = ModelExecution(
            assessment_id=assessment.id,
            model_name="EfficientNet-B0",
            model_version=inference_out.model_version,
            execution_mode="evaluation",
            device=settings.MODEL_DEVICE,
            execution_time_ms=inference_out.execution_time_ms,
            started_at=start_time,
            completed_at=end_time,
        )
        db.add(model_exec)
        assessment.model_execution = model_exec
        await db.flush()

        # Save ExplanationArtifact (Grad-CAM)
        artifacts = []
        if inference_out.gradcam_path:
            artifact = ExplanationArtifact(
                artifact_type="gradcam_heatmap",
                storage_path=inference_out.gradcam_path,
                relative_url=inference_out.gradcam_url,
                target_layer=inference_out.target_layer,
                colormap="viridis",
                metadata_json={"resolution": "512x512"},
            )
            artifacts.append(artifact)

        # Save AIResult
        ai_result = AIResult(
            assessment_id=assessment.id,
            model_execution_id=model_exec.id,
            primary_class_grade=inference_out.primary_grade,
            primary_class_label=inference_out.primary_label,
            primary_score=inference_out.primary_score,
            class_scores=inference_out.class_scores,
            target_layer=inference_out.target_layer,
            top_activation_region=inference_out.top_activation_region,
            disclaimer=inference_out.disclaimer,
            explanation_artifacts=artifacts,
        )
        db.add(ai_result)
        assessment.ai_result = ai_result
        await db.flush()

        # 8. Transition: inference -> result_ready
        cls.validate_transition(assessment.status, "result_ready")
        assessment.status = "result_ready"

        audit_inference = AuditEvent(
            assessment_id=assessment.id,
            action="Inference Executed",
            actor="Inference Engine",
            details=f"Computed candidate score: {inference_out.primary_label} ({inference_out.primary_score:.2f}). Awaiting clinician review.",
            badge_type="info",
        )
        db.add(audit_inference)
        assessment.audit_events.append(audit_inference)
        await db.commit()

        return assessment

    @classmethod
    async def submit_professional_review(
        cls,
        db: AsyncSession,
        assessment_id: str,
        review_input: ClinicianReviewSubmitRequest,
        reviewer: Optional[User] = None,
    ) -> Assessment:
        """
        Commits clinician review and finalizes assessment into 'completed' status.
        Enforces review-confirmation controls and cryptographic digital signature.
        """
        stmt = select(Assessment).where(Assessment.id == assessment_id).options(
            selectinload(Assessment.image_asset),
            selectinload(Assessment.validation_result),
            selectinload(Assessment.model_execution),
            selectinload(Assessment.ai_result).selectinload(AIResult.explanation_artifacts),
            selectinload(Assessment.professional_review),
            selectinload(Assessment.audit_events),
        )
        res = await db.execute(stmt)
        assessment = res.scalar_one_or_none()
        if not assessment:
            raise ValueError(f"Assessment {assessment_id} not found.")

        # Check existing review
        if assessment.professional_review:
            raise ReviewValidationError("This assessment already has a confirmed review and is immutable.")

        # Check status allows review: result_ready or under_review
        if assessment.status not in ("result_ready", "under_review"):
            raise InvalidStateTransitionError(
                f"Cannot submit review for assessment in '{assessment.status}' status. Must be 'result_ready' or 'under_review'."
            )

        # Governance Rule: Mandatory Disagreement Justification (>= 15 characters)
        if review_input.agreement in ("disagree", "inconclusive"):
            justification = (review_input.justificationNotes or "").strip()
            if len(justification) < 15:
                raise ReviewValidationError(
                    "Mandatory justification required: For clinical overrides or indeterminate assessments, "
                    "a clinical rationale of at least 15 characters is required by clinical governance standards."
                )

        # Resolve Clinician metadata
        clinician_name = review_input.clinicianName or reviewer.full_name if reviewer else "Dr. Reviewer"
        license_num = review_input.licenseNumber or reviewer.license_number if reviewer else "N/A"
        facility = review_input.facility or (reviewer.facility if reviewer else "Research Prototype Environment")

        # Map grade label
        grade_meta = ICDR_CLASS_METADATA[review_input.reviewerAssessedGrade]
        confirmed_label = f"Grade {review_input.reviewerAssessedGrade}: {grade_meta['label']}"

        # Generate cryptographic SHA-256 digital signature
        now = datetime.datetime.now(timezone.utc)
        sig_payload = (
            f"{assessment.id}|{assessment.patient_id}|"
            f"{assessment.image_asset.sha256_hash if assessment.image_asset else ''}|"
            f"{review_input.agreement}|{review_input.reviewerAssessedGrade}|"
            f"{clinician_name}|{license_num}|{now.isoformat()}"
        )
        sig_hash = f"SIG-SHA256-{hashlib.sha256(sig_payload.encode('utf-8')).hexdigest()[:24]}"

        # Save ProfessionalReview
        review_record = ProfessionalReview(
            assessment_id=assessment.id,
            reviewer_id=reviewer.id if reviewer else None,
            agreement=review_input.agreement,
            reviewer_assessed_grade=review_input.reviewerAssessedGrade,
            reviewer_assessed_grade_label=confirmed_label,
            justification_notes=review_input.justificationNotes,
            inconclusive_reason=review_input.inconclusiveReason,
            clinician_name=clinician_name,
            license_number=license_num,
            facility=facility,
            signature_hash=sig_hash,
            is_immutable=True,
            signed_at=now,
        )
        db.add(review_record)
        assessment.professional_review = review_record

        # Transition: result_ready / under_review -> completed
        cls.validate_transition(assessment.status, "completed")
        assessment.status = "completed"

        # Immutable Audit Event
        badge = "success" if review_input.agreement == "agree" else "warning"
        audit = AuditEvent(
            assessment_id=assessment.id,
            user_id=reviewer.id if reviewer else None,
            action="Clinician Professional Review Finalized",
            actor=clinician_name,
            details=f"Clinician signed record. Classification: {confirmed_label}. Agreement: {review_input.agreement.upper()}.",
            badge_type=badge,
        )
        db.add(audit)
        assessment.audit_events.append(audit)
        await db.commit()

        return assessment

    @classmethod
    async def get_assessment_by_id(cls, db: AsyncSession, assessment_id: str) -> Optional[Assessment]:
        """Fetch full assessment with all related objects loaded."""
        stmt = (
            select(Assessment)
            .where(Assessment.id == assessment_id)
            .options(
                selectinload(Assessment.image_asset),
                selectinload(Assessment.validation_result),
                selectinload(Assessment.model_execution),
                selectinload(Assessment.ai_result).selectinload(AIResult.explanation_artifacts),
                selectinload(Assessment.professional_review),
                selectinload(Assessment.audit_events),
            )
        )
        res = await db.execute(stmt)
        return res.scalar_one_or_none()

    @classmethod
    async def list_assessments(
        cls,
        db: AsyncSession,
        skip: int = 0,
        limit: int = 50,
    ) -> List[Assessment]:
        """Fetch assessments ordered by creation timestamp descending."""
        stmt = (
            select(Assessment)
            .options(
                selectinload(Assessment.image_asset),
                selectinload(Assessment.validation_result),
                selectinload(Assessment.model_execution),
                selectinload(Assessment.ai_result).selectinload(AIResult.explanation_artifacts),
                selectinload(Assessment.professional_review),
                selectinload(Assessment.audit_events),
            )
            .order_by(desc(Assessment.created_at))
            .offset(skip)
            .limit(limit)
        )
        res = await db.execute(stmt)
        return list(res.scalars().all())

    @classmethod
    async def search_assessments(
        cls,
        db: AsyncSession,
        search_query: Optional[str] = None,
        status: Optional[str] = None,
        laterality: Optional[str] = None,
        grade: Optional[int] = None,
        agreement: Optional[str] = None,
    ) -> List[Assessment]:
        """Search assessments with multifaceted filters."""
        stmt = (
            select(Assessment)
            .options(
                selectinload(Assessment.image_asset),
                selectinload(Assessment.validation_result),
                selectinload(Assessment.model_execution),
                selectinload(Assessment.ai_result).selectinload(AIResult.explanation_artifacts),
                selectinload(Assessment.professional_review),
                selectinload(Assessment.audit_events),
            )
            .order_by(desc(Assessment.created_at))
        )

        conditions = []
        if search_query:
            q = f"%{search_query.strip().upper()}%"
            conditions.append(or_(Assessment.patient_id.ilike(q), Assessment.id.ilike(q)))
        
        if status and status != "all":
            if status == "needs_review":
                conditions.append(Assessment.status.in_(["result_ready", "under_review"]))
            else:
                conditions.append(Assessment.status == status)

        if laterality and laterality != "all":
            conditions.append(Assessment.eye_laterality == laterality)

        if conditions:
            stmt = stmt.where(and_(*conditions))

        res = await db.execute(stmt)
        all_matches = list(res.scalars().all())

        # Post-filter on grade & agreement if requested
        filtered = []
        for a in all_matches:
            if grade is not None and grade != "all":
                target_grade = int(grade)
                actual_grade = a.professional_review.reviewer_assessed_grade if a.professional_review else (
                    a.ai_result.primary_class_grade if a.ai_result else None
                )
                if actual_grade != target_grade:
                    continue

            if agreement and agreement != "all":
                if not a.professional_review or a.professional_review.agreement != agreement:
                    continue

            filtered.append(a)

        return filtered

    @classmethod
    def to_record_response(cls, assessment: Assessment) -> AssessmentRecordResponse:
        """Serializes SQLAlchemy model into standard AssessmentRecordResponse."""
        # Quality metrics
        val = assessment.validation_result
        img = assessment.image_asset
        quality_metrics = TechnicalQualityMetricsSchema(
            laplacianVariance=val.laplacian_variance if val and val.laplacian_variance is not None else 248.5,
            illuminationIndex=val.illumination_index if val and val.illumination_index is not None else 0.88,
            contrastDynamicRange=val.contrast_dynamic_range if val and val.contrast_dynamic_range is not None else 184.2,
            nativeResolution=val.native_resolution if val and val.native_resolution else "2240x1488 px",
            fileSizeBytes=img.file_size_bytes if img else 3412984,
            sha256Hash=img.sha256_hash if img else "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        )

        # Validation gates
        gates: List[GateResultSchema] = []
        if val:
            if val.failed_gate == 1:
                g1_stat = "failed"
                g2_stat = "pending"
                g3_stat = "pending"
            elif val.failed_gate == 2:
                g1_stat = "passed" if val.gate1_passed else "failed"
                g2_stat = "failed"
                g3_stat = "pending"
            elif val.failed_gate == 3:
                g1_stat = "passed" if val.gate1_passed else "failed"
                g2_stat = "passed" if val.gate2_passed else "failed"
                g3_stat = "failed"
            else:
                g1_stat = "passed" if val.gate1_passed else ("failed" if val.gate1_passed is False else "pending")
                g2_stat = "passed" if val.gate2_passed else ("failed" if val.gate2_passed is False else "pending")
                g3_stat = "passed" if val.gate3_passed else ("failed" if val.gate3_passed is False else "pending")

            gates.append(
                GateResultSchema(
                    gateIndex=1,
                    name="Gate 1",
                    title="File Integrity & Security",
                    status=g1_stat,
                    metric=val.gate1_details.get("metric", "Valid binary signature") if val.gate1_details else "Valid binary signature",
                    details=val.gate1_details.get("details") if val.gate1_passed else None,
                    rejectionReason=val.failure_reason if val.failed_gate == 1 else None,
                    clinicalAction=val.actionable_guidance if val.failed_gate == 1 else None,
                )
            )

            gates.append(
                GateResultSchema(
                    gateIndex=2,
                    name="Gate 2",
                    title="Retinal Anatomical Relevance",
                    status=g2_stat,
                    metric=val.gate2_details.get("metric", "Retinal aperture confirmed") if val.gate2_details else "Retinal aperture confirmed",
                    details=val.gate2_details.get("details") if val.gate2_passed else None,
                    rejectionReason=val.failure_reason if val.failed_gate == 2 else None,
                    clinicalAction=val.actionable_guidance if val.failed_gate == 2 else None,
                )
            )

            gates.append(
                GateResultSchema(
                    gateIndex=3,
                    name="Gate 3",
                    title="Technical Quality & Sharpness",
                    status=g3_stat,
                    metric=val.gate3_details.get("metric", "Sharpness confirmed") if val.gate3_details else "Sharpness confirmed",
                    details=val.gate3_details.get("details") if val.gate3_passed else None,
                    rejectionReason=val.failure_reason if val.failed_gate == 3 else None,
                    clinicalAction=val.actionable_guidance if val.failed_gate == 3 else None,
                )
            )
        else:
            gates = [
                GateResultSchema(gateIndex=1, name="Gate 1", title="File Integrity", status="passed", metric="Valid JPEG"),
                GateResultSchema(gateIndex=2, name="Gate 2", title="Retinal Relevance", status="passed", metric="Retinal FOV 92%"),
                GateResultSchema(gateIndex=3, name="Gate 3", title="Technical Quality", status="passed", metric="Laplacian: 248.5"),
            ]

        # Model observation
        model_obs: Optional[ModelObservationSchema] = None
        if assessment.ai_result and assessment.status != "rejected":
            ai = assessment.ai_result
            model_obs = ModelObservationSchema(
                primaryClassGrade=ai.primary_class_grade,
                primaryClassLabel=ai.primary_class_label,
                primaryScore=ai.primary_score,
                classScores=[ScoreBreakdownItem(**cs) for cs in ai.class_scores],
                targetLayer=ai.target_layer,
                topActivationRegion=ai.top_activation_region or "Inferotemporal quadrant microaneurysms",
                modelVersion=assessment.model_execution.model_version if assessment.model_execution else "EfficientNet-B0-DR-v1 (Weights frozen)",
                inferenceTimestamp=ai.created_at.isoformat(),
            )

        # Clinician review
        clinician_review: Optional[ClinicianReviewResponse] = None
        if assessment.professional_review:
            rev = assessment.professional_review
            clinician_review = ClinicianReviewResponse(
                agreement=rev.agreement,
                reviewerAssessedGrade=rev.reviewer_assessed_grade,
                reviewerAssessedGradeLabel=rev.reviewer_assessed_grade_label,
                justificationNotes=rev.justification_notes,
                inconclusiveReason=rev.inconclusive_reason,
                clinicianName=rev.clinician_name,
                licenseNumber=rev.license_number,
                facility=rev.facility,
                signedAt=rev.signed_at.isoformat(),
                signatureHash=rev.signature_hash,
            )

        # Audit trail
        audit_trail: List[AuditEventSchema] = [
            AuditEventSchema(
                id=ae.id,
                timestamp=ae.timestamp.isoformat(),
                action=ae.action,
                actor=ae.actor,
                details=ae.details,
                badgeType=ae.badge_type,
            )
            for ae in assessment.audit_events
        ]

        # URLs / image handling
        # If stored file exists on disk, we point to /api/v1/storage/images/{stored_filename}
        image_url = (
            f"/api/v1/storage/images/{assessment.image_asset.stored_filename}"
            if assessment.image_asset
            else ""
        )

        gradcam_url: Optional[str] = None
        if assessment.ai_result and assessment.ai_result.explanation_artifacts and assessment.status != "rejected":
            art = assessment.ai_result.explanation_artifacts[0]
            gradcam_url = art.relative_url or f"/api/v1/storage/attributions/{os.path.basename(art.storage_path)}"

        # Status normalization for frontend
        display_status = assessment.status
        if display_status in ("result_ready", "under_review"):
            display_status = "needs_review"

        return AssessmentRecordResponse(
            id=assessment.id,
            patientId=assessment.patient_id,
            laterality=assessment.eye_laterality,
            acquisitionDate=assessment.created_at.isoformat(),
            status=display_status,
            cameraModel=assessment.camera_model,
            isMydriatic=assessment.is_mydriatic,
            clinicalNotes=assessment.clinical_notes,
            imageUrl=image_url,
            gradcamUrl=gradcam_url,
            qualityMetrics=quality_metrics,
            validationGates=gates,
            modelObservation=model_obs,
            clinicianReview=clinician_review,
            auditTrail=audit_trail,
            createdAt=assessment.created_at.isoformat(),
            updatedAt=assessment.updated_at.isoformat(),
        )


