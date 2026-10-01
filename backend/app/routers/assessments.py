import base64
import os
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query, Response, status
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.models.models import User, Assessment
# Clinical endpoints require a real authenticated session.
#
# This module previously imported `get_optional_current_user` under the name
# `get_current_user`, which returned the seeded demonstration account whenever
# no bearer token was supplied - so every route below served an authenticated
# session to an anonymous caller while appearing to be protected.
from app.routers.auth import get_current_user
from app.schemas.assessment import (
    AssessmentCreateRequest,
    AssessmentRecordResponse,
    ClinicianReviewSubmitRequest,
    AuditEventSchema,
    GateResultSchema,
    ModelObservationSchema,
)
from app.services.ai_service import ModelCheckpointError
from app.services.assessment_service import (
    AssessmentService,
    InvalidStateTransitionError,
    ReviewValidationError,
)
from app.services.report_service import ReportService

router = APIRouter(tags=["Assessments"])


def decode_image_data_url(data_url: str) -> bytes:
    """Decode a base64 image data URL into raw bytes."""
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    return base64.b64decode(data_url)


@router.post("/assessments", response_model=AssessmentRecordResponse, status_code=status.HTTP_201_CREATED)
async def create_assessment(
    payload: AssessmentCreateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Create a new assessment record.
    If imageDataUrl is supplied, automatically executes the 3-stage validation pipeline
    and preliminary AI inference (if validated).
    """
    # 1. Create base assessment in draft state
    assessment = await AssessmentService.create_assessment(
        db=db,
        payload=payload,
        creator=current_user,
    )

    # 2. If image data is provided, run ingestion, validation & inference pipeline
    if payload.imageDataUrl:
        try:
            image_bytes = decode_image_data_url(payload.imageDataUrl)
            filename = payload.filename or f"fundus_{payload.laterality}.jpg"
            assessment = await AssessmentService.process_and_validate_image(
                db=db,
                assessment_id=assessment.id,
                image_bytes=image_bytes,
                original_filename=filename,
                candidate_grade=payload.candidateGrade,
                simulate_gate_failure=payload.simulateGateFailure,
                actor=current_user,
            )
        except Exception as e:
            # If processing errors, mark failed if not rejected
            if assessment.status != "rejected":
                assessment.status = "failed"
                await db.commit()
            raise HTTPException(status_code=400, detail=str(e))

    return AssessmentService.to_record_response(assessment)


@router.post("/assessments/{assessment_id}/upload", response_model=AssessmentRecordResponse)
async def upload_assessment_image(
    assessment_id: str,
    file: UploadFile = File(...),
    candidate_grade: Optional[int] = Form(None),
    simulate_gate_failure: Optional[int] = Form(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Upload a retinal fundus image for an existing draft assessment.
    Triggers the Fail-Closed 3-Stage Technical Validation Pipeline and AI inference.
    """
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file payload uploaded.")

    try:
        assessment = await AssessmentService.process_and_validate_image(
            db=db,
            assessment_id=assessment_id,
            image_bytes=image_bytes,
            original_filename=file.filename or "fundus.jpg",
            candidate_grade=candidate_grade,
            simulate_gate_failure=simulate_gate_failure,
            actor=current_user,
        )
        return AssessmentService.to_record_response(assessment)
    except ModelCheckpointError as e:
        # Fail closed: the engine has no verified weights, so no grade is
        # produced. 503 signals a service-side defect, not a bad upload.
        raise HTTPException(
            status_code=503,
            detail=f"Inference engine unavailable: {e}",
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except InvalidStateTransitionError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing error: {str(e)}")


@router.get("/assessments", response_model=List[AssessmentRecordResponse])
async def list_assessments(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Fetch clinical triage worklist records."""
    assessments = await AssessmentService.list_assessments(db, skip=skip, limit=limit)
    return [AssessmentService.to_record_response(a) for a in assessments]


@router.get("/assessments/search", response_model=List[AssessmentRecordResponse])
async def search_assessments(
    searchQuery: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    laterality: Optional[str] = Query(None),
    grade: Optional[int] = Query(None),
    agreement: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Search and filter assessments by clinical parameters."""
    matches = await AssessmentService.search_assessments(
        db=db,
        search_query=searchQuery,
        status=status,
        laterality=laterality,
        grade=grade,
        agreement=agreement,
    )
    return [AssessmentService.to_record_response(a) for a in matches]


@router.get("/assessments/{assessment_id}", response_model=AssessmentRecordResponse)
async def get_assessment(
    assessment_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Retrieve full clinical assessment encounter details."""
    assessment = await AssessmentService.get_assessment_by_id(db, assessment_id)
    if not assessment:
        raise HTTPException(status_code=404, detail=f"Assessment {assessment_id} not found.")
    return AssessmentService.to_record_response(assessment)


@router.get("/assessments/{assessment_id}/status")
async def get_assessment_status(
    assessment_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Check current state machine status."""
    assessment = await AssessmentService.get_assessment_by_id(db, assessment_id)
    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment not found.")
    return {
        "id": assessment.id,
        "status": assessment.status,
        "updatedAt": assessment.updated_at.isoformat(),
    }


@router.get("/assessments/{assessment_id}/validation", response_model=List[GateResultSchema])
async def get_assessment_validation(
    assessment_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Retrieve 3-stage validation results."""
    assessment = await AssessmentService.get_assessment_by_id(db, assessment_id)
    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment not found.")
    record = AssessmentService.to_record_response(assessment)
    return record.validationGates


@router.get("/assessments/{assessment_id}/result", response_model=Optional[ModelObservationSchema])
async def get_assessment_result(
    assessment_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Retrieve model-generated class scores and Grad-CAM explanation."""
    assessment = await AssessmentService.get_assessment_by_id(db, assessment_id)
    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment not found.")
    record = AssessmentService.to_record_response(assessment)
    return record.modelObservation


@router.post("/assessments/{assessment_id}/review", response_model=AssessmentRecordResponse)
async def submit_review(
    assessment_id: str,
    review_data: ClinicianReviewSubmitRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Formal clinical governance checkpoint.
    Clinician signs off on ICDR classification, referral recommendation, and justification.
    Once submitted, the record transitions to 'completed' and becomes strictly immutable.
    """
    try:
        assessment = await AssessmentService.submit_professional_review(
            db=db,
            assessment_id=assessment_id,
            review_input=review_data,
            reviewer=current_user,
        )
        return AssessmentService.to_record_response(assessment)
    except ReviewValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except InvalidStateTransitionError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/assessments/{assessment_id}/audit", response_model=List[AuditEventSchema])
async def get_assessment_audit_trail(
    assessment_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Retrieve immutable audit ledger for a given assessment."""
    assessment = await AssessmentService.get_assessment_by_id(db, assessment_id)
    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment not found.")
    record = AssessmentService.to_record_response(assessment)
    return record.auditTrail


@router.get("/assessments/{assessment_id}/report")
async def download_report_pdf(
    assessment_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Generate and return a tamper-evident server-rendered clinical consultation report PDF."""
    assessment = await AssessmentService.get_assessment_by_id(db, assessment_id)
    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment not found.")

    pdf_bytes = ReportService.generate_pdf_report(assessment)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="DR-CDSS-Report-{assessment.id}.pdf"'},
    )


# Alias route matching frontend downloadReportPdf
@router.get("/reports/{assessment_id}/pdf")
async def download_report_pdf_alias(
    assessment_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Alias for report download matching frontend service call."""
    return await download_report_pdf(assessment_id, db, current_user)


# Static file streaming for private storage
@router.get("/storage/images/{filename}")
async def serve_image(filename: str):
    """Serve uploaded retinal images securely."""
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(settings.STORAGE_IMAGES_PATH, safe_filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image asset not found.")
    return FileResponse(file_path)


@router.get("/storage/attributions/{filename}")
async def serve_attribution(filename: str):
    """Serve Grad-CAM attribution heatmaps securely."""
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(settings.STORAGE_ATTRIBUTIONS_PATH, safe_filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Attribution heatmap not found.")
    return FileResponse(file_path, media_type="image/png")
