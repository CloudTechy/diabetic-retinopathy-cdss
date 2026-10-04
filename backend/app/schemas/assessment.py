from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field


EyeLaterality = Literal["OD", "OS"]
ReviewStatus = Literal["draft", "uploaded", "validating", "accepted", "rejected", "preprocessing", "inference", "result_ready", "needs_review", "under_review", "completed", "failed"]
GateStatus = Literal["pending", "in_progress", "passed", "failed"]
AgreementType = Literal["agree", "disagree", "inconclusive"]


class GateResultSchema(BaseModel):
    name: str
    gateIndex: int
    status: GateStatus
    title: str
    metric: Optional[str] = None
    details: Optional[str] = None
    rejectionReason: Optional[str] = None
    clinicalAction: Optional[str] = None


class TechnicalQualityMetricsSchema(BaseModel):
    laplacianVariance: float
    illuminationIndex: float
    contrastDynamicRange: float
    nativeResolution: str
    fileSizeBytes: int
    sha256Hash: str


class ScoreBreakdownItem(BaseModel):
    grade: int
    label: str
    score: float


class ModelObservationSchema(BaseModel):
    primaryClassGrade: int
    primaryClassLabel: str
    primaryScore: float
    classScores: List[ScoreBreakdownItem]
    targetLayer: str
    topActivationRegion: str
    modelVersion: str
    inferenceTimestamp: str


class ClinicianReviewSubmitRequest(BaseModel):
    agreement: AgreementType
    reviewerAssessedGrade: int
    reviewerAssessedGradeLabel: Optional[str] = None
    justificationNotes: Optional[str] = None
    inconclusiveReason: Optional[str] = None
    clinicianName: Optional[str] = None
    licenseNumber: Optional[str] = None
    facility: Optional[str] = None


class ClinicianReviewResponse(BaseModel):
    agreement: AgreementType
    reviewerAssessedGrade: int
    reviewerAssessedGradeLabel: str
    justificationNotes: Optional[str] = None
    inconclusiveReason: Optional[str] = None
    clinicianName: str
    licenseNumber: Optional[str] = None
    facility: Optional[str] = None
    signedAt: str
    signatureHash: str


class AuditEventSchema(BaseModel):
    id: str
    timestamp: str
    action: str
    actor: str
    details: str
    badgeType: Literal["info", "success", "warning", "error"]


class AssessmentCreateRequest(BaseModel):
    patientId: str
    laterality: EyeLaterality
    cameraModel: Optional[str] = "Topcon TRC-NW400 Non-Mydriatic"
    isMydriatic: Optional[bool] = False
    clinicalNotes: Optional[str] = None
    imageDataUrl: Optional[str] = None
    fileSizeBytes: Optional[int] = None
    filename: Optional[str] = None


class AssessmentRecordResponse(BaseModel):
    id: str
    patientId: str
    laterality: EyeLaterality
    acquisitionDate: str   # record creation time; the field name is historical
    status: str
    cameraModel: Optional[str] = None
    isMydriatic: Optional[bool] = False
    clinicalNotes: Optional[str] = None
    imageUrl: str
    gradcamUrl: Optional[str] = None
    qualityMetrics: TechnicalQualityMetricsSchema
    validationGates: List[GateResultSchema]
    modelObservation: Optional[ModelObservationSchema] = None
    clinicianReview: Optional[ClinicianReviewResponse] = None
    auditTrail: List[AuditEventSchema]
    createdAt: str
    updatedAt: str
