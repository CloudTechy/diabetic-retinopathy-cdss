import datetime
from datetime import timezone
import uuid
from typing import Optional, List, Dict, Any

from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    Boolean,
    DateTime,
    Text,
    JSON,
    ForeignKey,
)
from sqlalchemy.orm import relationship

from app.core.database import Base


def utcnow() -> datetime.datetime:
    return datetime.datetime.now(timezone.utc)


def generate_uuid() -> str:
    return str(uuid.uuid4())


class User(Base):
    """User account model supporting clinicians, technicians, and system administrators."""
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    username = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    full_name = Column(String(255), nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False, default="clinician")  # clinician, technician, admin
    license_number = Column(String(100), nullable=True)  # e.g., GMC-7492104
    facility = Column(String(255), nullable=True)  # Hospital / Clinic Unit
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)

    # Relationships
    created_assessments = relationship("Assessment", back_populates="creator", foreign_keys="Assessment.created_by_id")
    reviews = relationship("ProfessionalReview", back_populates="reviewer")
    audit_events = relationship("AuditEvent", back_populates="user")


class Assessment(Base):
    """Core assessment entity representing a single clinical eye encounter."""
    __tablename__ = "assessments"

    id = Column(String(64), primary_key=True, default=lambda: f"REC-2026-{uuid.uuid4().hex[:6].upper()}")
    patient_id = Column(String(100), nullable=False, index=True)
    eye_laterality = Column(String(10), nullable=False)  # 'OD' (Right) or 'OS' (Left)
    camera_model = Column(String(255), nullable=True, default="Topcon TRC-NW400 Non-Mydriatic")
    is_mydriatic = Column(Boolean, default=False, nullable=False)
    clinical_notes = Column(Text, nullable=True)
    
    # State Machine Status:
    # 'draft', 'uploaded', 'validating', 'accepted', 'rejected',
    # 'preprocessing', 'inference', 'result_ready', 'under_review',
    # 'completed', 'failed'
    status = Column(String(50), default="draft", nullable=False, index=True)

    created_by_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)

    # Relationships
    creator = relationship("User", back_populates="created_assessments", foreign_keys=[created_by_id], lazy="selectin")
    image_asset = relationship("ImageAsset", back_populates="assessment", uselist=False, cascade="all, delete-orphan", lazy="selectin")
    validation_result = relationship("ValidationResult", back_populates="assessment", uselist=False, cascade="all, delete-orphan", lazy="selectin")
    model_execution = relationship("ModelExecution", back_populates="assessment", uselist=False, cascade="all, delete-orphan", lazy="selectin")
    ai_result = relationship("AIResult", back_populates="assessment", uselist=False, cascade="all, delete-orphan", lazy="selectin")
    professional_review = relationship("ProfessionalReview", back_populates="assessment", uselist=False, cascade="all, delete-orphan", lazy="selectin")
    audit_events = relationship("AuditEvent", back_populates="assessment", cascade="all, delete-orphan", order_by="AuditEvent.timestamp", lazy="selectin")


class ImageAsset(Base):
    """Storage metadata and cryptographic integrity record for uploaded fundus photographs."""
    __tablename__ = "image_assets"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    assessment_id = Column(String(64), ForeignKey("assessments.id", ondelete="CASCADE"), unique=True, nullable=False)
    original_filename = Column(String(255), nullable=False)
    stored_filename = Column(String(255), nullable=False)
    storage_path = Column(String(500), nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    mime_type = Column(String(100), nullable=False)
    sha256_hash = Column(String(64), nullable=False)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    # Relationship
    assessment = relationship("Assessment", back_populates="image_asset")


class ValidationResult(Base):
    """Technical 3-stage validation results and machine-readable gating codes."""
    __tablename__ = "validation_results"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    assessment_id = Column(String(64), ForeignKey("assessments.id", ondelete="CASCADE"), unique=True, nullable=False)
    status = Column(String(50), nullable=False)  # 'passed' or 'rejected'
    
    gate1_passed = Column(Boolean, nullable=False)
    gate1_details = Column(JSON, nullable=False, default=dict)
    
    gate2_passed = Column(Boolean, nullable=True)
    gate2_details = Column(JSON, nullable=True, default=dict)
    
    gate3_passed = Column(Boolean, nullable=True)
    gate3_details = Column(JSON, nullable=True, default=dict)

    failed_gate = Column(Integer, nullable=True)  # 1, 2, or 3 if failed
    failure_code = Column(String(100), nullable=True)
    failure_reason = Column(Text, nullable=True)
    actionable_guidance = Column(Text, nullable=True)

    laplacian_variance = Column(Float, nullable=True)
    illumination_index = Column(Float, nullable=True)
    contrast_dynamic_range = Column(Float, nullable=True)
    native_resolution = Column(String(50), nullable=True)

    evaluated_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    # Relationship
    assessment = relationship("Assessment", back_populates="validation_result")


class ModelExecution(Base):
    """Audit record capturing PyTorch model invocation, execution mode, and compute runtime."""
    __tablename__ = "model_executions"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    assessment_id = Column(String(64), ForeignKey("assessments.id", ondelete="CASCADE"), unique=True, nullable=False)
    model_name = Column(String(100), default="EfficientNet-B0", nullable=False)
    model_version = Column(String(100), default="EfficientNet-B0-DR-v1 (Weights frozen)", nullable=False)
    execution_mode = Column(String(50), default="evaluation", nullable=False)  # Evaluation only; online learning forbidden
    device = Column(String(50), default="cpu", nullable=False)
    execution_time_ms = Column(Float, nullable=False, default=0.0)
    started_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    completed_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    # Relationship
    assessment = relationship("Assessment", back_populates="model_execution")
    ai_result = relationship("AIResult", back_populates="model_execution", uselist=False)


class AIResult(Base):
    """Bounded decision-support scores and candidate classification generated by the model."""
    __tablename__ = "ai_results"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    assessment_id = Column(String(64), ForeignKey("assessments.id", ondelete="CASCADE"), unique=True, nullable=False)
    model_execution_id = Column(String(36), ForeignKey("model_executions.id", ondelete="CASCADE"), unique=True, nullable=False)
    
    primary_class_grade = Column(Integer, nullable=False)  # 0 to 4
    primary_class_label = Column(String(100), nullable=False)  # e.g., 'Moderate NPDR'
    primary_score = Column(Float, nullable=False)  # e.g., 0.78
    
    # Full 5-class normalized score breakdown
    class_scores = Column(JSON, nullable=False)  # list of {grade: int, label: str, score: float}
    
    target_layer = Column(String(100), default="features.8 (Conv2d Bottleneck Residual)", nullable=False)
    top_activation_region = Column(String(255), nullable=True)
    disclaimer = Column(
        Text,
        default="NOTICE: Clinical Decision Support Only — Not for Independent Diagnosis. "
                "Scores represent feature activations from frozen EfficientNet-B0 and do not represent clinical certainty.",
        nullable=False
    )
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    # Relationships
    assessment = relationship("Assessment", back_populates="ai_result")
    model_execution = relationship("ModelExecution", back_populates="ai_result")
    explanation_artifacts = relationship("ExplanationArtifact", back_populates="ai_result", cascade="all, delete-orphan", lazy="selectin")


class ExplanationArtifact(Base):
    """Visual saliency artifacts (Grad-CAM heatmaps) linked directly to the model execution."""
    __tablename__ = "explanation_artifacts"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    ai_result_id = Column(String(36), ForeignKey("ai_results.id", ondelete="CASCADE"), nullable=False)
    artifact_type = Column(String(50), default="gradcam_heatmap", nullable=False)
    storage_path = Column(String(500), nullable=False)
    relative_url = Column(String(500), nullable=True)
    target_layer = Column(String(100), default="features.8", nullable=False)
    colormap = Column(String(50), default="viridis", nullable=False)
    metadata_json = Column(JSON, nullable=True, default=dict)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    # Relationship
    ai_result = relationship("AIResult", back_populates="explanation_artifacts")


class ProfessionalReview(Base):
    """Clinician-in-the-loop review record. Once signed, this record is strictly immutable."""
    __tablename__ = "professional_reviews"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    assessment_id = Column(String(64), ForeignKey("assessments.id", ondelete="CASCADE"), unique=True, nullable=False)
    reviewer_id = Column(String(36), ForeignKey("users.id", ondelete="RESTRICT"), nullable=True)
    
    # Tri-State Agreement: 'agree', 'disagree', 'inconclusive'
    agreement = Column(String(50), nullable=False)
    
    certified_grade = Column(Integer, nullable=False)  # 0 to 4
    certified_grade_label = Column(String(100), nullable=False)
    
    # Mandatory justification if disagree or inconclusive (>= 15 characters)
    justification_notes = Column(Text, nullable=True)
    inconclusive_reason = Column(String(255), nullable=True)
    referral_plan = Column(String(255), nullable=False)
    
    clinician_name = Column(String(255), nullable=False)
    license_number = Column(String(100), nullable=True)
    facility = Column(String(255), nullable=True)
    
    # Cryptographic SHA-256 digital signature of review content and image hash
    signature_hash = Column(String(128), nullable=False)
    is_immutable = Column(Boolean, default=True, nullable=False)
    signed_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    # Relationships
    assessment = relationship("Assessment", back_populates="professional_review")
    reviewer = relationship("User", back_populates="reviews")


class AuditEvent(Base):
    """Immutable audit trail log tracking clinical events, security actions, and state changes."""
    __tablename__ = "audit_events"

    id = Column(String(64), primary_key=True, default=lambda: f"AUD-{uuid.uuid4().hex[:6].upper()}")
    assessment_id = Column(String(64), ForeignKey("assessments.id", ondelete="SET NULL"), nullable=True, index=True)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    
    action = Column(String(150), nullable=False)
    actor = Column(String(150), nullable=False)
    details = Column(Text, nullable=False)
    badge_type = Column(String(30), default="info", nullable=False)  # 'info', 'success', 'warning', 'error'
    
    event_metadata = Column(JSON, nullable=True, default=dict)
    ip_address = Column(String(60), nullable=True)
    timestamp = Column(DateTime(timezone=True), default=utcnow, nullable=False, index=True)

    # Relationships
    assessment = relationship("Assessment", back_populates="audit_events")
    user = relationship("User", back_populates="audit_events")
