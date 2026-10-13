"""
Super Administrator console: physician accounts and the audit of their
administration. Every route depends on get_current_superadmin; a physician
receives 403 from all of them. No response here carries a password hash; a
temporary password appears only in the response to the request that issued it.
"""
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.models import ROLES, AuditEvent, User
from app.routers.auth import get_current_superadmin
from app.schemas.auth import (
    AccountSummaryResponse,
    AdminAuditEventResponse,
    BatchCreatePhysiciansRequest,
    BatchCreatePhysiciansResponse,
    CreatedPhysicianResponse,
    CreatePhysicianRequest,
    RejectRequest,
    ResetPasswordResponse,
    SuspendRequest,
    UserAccountResponse,
)
from app.services import account_service
from app.services.account_service import AccountError

router = APIRouter(prefix="/admin", tags=["Account Administration"])


def to_account_response(user: User, assessment_count: int = 0) -> UserAccountResponse:
    return UserAccountResponse(
        id=user.id,
        username=user.username,
        fullName=user.full_name,
        email=user.email,
        role=user.role,
        status=user.status,
        mustChangePassword=bool(user.must_change_password),
        licenseNumber=user.license_number,
        facility=user.facility,
        createdById=user.created_by_id,
        approvedById=user.approved_by_id,
        approvedAt=user.approved_at.isoformat() if user.approved_at else None,
        rejectionReason=user.rejection_reason,
        lastLoginAt=user.last_login_at.isoformat() if user.last_login_at else None,
        createdAt=user.created_at.isoformat(),
        updatedAt=user.updated_at.isoformat(),
        assessmentCount=assessment_count,
    )


def to_audit_response(event: AuditEvent) -> AdminAuditEventResponse:
    return AdminAuditEventResponse(
        id=event.id,
        timestamp=event.timestamp.isoformat(),
        eventType=event.event_type,
        action=event.action,
        actor=event.actor,
        actorUserId=event.user_id,
        targetUserId=event.target_user_id,
        assessmentId=event.assessment_id,
        outcome=event.outcome or "success",
        details=event.details,
        badgeType=event.badge_type,
    )


def _http(exc: AccountError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=str(exc))


@router.get("/summary", response_model=AccountSummaryResponse)
async def account_summary(db: AsyncSession = Depends(get_db), _: User = Depends(get_current_superadmin)):
    return AccountSummaryResponse(**await account_service.summary(db))


@router.get("/users", response_model=List[UserAccountResponse])
async def list_users(
    q: Optional[str] = Query(None, max_length=100, description="matches username, full name or email"),
    role: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(200, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_superadmin),
):
    """All accounts, with search and filters. Password hashes and tokens are never included."""
    if role and role not in ROLES:
        raise HTTPException(status_code=422, detail="role must be one of %s" % ", ".join(ROLES))
    try:
        rows = await account_service.list_users(db, query=q, role=role, status=status, limit=limit)
    except AccountError as exc:
        raise _http(exc)
    return [to_account_response(user, n) for user, n in rows]


@router.get("/users/pending", response_model=List[UserAccountResponse])
async def pending_registrations(db: AsyncSession = Depends(get_db), _: User = Depends(get_current_superadmin)):
    rows = await account_service.list_users(db, status="PENDING_APPROVAL")
    return [to_account_response(user, n) for user, n in rows]


@router.post("/users", response_model=CreatedPhysicianResponse, status_code=201)
async def create_physician(
    payload: CreatePhysicianRequest,
    db: AsyncSession = Depends(get_db),
    actor: User = Depends(get_current_superadmin),
):
    """Create a physician account. The temporary password is returned here and nowhere else."""
    try:
        user, password = await account_service.create_physician(
            db, actor=actor, full_name=payload.fullName.strip(), username=payload.username.strip(),
            email=str(payload.email).lower(), license_number=(payload.licenseNumber or "").strip() or None,
            facility=(payload.facility or "").strip() or None, temporary_password=payload.temporaryPassword,
            activate_immediately=payload.activateImmediately,
        )
    except AccountError as exc:
        raise _http(exc)
    return CreatedPhysicianResponse(user=to_account_response(user), temporaryPassword=password)


@router.post("/users/batch", response_model=BatchCreatePhysiciansResponse, status_code=201)
async def create_physicians_batch(
    payload: BatchCreatePhysiciansRequest,
    db: AsyncSession = Depends(get_db),
    actor: User = Depends(get_current_superadmin),
):
    """Up to ten numbered physician accounts for a supervised demonstration, each with its own temporary password."""
    try:
        created = await account_service.create_physicians_batch(
            db, actor=actor, count=payload.count, username_prefix=payload.usernamePrefix.strip(),
            full_name_prefix=payload.fullNamePrefix.strip(), email_domain=payload.emailDomain.strip().lower(),
            facility=(payload.facility or "").strip() or None,
        )
    except AccountError as exc:
        raise _http(exc)
    return BatchCreatePhysiciansResponse(
        created=[CreatedPhysicianResponse(user=to_account_response(u), temporaryPassword=p) for u, p in created])


@router.get("/users/{user_id}", response_model=UserAccountResponse)
async def get_user(user_id: str, db: AsyncSession = Depends(get_db), _: User = Depends(get_current_superadmin)):
    rows = await account_service.list_users(db, limit=500)
    for user, n in rows:
        if user.id == user_id:
            return to_account_response(user, n)
    raise HTTPException(status_code=404, detail="No account with id %s." % user_id)


@router.patch("/users/{user_id}/approve", response_model=UserAccountResponse)
async def approve_user(user_id: str, db: AsyncSession = Depends(get_db), actor: User = Depends(get_current_superadmin)):
    try:
        return to_account_response(await account_service.approve(db, actor=actor, user_id=user_id))
    except AccountError as exc:
        raise _http(exc)


@router.patch("/users/{user_id}/reject", response_model=UserAccountResponse)
async def reject_user(
    user_id: str, payload: Optional[RejectRequest] = None,
    db: AsyncSession = Depends(get_db), actor: User = Depends(get_current_superadmin),
):
    try:
        return to_account_response(await account_service.reject(
            db, actor=actor, user_id=user_id, reason=payload.reason if payload else None))
    except AccountError as exc:
        raise _http(exc)


@router.patch("/users/{user_id}/suspend", response_model=UserAccountResponse)
async def suspend_user(
    user_id: str, payload: Optional[SuspendRequest] = None,
    db: AsyncSession = Depends(get_db), actor: User = Depends(get_current_superadmin),
):
    try:
        return to_account_response(await account_service.suspend(
            db, actor=actor, user_id=user_id, reason=payload.reason if payload else None))
    except AccountError as exc:
        raise _http(exc)


@router.patch("/users/{user_id}/reactivate", response_model=UserAccountResponse)
async def reactivate_user(user_id: str, db: AsyncSession = Depends(get_db), actor: User = Depends(get_current_superadmin)):
    try:
        return to_account_response(await account_service.reactivate(db, actor=actor, user_id=user_id))
    except AccountError as exc:
        raise _http(exc)


@router.post("/users/{user_id}/reset-password", response_model=ResetPasswordResponse)
async def reset_password(user_id: str, db: AsyncSession = Depends(get_db), actor: User = Depends(get_current_superadmin)):
    """Issue a new temporary password (returned once) and invalidate the account's sessions."""
    try:
        user, password = await account_service.reset_password(db, actor=actor, user_id=user_id)
    except AccountError as exc:
        raise _http(exc)
    return ResetPasswordResponse(user=to_account_response(user), temporaryPassword=password)


@router.get("/audit-events", response_model=List[AdminAuditEventResponse])
async def audit_events(
    userId: Optional[str] = Query(None, description="events acting on or performed by this account"),
    eventType: Optional[str] = Query(None),
    limit: int = Query(200, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_superadmin),
):
    """Account-administration audit events, newest first. Read-only: no route edits or deletes an audit entry."""
    events = await account_service.admin_audit_events(
        db, target_user_id=userId, event_types=[eventType] if eventType else None, limit=limit)
    return [to_audit_response(e) for e in events]
