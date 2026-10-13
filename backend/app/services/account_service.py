"""
Account administration: roles, account states, the transitions between them,
and the audit record of every one of those transitions.

Rules enforced here (the routers only translate exceptions to HTTP):
- Only a Super Administrator creates, approves, rejects, suspends or
  reactivates a physician account; the acting user is always passed in.
- Self-registration produces PENDING_APPROVAL and nothing else.
- Nothing here ever creates a Super Administrator except bootstrap(), which
  refuses when one already exists.
- The last ACTIVE Super Administrator cannot be suspended.
- A temporary password is returned exactly once, to the administrator who
  requested it, and sets must_change_password.
"""
import datetime
import secrets
import string
from datetime import timezone
from typing import Iterable, List, Optional, Tuple

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.security import get_password_hash, verify_password
from app.models.models import (
    ACCOUNT_STATUSES,
    ROLE_PHYSICIAN,
    ROLE_SUPERADMIN,
    STATUS_ACTIVE,
    STATUS_PENDING,
    STATUS_REJECTED,
    STATUS_SUSPENDED,
    Assessment,
    AuditEvent,
    User,
    utcnow,
)

# Audit event codes. The `action` column keeps a human sentence; `event_type`
# is the stable code a report or a test filters on.
EV_SUPERADMIN_CREATED = "SUPERADMIN_CREATED"
EV_REGISTRATION_SUBMITTED = "PHYSICIAN_REGISTRATION_SUBMITTED"
EV_PHYSICIAN_CREATED = "PHYSICIAN_ACCOUNT_CREATED"
EV_ACCOUNT_APPROVED = "ACCOUNT_APPROVED"
EV_ACCOUNT_REJECTED = "ACCOUNT_REJECTED"
EV_ACCOUNT_SUSPENDED = "ACCOUNT_SUSPENDED"
EV_ACCOUNT_REACTIVATED = "ACCOUNT_REACTIVATED"
EV_TEMP_PASSWORD_ISSUED = "TEMPORARY_PASSWORD_ISSUED"
EV_PASSWORD_CHANGED = "PASSWORD_CHANGED"
EV_SIGN_IN_SUCCESS = "SIGN_IN_SUCCESS"
EV_SIGN_IN_FAILED = "SIGN_IN_FAILED"
EV_SIGN_OUT = "SIGN_OUT"
EV_REVIEW_CONFIRMED = "PROFESSIONAL_REVIEW_CONFIRMED"


class AccountError(Exception):
    """A rule of account administration was violated. `status_code` is the HTTP status the router answers with."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def record_event(
    db: AsyncSession,
    *,
    event_type: str,
    action: str,
    actor: Optional[User],
    details: str,
    target_user: Optional[User] = None,
    assessment_id: Optional[str] = None,
    outcome: str = "success",
    badge_type: str = "info",
    actor_label: Optional[str] = None,
) -> AuditEvent:
    """Append one audit event. Nothing in the application updates or deletes these rows."""
    event = AuditEvent(
        assessment_id=assessment_id,
        user_id=actor.id if actor else None,
        target_user_id=target_user.id if target_user else None,
        event_type=event_type,
        action=action,
        actor=actor_label or (actor.full_name if actor else "System"),
        details=details,
        outcome=outcome,
        badge_type=badge_type,
    )
    db.add(event)
    return event


def generate_temporary_password(length: int = 14) -> str:
    """A random password with at least one lower-case letter, one upper-case letter and one digit."""
    alphabet = string.ascii_letters + string.digits
    while True:
        candidate = "".join(secrets.choice(alphabet) for _ in range(length))
        if (any(c.islower() for c in candidate) and any(c.isupper() for c in candidate)
                and any(c.isdigit() for c in candidate)):
            return candidate


def check_password_policy(password: str) -> None:
    if len(password) < settings.PASSWORD_MIN_LENGTH:
        raise AccountError("The password must be at least %d characters long." % settings.PASSWORD_MIN_LENGTH, 422)
    if not any(c.isalpha() for c in password) or not any(c.isdigit() for c in password):
        raise AccountError("The password must contain both letters and digits.", 422)


def _sync_legacy_flag(user: User) -> None:
    user.is_active = user.status == STATUS_ACTIVE
    user.updated_at = utcnow()


async def _username_or_email_taken(db: AsyncSession, username: str, email: str) -> Optional[str]:
    res = await db.execute(select(User).where(or_(User.username == username, User.email == email)))
    existing = res.scalars().first()
    if existing is None:
        return None
    return "username" if existing.username == username else "email"


async def count_active_superadmins(db: AsyncSession) -> int:
    res = await db.execute(select(func.count()).select_from(User).where(
        User.role == ROLE_SUPERADMIN, User.status == STATUS_ACTIVE))
    return int(res.scalar_one())


async def get_user(db: AsyncSession, user_id: str) -> Optional[User]:
    res = await db.execute(select(User).where(User.id == user_id))
    return res.scalar_one_or_none()


async def find_by_login(db: AsyncSession, identifier: str) -> Optional[User]:
    res = await db.execute(select(User).where(or_(User.username == identifier, User.email == identifier)))
    return res.scalar_one_or_none()


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

async def bootstrap_superadmin(
    db: AsyncSession,
    *,
    username: str,
    email: str,
    full_name: str,
    initial_password: str,
    source: str = "environment",
) -> Tuple[Optional[User], str]:
    """
    Create the first Super Administrator. Idempotent: returns (None, reason)
    without writing when any Super Administrator already exists or the inputs
    are incomplete. The password is temporary and must be changed at the first
    sign-in. The creation is audited.
    """
    if not (username and email and initial_password):
        return None, "username, email and initial password are all required"
    res = await db.execute(select(func.count()).select_from(User).where(User.role == ROLE_SUPERADMIN))
    if int(res.scalar_one()) > 0:
        return None, "a Super Administrator already exists; nothing created"
    check_password_policy(initial_password)
    taken = await _username_or_email_taken(db, username, email)
    if taken:
        raise AccountError("The %s is already in use by another account." % taken, 409)
    user = User(
        username=username,
        email=email,
        full_name=full_name or "Super Administrator",
        hashed_password=get_password_hash(initial_password),
        role=ROLE_SUPERADMIN,
        status=STATUS_ACTIVE,
        must_change_password=True,
        is_active=True,
    )
    db.add(user)
    await db.flush()
    record_event(
        db, event_type=EV_SUPERADMIN_CREATED, action="Super Administrator Created", actor=None,
        actor_label="Bootstrap (%s)" % source, target_user=user,
        details="First Super Administrator '%s' created from the %s; the temporary password must be changed at first sign-in." % (username, source),
        badge_type="warning",
    )
    await db.commit()
    await db.refresh(user)
    return user, "created"


# ---------------------------------------------------------------------------
# Registration and creation
# ---------------------------------------------------------------------------

async def register_physician(
    db: AsyncSession, *, full_name: str, username: str, email: str, password: str,
    password_confirmation: str, license_number: Optional[str], facility: Optional[str],
) -> User:
    """Self-registration. The account is PENDING_APPROVAL; no request field can change that or the role."""
    if password != password_confirmation:
        raise AccountError("The password and its confirmation do not match.", 422)
    check_password_policy(password)
    taken = await _username_or_email_taken(db, username, email)
    if taken:
        raise AccountError("The %s is already in use by another account." % taken, 409)
    user = User(
        username=username,
        email=email,
        full_name=full_name,
        hashed_password=get_password_hash(password),
        role=ROLE_PHYSICIAN,
        status=STATUS_PENDING,
        must_change_password=False,
        is_active=False,
        license_number=license_number or None,
        facility=facility or None,
    )
    db.add(user)
    await db.flush()
    record_event(
        db, event_type=EV_REGISTRATION_SUBMITTED, action="Physician Registration Submitted",
        actor=user, target_user=user,
        details="Self-registration for '%s' submitted; the account is pending administrative approval." % username,
    )
    await db.commit()
    await db.refresh(user)
    return user


async def create_physician(
    db: AsyncSession, *, actor: User, full_name: str, username: str, email: str,
    license_number: Optional[str], facility: Optional[str],
    temporary_password: Optional[str], activate_immediately: bool,
) -> Tuple[User, str]:
    """An administrator creates a physician. Returns the account and the temporary password (once)."""
    _require_superadmin(actor)
    password = temporary_password or generate_temporary_password()
    check_password_policy(password)
    taken = await _username_or_email_taken(db, username, email)
    if taken:
        raise AccountError("The %s is already in use by another account." % taken, 409)
    now = utcnow()
    user = User(
        username=username,
        email=email,
        full_name=full_name,
        hashed_password=get_password_hash(password),
        role=ROLE_PHYSICIAN,
        status=STATUS_ACTIVE if activate_immediately else STATUS_PENDING,
        must_change_password=True,
        is_active=activate_immediately,
        license_number=license_number or None,
        facility=facility or None,
        created_by_id=actor.id,
        approved_by_id=actor.id if activate_immediately else None,
        approved_at=now if activate_immediately else None,
    )
    db.add(user)
    await db.flush()
    record_event(
        db, event_type=EV_PHYSICIAN_CREATED, action="Physician Account Created", actor=actor, target_user=user,
        details="Physician account '%s' created by the Super Administrator (%s); a temporary password was issued." % (
            username, "activated immediately" if activate_immediately else "pending approval"),
        badge_type="success",
    )
    record_event(
        db, event_type=EV_TEMP_PASSWORD_ISSUED, action="Temporary Password Issued", actor=actor, target_user=user,
        details="A temporary password was issued for '%s'; it must be changed at the first sign-in." % username,
        badge_type="warning",
    )
    await db.commit()
    await db.refresh(user)
    return user, password


async def create_physicians_batch(
    db: AsyncSession, *, actor: User, count: int, username_prefix: str, full_name_prefix: str,
    email_domain: str, facility: Optional[str],
) -> List[Tuple[User, str]]:
    """Accounts <prefix>-01.. for a supervised demonstration. Each one is audited like a single creation."""
    _require_superadmin(actor)
    created: List[Tuple[User, str]] = []
    # find the first free index so a second batch continues the numbering
    index = 1
    while len(created) < count:
        username = "%s-%02d" % (username_prefix, index)
        email = "%s@%s" % (username, email_domain)
        index += 1
        if await _username_or_email_taken(db, username, email):
            continue
        user, password = await create_physician(
            db, actor=actor, full_name="%s %02d" % (full_name_prefix, index - 1), username=username, email=email,
            license_number=None, facility=facility, temporary_password=None, activate_immediately=True,
        )
        created.append((user, password))
    return created


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------

def _require_superadmin(actor: User) -> None:
    if actor is None or actor.role != ROLE_SUPERADMIN or actor.status != STATUS_ACTIVE:
        raise AccountError("Only an active Super Administrator may administer accounts.", 403)


async def _load_target(db: AsyncSession, actor: User, user_id: str) -> User:
    _require_superadmin(actor)
    target = await get_user(db, user_id)
    if target is None:
        raise AccountError("No account with id %s." % user_id, 404)
    return target


async def approve(db: AsyncSession, *, actor: User, user_id: str) -> User:
    target = await _load_target(db, actor, user_id)
    if target.id == actor.id:
        raise AccountError("An account cannot approve itself.", 403)
    if target.status != STATUS_PENDING:
        raise AccountError("Only a PENDING_APPROVAL account can be approved (this one is %s)." % target.status, 409)
    target.status = STATUS_ACTIVE
    target.approved_by_id = actor.id
    target.approved_at = utcnow()
    target.rejection_reason = None
    _sync_legacy_flag(target)
    record_event(db, event_type=EV_ACCOUNT_APPROVED, action="Account Approved", actor=actor, target_user=target,
                 details="Physician account '%s' approved; it may now sign in." % target.username, badge_type="success")
    await db.commit()
    await db.refresh(target)
    return target


async def reject(db: AsyncSession, *, actor: User, user_id: str, reason: Optional[str]) -> User:
    target = await _load_target(db, actor, user_id)
    if target.id == actor.id:
        raise AccountError("An account cannot reject itself.", 403)
    if target.status != STATUS_PENDING:
        raise AccountError("Only a PENDING_APPROVAL account can be rejected (this one is %s)." % target.status, 409)
    target.status = STATUS_REJECTED
    target.rejection_reason = (reason or "").strip() or None
    target.token_version += 1
    _sync_legacy_flag(target)
    record_event(db, event_type=EV_ACCOUNT_REJECTED, action="Account Rejected", actor=actor, target_user=target,
                 details="Registration of '%s' rejected%s." % (target.username, (": " + reason) if reason else ""),
                 badge_type="error")
    await db.commit()
    await db.refresh(target)
    return target


async def suspend(db: AsyncSession, *, actor: User, user_id: str, reason: Optional[str]) -> User:
    target = await _load_target(db, actor, user_id)
    if target.status != STATUS_ACTIVE:
        raise AccountError("Only an ACTIVE account can be suspended (this one is %s)." % target.status, 409)
    if target.role == ROLE_SUPERADMIN and await count_active_superadmins(db) <= 1:
        raise AccountError("The last active Super Administrator cannot be suspended.", 409)
    target.status = STATUS_SUSPENDED
    target.token_version += 1          # sessions already issued stop working
    target.rejection_reason = (reason or "").strip() or None
    _sync_legacy_flag(target)
    record_event(db, event_type=EV_ACCOUNT_SUSPENDED, action="Account Suspended", actor=actor, target_user=target,
                 details="Account '%s' suspended%s; existing sessions were invalidated." % (
                     target.username, (": " + reason) if reason else ""), badge_type="warning")
    await db.commit()
    await db.refresh(target)
    return target


async def reactivate(db: AsyncSession, *, actor: User, user_id: str) -> User:
    target = await _load_target(db, actor, user_id)
    if target.status != STATUS_SUSPENDED:
        raise AccountError("Only a SUSPENDED account can be reactivated (this one is %s)." % target.status, 409)
    target.status = STATUS_ACTIVE
    target.rejection_reason = None
    _sync_legacy_flag(target)
    record_event(db, event_type=EV_ACCOUNT_REACTIVATED, action="Account Reactivated", actor=actor, target_user=target,
                 details="Account '%s' reactivated." % target.username, badge_type="success")
    await db.commit()
    await db.refresh(target)
    return target


async def reset_password(db: AsyncSession, *, actor: User, user_id: str) -> Tuple[User, str]:
    """Issue a new temporary password. Returned once; previous sessions are invalidated."""
    target = await _load_target(db, actor, user_id)
    if target.status not in (STATUS_ACTIVE, STATUS_SUSPENDED):
        raise AccountError("A temporary password can only be issued for an ACTIVE or SUSPENDED account.", 409)
    password = generate_temporary_password()
    target.hashed_password = get_password_hash(password)
    target.must_change_password = True
    target.token_version += 1
    target.updated_at = utcnow()
    record_event(db, event_type=EV_TEMP_PASSWORD_ISSUED, action="Temporary Password Issued", actor=actor, target_user=target,
                 details="A temporary password was issued for '%s'; it must be changed at the next sign-in." % target.username,
                 badge_type="warning")
    await db.commit()
    await db.refresh(target)
    return target, password


async def change_password(db: AsyncSession, *, user: User, current_password: str, new_password: str,
                          confirmation: str) -> User:
    """The signed-in account changes its own password; clears must_change_password; invalidates other sessions."""
    if not verify_password(current_password, user.hashed_password):
        raise AccountError("The current password is not correct.", 403)
    if new_password != confirmation:
        raise AccountError("The new password and its confirmation do not match.", 422)
    if new_password == current_password:
        raise AccountError("The new password must differ from the current one.", 422)
    check_password_policy(new_password)
    user.hashed_password = get_password_hash(new_password)
    user.must_change_password = False
    user.token_version += 1
    user.updated_at = utcnow()
    record_event(db, event_type=EV_PASSWORD_CHANGED, action="Password Changed", actor=user, target_user=user,
                 details="'%s' changed their password; sessions issued before the change were invalidated." % user.username,
                 badge_type="success")
    await db.commit()
    await db.refresh(user)
    return user


# ---------------------------------------------------------------------------
# Queries for the console
# ---------------------------------------------------------------------------

async def list_users(
    db: AsyncSession, *, query: Optional[str] = None, role: Optional[str] = None, status: Optional[str] = None,
    limit: int = 200,
) -> List[Tuple[User, int]]:
    """Accounts with the number of assessments each created, newest first."""
    counts = (
        select(Assessment.created_by_id.label("uid"), func.count(Assessment.id).label("n"))
        .group_by(Assessment.created_by_id).subquery()
    )
    stmt = select(User, func.coalesce(counts.c.n, 0)).outerjoin(counts, counts.c.uid == User.id)
    if query:
        like = "%%%s%%" % query.strip().lower()
        stmt = stmt.where(or_(func.lower(User.username).like(like), func.lower(User.full_name).like(like),
                              func.lower(User.email).like(like)))
    if role:
        stmt = stmt.where(User.role == role)
    if status:
        if status not in ACCOUNT_STATUSES:
            raise AccountError("Unknown status filter %r." % status, 422)
        stmt = stmt.where(User.status == status)
    stmt = stmt.order_by(User.created_at.desc()).limit(limit)
    res = await db.execute(stmt)
    return [(user, int(n)) for user, n in res.all()]


async def summary(db: AsyncSession) -> dict:
    res = await db.execute(select(User.role, User.status, func.count()).group_by(User.role, User.status))
    out = {"pending": 0, "active": 0, "suspended": 0, "rejected": 0, "superadmins": 0, "physicians": 0}
    for role, status, n in res.all():
        key = {STATUS_PENDING: "pending", STATUS_ACTIVE: "active", STATUS_SUSPENDED: "suspended",
               STATUS_REJECTED: "rejected"}.get(status)
        if key:
            out[key] += int(n)
        if role == ROLE_SUPERADMIN:
            out["superadmins"] += int(n)
        elif role == ROLE_PHYSICIAN:
            out["physicians"] += int(n)
    return out


async def admin_audit_events(
    db: AsyncSession, *, target_user_id: Optional[str] = None, event_types: Optional[Iterable[str]] = None,
    limit: int = 200,
) -> List[AuditEvent]:
    """Account-administration events (those carrying an event_type), newest first."""
    stmt = select(AuditEvent).where(AuditEvent.event_type.is_not(None))
    if target_user_id:
        stmt = stmt.where(or_(AuditEvent.target_user_id == target_user_id, AuditEvent.user_id == target_user_id))
    if event_types:
        stmt = stmt.where(AuditEvent.event_type.in_(list(event_types)))
    stmt = stmt.order_by(AuditEvent.timestamp.desc()).limit(limit)
    res = await db.execute(stmt)
    return list(res.scalars().all())
