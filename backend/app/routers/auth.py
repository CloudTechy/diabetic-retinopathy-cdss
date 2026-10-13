"""
Authentication: registration, sign-in, sign-out, profile and password change,
plus the dependencies every protected route uses to establish who is calling.

No account is seeded here. An earlier revision materialised a shared
demonstration physician on first use; accounts now come only from the Super
Administrator bootstrap (environment or command) and from the administration
and registration flows.
"""
import datetime
from datetime import timezone
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import create_access_token, decode_access_token, verify_password
from app.models.models import (
    ROLE_PHYSICIAN,
    ROLE_SUPERADMIN,
    STATUS_ACTIVE,
    STATUS_PENDING,
    STATUS_REJECTED,
    STATUS_SUSPENDED,
    User,
    utcnow,
)
from app.schemas.auth import (
    ChangePasswordRequest,
    ClinicianUserResponse,
    LoginRequest,
    RegisterRequest,
    RegisterResponse,
    TokenResponse,
)
from app.services import account_service
from app.services.account_service import AccountError

router = APIRouter(prefix="/auth", tags=["Authentication"])

# Messages for an account that exists but may not sign in. They name the state
# so the interface can say what to do; none of them reveals whether a password
# was right, because they are only reached after the password verified.
STATUS_MESSAGES = {
    STATUS_PENDING: "This account is awaiting administrative approval. You will be able to sign in once a Super Administrator approves it.",
    STATUS_REJECTED: "This registration was not approved. Contact the administrator if you believe this is an error.",
    STATUS_SUSPENDED: "This account is suspended. Contact the administrator to have it reactivated.",
}


def to_user_response(user: User, login_time: Optional[str] = None) -> ClinicianUserResponse:
    return ClinicianUserResponse(
        id=user.id,
        username=user.username,
        name=user.full_name,
        email=user.email,
        role=user.role,
        status=user.status,
        mustChangePassword=bool(user.must_change_password),
        licenseNumber=user.license_number,
        facility=user.facility,
        sessionTimeoutMinutes=15,
        loginTime=login_time or datetime.datetime.now(timezone.utc).isoformat(),
    )


def _unauthenticated(detail: str = "Could not validate credentials") -> HTTPException:
    return HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail, headers={"WWW-Authenticate": "Bearer"})


async def _resolve_token(authorization: Optional[str], db: AsyncSession) -> Optional[User]:
    """The account a bearer token names, if the token is valid, unexpired and not invalidated; else None."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    payload = decode_access_token(authorization.split(" ", 1)[1])
    if not payload or "sub" not in payload:
        return None
    user = await account_service.get_user(db, payload["sub"])
    if user is None:
        return None
    # A token outlives a password change, a suspension or a sign-out only if
    # its version still matches the account's.
    if int(payload.get("tv", -1)) != int(user.token_version or 0):
        return None
    return user


async def get_current_user(
    authorization: Optional[str] = Header(None),
    db: AsyncSession = Depends(get_db),
) -> User:
    """An authenticated ACTIVE account of either role. Used by /auth/me, /auth/logout and /auth/change-password."""
    if not authorization:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated",
                            headers={"WWW-Authenticate": "Bearer"})
    user = await _resolve_token(authorization, db)
    if user is None:
        raise _unauthenticated()
    if user.status != STATUS_ACTIVE:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail=STATUS_MESSAGES.get(user.status, "This account may not sign in."))
    return user


async def get_current_physician(current_user: User = Depends(get_current_user)) -> User:
    """
    An active PHYSICIAN whose password is not temporary. Every clinical route
    depends on this: a Super Administrator administers accounts and is not
    granted clinical content; an account with a temporary password must change
    it first.
    """
    if current_user.must_change_password:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="PASSWORD_CHANGE_REQUIRED: change the temporary password before using the system.")
    if current_user.role != ROLE_PHYSICIAN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="Clinical assessment operations are available to physician accounts only.")
    return current_user


async def get_current_superadmin(current_user: User = Depends(get_current_user)) -> User:
    """An active SUPERADMIN whose password is not temporary."""
    if current_user.must_change_password:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="PASSWORD_CHANGE_REQUIRED: change the temporary password before using the system.")
    if current_user.role != ROLE_SUPERADMIN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="Account administration is available to the Super Administrator only.")
    return current_user


async def get_optional_current_user(
    authorization: Optional[str] = Header(None),
    db: AsyncSession = Depends(get_db),
) -> Optional[User]:
    """Resolve a user from a bearer token, or None. Never fabricates a session."""
    user = await _resolve_token(authorization, db)
    if user is not None and user.status == STATUS_ACTIVE:
        return user
    # No valid bearer token means no authenticated user. An earlier revision
    # returned a seeded demonstration account here.
    return None


@router.post("/register", response_model=RegisterResponse, status_code=status.HTTP_201_CREATED)
async def register(payload: RegisterRequest, db: AsyncSession = Depends(get_db)):
    """
    Request a physician account. The account is created PENDING_APPROVAL and
    cannot sign in until a Super Administrator approves it. The request body
    has no role or status field; a public request can never produce an active
    account or a Super Administrator.
    """
    try:
        user = await account_service.register_physician(
            db, full_name=payload.fullName.strip(), username=payload.username.strip(), email=str(payload.email).lower(),
            password=payload.password, password_confirmation=payload.passwordConfirmation,
            license_number=(payload.licenseNumber or "").strip() or None, facility=(payload.facility or "").strip() or None,
        )
    except AccountError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
    return RegisterResponse(
        id=user.id, username=user.username, status=user.status,
        message="Your registration request has been submitted for administrative approval. "
                "You will be able to sign in once a Super Administrator approves the account.",
    )


@router.post("/login", response_model=TokenResponse)
async def login(credentials: LoginRequest, db: AsyncSession = Depends(get_db)):
    """Authenticate and issue a signed session token. The password is verified for every account, on every sign-in."""
    identifier = credentials.username.strip()
    user = await account_service.find_by_login(db, identifier)

    if user is None or not verify_password(credentials.password, user.hashed_password):
        # The attempted password is never written anywhere.
        account_service.record_event(
            db, event_type=account_service.EV_SIGN_IN_FAILED, action="Sign-In Failed", actor=None,
            actor_label=identifier[:150] or "unknown", target_user=user, outcome="failure",
            details="Sign-in failed for identifier '%s' (%s)." % (
                identifier[:100], "unknown account" if user is None else "wrong password"),
            badge_type="warning",
        )
        await db.commit()
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid clinical credentials or unauthorized account.")

    if user.status != STATUS_ACTIVE:
        account_service.record_event(
            db, event_type=account_service.EV_SIGN_IN_FAILED, action="Sign-In Refused", actor=user, target_user=user,
            outcome="failure", details="Sign-in refused for '%s': account status is %s." % (user.username, user.status),
            badge_type="warning",
        )
        await db.commit()
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail=STATUS_MESSAGES.get(user.status, "This account may not sign in."))

    # Claims: who (sub), what they may do (role) and which generation of
    # sessions this is (tv). No name, no email, no facility.
    token = create_access_token(subject=user.id, extra_claims={"role": user.role, "tv": int(user.token_version or 0)})

    now = utcnow()
    user.last_login_at = now
    account_service.record_event(
        db, event_type=account_service.EV_SIGN_IN_SUCCESS, action="Sign-In Succeeded", actor=user, target_user=user,
        details="'%s' signed in as %s%s." % (
            user.username, user.role, " (password change required)" if user.must_change_password else ""),
    )
    await db.commit()

    return TokenResponse(access_token=token, token_type="bearer", user=to_user_response(user, now.isoformat()))


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Invalidate every session of the signed-in account (its tokens stop being accepted)."""
    current_user.token_version = int(current_user.token_version or 0) + 1
    account_service.record_event(
        db, event_type=account_service.EV_SIGN_OUT, action="Signed Out", actor=current_user, target_user=current_user,
        details="'%s' signed out; the session token was invalidated." % current_user.username,
    )
    await db.commit()
    return None


@router.get("/me", response_model=ClinicianUserResponse)
async def get_my_profile(current_user: User = Depends(get_current_user)):
    """Return the authenticated account's profile."""
    return to_user_response(current_user)


@router.post("/change-password", response_model=ClinicianUserResponse)
async def change_password(
    payload: ChangePasswordRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Change the signed-in account's password. Required before anything else when the password is temporary."""
    try:
        user = await account_service.change_password(
            db, user=current_user, current_password=payload.currentPassword,
            new_password=payload.newPassword, confirmation=payload.newPasswordConfirmation,
        )
    except AccountError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))
    return to_user_response(user)
