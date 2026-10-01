import datetime
from datetime import timezone
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Header
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import verify_password, get_password_hash, create_access_token, decode_access_token
from app.models.models import User, AuditEvent
from app.schemas.auth import LoginRequest, TokenResponse, ClinicianUserResponse

router = APIRouter(prefix="/auth", tags=["Authentication"])

# Seeded demonstration account. Every value here is explicitly simulated: the
# system must never present a fabricated clinician identity, a fabricated
# professional registration number, or an affiliation with a real institution.
# An earlier revision seeded a plausible consultant name with a GMC-format
# number and an nhs.uk email address, which implied a real NHS affiliation.
DEFAULT_CLINICIAN_DATA = {
    "username": "demo.clinician",
    "email": "demo.clinician@research-prototype.invalid",
    "full_name": "Dr. Demo Clinician (Simulated)",
    "role": "Simulated Reviewer — Research Prototype",
    "license_number": "SIM-000001",
    "facility": "Research Prototype Environment",
}

# Usernames that may materialise the demonstration account on first login. They
# get an account created for them; they do NOT get to skip password checking.
DEMO_LOGIN_IDENTIFIERS = {
    "demo.clinician",
    "demo.clinician@research-prototype.invalid",
    "clinician",
}


async def get_or_create_seed_user(db: AsyncSession) -> User:
    """Ensure a default credentialed clinician exists for medical session."""
    stmt = select(User).where(User.username == DEFAULT_CLINICIAN_DATA["username"])
    res = await db.execute(stmt)
    user = res.scalar_one_or_none()
    if not user:
        user = User(
            id="USR-8821",
            username=DEFAULT_CLINICIAN_DATA["username"],
            email=DEFAULT_CLINICIAN_DATA["email"],
            full_name=DEFAULT_CLINICIAN_DATA["full_name"],
            hashed_password=get_password_hash("dr_secure_password_2026"),
            role=DEFAULT_CLINICIAN_DATA["role"],
            license_number=DEFAULT_CLINICIAN_DATA["license_number"],
            facility=DEFAULT_CLINICIAN_DATA["facility"],
            is_active=True,
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)
    return user


async def get_current_user(
    authorization: Optional[str] = Header(None),
    db: AsyncSession = Depends(get_db)
) -> User:
    """Strict dependency to extract authenticated user from Bearer JWT token."""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = authorization.split(" ")[1]
    payload = decode_access_token(token)
    if not payload or "sub" not in payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    stmt = select(User).where(User.id == payload["sub"])
    res = await db.execute(stmt)
    user = res.scalar_one_or_none()
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


async def get_optional_current_user(
    authorization: Optional[str] = Header(None),
    db: AsyncSession = Depends(get_db)
) -> Optional[User]:
    """Resolve a user from a bearer token, or None. Never fabricates a session."""
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]
        payload = decode_access_token(token)
        if payload and "sub" in payload:
            stmt = select(User).where(User.id == payload["sub"])
            res = await db.execute(stmt)
            user = res.scalar_one_or_none()
            if user and user.is_active:
                return user
    # No valid bearer token means no authenticated user.
    #
    # An earlier revision returned the seeded demonstration account here, so
    # every endpoint depending on this function served an authenticated session
    # to an anonymous caller. Returning None lets a route decide, and the
    # clinical routes now use the strict dependency instead.
    return None


@router.post("/login", response_model=TokenResponse)
async def login(credentials: LoginRequest, db: AsyncSession = Depends(get_db)):
    """Authenticate clinician credentials and issue signed clinical session JWT token."""
    stmt = select(User).where(
        (User.username == credentials.username) | (User.email == credentials.username)
    )
    res = await db.execute(stmt)
    user = res.scalar_one_or_none()

    # The demonstration account is materialised on first use so the prototype
    # can be opened without a separate migration step. It is created WITH a
    # password hash and authenticates against it exactly like any other account.
    if not user and credentials.username in DEMO_LOGIN_IDENTIFIERS:
        user = await get_or_create_seed_user(db)

    # The password is verified for every account, on every login.
    #
    # An earlier revision placed this check inside `if not user:`, so an account
    # that already existed skipped it entirely and any password issued a token.
    # Keep this unconditional: no username exemption, no plaintext comparison,
    # no "development credentials" branch.
    if user is None or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid clinical credentials or unauthorized account.",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This account is deactivated.",
        )

    token = create_access_token(subject=user.id, extra_claims={"role": user.role, "name": user.full_name})

    now = datetime.datetime.now(timezone.utc).isoformat()
    user_response = ClinicianUserResponse(
        id=user.id,
        name=user.full_name,
        email=user.email,
        role=user.role,
        licenseNumber=user.license_number,
        facility=user.facility,
        sessionTimeoutMinutes=15,
        loginTime=now,
    )

    # Log audit event
    audit = AuditEvent(
        user_id=user.id,
        action="Clinician Session Authenticated",
        actor=user.full_name,
        details=f"Authenticated session initiated from facility {user.facility}.",
        badge_type="info",
    )
    db.add(audit)
    await db.commit()

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        user=user_response,
    )


@router.get("/me", response_model=ClinicianUserResponse)
async def get_my_profile(current_user: User = Depends(get_current_user)):
    """Return authenticated clinician profile."""
    return ClinicianUserResponse(
        id=current_user.id,
        name=current_user.full_name,
        email=current_user.email,
        role=current_user.role,
        licenseNumber=current_user.license_number,
        facility=current_user.facility,
        sessionTimeoutMinutes=15,
        loginTime=datetime.datetime.now(timezone.utc).isoformat(),
    )
