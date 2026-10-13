from typing import List, Optional
from pydantic import BaseModel, EmailStr, Field


class LoginRequest(BaseModel):
    username: str
    password: str


class ClinicianUserResponse(BaseModel):
    """The signed-in account as the interface sees it. Never carries a hash or a password."""
    id: str
    username: str
    name: str
    email: str
    role: str                      # SUPERADMIN | PHYSICIAN
    status: str                    # PENDING_APPROVAL | ACTIVE | REJECTED | SUSPENDED
    mustChangePassword: bool = False
    # Typed by the applicant or an administrator; not verified against any register.
    licenseNumber: Optional[str] = None
    facility: Optional[str] = None
    sessionTimeoutMinutes: int = 15
    loginTime: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: ClinicianUserResponse


class RegisterRequest(BaseModel):
    """Self-registration by a prospective physician. Produces a PENDING_APPROVAL account; no role field exists."""
    fullName: str = Field(min_length=2, max_length=255)
    username: str = Field(min_length=3, max_length=100, pattern=r"^[A-Za-z0-9._-]+$")
    email: EmailStr
    password: str = Field(min_length=10, max_length=128)
    passwordConfirmation: str = Field(min_length=10, max_length=128)
    licenseNumber: Optional[str] = Field(default=None, max_length=100)
    facility: Optional[str] = Field(default=None, max_length=255)


class RegisterResponse(BaseModel):
    id: str
    username: str
    status: str
    message: str


class ChangePasswordRequest(BaseModel):
    currentPassword: str
    newPassword: str = Field(min_length=10, max_length=128)
    newPasswordConfirmation: str = Field(min_length=10, max_length=128)


class UserAccountResponse(BaseModel):
    """An account as listed to a Super Administrator. No hash, no password, no token."""
    id: str
    username: str
    fullName: str
    email: str
    role: str
    status: str
    mustChangePassword: bool
    licenseNumber: Optional[str] = None
    facility: Optional[str] = None
    createdById: Optional[str] = None
    approvedById: Optional[str] = None
    approvedAt: Optional[str] = None
    rejectionReason: Optional[str] = None
    lastLoginAt: Optional[str] = None
    createdAt: str
    updatedAt: str
    assessmentCount: int = 0


class CreatePhysicianRequest(BaseModel):
    fullName: str = Field(min_length=2, max_length=255)
    username: str = Field(min_length=3, max_length=100, pattern=r"^[A-Za-z0-9._-]+$")
    email: EmailStr
    licenseNumber: Optional[str] = Field(default=None, max_length=100)
    facility: Optional[str] = Field(default=None, max_length=255)
    # When omitted the server generates one. Either way it is temporary and
    # is returned ONCE, in the response to this request.
    temporaryPassword: Optional[str] = Field(default=None, min_length=10, max_length=128)
    activateImmediately: bool = True


class CreatedPhysicianResponse(BaseModel):
    user: UserAccountResponse
    temporaryPassword: str


class BatchCreatePhysiciansRequest(BaseModel):
    """Several accounts for a supervised demonstration: <prefix>-01, <prefix>-02, ..."""
    count: int = Field(ge=1, le=10)
    usernamePrefix: str = Field(min_length=2, max_length=40, pattern=r"^[A-Za-z0-9._-]+$")
    fullNamePrefix: str = Field(default="Demonstration Physician", min_length=2, max_length=200)
    emailDomain: str = Field(default="example.org", min_length=3, max_length=100)   # RFC 2606 reserved; no mail is sent
    facility: Optional[str] = Field(default=None, max_length=255)


class BatchCreatePhysiciansResponse(BaseModel):
    created: List[CreatedPhysicianResponse]


class RejectRequest(BaseModel):
    reason: Optional[str] = Field(default=None, max_length=500)


class SuspendRequest(BaseModel):
    reason: Optional[str] = Field(default=None, max_length=500)


class ResetPasswordResponse(BaseModel):
    user: UserAccountResponse
    temporaryPassword: str


class AccountSummaryResponse(BaseModel):
    pending: int
    active: int
    suspended: int
    rejected: int
    superadmins: int
    physicians: int


class AdminAuditEventResponse(BaseModel):
    id: str
    timestamp: str
    eventType: Optional[str] = None
    action: str
    actor: str
    actorUserId: Optional[str] = None
    targetUserId: Optional[str] = None
    assessmentId: Optional[str] = None
    outcome: str
    details: str
    badgeType: str
