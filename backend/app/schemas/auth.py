from typing import Optional, List, Any
from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    username: str
    password: str


class ClinicianUserResponse(BaseModel):
    id: str
    name: str
    email: str
    role: str
    licenseNumber: Optional[str] = None
    facility: Optional[str] = None
    sessionTimeoutMinutes: int = 15
    loginTime: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: ClinicianUserResponse
