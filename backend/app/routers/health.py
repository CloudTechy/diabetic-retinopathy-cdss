import os
from datetime import datetime, timezone
from fastapi import APIRouter
from pydantic import BaseModel

from app.core.config import settings

router = APIRouter(tags=["Health"])


class HealthResponse(BaseModel):
    status: str
    app_name: str
    environment: str
    timestamp: str
    storage_accessible: bool
    version: str = "1.0.0"


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check endpoint verifying service status and storage accessibility."""
    # Check if storage paths exist or are creatable
    storage_accessible = os.path.exists(settings.STORAGE_BASE_PATH) or os.path.exists("./storage")

    return HealthResponse(
        status="healthy",
        app_name=settings.APP_NAME,
        environment=settings.APP_ENV,
        timestamp=datetime.now(timezone.utc).isoformat(),
        storage_accessible=storage_accessible,
        version="1.0.0",
    )
