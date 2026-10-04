import os
from datetime import datetime, timezone
from fastapi import APIRouter
from typing import Optional

from pydantic import BaseModel

from app.core.config import settings
from app.services.ai_service import get_inference_health

router = APIRouter(tags=["Health"])


class HealthResponse(BaseModel):
    status: str
    app_name: str
    environment: str
    timestamp: str
    storage_accessible: bool
    inference_ready: bool
    inference_engine: str
    inference_detail: str
    # The SHA-256 the engine verified before loading; None for the simulated
    # engine or while not ready.
    checkpoint_sha256: Optional[str] = None
    version: str = "1.0.0"


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check endpoint verifying service status and storage accessibility."""
    # Check if storage paths exist or are creatable
    storage_accessible = os.path.exists(settings.STORAGE_BASE_PATH) or os.path.exists("./storage")

    # Surface inference readiness here so a missing or unverified checkpoint is
    # observable, rather than only discovered when a clinician submits a scan.
    inference = get_inference_health()

    return HealthResponse(
        status="healthy" if inference["ready"] else "degraded",
        app_name=settings.APP_NAME,
        environment=settings.APP_ENV,
        timestamp=datetime.now(timezone.utc).isoformat(),
        storage_accessible=storage_accessible,
        inference_ready=inference["ready"],
        inference_engine=inference["engine"],
        inference_detail=inference["detail"],
        checkpoint_sha256=inference.get("checkpoint_sha256"),
        version="1.0.0",
    )
