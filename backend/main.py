import logging
from contextlib import asynccontextmanager
import os
import uuid
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.database import init_db
from app.routers.health import router as health_router
from app.routers.auth import router as auth_router
from app.routers.assessments import router as assessments_router

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
)
logger = logging.getLogger("dr_cdss.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context for startup and shutdown routines."""
    logger.info("Initializing %s in [%s] mode...", settings.APP_NAME, settings.APP_ENV)
    # Ensure storage directories exist locally or in container
    for path in [
        settings.STORAGE_BASE_PATH,
        settings.STORAGE_IMAGES_PATH,
        settings.STORAGE_REPORTS_PATH,
        settings.STORAGE_ATTRIBUTIONS_PATH,
    ]:
        try:
            os.makedirs(path, exist_ok=True)
            logger.info("Verified storage path: %s", path)
        except Exception as e:
            logger.warning("Could not create storage path %s: %s", path, str(e))
    
    # Initialize database tables
    try:
        relaxed = await init_db()
        logger.info("Database tables initialized successfully.")
        if relaxed:
            logger.warning("Columns widened to accept NULL to match the models: %s", ", ".join(relaxed))
    except Exception as e:
        logger.error("Database initialization error: %s", str(e))

    yield
    logger.info("Shutting down %s...", settings.APP_NAME)


app = FastAPI(
    title=settings.APP_NAME,
    description="Clinical Decision Support System for Early Detection of Diabetic Retinopathy API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
    lifespan=lifespan,
)

# Configure CORS Middleware
origins = settings.CORS_ORIGINS if isinstance(settings.CORS_ORIGINS, list) else ["*"]
logger.info("Configuring CORS with allowed origins: %s", origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """
    An unhandled error is answered with a reason, never a bare 500.

    The default handler returns the text "Internal Server Error", so the client
    could only say "HTTP 500 with no explanation". The response names the kind
    of failure and carries a reference that is also written to the log beside
    the traceback. Database errors are summarised by their first line only: the
    full statement and its parameters stay in the log.
    """
    reference = uuid.uuid4().hex[:12]
    logger.exception("Unhandled error %s on %s %s", reference, request.method, request.url.path)
    kind = type(exc).__name__
    original = getattr(exc, "orig", None)
    summary = str(original if original is not None else exc).strip().splitlines()[0][:300] if str(exc).strip() else ""
    return JSONResponse(
        status_code=500,
        content={
            "detail": "The server failed while handling %s %s: %s%s. Reference %s."
                      % (request.method, request.url.path, kind, (" - " + summary) if summary else "", reference),
            "errorType": kind,
            "reference": reference,
        },
    )


# Mount Health router at root and under API v1 prefix
app.include_router(health_router)
app.include_router(health_router, prefix=settings.API_V1_PREFIX)

# Mount Clinical Routers under API v1 prefix
app.include_router(auth_router, prefix=settings.API_V1_PREFIX)
app.include_router(assessments_router, prefix=settings.API_V1_PREFIX)



@app.get("/", tags=["Root"])
async def root():
    """Root entrypoint returning basic system identity."""
    return {
        "name": settings.APP_NAME,
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "status": "operational",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
