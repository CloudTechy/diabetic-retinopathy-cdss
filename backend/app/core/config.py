import os
from typing import List, Union
from pydantic import AnyHttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "Diabetic Retinopathy CDSS"
    APP_ENV: str = "development"
    DEBUG: bool = False
    API_V1_PREFIX: str = "/api/v1"
    SECRET_KEY: str = "default-insecure-secret-key-change-in-production"

    # CORS settings: accepts list of strings or comma-separated string
    CORS_ORIGINS: Union[List[str], str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",") if i.strip()]
        elif isinstance(v, str) and v.startswith("["):
            import json
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
            except Exception:
                pass
        return v if isinstance(v, list) else []

    # Database
    POSTGRES_USER: str = "dr_user"
    POSTGRES_PASSWORD: str = "dr_secure_password_2026"
    POSTGRES_DB: str = "dr_cdss_db"
    POSTGRES_HOST: str = "db"
    POSTGRES_PORT: int = 5432
    DATABASE_URL: str = "postgresql+asyncpg://dr_user:dr_secure_password_2026@db:5432/dr_cdss_db"

    # Authentication & JWT
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours

    # Storage paths (private)
    STORAGE_BASE_PATH: str = "/app/storage" if os.path.exists("/app") else os.path.abspath("./storage")
    STORAGE_IMAGES_PATH: str = "/app/storage/images" if os.path.exists("/app") else os.path.abspath("./storage/images")
    STORAGE_REPORTS_PATH: str = "/app/storage/reports" if os.path.exists("/app") else os.path.abspath("./storage/reports")
    STORAGE_ATTRIBUTIONS_PATH: str = "/app/storage/attributions" if os.path.exists("/app") else os.path.abspath("./storage/attributions")
    MAX_UPLOAD_SIZE_MB: int = 15
    MAX_UPLOAD_SIZE_BYTES: int = 15 * 1024 * 1024  # 15 MB per Gate 1 spec

    # Technical Validation Thresholds
    MIN_IMAGE_DIMENSION: int = 480
    LAPLACIAN_BLUR_THRESHOLD: float = 4.3
    CONTRAST_THRESHOLD: float = 8.8
    ILLUMINATION_EXTREME_RATIO_MAX: float = 0.35
    RETINAL_MIN_COVERAGE: float = 0.20
    RETINAL_MAX_COVERAGE: float = 0.98
    RETINAL_RED_RATIO_MIN: float = 1.15
    # Longest side used when computing Gate 2/3 DISTRIBUTION statistics
    # (coverage ratios, channel means, contrast std, extreme-pixel proportions).
    # Nearest-neighbour subsampling leaves these unbiased while removing ~59%
    # of end-to-end request latency. Set to 0 to analyse at full resolution.
    # Gate 3's Laplacian variance is a spatial derivative and is NOT affected
    # by this; it keeps its own separate 1024px resize.
    VALIDATION_ANALYSIS_MAX_DIM: int = 512

    # AI Model
    # Must match where the weights actually land inside the container: compose
    # mounts ./backend at /app, so backend/models/weights/ -> /app/models/weights/.
    # Follows the same convention as STORAGE_* above: the container path when
    # running under compose, the repository path otherwise. A checkpoint path
    # that only resolves inside Docker makes every local run depend on an
    # override, and a missing override fails closed with a confusing message.
    MODEL_CHECKPOINT_PATH: str = (
        "/app/models/weights/efficientnet_b0_dr.pth" if os.path.exists("/app")
        else os.path.abspath(
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))), "models", "weights",
                "efficientnet_b0_dr.pth"))
    )
    MODEL_DEVICE: str = "cpu"
    MODEL_SCORE_THRESHOLD: float = 0.5
    # SHA-256 of the evaluated checkpoint. When set, the engine refuses to load
    # any other weights, tying served predictions to the graded artefact.
    MODEL_CHECKPOINT_SHA256: str = (
        "67d0b89641f08057126dd411e380b25575ef29f71ae37ee5796d472d9203dbf7"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )


settings = Settings()
