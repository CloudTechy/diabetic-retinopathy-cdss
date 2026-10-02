import io
import math
import os
import pathlib

# Workaround: on Windows the default pytest temp root
# (C:\Users\...\Temp\pytest-of-USER) can become inaccessible.
# Redirect to a guaranteed-writable location inside the project.
_pytest_tmp = pathlib.Path(__file__).resolve().parent.parent / ".pytest_tmp"
_pytest_tmp.mkdir(exist_ok=True)
os.environ.setdefault("PYTEST_DEBUG_TEMPROOT", str(_pytest_tmp))

# The suite exercises the clinical journey against the SIMULATED inference
# engine. This opt-in is explicit and must stay explicit: the production code
# path fails closed rather than silently substituting simulated grades, so
# without this the assessment endpoints correctly return 503.
os.environ.setdefault("AI_INFERENCE_ENGINE", "mock")

import numpy as np
import pytest
import pytest_asyncio
from PIL import Image, ImageDraw, ImageFilter
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.database import Base, get_db
from main import app


def create_synthetic_retinal_fundus(
    width: int = 512,
    height: int = 512,
    blur: bool = False,
    is_retinal: bool = True,
) -> Image.Image:
    """Generate deterministic synthetic fundus or non-retinal images for unit testing."""
    if not is_retinal:
        # Generate non-retinal image: cool blue landscape / sky image
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        arr[:, :, 0] = 50   # Low Red
        arr[:, :, 1] = 120  # Medium Green
        arr[:, :, 2] = 220  # High Blue
        img = Image.fromarray(arr, mode="RGB")
        draw = ImageDraw.Draw(img)
        draw.text((50, 50), "NON-RETINAL PHOTOGRAPH - CHEST X-RAY / FACE", fill=(255, 255, 255))
        return img

    # Authentic Retinal Fundus: Circular aperture with deep red/orange vascular tones
    img = Image.new("RGB", (width, height), (5, 5, 5))
    draw = ImageDraw.Draw(img)

    # Circular mask filling ~85% of frame
    margin = int(width * 0.08)
    bbox = [margin, margin, width - margin, height - margin]
    # Retinal base color: warm reddish-orange
    draw.ellipse(bbox, fill=(185, 65, 25))

    # Optic disc: yellowish ellipse
    disc_x, disc_y = int(width * 0.35), int(height * 0.50)
    disc_r = int(width * 0.07)
    draw.ellipse([disc_x - disc_r, disc_y - disc_r, disc_x + disc_r, disc_y + disc_r], fill=(240, 210, 110))

    # Fovea / Macula: darker reddish-brown spot
    fovea_x, fovea_y = int(width * 0.58), int(height * 0.52)
    fovea_r = int(width * 0.04)
    draw.ellipse([fovea_x - fovea_r, fovea_y - fovea_r, fovea_x + fovea_r, fovea_y + fovea_r], fill=(120, 30, 15))

    # Branching retinal blood vessels (sharp high-frequency edges for Laplacian sharpness)
    vessel_color = (110, 20, 15)
    for i in range(12):
        angle = (i / 12.0) * 2 * math.pi
        x_end = int(disc_x + math.cos(angle) * (width * 0.35))
        y_end = int(disc_y + math.sin(angle) * (height * 0.35))
        draw.line([disc_x, disc_y, x_end, y_end], fill=vessel_color, width=3)
        # Secondary branches
        mid_x = (disc_x + x_end) // 2
        mid_y = (disc_y + y_end) // 2
        draw.line([mid_x, mid_y, mid_x + 25, mid_y - 20], fill=vessel_color, width=2)
        draw.line([mid_x, mid_y, mid_x - 20, mid_y + 25], fill=vessel_color, width=2)

    if blur:
        # Apply heavy blur to trigger Gate 3 rejection
        img = img.filter(ImageFilter.GaussianBlur(radius=8.0))

    return img


def image_to_bytes(img: Image.Image, format: str = "JPEG") -> bytes:
    """Convert PIL image to encoded binary bytes."""
    buf = io.BytesIO()
    img.save(buf, format=format, quality=95)
    return buf.getvalue()


# SQLite in-memory test database fixture
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(
    TEST_DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False},
)

TestingSessionLocal = async_sessionmaker(
    bind=test_engine,
    class_=AsyncSession,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest_asyncio.fixture
async def async_db():
    """Yield a clean test database session with all tables created."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with TestingSessionLocal() as session:
        yield session

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture
def test_client():
    """Provide a TestClient with overridden database session."""
    # Ensure tables exist
    import asyncio
    async def create_tables():
        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    if loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            pool.submit(asyncio.run, create_tables()).result()
    else:
        loop.run_until_complete(create_tables())

    async def override_get_db():
        async with TestingSessionLocal() as session:
            try:
                yield session
            finally:
                await session.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


@pytest.fixture
def auth_headers(test_client):
    """
    Bearer header for an authenticated clinical session.

    The clinical endpoints require a real token. They previously accepted
    anonymous callers because the dependency returned the seeded demonstration
    account when no token was supplied, so tests passed without authenticating
    and the protection was never exercised.
    """
    res = test_client.post("/api/v1/auth/login", json={
        "username": "demo.clinician",
        "password": "dr_secure_password_2026",
    })
    assert res.status_code == 200, f"fixture login failed: {res.text}"
    return {"Authorization": f"Bearer {res.json()['access_token']}"}


@pytest.fixture
def authed_client(test_client, auth_headers):
    """A TestClient that sends the bearer token on every request."""
    test_client.headers.update(auth_headers)
    return test_client
