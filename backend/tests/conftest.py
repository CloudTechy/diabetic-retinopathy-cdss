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
        img = Image.fromarray(arr)
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


def pytest_sessionfinish(session, exitstatus):
    """
    Close the pooled in-memory connection so the run can exit.

    aiosqlite serves each connection from a non-daemon worker thread; while
    test_engine holds one open, the interpreter waits on that thread forever
    after printing the summary (tests/test_session_teardown.py).
    """
    import asyncio
    asyncio.run(test_engine.dispose())


@pytest.fixture(autouse=True)
def inference_double():
    """
    Install the test double (tests/inference_doubles.py) as the inference
    singleton for each test, then restore the previous one. The application has
    no other engine than EfficientNet-B0 and fails closed when it cannot load;
    tests that exercise the real engine replace the singleton themselves.
    """
    import app.services.ai_service as ai_svc
    from tests.inference_doubles import FixedOutputInferenceDouble
    previous = ai_svc._service_singleton
    ai_svc._service_singleton = FixedOutputInferenceDouble()
    yield ai_svc._service_singleton
    ai_svc._service_singleton = previous


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
    
    # get_event_loop() warns from Python 3.12 when no loop is running; a
    # fixture that owns its loop should create it outright.
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

    # Each test starts from an empty database. The suite used to share one
    # in-memory database across tests, which the seeded demonstration account
    # hid; with accounts bootstrapped per test, leftovers would collide.
    async def drop_tables():
        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
    loop.run_until_complete(drop_tables())
    loop.close()


# ---------------------------------------------------------------------------
# Accounts. No account is seeded by the application: the suite bootstraps a
# Super Administrator exactly as a deployment does (the service the start-up
# and the bootstrap command call), then creates physicians through the
# administration API. Every password below is a test value that exists only
# in this file.
# ---------------------------------------------------------------------------
SUPERADMIN_CREDENTIALS = {"username": "root.admin", "email": "root.admin@example.org",
                          "temporary": "Temp-Admin-Pass-2026", "password": "Perm-Admin-Pass-2026"}


def _run(coro):
    import asyncio
    return asyncio.run(coro)


async def _bootstrap_superadmin():
    from app.services import account_service
    async with TestingSessionLocal() as session:
        user, outcome = await account_service.bootstrap_superadmin(
            session, username=SUPERADMIN_CREDENTIALS["username"], email=SUPERADMIN_CREDENTIALS["email"],
            full_name="Test Super Administrator", initial_password=SUPERADMIN_CREDENTIALS["temporary"],
            source="test fixture")
        return user.id if user else None


def sign_in(client, username, password):
    res = client.post("/api/v1/auth/login", json={"username": username, "password": password})
    assert res.status_code == 200, f"fixture login failed for {username}: {res.text}"
    return res.json()


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


def change_password(client, token, current, new):
    res = client.post("/api/v1/auth/change-password", headers=bearer(token), json={
        "currentPassword": current, "newPassword": new, "newPasswordConfirmation": new})
    assert res.status_code == 200, f"fixture password change failed: {res.text}"
    return res.json()


@pytest.fixture
def superadmin_account(test_client):
    """The first Super Administrator, bootstrapped, signed in and past the mandatory password change."""
    user_id = _run(_bootstrap_superadmin())
    assert user_id, "bootstrap created nothing"
    first = sign_in(test_client, SUPERADMIN_CREDENTIALS["username"], SUPERADMIN_CREDENTIALS["temporary"])
    assert first["user"]["mustChangePassword"] is True
    change_password(test_client, first["access_token"], SUPERADMIN_CREDENTIALS["temporary"], SUPERADMIN_CREDENTIALS["password"])
    final = sign_in(test_client, SUPERADMIN_CREDENTIALS["username"], SUPERADMIN_CREDENTIALS["password"])
    return {"id": user_id, "username": SUPERADMIN_CREDENTIALS["username"], "password": SUPERADMIN_CREDENTIALS["password"],
            "token": final["access_token"], "headers": bearer(final["access_token"])}


@pytest.fixture
def superadmin_headers(superadmin_account):
    return superadmin_account["headers"]


def create_physician(client, admin_headers, username, full_name=None, email=None, activate=True):
    """Create a physician through the administration API; returns its id, temporary password and the response."""
    res = client.post("/api/v1/admin/users", headers=admin_headers, json={
        "fullName": full_name or f"Test Physician {username}",
        "username": username,
        "email": email or f"{username}@example.org",
        "licenseNumber": None,
        "facility": "Test Clinic",
        "activateImmediately": activate,
    })
    assert res.status_code == 201, f"fixture physician creation failed: {res.text}"
    body = res.json()
    return body["user"]["id"], body["temporaryPassword"], body


def activated_physician(client, admin_headers, username, password):
    """Create a physician, sign in with the temporary password, change it, sign in again. Returns the account dict."""
    user_id, temporary, _ = create_physician(client, admin_headers, username)
    first = sign_in(client, username, temporary)
    assert first["user"]["mustChangePassword"] is True
    change_password(client, first["access_token"], temporary, password)
    final = sign_in(client, username, password)
    return {"id": user_id, "username": username, "password": password, "token": final["access_token"],
            "headers": bearer(final["access_token"])}


@pytest.fixture
def physician_account(test_client, superadmin_headers):
    """An ACTIVE physician created by the Super Administrator, past the mandatory password change."""
    return activated_physician(test_client, superadmin_headers, "physician.one", "Phys-One-Pass-2026")


@pytest.fixture
def second_physician_account(test_client, superadmin_headers):
    return activated_physician(test_client, superadmin_headers, "physician.two", "Phys-Two-Pass-2026")


@pytest.fixture
def auth_headers(physician_account):
    """
    Bearer header for an authenticated physician session.

    The clinical endpoints require a real token. They previously accepted
    anonymous callers because the dependency returned the seeded demonstration
    account when no token was supplied, so tests passed without authenticating
    and the protection was never exercised.
    """
    return physician_account["headers"]


@pytest.fixture
def authed_client(test_client, auth_headers):
    """A TestClient that sends the physician's bearer token on every request."""
    test_client.headers.update(auth_headers)
    return test_client
