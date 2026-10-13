"""
Create the first Super Administrator.

    python backend/scripts/bootstrap_superadmin.py                 # interactive prompts
    SUPERADMIN_USERNAME=... SUPERADMIN_EMAIL=... SUPERADMIN_INITIAL_PASSWORD=... \
        python backend/scripts/bootstrap_superadmin.py             # from the environment

Idempotent: if any Super Administrator exists, nothing is created and the
command says so. The password given here is temporary and must be changed at
the first sign-in. No credential is written anywhere but the password hash.
"""
import asyncio
import getpass
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings                      # noqa: E402
from app.core.database import AsyncSessionLocal, init_db  # noqa: E402
import app.models.models                                   # noqa: E402,F401  (register the tables)
from app.services import account_service                   # noqa: E402
from app.services.account_service import AccountError      # noqa: E402


async def main() -> int:
    await init_db()
    username = settings.SUPERADMIN_USERNAME or input("Super Administrator username: ").strip()
    email = settings.SUPERADMIN_EMAIL or input("Super Administrator email: ").strip()
    full_name = settings.SUPERADMIN_FULL_NAME or "Super Administrator"
    password = settings.SUPERADMIN_INITIAL_PASSWORD or getpass.getpass("Temporary password (changed at first sign-in): ")
    source = "environment" if settings.SUPERADMIN_INITIAL_PASSWORD else "interactive bootstrap command"
    async with AsyncSessionLocal() as db:
        try:
            user, outcome = await account_service.bootstrap_superadmin(
                db, username=username, email=email, full_name=full_name, initial_password=password, source=source)
        except AccountError as exc:
            print("REFUSED: %s" % exc)
            return 1
    if user is None:
        print("NOTHING CREATED: %s" % outcome)
        return 0
    print("CREATED Super Administrator '%s' (id %s). The password is temporary: change it at the first sign-in." % (
        user.username, user.id))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
