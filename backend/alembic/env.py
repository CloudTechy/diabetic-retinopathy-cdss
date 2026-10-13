"""
Alembic environment. The database URL comes from the application settings so
the migration runs against whatever the application itself would open:

    cd backend && alembic upgrade head

The async driver suffix is dropped because Alembic runs synchronously
(postgresql+asyncpg -> postgresql via psycopg2; sqlite+aiosqlite -> sqlite).
"""
import os
import sys
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings          # noqa: E402
from app.core.database import Base            # noqa: E402
import app.models.models                      # noqa: E402,F401

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def sync_url() -> str:
    url = os.environ.get("ALEMBIC_DATABASE_URL") or settings.DATABASE_URL
    return url.replace("+asyncpg", "").replace("+aiosqlite", "")


def run_migrations_offline() -> None:
    context.configure(url=sync_url(), target_metadata=target_metadata, literal_binds=True,
                      dialect_opts={"paramstyle": "named"}, render_as_batch=True)
    with context.begin_transaction():
        context.run_migrations()


def _migrate(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata, render_as_batch=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    # At application start-up the connection is handed in (app/core/database.py);
    # from the command line one is opened here.
    connection = config.attributes.get("connection")
    if connection is not None:
        _migrate(connection)
        return
    connectable = engine_from_config({"sqlalchemy.url": sync_url()}, prefix="sqlalchemy.", poolclass=pool.NullPool)
    with connectable.connect() as connection:
        _migrate(connection)
        connection.commit()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
