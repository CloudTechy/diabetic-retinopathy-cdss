import os
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base

from app.core.config import settings

connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    connect_args["check_same_thread"] = False

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    future=True,
    pool_pre_ping=not settings.DATABASE_URL.startswith("sqlite"),
    connect_args=connect_args,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)

Base = declarative_base()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency injection helper for database session yielding."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


def _relax_stale_not_null(sync_conn) -> list:
    """
    Drop NOT NULL from columns the models now declare nullable.

    `create_all` creates missing tables and never alters an existing one. When
    `assessments.camera_model` and `assessments.is_mydriatic` became nullable
    ("not recorded"), a database created earlier kept both NOT NULL, and every
    new assessment failed with a NotNullViolation - an HTTP 500 on the first
    request of a deployment whose database predated the change. Widening a
    column to accept NULL loses nothing, so it is done at start-up; nothing is
    ever narrowed or dropped here.
    """
    from sqlalchemy import inspect, text

    if sync_conn.dialect.name != "postgresql":
        return []
    inspector = inspect(sync_conn)
    existing = set(inspector.get_table_names())
    relaxed = []
    for table in Base.metadata.sorted_tables:
        if table.name not in existing:
            continue
        in_db = {c["name"]: c for c in inspector.get_columns(table.name)}
        for column in table.columns:
            found = in_db.get(column.name)
            if found is not None and column.nullable and not found["nullable"] and not column.primary_key:
                sync_conn.execute(text('ALTER TABLE "%s" ALTER COLUMN "%s" DROP NOT NULL' % (table.name, column.name)))
                relaxed.append("%s.%s" % (table.name, column.name))
    return relaxed


def _run_migrations(sync_conn) -> str:
    """
    Bring the schema to the newest Alembic revision.

    Every database so far was created by create_all and carries no Alembic
    version. Such a database is stamped at the baseline and then upgraded, so
    the account-administration columns are added in place and existing rows
    are backfilled; nothing is dropped or re-created. A database that create_all
    has just built from the current models already has every column, and the
    revision's ADD COLUMN steps skip what exists.
    """
    from alembic import command
    from alembic.config import Config
    from alembic.runtime.migration import MigrationContext
    from sqlalchemy import inspect

    cfg = Config(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "alembic.ini"))
    cfg.attributes["connection"] = sync_conn
    inspector = inspect(sync_conn)
    current = MigrationContext.configure(sync_conn).get_current_revision()
    if current is None and "users" in inspector.get_table_names():
        command.stamp(cfg, "0001_baseline")
    command.upgrade(cfg, "head")
    return MigrationContext.configure(sync_conn).get_current_revision() or ""


async def init_db() -> list:
    """Create missing tables, run the migrations, then widen columns the models have made nullable."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        if not settings.DATABASE_URL.startswith("sqlite"):
            await conn.run_sync(_run_migrations)
        return await conn.run_sync(_relax_stale_not_null)

