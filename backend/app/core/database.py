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


async def init_db() -> list:
    """Create missing tables, then widen columns the models have made nullable. Returns the columns widened."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        return await conn.run_sync(_relax_stale_not_null)

