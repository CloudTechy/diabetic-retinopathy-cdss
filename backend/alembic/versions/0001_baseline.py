"""Baseline: the schema as created by Base.metadata.create_all before account administration.

Every database that exists today was created by create_all, never by Alembic,
so this revision creates nothing. It marks the state that 0002 migrates from.
A database that is still empty is created by create_all at start-up and then
stamped at the newest revision.

Revision ID: 0001_baseline
Revises:
Create Date: 2026-10-12
"""
revision = "0001_baseline"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
