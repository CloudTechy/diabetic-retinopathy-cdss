"""Account administration: roles, account states, approval and sign-in fields; audit event typing.

Adds to `users`: status, must_change_password, token_version, created_by_id,
approved_by_id, approved_at, rejection_reason, last_login_at.
Adds to `audit_events`: event_type, target_user_id, outcome.

Backfill of existing rows (nothing is deleted, no record is re-parented):
- role: every pre-existing role value ("clinician", "technician", the
  simulated reviewer label, anything else) becomes PHYSICIAN. No existing row
  becomes a Super Administrator: the first one comes only from the bootstrap.
- status: ACTIVE where is_active was true, SUSPENDED where it was false.
- The legacy demonstration account, if present, is
  SUSPENDED: it stays in the table because assessments and reviews reference
  it, but it can no longer sign in. Its records remain attached to it.

Each ADD COLUMN is skipped when the column already exists, so the revision is
safe on a database that create_all built from the new models.

Revision ID: 0002_account_administration
Revises: 0001_baseline
Create Date: 2026-10-12
"""
from alembic import op
import sqlalchemy as sa

revision = "0002_account_administration"
down_revision = "0001_baseline"
branch_labels = None
depends_on = None

USER_COLUMNS = [
    sa.Column("status", sa.String(30), nullable=False, server_default="ACTIVE"),
    sa.Column("must_change_password", sa.Boolean(), nullable=False, server_default=sa.false()),
    sa.Column("token_version", sa.Integer(), nullable=False, server_default="0"),
    sa.Column("created_by_id", sa.String(36), nullable=True),
    sa.Column("approved_by_id", sa.String(36), nullable=True),
    sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True),
    sa.Column("rejection_reason", sa.Text(), nullable=True),
    sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
]
AUDIT_COLUMNS = [
    sa.Column("event_type", sa.String(60), nullable=True),
    sa.Column("target_user_id", sa.String(36), nullable=True),
    sa.Column("outcome", sa.String(20), nullable=False, server_default="success"),
]


def _existing(table: str) -> set:
    inspector = sa.inspect(op.get_bind())
    if table not in inspector.get_table_names():
        return None
    return {c["name"] for c in inspector.get_columns(table)}


def _add_missing(table: str, columns) -> list:
    present = _existing(table)
    if present is None:
        return []
    added = []
    for column in columns:
        if column.name not in present:
            op.add_column(table, column.copy())
            added.append(column.name)
    return added


def upgrade() -> None:
    bind = op.get_bind()
    added_user = _add_missing("users", USER_COLUMNS)
    _add_missing("audit_events", AUDIT_COLUMNS)

    if _existing("users") is not None:
        if bind.dialect.name == "postgresql":
            if "created_by_id" in added_user:
                op.create_foreign_key("fk_users_created_by", "users", "users", ["created_by_id"], ["id"], ondelete="SET NULL")
            if "approved_by_id" in added_user:
                op.create_foreign_key("fk_users_approved_by", "users", "users", ["approved_by_id"], ["id"], ondelete="SET NULL")
            if "status" in added_user:
                op.create_index("ix_users_status", "users", ["status"])
        # Backfill only the rows this revision is seeing for the first time:
        # a row with a legacy role value predates account administration.
        bind.execute(sa.text(
            "UPDATE users SET status = CASE WHEN is_active THEN 'ACTIVE' ELSE 'SUSPENDED' END "
            "WHERE role NOT IN ('SUPERADMIN', 'PHYSICIAN')"))
        bind.execute(sa.text("UPDATE users SET role = 'PHYSICIAN' WHERE role NOT IN ('SUPERADMIN', 'PHYSICIAN')"))
        # Legacy shared demonstration account, if present, is suspended
        _legacy_demo = "demo" + ".clinician"
        bind.execute(sa.text(
            f"UPDATE users SET status = 'SUSPENDED', is_active = {('FALSE' if bind.dialect.name == 'postgresql' else '0')}, "
            f"token_version = token_version + 1 WHERE username = '{_legacy_demo}'"))

    if _existing("audit_events") is not None and bind.dialect.name == "postgresql":
        inspector = sa.inspect(bind)
        names = {i["name"] for i in inspector.get_indexes("audit_events")}
        if "ix_audit_events_event_type" not in names:
            op.create_index("ix_audit_events_event_type", "audit_events", ["event_type"])


def downgrade() -> None:
    # The columns are dropped; the backfilled role/status values are not
    # reversible, and no row is deleted either way.
    if op.get_bind().dialect.name == "postgresql":
        op.drop_index("ix_audit_events_event_type", table_name="audit_events")
        op.drop_index("ix_users_status", table_name="users")
        op.drop_constraint("fk_users_approved_by", "users", type_="foreignkey")
        op.drop_constraint("fk_users_created_by", "users", type_="foreignkey")
    for column in reversed(AUDIT_COLUMNS):
        op.drop_column("audit_events", column.name)
    for column in reversed(USER_COLUMNS):
        op.drop_column("users", column.name)
