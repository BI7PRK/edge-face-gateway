"""Add system users and revocable web sessions."""
from alembic import op
import sqlalchemy as sa

revision = "0004_web_auth"
down_revision = "0003_mqtt_outbox"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "system_users",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("username", sa.String(length=128), nullable=False, unique=True),
        sa.Column("password_hash", sa.Text(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("must_change_password", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at_ms", sa.Integer(), nullable=False),
        sa.Column("updated_at_ms", sa.Integer(), nullable=False),
    )
    op.create_index("ix_system_users_active", "system_users", ["is_active", "username"])
    op.create_table(
        "web_sessions",
        sa.Column("token_hash", sa.String(length=64), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("system_users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("created_at_ms", sa.Integer(), nullable=False),
        sa.Column("expires_at_ms", sa.Integer(), nullable=False),
    )
    op.create_index("ix_web_sessions_user", "web_sessions", ["user_id"])
    op.create_index("ix_web_sessions_expiry", "web_sessions", ["expires_at_ms"])


def downgrade():
    op.drop_index("ix_web_sessions_expiry", table_name="web_sessions")
    op.drop_index("ix_web_sessions_user", table_name="web_sessions")
    op.drop_table("web_sessions")
    op.drop_index("ix_system_users_active", table_name="system_users")
    op.drop_table("system_users")
