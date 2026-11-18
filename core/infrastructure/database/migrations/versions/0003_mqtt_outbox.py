"""Add durable MQTT event outbox."""
from alembic import op
import sqlalchemy as sa
import json

revision = "0003_mqtt_outbox"
down_revision = "0002_cloud_storage"
branch_labels = None
depends_on = None

def upgrade():
    op.create_table(
        "mqtt_outbox",
        sa.Column("event_id", sa.String(64), primary_key=True),
        sa.Column("topic", sa.Text(), nullable=False),
        sa.Column("payload_json", sa.Text(), nullable=False),
        sa.Column("qos", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("retain", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("next_attempt_at_ms", sa.Integer(), nullable=False),
        sa.Column("last_error", sa.Text()),
        sa.Column("created_at_ms", sa.Integer(), nullable=False),
        sa.Column("published_at_ms", sa.Integer()),
    )
    op.create_index("ix_mqtt_outbox_due", "mqtt_outbox", ["status", "next_attempt_at_ms", "created_at_ms"])

    # Remove obsolete encrypted RabbitMQ credentials/settings while preserving S3 settings.
    connection = op.get_bind()
    row = connection.execute(sa.text("SELECT value_json FROM system_settings WHERE key = 'cloud_services'")).first()
    if row is not None:
        try:
            config = json.loads(row[0])
            if isinstance(config, dict) and "rmq" in config:
                config.pop("rmq", None)
                connection.execute(sa.text(
                    "UPDATE system_settings SET value_json = :value WHERE key = 'cloud_services'"
                ), {"value": json.dumps(config, ensure_ascii=False)})
        except (TypeError, ValueError, json.JSONDecodeError):
            # Keep an unrecognized legacy settings blob untouched; runtime ignores rmq keys.
            pass

def downgrade():
    op.drop_index("ix_mqtt_outbox_due", table_name="mqtt_outbox")
    op.drop_table("mqtt_outbox")
