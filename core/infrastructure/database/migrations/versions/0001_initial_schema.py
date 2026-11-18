"""Initial edge-face-gateway persistence schema."""
from alembic import op
import sqlalchemy as sa
revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    op.create_table("cameras",
        sa.Column("id", sa.String(128), primary_key=True), sa.Column("name", sa.String(200), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False), sa.Column("main_url_encrypted", sa.Text(), nullable=False),
        sa.Column("sub_url_encrypted", sa.Text()), sa.Column("stream_mode", sa.String(16), nullable=False),
        sa.Column("detection_fps", sa.Integer(), nullable=False), sa.Column("config_json", sa.Text(), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False), sa.Column("created_at_ms", sa.Integer(), nullable=False),
        sa.Column("updated_at_ms", sa.Integer(), nullable=False))
    op.create_table("camera_status",
        sa.Column("camera_id", sa.String(128), sa.ForeignKey("cameras.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("status", sa.String(24), nullable=False), sa.Column("last_connected_at_ms", sa.Integer()),
        sa.Column("last_disconnected_at_ms", sa.Integer()), sa.Column("last_error", sa.Text()),
        sa.Column("reconnect_count", sa.Integer(), nullable=False), sa.Column("current_fps", sa.Float(), nullable=False),
        sa.Column("worker_state", sa.String(24), nullable=False), sa.Column("updated_at_ms", sa.Integer(), nullable=False))
    op.create_table("capture_images",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True), sa.Column("capture_id", sa.String(64), nullable=False),
        sa.Column("camera_id", sa.String(128), nullable=False), sa.Column("parent_image_id", sa.Integer(), sa.ForeignKey("capture_images.id", ondelete="SET NULL")),
        sa.Column("image_type", sa.String(24), nullable=False), sa.Column("file_path", sa.Text(), nullable=False, unique=True),
        sa.Column("captured_at_ms", sa.Integer(), nullable=False), sa.Column("confidence", sa.Float()),
        sa.Column("quality_score", sa.Float()), sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False), sa.Column("file_size", sa.Integer(), nullable=False),
        sa.Column("created_at_ms", sa.Integer(), nullable=False))
    op.create_index("ix_capture_images_capture_id", "capture_images", ["capture_id"])
    op.create_index("ix_capture_images_camera_id", "capture_images", ["camera_id"])
    op.create_index("ix_capture_camera_time", "capture_images", ["camera_id", "captured_at_ms", "id"])
    op.create_index("ix_capture_type_time", "capture_images", ["image_type", "captured_at_ms", "id"])
    op.create_table("application_logs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True), sa.Column("timestamp_ms", sa.Integer(), nullable=False),
        sa.Column("level", sa.String(16), nullable=False), sa.Column("logger", sa.String(200), nullable=False),
        sa.Column("category", sa.String(32), nullable=False), sa.Column("camera_id", sa.String(128)),
        sa.Column("message", sa.Text(), nullable=False), sa.Column("details_json", sa.Text()), sa.Column("exception", sa.Text()))
    op.create_index("ix_logs_time", "application_logs", ["timestamp_ms", "id"])
    op.create_index("ix_logs_camera_time", "application_logs", ["camera_id", "timestamp_ms", "id"])
    op.create_index("ix_logs_level_time", "application_logs", ["level", "timestamp_ms", "id"])
    op.create_table("business_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True), sa.Column("event_type", sa.String(40), nullable=False),
        sa.Column("timestamp_ms", sa.Integer(), nullable=False), sa.Column("camera_id", sa.String(128)),
        sa.Column("capture_id", sa.String(64)), sa.Column("status", sa.String(24)), sa.Column("payload_json", sa.Text(), nullable=False))
    op.create_index("ix_events_time", "business_events", ["timestamp_ms", "id"])
    op.create_index("ix_events_type_time", "business_events", ["event_type", "timestamp_ms", "id"])
    op.create_index("ix_events_camera_time", "business_events", ["camera_id", "timestamp_ms", "id"])
    op.create_table("system_settings", sa.Column("key", sa.String(128), primary_key=True),
                    sa.Column("value_json", sa.Text(), nullable=False), sa.Column("updated_at_ms", sa.Integer(), nullable=False))

def downgrade():
    op.drop_table("system_settings")
    op.drop_index("ix_events_camera_time", table_name="business_events"); op.drop_index("ix_events_type_time", table_name="business_events"); op.drop_index("ix_events_time", table_name="business_events"); op.drop_table("business_events")
    op.drop_index("ix_logs_level_time", table_name="application_logs"); op.drop_index("ix_logs_camera_time", table_name="application_logs"); op.drop_index("ix_logs_time", table_name="application_logs"); op.drop_table("application_logs")
    op.drop_index("ix_capture_type_time", table_name="capture_images"); op.drop_index("ix_capture_camera_time", table_name="capture_images"); op.drop_index("ix_capture_images_camera_id", table_name="capture_images"); op.drop_index("ix_capture_images_capture_id", table_name="capture_images"); op.drop_table("capture_images")
    op.drop_table("camera_status"); op.drop_table("cameras")
