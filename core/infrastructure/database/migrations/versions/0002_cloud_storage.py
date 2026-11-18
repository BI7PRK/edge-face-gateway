"""Track cloud storage upload state for captured images."""
from alembic import op
import sqlalchemy as sa

revision = "0002_cloud_storage"
down_revision = "0001_initial_schema"
branch_labels = None
depends_on = None

def upgrade():
    op.add_column("capture_images", sa.Column("cloud_key", sa.Text(), nullable=True))
    op.add_column("capture_images", sa.Column("cloud_uploaded_at_ms", sa.Integer(), nullable=True))
    op.add_column("capture_images", sa.Column("cloud_upload_status", sa.String(24), nullable=True))

def downgrade():
    op.drop_column("capture_images", "cloud_upload_status")
    op.drop_column("capture_images", "cloud_uploaded_at_ms")
    op.drop_column("capture_images", "cloud_key")
