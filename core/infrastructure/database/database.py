from __future__ import annotations

import contextlib
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, event, inspect
from sqlalchemy.engine import URL
from sqlalchemy.orm import sessionmaker

from .models import Base


# Schema signatures are intentionally explicit. A database without alembic_version
# is stamped only when its physical schema matches one of these known revisions.
_REQUIRED_COLUMNS: dict[str, set[str]] = {
    "cameras": {"id", "name", "enabled", "main_url_encrypted", "sub_url_encrypted", "stream_mode",
                "detection_fps", "config_json", "revision", "created_at_ms", "updated_at_ms"},
    "camera_status": {"camera_id", "status", "last_connected_at_ms", "last_disconnected_at_ms", "last_error",
                       "reconnect_count", "current_fps", "worker_state", "updated_at_ms"},
    "capture_images": {"id", "capture_id", "camera_id", "parent_image_id", "image_type", "file_path",
                        "captured_at_ms", "confidence", "quality_score", "width", "height", "file_size", "created_at_ms"},
    "application_logs": {"id", "timestamp_ms", "level", "logger", "category", "camera_id", "message", "details_json", "exception"},
    "business_events": {"id", "event_type", "timestamp_ms", "camera_id", "capture_id", "status", "payload_json"},
    "system_settings": {"key", "value_json", "updated_at_ms"},
}
_CLOUD_COLUMNS = {"cloud_key", "cloud_uploaded_at_ms", "cloud_upload_status"}
_AUTH_COLUMNS: dict[str, set[str]] = {
    "system_users": {"id", "username", "password_hash", "is_active", "must_change_password",
                     "created_at_ms", "updated_at_ms"},
    "web_sessions": {"token_hash", "user_id", "created_at_ms", "expires_at_ms"},
}
_AUTH_INDEXES = {
    "system_users": {"ix_system_users_active"},
    "web_sessions": {"ix_web_sessions_user", "ix_web_sessions_expiry"},
}
_REQUIRED_INDEXES = {
    "capture_images": {"ix_capture_images_capture_id", "ix_capture_images_camera_id", "ix_capture_camera_time", "ix_capture_type_time"},
    "application_logs": {"ix_logs_time", "ix_logs_camera_time", "ix_logs_level_time"},
    "business_events": {"ix_events_time", "ix_events_type_time", "ix_events_camera_time"},
}
_KNOWN_REVISIONS = ("0001_initial_schema", "0002_cloud_storage", "0003_mqtt_outbox", "0004_web_auth")
log = logging.getLogger(__name__)


class DatabaseMigrationError(RuntimeError):
    """Raised when an existing database cannot be safely migrated automatically."""


@contextlib.contextmanager
def _migration_file_lock(path: Path, timeout: float = 120.0) -> Iterator[None]:
    """Cross-process lock; prevents two application instances migrating at once."""
    lock_path = path.with_name(path.name + ".migrate.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    deadline = time.monotonic() + timeout
    locked = False
    try:
        while not locked:
            try:
                if os.name == "nt":
                    import msvcrt
                    handle.seek(0)
                    # Ensure a byte exists for the Windows byte-range lock.
                    if handle.read(1) == b"":
                        handle.seek(0)
                        handle.write(b"0")
                        handle.flush()
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
            except (BlockingIOError, OSError):
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for database migration lock: {lock_path}")
                time.sleep(0.2)
        yield
    finally:
        if locked:
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(
            URL.create("sqlite", database=str(self.path)),
            connect_args={"check_same_thread": False, "timeout": 5.0},
            pool_pre_ping=True,
        )

        @event.listens_for(self.engine, "connect")
        def _sqlite_pragmas(dbapi_connection, _record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.close()

        self.sessions = sessionmaker(bind=self.engine, expire_on_commit=False)

    def _alembic_config(self) -> Config:
        root = Path(__file__).resolve().parents[3]
        cfg = Config(str(root / "alembic.ini"))
        url = URL.create("sqlite", database=str(self.path)).render_as_string(hide_password=False)
        # ConfigParser treats '%' as interpolation syntax.
        cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
        return cfg

    def _tables(self) -> set[str]:
        with self.engine.connect() as connection:
            return set(inspect(connection).get_table_names())

    def _detect_unversioned_revision(self) -> str | None:
        """Return a known revision only for a complete, matching legacy schema."""
        with self.engine.connect() as connection:
            inspector = inspect(connection)
            tables = set(inspector.get_table_names())
            if not tables:
                return None

            missing_tables = set(_REQUIRED_COLUMNS) - tables
            if missing_tables:
                raise DatabaseMigrationError(
                    "Existing SQLite database has no alembic_version and does not match a known schema. "
                    f"Missing tables: {', '.join(sorted(missing_tables))}. "
                    "The database was left unchanged; restore a backup or provide a schema-specific migration."
                )

            missing_columns: list[str] = []
            for table, expected in _REQUIRED_COLUMNS.items():
                actual = {column["name"] for column in inspector.get_columns(table)}
                absent = expected - actual
                if absent:
                    missing_columns.extend(f"{table}.{name}" for name in sorted(absent))
            if missing_columns:
                raise DatabaseMigrationError(
                    "Existing SQLite database has no alembic_version and is structurally incomplete. "
                    f"Missing columns: {', '.join(missing_columns)}. Database was left unchanged."
                )

            missing_indexes: list[str] = []
            for table, expected in _REQUIRED_INDEXES.items():
                actual = {index["name"] for index in inspector.get_indexes(table)}
                missing_indexes.extend(f"{table}.{name}" for name in sorted(expected - actual))
            if missing_indexes:
                raise DatabaseMigrationError(
                    "Existing SQLite database is missing indexes required by the known schema: "
                    f"{', '.join(missing_indexes)}. Database was left unchanged."
                )

            present_auth_tables = set(_AUTH_COLUMNS) & tables
            has_auth_schema = bool(present_auth_tables)
            if present_auth_tables:
                if present_auth_tables != set(_AUTH_COLUMNS):
                    raise DatabaseMigrationError("Existing SQLite database has a partial authentication schema; database was left unchanged.")
                missing_auth_columns = []
                missing_auth_indexes = []
                for table, expected in _AUTH_COLUMNS.items():
                    actual_columns = {column["name"] for column in inspector.get_columns(table)}
                    missing_auth_columns.extend(f"{table}.{name}" for name in sorted(expected - actual_columns))
                    actual_indexes = {index["name"] for index in inspector.get_indexes(table)}
                    missing_auth_indexes.extend(f"{table}.{name}" for name in sorted(_AUTH_INDEXES[table] - actual_indexes))
                if missing_auth_columns or missing_auth_indexes:
                    details = missing_auth_columns + missing_auth_indexes
                    raise DatabaseMigrationError(
                        f"Existing SQLite authentication schema is incomplete ({', '.join(details)}); database was left unchanged."
                    )

            capture_columns = {column["name"] for column in inspector.get_columns("capture_images")}
            present_cloud_columns = _CLOUD_COLUMNS & capture_columns
            if present_cloud_columns and present_cloud_columns != _CLOUD_COLUMNS:
                raise DatabaseMigrationError(
                    "Existing SQLite database has a partial cloud-storage schema "
                    f"({', '.join(sorted(present_cloud_columns))}). Database was left unchanged."
                )
            if present_cloud_columns == _CLOUD_COLUMNS:
                if "mqtt_outbox" in tables:
                    outbox_columns = {column["name"] for column in inspector.get_columns("mqtt_outbox")}
                    expected_outbox = {"event_id", "topic", "payload_json", "qos", "retain", "status",
                                       "attempts", "next_attempt_at_ms", "last_error", "created_at_ms", "published_at_ms"}
                    if not expected_outbox.issubset(outbox_columns):
                        raise DatabaseMigrationError("Existing mqtt_outbox table is incomplete; database was left unchanged.")
                    outbox_indexes = {index["name"] for index in inspector.get_indexes("mqtt_outbox")}
                    if "ix_mqtt_outbox_due" not in outbox_indexes:
                        raise DatabaseMigrationError("Existing mqtt_outbox table is missing ix_mqtt_outbox_due; database was left unchanged.")
                    return "0004_web_auth" if has_auth_schema else "0003_mqtt_outbox"
                if has_auth_schema:
                    raise DatabaseMigrationError("Authentication tables require the MQTT outbox schema; database was left unchanged.")
                return "0002_cloud_storage"
            if has_auth_schema:
                raise DatabaseMigrationError("Authentication tables require the MQTT outbox schema; database was left unchanged.")
            return "0001_initial_schema"

    def _backup(self) -> Path:
        """Create a consistent online backup, including committed WAL contents."""
        backup_dir = self.path.parent / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        backup_path = backup_dir / f"{self.path.stem}-{stamp}.sqlite3"
        source = sqlite3.connect(str(self.path), timeout=30)
        destination = sqlite3.connect(str(backup_path))
        try:
            source.backup(destination)
            result = destination.execute("PRAGMA quick_check").fetchone()
            if not result or result[0] != "ok":
                raise DatabaseMigrationError(f"Database backup integrity check failed: {result}")
        except Exception:
            destination.close()
            source.close()
            backup_path.unlink(missing_ok=True)
            raise
        else:
            destination.close()
            source.close()
        return backup_path

    def _has_user_schema(self) -> bool:
        return bool(self._tables() - {"alembic_version", "sqlite_sequence"})

    def migrate(self) -> None:
        """Safely migrate without dropping/recreating an existing application database."""
        with _migration_file_lock(self.path):
            cfg = self._alembic_config()
            tables = self._tables()
            has_version_table = "alembic_version" in tables
            needs_backup = self._has_user_schema()

            if has_version_table:
                with self.engine.connect() as connection:
                    rows = connection.exec_driver_sql("SELECT version_num FROM alembic_version").fetchall()
                if len(rows) == 0:
                    # Older startup code could create application tables while failing
                    # to commit Alembic's version row. Treat an empty marker table as
                    # unversioned and identify the real schema before stamping it.
                    has_version_table = False
                    log.warning("Found empty alembic_version table; inspecting the existing schema")
                elif len(rows) != 1 or rows[0][0] not in _KNOWN_REVISIONS:
                    raise DatabaseMigrationError(
                        f"Unsupported or invalid alembic_version state: {[row[0] for row in rows]}. "
                        "Database was not modified."
                    )
                else:
                    detected = self._detect_unversioned_revision()
                    if detected != rows[0][0]:
                        raise DatabaseMigrationError(
                            f"Database schema does not match its alembic_version marker "
                            f"({rows[0][0]} vs detected {detected}). Database was not modified."
                        )
            if not has_version_table and needs_backup:
                detected = self._detect_unversioned_revision()
                if detected is None:
                    raise DatabaseMigrationError("Could not identify existing database schema; refusing to initialize over it.")
                backup_path = self._backup()
                command.stamp(cfg, detected)
                # Keep the backup path in logs/diagnostics and continue the forward migration.
                log.warning("Recognized unversioned database as %s; backup created at %s", detected, backup_path)
            # New database (or an empty version marker left by an interrupted first
            # startup): no application tables exist, so the initial migration is safe.

            # Back up before any forward schema change. Do not make redundant backups
            # when the database is already at head.
            current = None
            if "alembic_version" in self._tables():
                with self.engine.connect() as connection:
                    row = connection.exec_driver_sql("SELECT version_num FROM alembic_version").first()
                    current = row[0] if row else None
            if current != _KNOWN_REVISIONS[-1] and needs_backup and has_version_table:
                backup_path = self._backup()
                log.info("Database backup created before migration: %s", backup_path)

            command.upgrade(cfg, "head")
            with self.engine.connect() as connection:
                check = connection.exec_driver_sql("PRAGMA quick_check").scalar_one()
                if check != "ok":
                    raise DatabaseMigrationError(f"SQLite integrity check failed after migration: {check}")
                foreign_key_errors = connection.exec_driver_sql("PRAGMA foreign_key_check").fetchall()
                if foreign_key_errors:
                    raise DatabaseMigrationError(
                        f"SQLite foreign key check failed after migration: {foreign_key_errors[:5]}"
                    )

    def create_schema(self) -> None:
        # Test helper only. Production startup must use Alembic migrations.
        Base.metadata.create_all(self.engine)

    def close(self) -> None:
        self.engine.dispose()
