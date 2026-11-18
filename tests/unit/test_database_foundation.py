import logging
from pathlib import Path

from core.infrastructure.database.database import Database
from core.infrastructure.database.camera_repository import CameraRepository
from core.infrastructure.database.models import ApplicationLog, CaptureImage, Camera
from core.infrastructure.database.pagination import Page, paginate
from core.infrastructure.logging.sqlite_handler import SQLiteLogHandler
from core.infrastructure.security import SecretCipher
from sqlalchemy import select


def test_alembic_migration_creates_schema_and_camera_seed_is_database_backed(tmp_path: Path):
    db = Database(tmp_path / "database" / "gateway.db")
    db.migrate()
    cipher = SecretCipher(tmp_path / "secrets" / "camera.key")
    repo = CameraRepository(cipher)
    config = {"cameras": [{"id": "front-01", "name": "Front", "enabled": True,
        "streams": {"main": {"url": "rtsp://user:secret@127.0.0.1/main", "enabled": True},
                    "sub": {"url": "rtsp://127.0.0.1/sub", "enabled": True}}}]}
    with db.sessions.begin() as session:
        assert repo.seed_from_legacy_config(session, config) == 1
    with db.sessions() as session:
        row = session.get(Camera, "front-01")
        assert row is not None
        assert "secret" not in row.main_url_encrypted
        cameras = repo.list_enabled(session)
        assert cameras[0].main_url == "rtsp://user:secret@127.0.0.1/main"
        assert cameras[0].mode == "dual"
    with db.sessions.begin() as session:
        assert repo.seed_from_legacy_config(session, config) == 0
    db.close()


def test_pagination_is_stable_and_has_metadata(tmp_path: Path):
    db = Database(tmp_path / "gateway.db")
    db.create_schema()
    with db.sessions.begin() as session:
        for i in range(5):
            session.add(ApplicationLog(timestamp_ms=i, level="INFO", logger="test", category="SYSTEM", message=str(i)))
    with db.sessions() as session:
        result = paginate(session, ApplicationLog, Page(2, 2), order_by=(ApplicationLog.timestamp_ms, ApplicationLog.id))
        assert [x.message for x in result["items"]] == ["2", "3"]
        assert result["pagination"] == {"page": 2, "page_size": 2, "total": 5,
            "total_pages": 3, "has_previous": True, "has_next": True}
    db.close()


def test_sqlite_logging_handler_persists_records(tmp_path: Path):
    db = Database(tmp_path / "gateway.db")
    db.create_schema()
    handler = SQLiteLogHandler(db, capacity=100, batch_size=10)
    logger = logging.getLogger("test.edge-face.db")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.info("persistent log", extra={"category": "CAMERA", "camera_id": "front-01"})
    handler.flush()
    with db.sessions() as session:
        rows = session.scalars(select(ApplicationLog)).all()
        assert len(rows) == 1
        assert rows[0].message == "persistent log"
        assert rows[0].category == "CAMERA"
        assert rows[0].camera_id == "front-01"
    logger.removeHandler(handler)
    handler.close(); db.close()


def test_image_repository_persists_metadata_and_relative_path(tmp_path: Path):
    from core.infrastructure.storage.image_repository import ImageRepository
    from core.infrastructure.database.models import CaptureImage
    db = Database(tmp_path / "gateway.db")
    db.create_schema()
    root = tmp_path / "images"
    image_dir = root / "camera-01"
    image_dir.mkdir(parents=True)
    image = image_dir / "face.jpg"
    image.write_bytes(b"jpeg-placeholder")
    repo = ImageRepository(db, root)
    image_id = repo.register(capture_id="capture-1", camera_id="camera-01", image_type="face",
        path=image, captured_at_ms=1234, width=20, height=30, confidence=0.95, quality_score=88.0)
    with db.sessions() as session:
        row = session.get(CaptureImage, image_id)
        assert row.file_path == "camera-01/face.jpg"
        assert row.file_size == len(b"jpeg-placeholder")
        assert row.confidence == 0.95
    db.close()


def test_legacy_image_and_jsonl_events_are_imported_once(tmp_path: Path):
    import json
    import cv2
    import numpy as np
    from core.infrastructure.database.event_repository import SQLiteEventLogger
    from core.infrastructure.database.models import BusinessEvent
    from core.infrastructure.storage.image_repository import ImageRepository
    db = Database(tmp_path / "gateway.db")
    db.create_schema()
    root = tmp_path / "data"
    camera = root / "camera-01"
    camera.mkdir(parents=True)
    stamp = "20261118T100000_000000Z"
    for suffix in ("_frame.jpg", "_detections.jpg", "_face_001_score_0.930.jpg"):
        assert cv2.imwrite(str(camera / f"{stamp}{suffix}"), np.zeros((40, 50, 3), dtype=np.uint8))
    images = ImageRepository(db, root)
    assert images.import_legacy_files() == 3
    assert images.import_legacy_files() == 0
    legacy = tmp_path / "runtime.jsonl"
    legacy.write_text(json.dumps({"event_type": "CaptureEvent", "ts": "2026-11-18T10:00:00Z", "camera_id": "camera-01"}) + "\n", encoding="utf-8")
    events = SQLiteEventLogger(db)
    assert events.import_legacy_jsonl(str(legacy)) == 1
    assert events.import_legacy_jsonl(str(legacy)) == 0
    with db.sessions() as session:
        rows = session.scalars(select(CaptureImage).order_by(CaptureImage.id)).all()
        assert [row.image_type for row in rows] == ["frame", "annotated", "face"]
        assert rows[2].parent_image_id == rows[0].id
        assert session.query(BusinessEvent).count() == 1
    db.close()


def test_existing_unversioned_v0001_database_is_stamped_and_upgraded_without_recreating_tables(tmp_path: Path):
    from alembic import command
    db_path = tmp_path / "database" / "gateway.db"
    db = Database(db_path)
    cfg = db._alembic_config()
    command.upgrade(cfg, "0001_initial_schema")
    with db.engine.begin() as connection:
        connection.exec_driver_sql("INSERT INTO cameras (id, name, enabled, main_url_encrypted, stream_mode, detection_fps, config_json, revision, created_at_ms, updated_at_ms) VALUES ('front', 'Front', 1, 'ciphertext', 'single', 3, '{}', 1, 1, 1)")
        connection.exec_driver_sql("DROP TABLE alembic_version")

    db.migrate()

    with db.engine.connect() as connection:
        assert connection.exec_driver_sql("SELECT version_num FROM alembic_version").scalar_one() == "0003_mqtt_outbox"
        assert connection.exec_driver_sql("SELECT name FROM cameras WHERE id='front'").scalar_one() == "Front"
        columns = {row[1] for row in connection.exec_driver_sql("PRAGMA table_info(capture_images)")}
        assert {"cloud_key", "cloud_uploaded_at_ms", "cloud_upload_status"}.issubset(columns)
    backups = list((db_path.parent / "backups").glob("gateway-*.sqlite3"))
    assert len(backups) == 1
    # A second startup must be idempotent and must not create another backup.
    db.migrate()
    assert len(list((db_path.parent / "backups").glob("gateway-*.sqlite3"))) == 1
    db.close()


def test_existing_unversioned_current_database_is_stamped_without_running_initial_migration(tmp_path: Path):
    db_path = tmp_path / "gateway.db"
    db = Database(db_path)
    db.create_schema()
    with db.engine.begin() as connection:
        connection.exec_driver_sql("INSERT INTO cameras (id, name, enabled, main_url_encrypted, stream_mode, detection_fps, config_json, revision, created_at_ms, updated_at_ms) VALUES ('rear', 'Rear', 1, 'ciphertext', 'single', 3, '{}', 1, 1, 1)")
        connection.exec_driver_sql("DROP TABLE alembic_version") if "alembic_version" in {r[0] for r in connection.exec_driver_sql("SELECT name FROM sqlite_master WHERE type='table'")} else None

    db.migrate()

    with db.engine.connect() as connection:
        assert connection.exec_driver_sql("SELECT version_num FROM alembic_version").scalar_one() == "0003_mqtt_outbox"
        assert connection.exec_driver_sql("SELECT name FROM cameras WHERE id='rear'").scalar_one() == "Rear"
    db.close()


def test_unrecognized_partial_database_fails_safely_without_dropping_existing_table(tmp_path: Path):
    from core.infrastructure.database.database import DatabaseMigrationError
    db = Database(tmp_path / "gateway.db")
    with db.engine.begin() as connection:
        connection.exec_driver_sql("CREATE TABLE cameras (id TEXT PRIMARY KEY)")
        connection.exec_driver_sql("INSERT INTO cameras (id) VALUES ('keep-me')")

    import pytest
    with pytest.raises(DatabaseMigrationError, match="does not match a known schema"):
        db.migrate()

    with db.engine.connect() as connection:
        assert connection.exec_driver_sql("SELECT id FROM cameras").scalar_one() == "keep-me"
        assert "alembic_version" not in {row[0] for row in connection.exec_driver_sql("SELECT name FROM sqlite_master WHERE type='table'")}
    db.close()


def test_concurrent_startup_migration_is_serialized(tmp_path: Path):
    from concurrent.futures import ThreadPoolExecutor
    path = tmp_path / "gateway.db"
    first = Database(path)
    second = Database(path)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda database: database.migrate(), (first, second)))
    assert results == [None, None]
    with first.engine.connect() as connection:
        assert connection.exec_driver_sql("SELECT version_num FROM alembic_version").scalar_one() == "0003_mqtt_outbox"
    first.close()
    second.close()


def test_empty_alembic_version_table_with_existing_schema_is_recovered(tmp_path: Path):
    from alembic import command
    db_path = tmp_path / "gateway.db"
    db = Database(db_path)
    command.upgrade(db._alembic_config(), "0001_initial_schema")
    with db.engine.begin() as connection:
        connection.exec_driver_sql("INSERT INTO cameras (id, name, enabled, main_url_encrypted, stream_mode, detection_fps, config_json, revision, created_at_ms, updated_at_ms) VALUES ('existing', 'Existing', 1, 'ciphertext', 'single', 3, '{}', 1, 1, 1)")
        connection.exec_driver_sql("DELETE FROM alembic_version")

    db.migrate()

    with db.engine.connect() as connection:
        assert connection.exec_driver_sql("SELECT version_num FROM alembic_version").scalar_one() == "0003_mqtt_outbox"
        assert connection.exec_driver_sql("SELECT name FROM cameras WHERE id='existing'").scalar_one() == "Existing"
    assert len(list((db_path.parent / "backups").glob("gateway-*.sqlite3"))) == 1
    db.close()


def test_empty_database_with_empty_version_marker_can_initialize(tmp_path: Path):
    db = Database(tmp_path / "gateway.db")
    with db.engine.begin() as connection:
        connection.exec_driver_sql("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)")

    db.migrate()

    with db.engine.connect() as connection:
        assert connection.exec_driver_sql("SELECT version_num FROM alembic_version").scalar_one() == "0003_mqtt_outbox"
        assert "cameras" in {row[0] for row in connection.exec_driver_sql("SELECT name FROM sqlite_master WHERE type='table'")}
    db.close()


def test_v041_database_upgrades_to_mqtt_outbox_and_removes_legacy_rmq_setting(tmp_path: Path):
    import json
    from alembic import command
    from core.infrastructure.database.models import SystemSetting

    path = tmp_path / "gateway.db"
    db = Database(path)
    command.upgrade(db._alembic_config(), "0002_cloud_storage")
    with db.sessions.begin() as session:
        session.add(SystemSetting(key="cloud_services", value_json=json.dumps({
            "rmq": {"enabled": True, "password_enc": "old-encrypted-secret"},
            "s3": {"enabled": True, "bucket": "faces"}, "auto_upload": True,
        }), updated_at_ms=1))

    db.migrate()

    with db.engine.connect() as connection:
        assert connection.exec_driver_sql("SELECT version_num FROM alembic_version").scalar_one() == "0003_mqtt_outbox"
        tables = {row[0] for row in connection.exec_driver_sql("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "mqtt_outbox" in tables
    with db.sessions() as session:
        config = json.loads(session.get(SystemSetting, "cloud_services").value_json)
    assert "rmq" not in config
    assert config["s3"]["bucket"] == "faces"
    assert config["auto_upload"] is True
    db.close()
