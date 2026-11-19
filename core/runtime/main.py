from __future__ import annotations

import logging
import os
import signal
import threading
from pathlib import Path

import uvicorn

from .config import load_config
from core.infrastructure.database.database import Database
from core.infrastructure.database.camera_repository import CameraRepository
from core.infrastructure.security import SecretCipher
from core.infrastructure.logging.sqlite_handler import SQLiteLogHandler
from core.infrastructure.storage.image_repository import ImageRepository
from core.infrastructure.services.cloud_services import CloudServices
from core.infrastructure.database.event_repository import SQLiteEventLogger
from .engine import RuntimeEngine
from .web import create_app


def main() -> None:
    config_path = os.getenv("FACE_CONFIG", "/app/config/cameras.yaml")
    config = load_config(config_path)
    data_dir = Path(os.getenv("FACE_DATA_DIR", "/app/data"))
    database_path = Path(os.getenv("FACE_DATABASE_PATH", str(data_dir / "database" / "gateway.db")))
    database = Database(database_path)
    database.migrate()
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, os.getenv("FACE_LOG_LEVEL", "INFO").upper(), logging.INFO))
    db_handler = SQLiteLogHandler(database)
    db_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(db_handler)
    log = logging.getLogger(__name__)
    cipher = SecretCipher(os.getenv("FACE_SECRET_PATH", str(data_dir / "secrets" / "camera.key")))
    camera_repository = CameraRepository(cipher)
    image_repository = ImageRepository(database, Path(os.getenv("FACE_OUTPUT_DIR") or config.get("capture", {}).get("output_dir", "/app/data")))
    legacy_images = image_repository.import_legacy_files()
    legacy_events = SQLiteEventLogger(database).import_legacy_jsonl(os.getenv("FACE_LEGACY_EVENTS", "/app/logs/runtime.jsonl"))
    with database.sessions() as session:
        cameras = camera_repository.list_enabled(session)
    log.info("database initialized path=%s legacy_images=%s legacy_events=%s active_cameras=%s",
             database_path, legacy_images, legacy_events, len(cameras))
    cloud_services = CloudServices(database, cipher, image_repository.root)
    engine = RuntimeEngine(cameras, config, database=database, camera_repository=camera_repository, cloud_services=cloud_services)
    engine.start()
    server_cfg = config.get("server", {})
    server = uvicorn.Server(uvicorn.Config(create_app(engine), host=server_cfg.get("host", "0.0.0.0"),
                                           port=int(server_cfg.get("port", 8080)), log_level="info", log_config=None))
    stop = threading.Event()

    def shutdown(*_args):
        if not stop.is_set():
            stop.set()
            server.should_exit = True
            engine.stop()
            db_handler.flush()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    try:
        server.run()
    finally:
        engine.stop()
        db_handler.flush()
        root_logger.removeHandler(db_handler)
        db_handler.close()
        database.close()


if __name__ == "__main__":
    main()
