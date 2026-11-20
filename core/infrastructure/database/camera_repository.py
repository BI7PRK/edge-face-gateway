from __future__ import annotations
import json
import time
from sqlalchemy import select, delete
from sqlalchemy.orm import Session
from .models import Camera, CameraStatus
from ..security import SecretCipher

class CameraRepository:
    def __init__(self, cipher: SecretCipher):
        self.cipher = cipher

    def list_enabled(self, session: Session):
        rows = session.scalars(select(Camera).where(Camera.enabled.is_(True)).order_by(Camera.id)).all()
        from core.runtime.config import CameraConfig
        result = []
        for row in rows:
            main = self.cipher.decrypt(row.main_url_encrypted)
            sub = self.cipher.decrypt(row.sub_url_encrypted)
            if main:
                camera_config = json.loads(row.config_json or "{}")
                result.append(CameraConfig(
                    camera_id=row.id, name=row.name, main_url=main, sub_url=sub,
                    detection_fps=row.detection_fps,
                    min_capture_score=float(camera_config.get("min_capture_score", 0.78)),
                ))
        return result

    def get(self, session: Session, camera_id: str):
        return session.get(Camera, camera_id)

    def create(self, session: Session, *, camera_id: str, name: str, main_url: str, sub_url: str | None,
               detection_fps: int, min_capture_score: float = 0.78, enabled: bool = True,
               config_json: str = "{}") -> Camera:
        now = int(time.time() * 1000)
        camera_config = json.loads(config_json or "{}")
        camera_config["min_capture_score"] = min_capture_score
        row = Camera(id=camera_id, name=name, enabled=enabled, main_url_encrypted=self.cipher.encrypt(main_url),
                     sub_url_encrypted=self.cipher.encrypt(sub_url), stream_mode="dual" if sub_url else "single",
                     detection_fps=detection_fps, config_json=json.dumps(camera_config),
                     revision=1, created_at_ms=now, updated_at_ms=now)
        session.add(row)
        session.flush()
        session.add(CameraStatus(camera_id=camera_id, status="STOPPED", worker_state="STOPPED", reconnect_count=0, current_fps=0.0, updated_at_ms=now))
        return row

    def update(self, session: Session, row: Camera, *, name: str, main_url: str | None, sub_url: str | None,
               keep_sub: bool, detection_fps: int, enabled: bool, expected_revision: int,
               min_capture_score: float | None = None) -> Camera:
        if row.revision != expected_revision:
            raise RuntimeError("revision_conflict")
        if main_url:
            row.main_url_encrypted = self.cipher.encrypt(main_url)
        if not keep_sub:
            row.sub_url_encrypted = self.cipher.encrypt(sub_url)
        row.name = name
        row.detection_fps = detection_fps
        if min_capture_score is not None:
            camera_config = json.loads(row.config_json or "{}")
            camera_config["min_capture_score"] = min_capture_score
            row.config_json = json.dumps(camera_config)
        row.enabled = enabled
        row.stream_mode = "dual" if (self.cipher.decrypt(row.sub_url_encrypted) or "") else "single"
        row.revision += 1
        row.updated_at_ms = int(time.time() * 1000)
        return row

    def set_enabled(self, session: Session, row: Camera, enabled: bool, expected_revision: int) -> Camera:
        if row.revision != expected_revision:
            raise RuntimeError("revision_conflict")
        row.enabled = enabled
        row.revision += 1
        row.updated_at_ms = int(time.time() * 1000)
        return row
