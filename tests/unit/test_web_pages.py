import json
import threading
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
import cv2
import numpy as np

from core.infrastructure.database.database import Database
from core.infrastructure.database.models import BusinessEvent, CaptureImage
from core.runtime.web import create_app


def test_album_lists_capture_artifacts_and_media_is_served(tmp_path: Path, authenticate_client):
    camera_dir = tmp_path / "camera-01"
    camera_dir.mkdir()
    image_path = camera_dir / "20261118T100000_000000Z_frame.jpg"
    assert cv2.imwrite(str(image_path), np.zeros((1200, 1600, 3), dtype=np.uint8))
    logs = tmp_path / "runtime.jsonl"
    logs.write_text("", encoding="utf-8")
    db = Database(tmp_path / "gateway.db")
    db.create_schema()
    with db.sessions.begin() as session:
        session.add(CaptureImage(capture_id="cap-1", camera_id="camera-01", image_type="frame",
            file_path=f"camera-01/{image_path.name}", captured_at_ms=1000, width=1600, height=1200,
            file_size=image_path.stat().st_size, created_at_ms=1000))
    engine = SimpleNamespace(output_dir=tmp_path, host_started_mono=0.0, runtimes=[],
                             stop_event=threading.Event(), event_log=SimpleNamespace(path=logs), database=db)
    client = TestClient(create_app(engine))
    authenticate_client(client)
    response = client.get("/api/captures")
    assert response.status_code == 200
    assert response.json()["total"] == 1
    assert response.json()["items"][0]["kind"] == "frame"
    assert "thumbnail_url" in response.json()["items"][0]
    assert client.get("/album").status_code == 200
    logs_page = client.get("/logs").text
    assert "quality.face_rejected" in logs_page
    assert "capture.score_rejected" in logs_page
    assert "QualityEvent" not in logs_page
    assert "创建时间：" in logs_page
    assert "推送时间：" in logs_page
    thumbnail = client.get(response.json()["items"][0]["thumbnail_url"])
    assert thumbnail.status_code == 200
    decoded = cv2.imdecode(np.frombuffer(thumbnail.content, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert max(decoded.shape[:2]) <= 480
    db.close()


def test_event_api_filters_typed_events(tmp_path: Path, authenticate_client):
    db = Database(tmp_path / "gateway.db")
    db.create_schema()
    with db.sessions.begin() as session:
        session.add_all([
            BusinessEvent(event_type="capture.detected", timestamp_ms=1, camera_id="cam-1",
                payload_json=json.dumps({"event_type": "capture.detected", "camera_id": "cam-1"})),
            BusinessEvent(event_type="quality.face_accepted", timestamp_ms=2, camera_id="cam-1",
                payload_json=json.dumps({"event_type": "quality.face_accepted", "camera_id": "cam-1"})),
            BusinessEvent(event_type="artifact.saved", timestamp_ms=3, camera_id="cam-1",
                payload_json=json.dumps({"event_type": "artifact.saved", "camera_id": "cam-1"})),
            BusinessEvent(event_type="upload.complete", timestamp_ms=4, camera_id="cam-1",
                payload_json=json.dumps({"event_type": "upload.complete", "camera_id": "cam-1"})),
        ])
    engine = SimpleNamespace(output_dir=tmp_path, host_started_mono=0.0, runtimes=[],
                             stop_event=threading.Event(), event_log=SimpleNamespace(path=None), database=db)
    client = authenticate_client(TestClient(create_app(engine)))
    response = client.get("/api/events?event_type=quality.face_accepted")
    assert response.status_code == 200
    assert response.json()["total"] == 1
    assert response.json()["items"][0]["event_type"] == "quality.face_accepted"
    db.close()
