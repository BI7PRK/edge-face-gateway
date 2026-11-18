import json
import threading
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
import cv2
import numpy as np

from core.runtime.web import create_app


def test_album_lists_capture_artifacts_and_media_is_served(tmp_path: Path):
    camera_dir = tmp_path / "camera-01"
    camera_dir.mkdir()
    image_path = camera_dir / "20261118T100000_000000Z_frame.jpg"
    assert cv2.imwrite(str(image_path), np.zeros((1200, 1600, 3), dtype=np.uint8))
    logs = tmp_path / "runtime.jsonl"
    logs.write_text("", encoding="utf-8")
    engine = SimpleNamespace(output_dir=tmp_path, host_started_mono=0.0, runtimes=[],
                             stop_event=threading.Event(), event_log=SimpleNamespace(path=logs))
    client = TestClient(create_app(engine))
    response = client.get("/api/captures")
    assert response.status_code == 200
    assert response.json()["total"] == 1
    assert response.json()["items"][0]["kind"] == "frame"
    assert "thumbnail_url" in response.json()["items"][0]
    assert client.get("/album").status_code == 200
    assert "QualityEvent" in client.get("/logs").text
    thumbnail = client.get(response.json()["items"][0]["thumbnail_url"])
    assert thumbnail.status_code == 200
    decoded = cv2.imdecode(np.frombuffer(thumbnail.content, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert max(decoded.shape[:2]) <= 480


def test_event_api_filters_typed_events(tmp_path: Path):
    logs = tmp_path / "runtime.jsonl"
    logs.write_text("\n".join(json.dumps(x) for x in [
        {"event_type": "CaptureEvent", "ts": "2026-11-18T10:00:00Z", "camera_id": "cam-1"},
        {"event_type": "QualityEvent", "ts": "2026-11-18T10:00:01Z", "camera_id": "cam-1"},
        {"event_type": "SaveEvent", "ts": "2026-11-18T10:00:02Z", "camera_id": "cam-1"},
        {"event_type": "UploadEvent", "ts": "2026-11-18T10:00:03Z", "camera_id": "cam-1"},
    ]), encoding="utf-8")
    engine = SimpleNamespace(output_dir=tmp_path, host_started_mono=0.0, runtimes=[],
                             stop_event=threading.Event(), event_log=SimpleNamespace(path=logs))
    client = TestClient(create_app(engine))
    response = client.get("/api/events?event_type=QualityEvent")
    assert response.status_code == 200
    assert response.json()["total"] == 1
    assert response.json()["items"][0]["event_type"] == "QualityEvent"
