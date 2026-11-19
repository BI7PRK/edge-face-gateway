import json
import threading
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from fastapi.testclient import TestClient

from core.infrastructure.database.database import Database
from core.infrastructure.database.models import ApplicationLog, BusinessEvent, CaptureImage, MqttOutboxEvent
from core.runtime.web import create_app


def test_database_backed_capture_event_and_log_apis_are_paginated(tmp_path: Path):
    db = Database(tmp_path / "gateway.db")
    db.create_schema()
    image_dir = tmp_path / "cam-01"
    image_dir.mkdir()
    image = image_dir / "frame.jpg"
    assert cv2.imwrite(str(image), np.zeros((120, 160, 3), dtype=np.uint8))
    with db.sessions.begin() as session:
        session.add(CaptureImage(capture_id="cap-1", camera_id="cam-01", image_type="frame",
            file_path="cam-01/frame.jpg", captured_at_ms=1000, width=160, height=120,
            file_size=image.stat().st_size, created_at_ms=1000))
        for i in range(3):
            session.add(BusinessEvent(event_type="capture.detected", timestamp_ms=i,
                payload_json=json.dumps({"event_type": "capture.detected", "ts": str(i)})))
            session.add(ApplicationLog(timestamp_ms=i, level="INFO", logger="test", category="SYSTEM", message=str(i)))
    engine = SimpleNamespace(output_dir=tmp_path, host_started_mono=0.0, runtimes=[],
        stop_event=threading.Event(), event_log=SimpleNamespace(path=None), database=db)
    client = TestClient(create_app(engine))
    captures = client.get("/api/captures?page=1&page_size=1").json()
    assert captures["total"] == 1 and len(captures["items"]) == 1
    assert captures["items"][0]["kind"] == "frame"
    events = client.get("/api/events?page=2&page_size=1").json()
    assert events["total"] == 3 and len(events["items"]) == 1
    logs = client.get("/api/logs?page=2&page_size=2").json()
    assert logs["pagination"]["total"] == 3 and len(logs["items"]) == 1
    assert client.get("/system-logs").status_code == 200
    db.close()


def test_event_api_exposes_creation_and_mqtt_publish_times(tmp_path: Path):
    db = Database(tmp_path / "gateway.db")
    db.create_schema()
    with db.sessions.begin() as session:
        event_id = "event-1"
        session.add(BusinessEvent(
            event_type="capture.created", timestamp_ms=123456,
            payload_json=json.dumps({"event_type": "capture.created", "event_id": event_id}),
        ))
        session.add(MqttOutboxEvent(
            event_id=event_id, topic="gateway/events", payload_json="{}",
            qos=1, retain=False, status="published", attempts=1,
            next_attempt_at_ms=123456, created_at_ms=123456, published_at_ms=234567,
        ))
    engine = SimpleNamespace(output_dir=tmp_path, host_started_mono=0.0, runtimes=[],
        stop_event=threading.Event(), event_log=SimpleNamespace(path=None), database=db)
    item = TestClient(create_app(engine)).get("/api/events").json()["items"][0]
    assert item["created_at_ms"] == 123456
    assert item["published_at_ms"] == 234567
    assert item["push_status"] == "published"
    db.close()
