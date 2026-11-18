import json
from pathlib import Path

from core.infrastructure.database.database import Database
from core.infrastructure.database.event_repository import SQLiteEventLogger
from core.infrastructure.database.models import BusinessEvent, MqttOutboxEvent, SystemSetting, CaptureImage
from core.infrastructure.security import SecretCipher
from core.infrastructure.services.cloud_services import CloudServices


def make_services(tmp_path):
    db = Database(tmp_path / "database" / "gateway.db")
    db.migrate()
    services = CloudServices(db, SecretCipher(tmp_path / "secrets" / "camera.key"), tmp_path / "images")
    return db, services


def test_mqtt_and_s3_credentials_are_encrypted_and_masked(tmp_path: Path):
    db, services = make_services(tmp_path)
    public = services.save({
        "mqtt": {"enabled": True, "host": "mqtt.internal", "port": 1883,
                  "username": "edge", "password": "mqtt-password", "gateway_id": "gw-01"},
        "s3": {"enabled": True, "endpoint_url": "http://minio:9000", "bucket": "faces",
               "access_key": "access-secret", "secret_key": "s3-secret", "use_ssl": False},
        "auto_upload": True,
    })
    assert public["mqtt"]["has_password"] is True
    assert public["mqtt"]["host"] == "mqtt.internal"
    assert public["s3"]["has_secret_key"] is True
    assert public["s3"]["has_access_key"] is True
    assert "mqtt-password" not in json.dumps(public)
    assert "s3-secret" not in json.dumps(public)
    assert "access-secret" not in json.dumps(public)
    with db.sessions() as session:
        raw = session.get(SystemSetting, "cloud_services").value_json
    assert "mqtt-password" not in raw
    assert "s3-secret" not in raw
    assert "access-secret" not in raw
    services.save({"mqtt": {"host": "mqtt2.internal", "password": ""}})
    cfg, password = services._mqtt_parameters()
    assert cfg["host"] == "mqtt2.internal"
    assert password == "mqtt-password"
    db.close()


def test_cloud_service_auto_upload_setting_defaults_off(tmp_path: Path):
    db, services = make_services(tmp_path)
    assert services.public_config()["auto_upload"] is False
    services.save({"s3": {"enabled": True, "bucket": "images"}, "auto_upload": True})
    assert services.public_config()["auto_upload"] is True
    db.close()


def test_supported_event_is_atomically_added_to_mqtt_outbox(tmp_path: Path):
    db, services = make_services(tmp_path)
    services.save({"mqtt": {"enabled": True, "gateway_id": "gw-test", "topic_prefix": "efg"}})
    SQLiteEventLogger(db, publisher=services).write({
        "event_type": "capture.created", "status": "created", "timestamp_ms": 123456,
        "capture_id": "capture-1", "camera_id": "cam-1",
        "data": {"capture_id": "capture-1", "camera_id": "cam-1"},
    })
    with db.sessions() as session:
        business = session.query(BusinessEvent).filter_by(event_type="capture.created").one()
        outbox = session.query(MqttOutboxEvent).one()
        payload = json.loads(outbox.payload_json)
        local_payload = json.loads(business.payload_json)
    assert business.capture_id == "capture-1"
    assert outbox.topic == "efg/gw-test/events"
    assert outbox.status == "pending"
    assert payload["event_type"] == "capture.created"
    assert payload["gateway_id"] == "gw-test"
    assert payload["data"]["camera_id"] == "cam-1"
    assert payload["event_id"] == outbox.event_id
    assert local_payload["event_id"] == outbox.event_id
    db.close()


def test_non_contract_events_are_not_sent_to_mqtt(tmp_path: Path):
    db, services = make_services(tmp_path)
    services.save({"mqtt": {"enabled": True}})
    SQLiteEventLogger(db, publisher=services).write({"event_type": "QualityEvent", "event": "face_accepted"})
    with db.sessions() as session:
        assert session.query(BusinessEvent).count() == 1
        assert session.query(MqttOutboxEvent).count() == 0
    db.close()


def test_manual_s3_upload_emits_upload_complete_with_sha256(tmp_path: Path, monkeypatch):
    class FakeS3:
        def __init__(self):
            self.calls = []
        def upload_file(self, filename, bucket, key, ExtraArgs=None):
            self.calls.append((Path(filename).read_bytes(), bucket, key, ExtraArgs))

    db, services = make_services(tmp_path)
    root = tmp_path / "images"
    image_dir = root / "cam-01"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "capture.jpg"
    image_path.write_bytes(b"jpeg-data")
    with db.sessions.begin() as session:
        session.add(CaptureImage(capture_id="capture-01", camera_id="cam-01", image_type="face",
            file_path="cam-01/capture.jpg", captured_at_ms=123456, width=20, height=10,
            file_size=9, created_at_ms=123456))
    services.save({"mqtt": {"enabled": True}, "s3": {"enabled": True, "bucket": "face-images", "prefix": "gateway"}})
    fake = FakeS3()
    monkeypatch.setattr(services, "_s3_client", lambda: ({"bucket": "face-images", "prefix": "gateway"}, fake))
    result = services.upload_image(1)
    assert result["ok"] is True
    with db.sessions() as session:
        image = session.get(CaptureImage, 1)
        event = session.query(BusinessEvent).filter_by(event_type="upload.complete").one()
        outbox = session.query(MqttOutboxEvent).one()
        payload = json.loads(outbox.payload_json)
    assert image.cloud_upload_status == "uploaded"
    assert image.cloud_key == result["key"]
    assert payload["event_type"] == "upload.complete"
    assert payload["data"]["image_type"] == "face"
    assert payload["data"]["image_sha256"]
    assert "upload.complete" in event.payload_json
    db.close()


def test_upload_failure_emits_upload_error(tmp_path: Path, monkeypatch):
    db, services = make_services(tmp_path)
    root = tmp_path / "images"
    image_dir = root / "cam-01"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "capture.jpg"
    image_path.write_bytes(b"jpeg-data")
    with db.sessions.begin() as session:
        session.add(CaptureImage(capture_id="capture-02", camera_id="cam-01", image_type="frame",
            file_path="cam-01/capture.jpg", captured_at_ms=123456, width=20, height=10,
            file_size=9, created_at_ms=123456))
    services.save({"mqtt": {"enabled": True}, "s3": {"enabled": True, "bucket": "faces"}})
    def fail_s3():
        raise RuntimeError("broker-independent S3 failure")
    monkeypatch.setattr(services, "_s3_client", fail_s3)
    try:
        services.upload_image(1)
        assert False, "expected upload failure"
    except RuntimeError:
        pass
    with db.sessions() as session:
        row = session.get(CaptureImage, 1)
        event = session.query(BusinessEvent).filter_by(event_type="upload.error").one()
        outbox = session.query(MqttOutboxEvent).one()
        payload = json.loads(outbox.payload_json)
    assert row.cloud_upload_status == "failed"
    assert payload["event_type"] == "upload.error"
    assert payload["data"]["error_code"] == "RuntimeError"
    assert "upload.error" in event.payload_json
    db.close()


def test_mqtt_test_button_publishes_a_real_test_message(tmp_path: Path, monkeypatch):
    import sys
    import types

    calls = []

    class FakeInfo:
        rc = 0
        def wait_for_publish(self, timeout=None):
            calls.append(("wait", timeout))
        def is_published(self):
            return True

    class FakeClient:
        def __init__(self, client_id, protocol):
            calls.append(("client", client_id, protocol))
        def connect(self, host, port, keepalive):
            calls.append(("connect", host, port, keepalive))
        def loop_start(self): pass
        def loop_stop(self): pass
        def is_connected(self): return True
        def publish(self, topic, payload, qos, retain):
            calls.append(("publish", topic, json.loads(payload), qos, retain))
            return FakeInfo()
        def disconnect(self): pass

    paho = types.ModuleType("paho")
    mqtt_pkg = types.ModuleType("paho.mqtt")
    client_mod = types.ModuleType("paho.mqtt.client")
    client_mod.Client = FakeClient
    client_mod.MQTTv311 = 4
    monkeypatch.setitem(sys.modules, "paho", paho)
    monkeypatch.setitem(sys.modules, "paho.mqtt", mqtt_pkg)
    monkeypatch.setitem(sys.modules, "paho.mqtt.client", client_mod)

    db, services = make_services(tmp_path)
    services.save({"mqtt": {"enabled": True, "host": "broker", "gateway_id": "gw-a",
                             "topic_prefix": "edge-face-gateway", "qos": 1}})
    result = services.test_mqtt()
    assert result["ok"] is True
    published = next(call for call in calls if call[0] == "publish")
    assert published[1] == "edge-face-gateway/gw-a/test"
    assert published[2]["event_type"] == "gateway.test"
    assert published[3:] == (1, False)
    db.close()


def test_outbox_worker_marks_confirmed_message_published(tmp_path: Path, monkeypatch):
    import threading
    import time

    db, services = make_services(tmp_path)
    services.save({"mqtt": {"enabled": True}})
    SQLiteEventLogger(db, publisher=services).write({"event_type": "camera.online", "camera_id": "cam-1"})

    def publish_then_stop(event_id, topic, payload_json, qos):
        assert topic.endswith("/events")
        assert json.loads(payload_json)["event_type"] == "camera.online"
        services._stop.set()

    monkeypatch.setattr(services, "_ensure_mqtt_connection", lambda: None)
    monkeypatch.setattr(services, "_publish_outbox_event", publish_then_stop)
    services._stop.clear()
    worker = threading.Thread(target=services._mqtt_loop)
    worker.start()
    worker.join(timeout=3)
    assert not worker.is_alive()
    with db.sessions() as session:
        row = session.query(MqttOutboxEvent).one()
        assert row.status == "published"
        assert row.published_at_ms is not None
    db.close()


def test_outbox_worker_keeps_failed_message_and_schedules_retry(tmp_path: Path, monkeypatch):
    import threading

    db, services = make_services(tmp_path)
    services.save({"mqtt": {"enabled": True}})
    SQLiteEventLogger(db, publisher=services).write({"event_type": "upload.complete", "camera_id": "cam-1"})

    def fail_then_stop(*args):
        services._stop.set()
        raise RuntimeError("temporary broker outage")

    monkeypatch.setattr(services, "_ensure_mqtt_connection", lambda: None)
    monkeypatch.setattr(services, "_publish_outbox_event", fail_then_stop)
    services._stop.clear()
    worker = threading.Thread(target=services._mqtt_loop)
    worker.start()
    worker.join(timeout=3)
    assert not worker.is_alive()
    with db.sessions() as session:
        row = session.query(MqttOutboxEvent).one()
        assert row.status == "pending"
        assert row.attempts == 1
        assert row.next_attempt_at_ms > row.created_at_ms
        assert "temporary broker outage" in row.last_error
    db.close()
