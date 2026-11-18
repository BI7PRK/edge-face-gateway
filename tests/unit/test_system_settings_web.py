import threading
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from core.infrastructure.database.database import Database
from core.infrastructure.security import SecretCipher
from core.infrastructure.services.cloud_services import CloudServices
from core.runtime.web import create_app


def test_system_settings_page_and_api_mask_secrets(tmp_path: Path):
    db = Database(tmp_path / "gateway.db")
    db.migrate()
    cloud = CloudServices(db, SecretCipher(tmp_path / "camera.key"), tmp_path)
    engine = SimpleNamespace(output_dir=tmp_path, host_started_mono=0.0, runtimes=[],
        stop_event=threading.Event(), event_log=SimpleNamespace(path=None), database=db,
        cloud_services=cloud)
    client = TestClient(create_app(engine))
    page = client.get("/settings").text
    assert "MQTT 消息服务" in page
    assert "RabbitMQ" not in page
    assert "S3" in page
    saved = client.put("/api/system-settings", json={
        "mqtt": {"enabled": True, "host": "mqtt", "port": 1883,
                  "username": "edge", "password": "secret", "client_id": "efg-gw",
                  "gateway_id": "gw-01", "topic_prefix": "edge-face-gateway", "qos": 1},
        "s3": {"enabled": True, "endpoint_url": "http://minio:9000", "bucket": "captures",
               "access_key": "access", "secret_key": "secret-key", "use_ssl": False},
    })
    assert saved.status_code == 200
    body = saved.json()
    assert body["mqtt"]["has_password"] is True
    assert body["s3"]["has_secret_key"] is True
    assert "secret-key" not in saved.text
    assert '"password":"secret"' not in saved.text
    assert '"password": "secret"' not in saved.text
    enabled = client.put("/api/system-settings/auto-upload", json={"enabled": True})
    assert enabled.status_code == 200 and enabled.json()["auto_upload"] is True
    status = client.get("/api/system-settings/mqtt-status")
    assert status.status_code == 200
    assert status.json()["enabled"] is True
    db.close()


def test_mqtt_settings_validation_rejects_invalid_qos(tmp_path: Path):
    db = Database(tmp_path / "gateway.db")
    db.migrate()
    cloud = CloudServices(db, SecretCipher(tmp_path / "camera.key"), tmp_path)
    engine = SimpleNamespace(output_dir=tmp_path, host_started_mono=0.0, runtimes=[],
        stop_event=threading.Event(), event_log=SimpleNamespace(path=None), database=db,
        cloud_services=cloud)
    client = TestClient(create_app(engine))
    response = client.put("/api/system-settings", json={"mqtt": {"enabled": True, "host": "mqtt", "qos": 3}})
    assert response.status_code == 422
    db.close()
