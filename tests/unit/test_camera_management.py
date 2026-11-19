from pathlib import Path
from threading import Event
from types import SimpleNamespace

from fastapi.testclient import TestClient

from core.infrastructure.database.camera_repository import CameraRepository
from core.infrastructure.database.database import Database
from core.infrastructure.database.models import Camera
from core.infrastructure.security import SecretCipher
from core.runtime.web import create_app


class FakeEngine:
    def __init__(self, db, repo):
        self.database = db
        self.camera_repository = repo
        self.output_dir = Path(".")
        self.host_started_mono = 0.0
        self.runtimes = []
        self.stop_event = Event()
        self.event_log = SimpleNamespace(path=None)
        self.active = {}

    def get_runtime(self, camera_id):
        return self.active.get(camera_id)

    def add_or_replace_camera(self, cfg):
        if cfg.enabled:
            self.active[cfg.camera_id] = SimpleNamespace(main=SimpleNamespace(connected=False, thread=None), sub=None)
        else:
            self.active.pop(cfg.camera_id, None)

    def remove_camera(self, camera_id):
        self.active.pop(camera_id, None)

    @staticmethod
    def test_stream(url):
        return {"ok": True, "width": 1920, "height": 1080, "elapsed_ms": 10.0}


def test_camera_crud_masks_rtsp_credentials_and_enforces_revision(tmp_path: Path, authenticate_client):
    db = Database(tmp_path / "gateway.db")
    db.create_schema()
    repo = CameraRepository(SecretCipher(tmp_path / "secret.key"))
    engine = FakeEngine(db, repo)
    client = TestClient(create_app(engine))
    authenticate_client(client)

    created = client.post("/api/cameras", json={
        "id": "gate-01", "name": "Gate", "main_url": "rtsp://admin:super-secret@10.0.0.2/main",
        "sub_url": "rtsp://admin:sub-secret@10.0.0.2/sub", "detection_fps": 5,
    })
    assert created.status_code == 201
    assert created.json()["stream_mode"] == "dual"
    listing = client.get("/api/cameras").json()["items"][0]
    assert listing["has_main_url"] is True
    assert "super-secret" not in str(listing)
    assert listing["revision"] == 1

    updated = client.put("/api/cameras/gate-01", json={
        "name": "Gate North", "main_url": None, "sub_url": None, "keep_substream": True,
        "detection_fps": 8, "enabled": True, "revision": 1,
    })
    assert updated.status_code == 200
    assert updated.json()["revision"] == 2
    assert updated.json()["stream_mode"] == "dual"

    conflict = client.put("/api/cameras/gate-01", json={
        "name": "Stale", "main_url": None, "sub_url": None, "keep_substream": True,
        "detection_fps": 8, "enabled": True, "revision": 1,
    })
    assert conflict.status_code == 409

    disabled = client.put("/api/cameras/gate-01/enabled", json={"enabled": False, "revision": 2})
    assert disabled.status_code == 200
    assert "gate-01" not in engine.active
    with db.sessions() as session:
        row = session.get(Camera, "gate-01")
        assert row.enabled is False
        assert repo.cipher.decrypt(row.main_url_encrypted) == "rtsp://admin:super-secret@10.0.0.2/main"
    db.close()


def test_camera_create_rejects_non_rtsp_url(tmp_path: Path, authenticate_client):
    db = Database(tmp_path / "gateway.db")
    db.create_schema()
    repo = CameraRepository(SecretCipher(tmp_path / "secret.key"))
    client = TestClient(create_app(FakeEngine(db, repo)))
    authenticate_client(client)
    response = client.post("/api/cameras", json={"id": "x", "name": "x", "main_url": "http://example.com"})
    assert response.status_code == 422
    db.close()


def test_camera_update_can_switch_dual_stream_to_single_stream(tmp_path: Path, authenticate_client):
    db = Database(tmp_path / "gateway.db")
    db.create_schema()
    repo = CameraRepository(SecretCipher(tmp_path / "secret.key"))
    client = TestClient(create_app(FakeEngine(db, repo)))
    authenticate_client(client)
    assert client.post("/api/cameras", json={
        "id": "dual", "name": "Dual", "main_url": "rtsp://cam/main", "sub_url": "rtsp://cam/sub",
    }).status_code == 201
    response = client.put("/api/cameras/dual", json={
        "name": "Single", "main_url": None, "sub_url": None, "keep_substream": False,
        "detection_fps": 3, "enabled": True, "revision": 1,
    })
    assert response.status_code == 200
    assert response.json()["stream_mode"] == "single"
    with db.sessions() as session:
        row = session.get(Camera, "dual")
        assert repo.cipher.decrypt(row.sub_url_encrypted) is None
    db.close()


def test_runtime_engine_replaces_only_target_camera(monkeypatch):
    import threading
    from core.runtime import engine as engine_module
    from core.runtime.config import CameraConfig

    stopped = []

    class StubRuntime:
        def __init__(self, cfg, _capture_cfg):
            self.cfg = cfg
            self.started = False
        def start(self):
            self.started = True
        def stop(self):
            stopped.append(self.cfg.camera_id)

    monkeypatch.setattr(engine_module, "CameraRuntime", StubRuntime)
    runtime_engine = engine_module.RuntimeEngine.__new__(engine_module.RuntimeEngine)
    runtime_engine._runtime_lock = threading.RLock()
    runtime_engine._runtimes = {
        "cam-a": StubRuntime(CameraConfig("cam-a", "A", "rtsp://a", None), {}),
        "cam-b": StubRuntime(CameraConfig("cam-b", "B", "rtsp://b", None), {}),
    }
    runtime_engine.capture_cfg = {}
    runtime_engine._started = True
    runtime_engine.database = None

    runtime_engine.add_or_replace_camera(CameraConfig("cam-a", "A2", "rtsp://a2", None, True, 6))
    assert stopped == ["cam-a"]
    assert runtime_engine.get_runtime("cam-b") is not None
    assert runtime_engine.get_runtime("cam-a").started is True
    assert runtime_engine.get_runtime("cam-a").cfg.name == "A2"
