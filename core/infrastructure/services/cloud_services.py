from __future__ import annotations

import hashlib
import json
import logging
import queue
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy import select

from core.infrastructure.database.models import CaptureImage, MqttOutboxEvent, SystemSetting

log = logging.getLogger(__name__)

DEFAULTS: dict[str, Any] = {
    "mqtt": {
        "enabled": False, "host": "localhost", "port": 1883, "username": "", "password_enc": "",
        "client_id": "edge-face-gateway", "gateway_id": "gateway-001",
        "topic_prefix": "edge-face-gateway", "qos": 1, "keep_alive": 60,
        "use_tls": False, "heartbeat": 30,
    },
    "s3": {"enabled": False, "endpoint_url": "", "region": "us-east-1", "bucket": "",
           "access_key_enc": "", "secret_key_enc": "", "use_ssl": True, "prefix": "edge-face"},
    "auto_upload": False,
}

MQTT_EVENT_TYPES = {
    "camera.online", "camera.offline", "capture.created", "capture.saved", "upload.complete", "upload.error"
}


class CloudServices:
    """SQLite-backed S3/MQTT services with a durable MQTT outbox."""

    def __init__(self, database, cipher, image_root: str | Path):
        self.database, self.cipher = database, cipher
        self.image_root = Path(image_root).resolve()
        self._queue: queue.Queue = queue.Queue(maxsize=2000)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._mqtt_client = None
        self._mqtt_client_lock = threading.RLock()
        self._mqtt_connected = False

    def _read(self) -> dict[str, Any]:
        with self.database.sessions() as session:
            row = session.get(SystemSetting, "cloud_services")
            saved = json.loads(row.value_json) if row else {}
        cfg = {**DEFAULTS, **saved}
        cfg["mqtt"] = {**DEFAULTS["mqtt"], **saved.get("mqtt", {})}
        cfg["s3"] = {**DEFAULTS["s3"], **saved.get("s3", {})}
        return cfg

    def public_config(self) -> dict[str, Any]:
        cfg = self._read()
        return {
            "mqtt": {k: v for k, v in cfg["mqtt"].items() if k != "password_enc"} |
                    {"has_password": bool(cfg["mqtt"].get("password_enc"))},
            "s3": {k: v for k, v in cfg["s3"].items() if k not in {"secret_key_enc", "access_key_enc"}} |
                  {"has_secret_key": bool(cfg["s3"].get("secret_key_enc")),
                   "has_access_key": bool(cfg["s3"].get("access_key_enc"))},
            "auto_upload": bool(cfg.get("auto_upload", False)),
        }

    def save(self, payload: dict[str, Any]) -> dict[str, Any]:
        old = self._read()
        mqtt_cfg = {**old["mqtt"], **payload.get("mqtt", {})}
        s3 = {**old["s3"], **payload.get("s3", {})}
        password = payload.get("mqtt", {}).get("password")
        secret = payload.get("s3", {}).get("secret_key")
        access = payload.get("s3", {}).get("access_key")
        mqtt_cfg.pop("password", None)
        s3.pop("secret_key", None)
        s3.pop("access_key", None)
        if password:
            mqtt_cfg["password_enc"] = self.cipher.encrypt(password)
        if secret:
            s3["secret_key_enc"] = self.cipher.encrypt(secret)
        if access:
            s3["access_key_enc"] = self.cipher.encrypt(access)
        if payload.get("mqtt", {}).get("clear_password"):
            mqtt_cfg["password_enc"] = ""
        if payload.get("s3", {}).get("clear_secret_key"):
            s3["secret_key_enc"] = ""
        if payload.get("s3", {}).get("clear_access_key"):
            s3["access_key_enc"] = ""
        auto_upload = bool(payload.get("auto_upload", old.get("auto_upload", False))) and bool(s3.get("enabled"))
        cfg = {"mqtt": mqtt_cfg, "s3": s3, "auto_upload": auto_upload}
        with self.database.sessions.begin() as session:
            row = session.get(SystemSetting, "cloud_services")
            now = int(time.time() * 1000)
            if row is None:
                session.add(SystemSetting(key="cloud_services", value_json=json.dumps(cfg), updated_at_ms=now))
            else:
                row.value_json, row.updated_at_ms = json.dumps(cfg), now
        self._close_mqtt_client()
        return self.public_config()

    def _mqtt_parameters(self):
        cfg = self._read()["mqtt"]
        if not cfg.get("enabled"):
            raise RuntimeError("MQTT 服务未启用")
        password = self.cipher.decrypt(cfg.get("password_enc")) or ""
        return cfg, password

    def _s3_client(self):
        cfg = self._read()["s3"]
        if not cfg.get("enabled"):
            raise RuntimeError("S3 云存储服务未启用")
        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:
            raise RuntimeError("S3 SDK 未安装") from exc
        secret = self.cipher.decrypt(cfg.get("secret_key_enc")) or ""
        access = self.cipher.decrypt(cfg.get("access_key_enc")) or ""
        return cfg, boto3.client("s3", endpoint_url=cfg.get("endpoint_url") or None,
            region_name=cfg.get("region") or None, aws_access_key_id=access or None,
            aws_secret_access_key=secret or None, use_ssl=bool(cfg.get("use_ssl", True)),
            config=Config(connect_timeout=4, read_timeout=10, retries={"max_attempts": 2}))

    @staticmethod
    def _mqtt_module():
        try:
            import paho.mqtt.client as mqtt
            return mqtt
        except ImportError as exc:
            raise RuntimeError("MQTT SDK 未安装（paho-mqtt）") from exc

    def _new_mqtt_client(self, cfg: dict[str, Any], password: str):
        mqtt = self._mqtt_module()
        client_id = str(cfg.get("client_id") or "edge-face-gateway")
        client = mqtt.Client(client_id=client_id, protocol=mqtt.MQTTv311)
        if cfg.get("username"):
            client.username_pw_set(str(cfg["username"]), password)
        if cfg.get("use_tls"):
            client.tls_set()
        client.connect(str(cfg["host"]), int(cfg["port"]), keepalive=int(cfg.get("keep_alive", 60)))
        client.loop_start()
        deadline = time.monotonic() + 8
        while not client.is_connected() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not client.is_connected():
            client.loop_stop()
            raise RuntimeError("MQTT 连接超时")
        return client

    def _close_mqtt_client(self) -> None:
        with self._mqtt_client_lock:
            client, self._mqtt_client = self._mqtt_client, None
            self._mqtt_connected = False
        if client is not None:
            try:
                client.disconnect()
                client.loop_stop()
            except Exception:
                log.debug("MQTT client close failed", exc_info=True)

    def test_mqtt(self) -> dict[str, Any]:
        client = None
        try:
            cfg, password = self._mqtt_parameters()
            test_cfg = dict(cfg)
            test_cfg["client_id"] = (str(cfg.get("client_id") or "edge-face-gateway") + "-test")[:128]
            client = self._new_mqtt_client(test_cfg, password)
            topic = "/".join(x.strip("/") for x in [cfg["topic_prefix"], cfg["gateway_id"], "test"] if x)
            payload = {"event_type": "gateway.test", "gateway_id": cfg["gateway_id"],
                       "timestamp_ms": int(time.time() * 1000), "message": "MQTT publish test"}
            info = client.publish(topic, json.dumps(payload, ensure_ascii=False), qos=int(cfg.get("qos", 1)), retain=False)
            info.wait_for_publish(timeout=10)
            if info.rc != 0 or not info.is_published():
                raise RuntimeError("MQTT publish acknowledgement timeout")
            return {"ok": True, "message": f"MQTT 连接成功，测试消息已发布并收到 QoS {cfg.get('qos', 1)} 确认", "topic": topic}
        except Exception as exc:
            return {"ok": False, "message": f"MQTT 测试失败：{type(exc).__name__}: {exc}"}
        finally:
            if client is not None:
                try:
                    client.disconnect()
                    client.loop_stop()
                except Exception:
                    pass

    def test_s3(self) -> dict[str, Any]:
        try:
            cfg, client = self._s3_client()
            client.head_bucket(Bucket=cfg["bucket"])
            return {"ok": True, "message": "S3 Bucket 访问成功"}
        except Exception:
            return {"ok": False, "message": "S3 连接失败，请检查 Endpoint、Bucket、访问密钥及权限"}

    def add_event_to_outbox(self, session, payload: dict[str, Any], timestamp_ms: int) -> None:
        """Add supported events in the same transaction as the local business event."""
        event_type = str(payload.get("event_type", ""))
        if event_type not in MQTT_EVENT_TYPES:
            return
        setting = session.get(SystemSetting, "cloud_services")
        saved = json.loads(setting.value_json) if setting else {}
        cfg = {**DEFAULTS["mqtt"], **saved.get("mqtt", {})}
        if not cfg.get("enabled"):
            return
        event_id = str(payload.get("event_id") or uuid.uuid4())
        gateway_id = str(cfg.get("gateway_id") or "gateway-001")
        prefix = str(cfg.get("topic_prefix") or "edge-face-gateway").strip("/")
        topic = f"{prefix}/{gateway_id}/events"
        data = payload.get("data")
        if not isinstance(data, dict):
            excluded = {"event_id", "event_type", "gateway_id", "timestamp_ms", "ts", "status"}
            data = {key: value for key, value in payload.items() if key not in excluded}
        envelope = {
            "event_id": event_id,
            "event_type": event_type,
            "gateway_id": gateway_id,
            "timestamp_ms": int(timestamp_ms),
            "data": data,
        }
        session.add(MqttOutboxEvent(
            event_id=event_id, topic=topic, payload_json=json.dumps(envelope, ensure_ascii=False, default=str),
            qos=max(0, min(2, int(cfg.get("qos", 1)))), retain=False, status="pending", attempts=0,
            next_attempt_at_ms=int(time.time() * 1000), created_at_ms=int(time.time() * 1000)))

    def mqtt_status(self) -> dict[str, Any]:
        cfg = self._read()["mqtt"]
        with self.database.sessions() as session:
            pending = int(session.query(MqttOutboxEvent).filter(MqttOutboxEvent.status == "pending").count())
        return {"enabled": bool(cfg.get("enabled")), "connected": bool(self._mqtt_connected),
                "pending_events": pending, "gateway_id": cfg.get("gateway_id"),
                "topic": f"{str(cfg.get('topic_prefix', 'edge-face-gateway')).strip('/')}/{cfg.get('gateway_id', 'gateway-001')}/events"}

    def _ensure_mqtt_connection(self) -> None:
        cfg, password = self._mqtt_parameters()
        with self._mqtt_client_lock:
            if self._mqtt_client is not None and self._mqtt_client.is_connected():
                self._mqtt_connected = True
                return
            self._close_mqtt_client()
            self._mqtt_client = self._new_mqtt_client(cfg, password)
            self._mqtt_connected = bool(self._mqtt_client.is_connected())

    def _publish_outbox_event(self, event_id: str, topic: str, payload_json: str, qos: int) -> None:
        self._ensure_mqtt_connection()
        with self._mqtt_client_lock:
            client = self._mqtt_client
            self._mqtt_connected = bool(client and client.is_connected())
        if client is None:
            raise RuntimeError("MQTT client is not connected")
        info = client.publish(topic, payload_json, qos=qos, retain=False)
        info.wait_for_publish(timeout=15)
        if info.rc != 0 or not info.is_published():
            raise RuntimeError(f"MQTT publish was not acknowledged (rc={info.rc})")

    def _mqtt_loop(self) -> None:
        while not self._stop.is_set():
            try:
                if not self._read()["mqtt"].get("enabled"):
                    self._mqtt_connected = False
                    self._close_mqtt_client()
                    self._stop.wait(1)
                    continue
                self._ensure_mqtt_connection()
                now = int(time.time() * 1000)
                with self.database.sessions() as session:
                    rows = session.scalars(select(MqttOutboxEvent)
                        .where(MqttOutboxEvent.status == "pending", MqttOutboxEvent.next_attempt_at_ms <= now)
                        .order_by(MqttOutboxEvent.created_at_ms).limit(20)).all()
                    batch = [(row.event_id, row.topic, row.payload_json, row.qos) for row in rows]
                if not batch:
                    self._stop.wait(0.5)
                    continue
                for event_id, topic, payload_json, qos in batch:
                    if self._stop.is_set():
                        break
                    try:
                        self._publish_outbox_event(event_id, topic, payload_json, qos)
                        with self.database.sessions.begin() as session:
                            row = session.get(MqttOutboxEvent, event_id)
                            if row is not None:
                                row.status = "published"
                                row.published_at_ms = int(time.time() * 1000)
                                row.last_error = None
                        self._mqtt_connected = True
                    except Exception as exc:
                        log.warning("MQTT event publish failed event_id=%s: %s", event_id, exc)
                        self._mqtt_connected = False
                        self._close_mqtt_client()
                        with self.database.sessions.begin() as session:
                            row = session.get(MqttOutboxEvent, event_id)
                            if row is not None:
                                row.attempts += 1
                                delay_seconds = min(3600, 2 ** min(row.attempts, 12))
                                row.next_attempt_at_ms = int(time.time() * 1000) + delay_seconds * 1000
                                row.last_error = str(exc)[:2000]
                        self._stop.wait(0.25)
            except Exception:
                log.exception("MQTT outbox worker failed")
                self._mqtt_connected = False
                self._close_mqtt_client()
                self._stop.wait(1)

    def _record_upload_error(self, *, image_id: int, snapshot: dict[str, Any] | None, exc: Exception) -> None:
        from core.infrastructure.database.event_repository import SQLiteEventLogger
        with self.database.sessions.begin() as session:
            row = session.get(CaptureImage, image_id)
            if row is not None:
                row.cloud_upload_status = "failed"
                snapshot = {"id": row.id, "capture_id": row.capture_id, "camera_id": row.camera_id,
                            "image_type": row.image_type}
            event = {
                "event_type": "upload.error", "status": "failed", "timestamp_ms": int(time.time() * 1000),
                "capture_id": snapshot.get("capture_id") if snapshot else None,
                "camera_id": snapshot.get("camera_id") if snapshot else None,
                "data": {"image_id": image_id, "capture_id": snapshot.get("capture_id") if snapshot else None,
                    "camera_id": snapshot.get("camera_id") if snapshot else None,
                    "image_type": snapshot.get("image_type") if snapshot else None,
                    "error_code": type(exc).__name__, "message": str(exc)[:500]},
            }
            SQLiteEventLogger(self.database, publisher=self).write_in_session(session, event)

    def upload_image(self, image_id: int) -> dict[str, Any]:
        snapshot = None
        try:
            with self.database.sessions() as session:
                row = session.get(CaptureImage, image_id)
                if row is None:
                    raise ValueError("图片记录不存在")
                path = (self.image_root / row.file_path).resolve()
                if self.image_root not in path.parents or not path.is_file():
                    raise FileNotFoundError("本地图片文件不存在")
                snapshot = {"id": row.id, "capture_id": row.capture_id, "camera_id": row.camera_id,
                            "captured_at_ms": row.captured_at_ms, "image_type": row.image_type,
                            "file_path": row.file_path, "file_size": row.file_size}
            cfg, client = self._s3_client()
            key = "/".join(x.strip("/") for x in [cfg.get("prefix", "edge-face"), snapshot["camera_id"],
                str(snapshot["captured_at_ms"]), Path(snapshot["file_path"]).name] if x)
            client.upload_file(str(path), cfg["bucket"], key, ExtraArgs={"ContentType": "image/jpeg"})
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        except Exception as exc:
            self._record_upload_error(image_id=image_id, snapshot=snapshot, exc=exc)
            raise

        uploaded_at_ms = int(time.time() * 1000)
        from core.infrastructure.database.event_repository import SQLiteEventLogger
        event = {
            "event_type": "upload.complete", "status": "success", "timestamp_ms": uploaded_at_ms,
            "capture_id": snapshot["capture_id"], "camera_id": snapshot["camera_id"],
            "data": {"capture_id": snapshot["capture_id"], "camera_id": snapshot["camera_id"],
                "image_id": snapshot["id"], "image_type": snapshot["image_type"],
                "image_key": key, "bucket": cfg["bucket"], "image_sha256": digest.hexdigest(),
                "file_size": snapshot["file_size"], "content_type": "image/jpeg",
                "storage_provider": "s3_compatible", "uploaded_at_ms": uploaded_at_ms},
        }
        # Persist upload state, local event and MQTT outbox row atomically. If this
        # transaction fails after S3 accepted the object, do not misreport an upload error.
        with self.database.sessions.begin() as session:
            row = session.get(CaptureImage, image_id)
            if row is not None:
                row.cloud_key, row.cloud_uploaded_at_ms = key, uploaded_at_ms
                row.cloud_upload_status = "uploaded"
            SQLiteEventLogger(self.database, publisher=self).write_in_session(session, event)
        return {"ok": True, "key": key}

    def enqueue_image(self, image_id: int) -> None:
        if not self._read().get("auto_upload"):
            return
        try:
            with self.database.sessions.begin() as session:
                row = session.get(CaptureImage, image_id)
                if row is not None:
                    row.cloud_upload_status = "queued"
            self._queue.put_nowait(("image", image_id))
        except queue.Full:
            log.error("cloud upload queue full; image_id=%s remains local", image_id)
            snapshot = None
            with self.database.sessions.begin() as session:
                row = session.get(CaptureImage, image_id)
                if row is not None:
                    row.cloud_upload_status = "failed"
                    snapshot = {"id": row.id, "capture_id": row.capture_id, "camera_id": row.camera_id,
                                "image_type": row.image_type}
            self._record_upload_error(image_id=image_id, snapshot=snapshot,
                                      exc=RuntimeError("cloud upload queue is full"))

    def start(self) -> None:
        if any(thread.is_alive() for thread in self._threads):
            return
        self._stop.clear()
        self._threads = [
            threading.Thread(target=self._upload_loop, name="cloud-upload-worker", daemon=True),
            threading.Thread(target=self._mqtt_loop, name="mqtt-outbox-worker", daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def stop(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=5)
        self._threads.clear()
        self._close_mqtt_client()

    def _upload_loop(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                kind, payload = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                if kind == "image":
                    self.upload_image(int(payload))
            except Exception:
                log.exception("cloud background task failed kind=%s", kind)
            finally:
                self._queue.task_done()
