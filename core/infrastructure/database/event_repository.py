from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from core.infrastructure.database.models import BusinessEvent

MQTT_EVENT_TYPES = {
    "camera.online",
    "camera.offline",
    "capture.created",
    "capture.saved",
    "upload.complete",
    "upload.error",
}


def _timestamp_ms(payload: dict) -> int:
    value = payload.get("timestamp_ms")
    if isinstance(value, (int, float)):
        return int(value)
    stamp = payload.get("ts")
    if isinstance(stamp, (int, float)):
        return int(stamp * 1000)
    if stamp:
        try:
            return int(datetime.fromisoformat(str(stamp).replace("Z", "+00:00")).timestamp() * 1000)
        except (ValueError, TypeError):
            pass
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _event_type(payload: dict) -> str:
    return str(payload.get("event_type") or "runtime.unknown")[:40]


class SQLiteEventLogger:
    """Persists local business events and atomically stages supported MQTT events."""

    def __init__(self, database, publisher=None):
        self.database = database
        self.publisher = publisher
        self.path = None

    def write(self, payload: dict) -> None:
        with self.database.sessions.begin() as session:
            self.write_in_session(session, payload)

    def write_in_session(self, session, payload: dict) -> None:
        """Persist a business event and its MQTT outbox row in a caller-owned transaction."""
        payload = dict(payload)
        event_type = _event_type(payload)
        if event_type in MQTT_EVENT_TYPES:
            payload.setdefault("event_id", str(uuid.uuid4()))
        timestamp_ms = _timestamp_ms(payload)
        session.add(BusinessEvent(
            event_type=event_type, timestamp_ms=timestamp_ms, camera_id=payload.get("camera_id"),
            capture_id=payload.get("capture_id"), status=payload.get("status"),
            payload_json=json.dumps(payload, ensure_ascii=False, default=str),
        ))
        if self.publisher is not None and event_type in MQTT_EVENT_TYPES:
            self.publisher.add_event_to_outbox(session, payload, timestamp_ms)

    def import_legacy_jsonl(self, path: str) -> int:
        """One-time import of v0.1.x JSONL business events; legacy records are not re-published."""
        source = __import__("pathlib").Path(path)
        if not source.is_file():
            return 0
        from core.infrastructure.database.models import SystemSetting
        with self.database.sessions.begin() as session:
            if session.get(SystemSetting, "legacy_jsonl_imported") is not None:
                return 0
            imported = 0
            batch = []
            with source.open("r", encoding="utf-8") as stream:
                for line in stream:
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    batch.append(BusinessEvent(
                        event_type=_event_type(payload), timestamp_ms=_timestamp_ms(payload),
                        camera_id=payload.get("camera_id"), capture_id=payload.get("capture_id"),
                        status=payload.get("status"),
                        payload_json=json.dumps(payload, ensure_ascii=False, default=str)))
                    if len(batch) >= 500:
                        session.add_all(batch)
                        imported += len(batch)
                        batch.clear()
                        session.flush()
            if batch:
                session.add_all(batch)
                imported += len(batch)
            session.add(SystemSetting(key="legacy_jsonl_imported", value_json=json.dumps({"count": imported}),
                                      updated_at_ms=int(datetime.now(timezone.utc).timestamp() * 1000)))
            return imported
