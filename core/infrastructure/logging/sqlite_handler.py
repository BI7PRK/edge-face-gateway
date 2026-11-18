from __future__ import annotations
import json
import logging
import queue
import threading
import time
import traceback
from pathlib import Path
from typing import Any
from core.infrastructure.database.models import ApplicationLog


class SQLiteLogHandler(logging.Handler):
    """Asynchronous batched SQLite logging with a disk spool for queue/DB outages."""
    def __init__(self, database, capacity: int = 10000, batch_size: int = 100):
        super().__init__()
        self.database = database
        self.queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=capacity)
        self.batch_size = batch_size
        self.spool_path = database.path.parent / "pending-logs.jsonl"
        self.spool_path.parent.mkdir(parents=True, exist_ok=True)
        self._spool_lock = threading.Lock()
        self._closed_event = threading.Event()
        self._inflight = False
        self._state_lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="sqlite-log-writer", daemon=True)
        self._thread.start()
        self.dropped = 0

    def _spool(self, payload: dict[str, Any]) -> None:
        with self._spool_lock, self.spool_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
            stream.flush()

    def _take_spooled(self, limit: int) -> list[dict[str, Any]]:
        if not self.spool_path.exists():
            return []
        with self._spool_lock:
            if not self.spool_path.exists():
                return []
            lines = self.spool_path.read_text(encoding="utf-8").splitlines()
            selected, remaining = lines[:limit], lines[limit:]
            if remaining:
                temp = self.spool_path.with_suffix(".tmp")
                temp.write_text("\n".join(remaining) + "\n", encoding="utf-8")
                temp.replace(self.spool_path)
            else:
                self.spool_path.unlink(missing_ok=True)
        result = []
        for line in selected:
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                # Keep corrupt lines for manual inspection rather than silently discarding evidence.
                corrupt = self.spool_path.with_suffix(".corrupt.jsonl")
                with corrupt.open("a", encoding="utf-8") as stream:
                    stream.write(line + "\n")
                continue
        return result

    def emit(self, record: logging.LogRecord) -> None:
        try:
            exc = "".join(traceback.format_exception(*record.exc_info)) if record.exc_info else None
            payload = {"timestamp_ms": int(record.created * 1000), "level": record.levelname,
                "logger": record.name, "category": getattr(record, "category", "SYSTEM"),
                "camera_id": getattr(record, "camera_id", None), "message": record.getMessage(),
                "details_json": json.dumps(getattr(record, "details", None), ensure_ascii=False)
                    if getattr(record, "details", None) is not None else None,
                "exception": exc}
            try:
                self.queue.put_nowait(payload)
            except queue.Full:
                # Disk spool is the overflow path; the writer replays it after the DB is available.
                self._spool(payload)
        except Exception:
            self.handleError(record)

    def _requeue(self, batch: list[dict[str, Any]]) -> None:
        for item in batch:
            try:
                self.queue.put_nowait(item)
            except queue.Full:
                self._spool(item)

    def _run(self):
        while not self._closed_event.is_set() or not self.queue.empty():
            batch = self._take_spooled(self.batch_size)
            if not batch:
                try:
                    item = self.queue.get(timeout=0.2)
                    if item is None:
                        self._closed_event.set()
                        continue
                    batch.append(item)
                except queue.Empty:
                    continue
            while len(batch) < self.batch_size:
                try:
                    item = self.queue.get_nowait()
                    if item is None:
                        self._closed_event.set()
                        break
                    batch.append(item)
                except queue.Empty:
                    break
            with self._state_lock:
                self._inflight = True
            try:
                with self.database.sessions.begin() as session:
                    session.add_all([ApplicationLog(**item) for item in batch])
            except Exception as exc:
                try:
                    import sys
                    sys.stderr.write(f"SQLite log persistence failure; retaining batch for retry: {exc}\n")
                except Exception:
                    pass
                time.sleep(0.25)
                self._requeue(batch)
            finally:
                with self._state_lock:
                    self._inflight = False

    def flush(self):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            spool_pending = self.spool_path.exists() and self.spool_path.stat().st_size > 0
            with self._state_lock:
                active = self._inflight
            if self.queue.empty() and not spool_pending and not active:
                return
            time.sleep(0.02)

    def close(self):
        if self._thread.is_alive():
            # A sentinel stops the worker after queued entries. Any remaining spool is durable
            # and will be replayed at the next process start.
            try:
                self.queue.put(None, timeout=1)
            except queue.Full:
                self._closed_event.set()
            self._thread.join(timeout=5)
        super().close()
