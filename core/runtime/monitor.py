from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any


class CameraMetrics:
    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        self.lock = threading.Lock()
        self.started_at = time.time()
        self.inference_times: deque[float] = deque(maxlen=300)
        self.detection_fps_samples: deque[float] = deque(maxlen=60)
        self.last_inference_at = 0.0
        self.last_frame_at = 0.0
        self.inferences = 0
        self.frames_processed = 0
        self.faces_detected = 0
        self.captures_saved = 0
        self.capture_misses = 0
        self.faces_quality_accepted = 0
        self.faces_quality_rejected = 0
        self.last_face_count = 0
        self.last_error: str | None = None
        self.last_detections: list[dict[str, Any]] = []
        self.last_preview_jpeg: bytes | None = None

    def update(self, inference_ms: float, face_count: int, detections: list[dict[str, Any]], preview_jpeg: bytes) -> None:
        now = time.time()
        with self.lock:
            if self.last_inference_at:
                delta = now - self.last_inference_at
                if delta > 0:
                    self.detection_fps_samples.append(1.0 / delta)
            self.last_inference_at = now
            self.last_frame_at = now
            self.inferences += 1
            self.frames_processed += 1
            self.faces_detected += face_count
            self.last_face_count = face_count
            self.inference_times.append(inference_ms)
            self.last_detections = detections
            self.last_preview_jpeg = preview_jpeg
            self.last_error = None

    def increment(self, field: str, amount: int = 1) -> None:
        with self.lock:
            setattr(self, field, getattr(self, field) + amount)

    def error(self, message: str) -> None:
        with self.lock:
            self.last_error = message

    def snapshot(self, mode: str, main_reader: Any, sub_reader: Any = None) -> dict[str, Any]:
        with self.lock:
            times = sorted(self.inference_times)
            p95 = times[min(len(times) - 1, int(len(times) * 0.95))] if times else 0.0
            avg = sum(times) / len(times) if times else 0.0
            fps = sum(self.detection_fps_samples) / len(self.detection_fps_samples) if self.detection_fps_samples else 0.0
            result = {
                "camera_id": self.camera_id, "mode": mode,
                "inference_fps": round(fps, 2), "last_inference_ms": round(self.inference_times[-1], 2) if self.inference_times else 0,
                "avg_inference_ms": round(avg, 2), "p95_inference_ms": round(p95, 2),
                "inferences": self.inferences, "faces_detected_total": self.faces_detected,
                "last_face_count": self.last_face_count, "captures_saved": self.captures_saved,
                "capture_misses": self.capture_misses,
                "faces_quality_accepted": self.faces_quality_accepted,
                "faces_quality_rejected": self.faces_quality_rejected,
                "last_error": self.last_error,
                "last_frame_at": self.last_frame_at or None,
                "main_connected": bool(main_reader.connected), "main_frames": main_reader.frames_read,
                "main_reconnects": main_reader.reconnects,
                "sub_connected": bool(sub_reader.connected) if sub_reader else None,
                "sub_frames": sub_reader.frames_read if sub_reader else None,
                "sub_reconnects": sub_reader.reconnects if sub_reader else None,
            }
            return result


class JsonlLogger:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def write(self, payload: dict[str, Any]) -> None:
        with self.lock, self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
