from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Frame:
    image: np.ndarray | None
    monotonic_ts: float
    wall_ts: float
    jpeg: bytes | None = None


class RTSPReader:
    """One reconnecting reader thread per configured RTSP stream."""

    def __init__(self, camera_id: str, stream_name: str, url: str, buffer_seconds: float = 1.5,
                 ring_sample_interval_ms: int = 200, ring_jpeg_quality: int = 96):
        self.camera_id = camera_id
        self.stream_name = stream_name
        self.url = url
        self.buffer_seconds = max(0.2, buffer_seconds)
        self.ring_sample_interval = max(50, ring_sample_interval_ms) / 1000.0
        self.ring_jpeg_quality = max(80, min(100, ring_jpeg_quality))
        self.last_ring_sample_ts = 0.0
        self.latest: Optional[Frame] = None
        self.ring: deque[Frame] = deque()
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.connected = False
        self.reconnects = 0
        self.frames_read = 0
        self.last_error: Optional[str] = None

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.thread = threading.Thread(target=self._run, name=f"rtsp-{self.camera_id}-{self.stream_name}", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5)

    def _run(self) -> None:
        backoff = 1.0
        while not self.stop_event.is_set():
            try:
                cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG, [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000,
                ])
            except TypeError:  # compatibility with OpenCV builds lacking open-parameter overload
                cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                self.connected = False
                self.last_error = "open_failed"
                self.reconnects += 1
                cap.release()
                self.stop_event.wait(backoff)
                backoff = min(backoff * 2, 15.0)
                continue
            self.connected = True
            self.last_error = None
            backoff = 1.0
            log.info("RTSP connected camera=%s stream=%s", self.camera_id, self.stream_name)
            failures = 0
            try:
                while not self.stop_event.is_set():
                    ok, image = cap.read()
                    if not ok or image is None:
                        failures += 1
                        if failures >= 5:
                            break
                        self.stop_event.wait(0.05)
                        continue
                    failures = 0
                    now_mono, now_wall = time.monotonic(), time.time()
                    frame = Frame(image=image, monotonic_ts=now_mono, wall_ts=now_wall)
                    with self.lock:
                        self.latest = frame
                        self.frames_read += 1
                        if self.stream_name == "main":
                            if now_mono - self.last_ring_sample_ts >= self.ring_sample_interval:
                                ok_jpeg, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self.ring_jpeg_quality])
                                if ok_jpeg:
                                    # Store compressed samples; raw 2K frames in a 1.5s ring can consume hundreds of MB.
                                    self.ring.append(Frame(image=None, monotonic_ts=now_mono,
                                                           wall_ts=now_wall, jpeg=encoded.tobytes()))
                                    self.last_ring_sample_ts = now_mono
                            cutoff = now_mono - self.buffer_seconds
                            while self.ring and self.ring[0].monotonic_ts < cutoff:
                                self.ring.popleft()
            finally:
                self.connected = False
                cap.release()
            if not self.stop_event.is_set():
                self.reconnects += 1
                log.warning("RTSP disconnected; reconnecting camera=%s stream=%s", self.camera_id, self.stream_name)
                self.stop_event.wait(backoff)
                backoff = min(backoff * 2, 15.0)

    def get_latest(self) -> Optional[Frame]:
        with self.lock:
            return self.latest

    def nearest(self, monotonic_ts: float, tolerance_ms: int) -> Optional[Frame]:
        with self.lock:
            if not self.ring:
                return None
            candidate = min(self.ring, key=lambda f: abs(f.monotonic_ts - monotonic_ts))
        if abs(candidate.monotonic_ts - monotonic_ts) * 1000 > tolerance_ms:
            return None
        if candidate.image is None and candidate.jpeg:
            decoded = cv2.imdecode(np.frombuffer(candidate.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if decoded is None:
                return None
            return Frame(image=decoded, monotonic_ts=candidate.monotonic_ts, wall_ts=candidate.wall_ts)
        return candidate
