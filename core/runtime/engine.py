from __future__ import annotations

import logging
import math
import os
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from core.capture.stream import Frame, RTSPReader
from core.capture.artifacts import annotate_faces, crop_face, map_faces
from core.capture.quality import assess_face_quality
from core.inference.onnx_cpu import OnnxCpuBackend
from .config import CameraConfig
from .monitor import CameraMetrics, JsonlLogger
from core.infrastructure.database.event_repository import SQLiteEventLogger
from core.infrastructure.storage.image_repository import ImageRepository

log = logging.getLogger(__name__)


class CameraRuntime:
    def __init__(self, cfg: CameraConfig, capture_cfg: dict[str, Any]):
        self.cfg = cfg
        buffer_seconds = float(capture_cfg.get("main_stream_buffer_seconds", 1.5))
        ring_sample_ms = int(capture_cfg.get("main_stream_ring_sample_interval_ms", 200))
        ring_quality = int(capture_cfg.get("main_stream_ring_jpeg_quality", 96))
        self.main = RTSPReader(cfg.camera_id, "main", cfg.main_url, buffer_seconds, ring_sample_ms, ring_quality)
        self.sub = RTSPReader(cfg.camera_id, "sub", cfg.sub_url, buffer_seconds) if cfg.sub_url else None
        self.detection_reader = self.sub or self.main
        self.metrics = CameraMetrics(cfg.camera_id)
        self.latest_preview: bytes | None = None
        self.last_processed_ts = 0.0
        self.last_capture_mono = 0.0
        self.capture_cfg = capture_cfg
        self.mode = cfg.mode
        self.detection_fps = max(1, min(30, int(getattr(cfg, "detection_fps", 3))))

    def start(self) -> None:
        self.main.start()
        if self.sub:
            self.sub.start()

    def stop(self) -> None:
        if self.sub:
            self.sub.stop()
        self.main.stop()

    def next_frame(self) -> Frame | None:
        frame = self.detection_reader.get_latest()
        if frame is None or frame.monotonic_ts <= self.last_processed_ts:
            return None
        self.last_processed_ts = frame.monotonic_ts
        return frame

    def capture_source(self, detection_frame: Frame) -> Frame | None:
        if self.mode == "single":
            return detection_frame
        tolerance = int(self.capture_cfg.get("main_stream_match_tolerance_ms", 250))
        return self.main.nearest(detection_frame.monotonic_ts, tolerance)


class RuntimeEngine:
    def __init__(self, cameras: list[CameraConfig], config: dict[str, Any], database=None, camera_repository=None, cloud_services=None):
        inference_cfg = config.get("inference", {})
        capture_cfg = config.get("capture", {})
        model_path = os.getenv("FACE_MODEL_PATH") or inference_cfg.get("model_path") or "/app/models/source/yolov12n-face.onnx"
        self.backend = OnnxCpuBackend(
            model_path=model_path,
            input_size=int(inference_cfg.get("input_size", 640)),
            confidence_threshold=float(inference_cfg.get("confidence_threshold", 0.45)),
            intra_op_num_threads=int(inference_cfg.get("intra_op_num_threads", 2)),
            output_format=str(inference_cfg.get("output_format", "xyxy_score_class")),
            normalized_boxes=bool(inference_cfg.get("normalized_boxes", False)),
        )
        self._runtimes = {c.camera_id: CameraRuntime(c, capture_cfg) for c in cameras}
        self._runtime_lock = threading.RLock()
        self._started = False
        self.max_fps = max(0.1, float(inference_cfg.get("max_fps_per_camera", 3.0)))
        self.capture_cfg = capture_cfg
        self.output_dir = Path(os.getenv("FACE_OUTPUT_DIR") or capture_cfg.get("output_dir", "/app/data"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.database = database
        self.camera_repository = camera_repository
        self.cloud_services = cloud_services
        self.event_log = SQLiteEventLogger(database, publisher=cloud_services) if database is not None else JsonlLogger("/tmp/edge-face-runtime.jsonl")
        self.image_repository = ImageRepository(database, self.output_dir) if database is not None else None
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.host_started = time.time()
        self.host_started_mono = time.monotonic()
        self._camera_connection_states: dict[str, bool | None] = {camera_id: None for camera_id in self._runtimes}

    @property
    def runtimes(self) -> list[CameraRuntime]:
        with self._runtime_lock:
            return list(self._runtimes.values())

    def get_runtime(self, camera_id: str) -> CameraRuntime | None:
        with self._runtime_lock:
            return self._runtimes.get(camera_id)

    def _persist_worker_state(self, camera_id: str, state: str) -> None:
        if self.database is None:
            return
        try:
            from core.infrastructure.database.models import CameraStatus
            with self.database.sessions.begin() as session:
                row = session.get(CameraStatus, camera_id)
                if row is not None:
                    row.worker_state = state
                    row.status = state
                    row.updated_at_ms = int(time.time() * 1000)
        except Exception:
            log.exception("failed to persist camera worker state camera=%s", camera_id)

    def add_or_replace_camera(self, cfg: CameraConfig) -> None:
        # Only the selected camera's readers are stopped/replaced. The shared inference
        # scheduler and every other camera runtime remain alive.
        with self._runtime_lock:
            previous = self._runtimes.pop(cfg.camera_id, None)
            if previous is not None:
                previous.stop()
            if cfg.enabled:
                runtime = CameraRuntime(cfg, self.capture_cfg)
                self._runtimes[cfg.camera_id] = runtime
                if self._started:
                    runtime.start()
                    self._persist_worker_state(cfg.camera_id, "CONNECTING")
            else:
                self._persist_worker_state(cfg.camera_id, "STOPPED")

    def remove_camera(self, camera_id: str) -> None:
        with self._runtime_lock:
            runtime = self._runtimes.pop(camera_id, None)
            if runtime is not None:
                runtime.stop()

    def start(self) -> None:
        if self.cloud_services is not None:
            self.cloud_services.start()
        for runtime in self.runtimes:
            runtime.start()
            self._persist_worker_state(runtime.cfg.camera_id, "CONNECTING")
        self._started = True
        self.thread = threading.Thread(target=self._inference_loop, name="inference-scheduler", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=10)
        for runtime in self.runtimes:
            runtime.stop()
        self._started = False
        if self.cloud_services is not None:
            self.cloud_services.stop()

    @staticmethod
    def test_stream(url: str, timeout_ms: int = 5000) -> dict[str, Any]:
        # Bounded one-shot probe; no persistent worker is created.
        timeout_ms = max(1000, min(10000, int(timeout_ms)))
        cap = None
        started = time.monotonic()
        try:
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG, [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms,
            ])
            if not cap.isOpened():
                return {"ok": False, "error": "无法打开视频流，请检查地址、账号密码及网络"}
            ok, frame = cap.read()
            if not ok or frame is None:
                return {"ok": False, "error": "已连接但无法读取视频帧"}
            return {"ok": True, "width": int(frame.shape[1]), "height": int(frame.shape[0]),
                    "elapsed_ms": round((time.monotonic() - started) * 1000, 1)}
        except Exception:
            return {"ok": False, "error": "视频流测试失败，请检查 RTSP 配置"}
        finally:
            if cap is not None:
                cap.release()

    def _poll_camera_connection_events(self) -> None:
        """Emit only real connection state transitions; startup while disconnected is not an offline alert."""
        now_ms = int(time.time() * 1000)
        for runtime in self.runtimes:
            camera_id = runtime.cfg.camera_id
            connected = bool(runtime.main.connected and (runtime.sub is None or runtime.sub.connected))
            previous = self._camera_connection_states.get(camera_id)
            if connected and previous is not True:
                self.event_log.write({
                    "event_type": "camera.online", "status": "online", "timestamp_ms": now_ms,
                    "camera_id": camera_id, "data": {"camera_id": camera_id, "stream_mode": runtime.mode},
                })
            elif previous is True and not connected:
                self.event_log.write({
                    "event_type": "camera.offline", "status": "offline", "timestamp_ms": now_ms,
                    "camera_id": camera_id, "data": {"camera_id": camera_id, "stream_mode": runtime.mode},
                })
            self._camera_connection_states[camera_id] = connected

    def _inference_loop(self) -> None:
        next_allowed: dict[str, float] = {}
        while not self.stop_event.is_set():
            self._poll_camera_connection_events()
            did_work = False
            now = time.monotonic()
            for runtime in self.runtimes:
                interval = 1.0 / max(1, min(30, runtime.detection_fps))
                if now < next_allowed.get(runtime.cfg.camera_id, 0.0):
                    continue
                frame = runtime.next_frame()
                if frame is None:
                    continue
                did_work = True
                next_allowed[runtime.cfg.camera_id] = now + interval
                try:
                    detections, elapsed_ms = self.backend.infer_with_timing(frame.image)
                    self._handle_result(runtime, frame, detections, elapsed_ms)
                except Exception as exc:  # keep one bad frame from killing other cameras
                    runtime.metrics.error(str(exc))
                    log.exception("inference failed camera=%s", runtime.cfg.camera_id, extra={"category": "DETECTION", "camera_id": runtime.cfg.camera_id})
            if not did_work:
                self.stop_event.wait(0.01)

    def _handle_result(self, runtime: CameraRuntime, detection_frame: Frame, detections: list[Any], elapsed_ms: float) -> None:
        preview = detection_frame.image.copy()
        h, w = preview.shape[:2]
        det_json: list[dict[str, Any]] = []
        for det in detections:
            x1, y1, x2, y2 = map(int, (det.x1, det.y1, det.x2, det.y2))
            cv2.rectangle(preview, (x1, y1), (x2, y2), (40, 220, 40), 2)
            cv2.putText(preview, f"face {det.score:.2f}", (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 220, 40), 2)
            det_json.append({"x1": det.x1, "y1": det.y1, "x2": det.x2, "y2": det.y2, "score": det.score, "class_id": det.class_id})
        cv2.putText(preview, f"{runtime.cfg.camera_id} | {elapsed_ms:.1f} ms | {len(detections)} faces", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)
        ok, encoded = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 82])
        jpeg = encoded.tobytes() if ok else b""
        runtime.metrics.update(elapsed_ms, len(detections), det_json, jpeg)
        runtime.latest_preview = jpeg

        if detections:
            self._maybe_capture(runtime, detection_frame, preview, detections)

    def _maybe_capture(self, runtime: CameraRuntime, detection_frame: Frame, annotated: np.ndarray, detections: list[Any]) -> None:
        cooldown = float(self.capture_cfg.get("cooldown_seconds", 3.0))
        now = time.monotonic()
        if now - runtime.last_capture_mono < cooldown:
            return
        source = runtime.capture_source(detection_frame)
        if source is None or source.image is None:
            runtime.metrics.increment("capture_misses")
            self.event_log.write({"event_type": "capture.missed", "ts": datetime.now(timezone.utc).isoformat(),
                                  "camera_id": runtime.cfg.camera_id, "status": "failed",
                                  "reason": "no_main_frame_within_timestamp_tolerance"})
            return
        runtime.last_capture_mono = now
        capture_id = uuid.uuid4().hex
        quality = max(80, min(100, int(self.capture_cfg.get("jpeg_quality", 96))))
        face_quality = max(80, min(100, int(self.capture_cfg.get("face_jpeg_quality", 98))))
        padding = float(self.capture_cfg.get("face_crop_padding_ratio", 0.25))
        faces = map_faces(detections, detection_frame.image.shape, source.image.shape)
        threshold_raw = getattr(runtime.cfg, "min_capture_score", 0.78)
        try:
            capture_score_threshold = float(threshold_raw)
            if not math.isfinite(capture_score_threshold):
                capture_score_threshold = 0.78
            capture_score_threshold = min(1.0, max(0.0, capture_score_threshold))
        except (TypeError, ValueError):
            capture_score_threshold = 0.78
        score_rejected_faces = [
            {"face_index": index, "confidence": float(face.score)}
            for index, face in enumerate(faces, start=1)
            if face.score < capture_score_threshold
        ]
        if score_rejected_faces:
            self.event_log.write({
                "event_type": "capture.score_rejected", "status": "rejected",
                "timestamp_ms": int(source.wall_ts * 1000), "capture_id": capture_id,
                "camera_id": runtime.cfg.camera_id, "min_capture_score": capture_score_threshold,
                "face_count": len(faces), "rejected_faces": score_rejected_faces,
                "ts": datetime.now(timezone.utc).isoformat(),
            })
        if faces and len(score_rejected_faces) == len(faces):
            runtime.metrics.increment("faces_quality_rejected", len(score_rejected_faces))
            return

        camera_dir = self.output_dir / re.sub(r"[^A-Za-z0-9_.-]", "_", runtime.cfg.camera_id)
        camera_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.fromtimestamp(source.wall_ts, timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        original_path = camera_dir / f"{stamp}_frame.jpg"
        annotated_path = camera_dir / f"{stamp}_detections.jpg"

        # Evaluate quality on the original-resolution face ROI, before adding context padding.
        quality_cfg = self.capture_cfg.get("quality_assessment", {})
        quality_results = [assess_face_quality(source.image, face, quality_cfg) for face in faces]
        capture_annotated = annotate_faces(
            source.image, faces, runtime.cfg.camera_id, quality_results, capture_score_threshold
        )
        saved_files: list[dict[str, Any]] = []
        if cv2.imwrite(str(original_path), source.image, [cv2.IMWRITE_JPEG_QUALITY, quality]):
            saved_files.append({"kind": "frame", "path": original_path, "width": source.image.shape[1],
                                "height": source.image.shape[0], "quality": quality})
        else:
            self.event_log.write({"event_type": "artifact.save_failed", "status": "failed", "capture_id": capture_id,
                                  "camera_id": runtime.cfg.camera_id, "artifact": "frame", "path": str(original_path),
                                  "error": "jpeg_write_failed", "ts": datetime.now(timezone.utc).isoformat()})
            self.event_log.write({"event_type": "capture.failed", "status": "failed",
                                  "capture_id": capture_id, "camera_id": runtime.cfg.camera_id,
                                  "ts": datetime.now(timezone.utc).isoformat(), "reason": "frame_save_failed"})
            runtime.metrics.increment("capture_misses")
            return

        self.event_log.write({
            "event_type": "capture.created", "status": "created",
            "timestamp_ms": int(source.wall_ts * 1000), "capture_id": capture_id,
            "camera_id": runtime.cfg.camera_id,
            "data": {"capture_id": capture_id, "camera_id": runtime.cfg.camera_id,
                     "captured_at_ms": int(source.wall_ts * 1000), "frame_width": source.image.shape[1],
                     "frame_height": source.image.shape[0]},
        })

        if cv2.imwrite(str(annotated_path), capture_annotated, [cv2.IMWRITE_JPEG_QUALITY, quality]):
            saved_files.append({"kind": "annotated", "path": annotated_path, "width": source.image.shape[1],
                                "height": source.image.shape[0], "quality": quality})
        else:
            self.event_log.write({"event_type": "artifact.save_failed", "status": "failed",
                                  "capture_id": capture_id, "camera_id": runtime.cfg.camera_id,
                                  "artifact": "annotated", "path": str(annotated_path),
                                  "ts": datetime.now(timezone.utc).isoformat(), "error": "jpeg_write_failed"})

        face_records: list[dict[str, Any]] = []
        accepted_count = 0
        rejected_count = 0
        for index, (face, quality_result) in enumerate(zip(faces, quality_results), start=1):
            score_rejected = face.score < capture_score_threshold
            record = {"index": index, "score": round(face.score, 4),
                      "box": [face.x1, face.y1, face.x2, face.y2],
                      "accepted": quality_result.accepted and not score_rejected,
                      "quality": quality_result.to_dict(),
                      "min_capture_score": capture_score_threshold}
            if score_rejected:
                rejected_count += 1
                record["accepted"] = False
                record["quality"]["reasons"] = list(record["quality"]["reasons"]) + ["detection_score_too_low"]
                face_records.append(record)
                continue
            if not quality_result.accepted:
                rejected_count += 1
                face_records.append(record)
                self.event_log.write({"event_type": "quality.face_rejected",
                                      "status": "rejected", "capture_id": capture_id,
                                      "camera_id": runtime.cfg.camera_id, "face_index": index,
                                      "score": quality_result.score, "reasons": list(quality_result.reasons),
                                      "metrics": quality_result.to_dict(),
                                      "ts": datetime.now(timezone.utc).isoformat()})
                continue

            crop = crop_face(source.image, face, padding)
            if crop is None:
                rejected_count += 1
                record["accepted"] = False
                record["quality"]["reasons"] = ["crop_failed"]
                face_records.append(record)
                continue
            crop_path = camera_dir / f"{stamp}_face_{index:03d}_score_{face.score:.3f}.jpg"
            if cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, face_quality]):
                accepted_count += 1
                saved_files.append({"kind": "face", "path": crop_path, "width": crop.shape[1],
                                    "height": crop.shape[0], "quality": face_quality,
                                    "confidence": float(face.score), "quality_score": float(quality_result.score)})
                record.update({"image": str(crop_path), "width": crop.shape[1], "height": crop.shape[0]})
                self.event_log.write({"event_type": "quality.face_accepted",
                                      "status": "accepted", "capture_id": capture_id,
                                      "camera_id": runtime.cfg.camera_id, "face_index": index,
                                      "score": quality_result.score, "metrics": quality_result.to_dict(),
                                      "ts": datetime.now(timezone.utc).isoformat()})
            else:
                rejected_count += 1
                record["accepted"] = False
                record["quality"]["reasons"] = ["jpeg_write_failed"]
                self.event_log.write({"event_type": "artifact.save_failed", "status": "failed",
                                      "capture_id": capture_id, "camera_id": runtime.cfg.camera_id,
                                      "artifact": "face", "path": str(crop_path),
                                      "ts": datetime.now(timezone.utc).isoformat(), "error": "jpeg_write_failed"})
            face_records.append(record)

        if getattr(self, "image_repository", None) is not None:
            captured_at_ms = int(source.wall_ts * 1000)
            frame_image_id = None
            registered_images: list[dict[str, Any]] = []
            for item in saved_files:
                try:
                    image_id = self.image_repository.register(
                        capture_id=capture_id, camera_id=runtime.cfg.camera_id,
                        image_type=item["kind"], path=item["path"], captured_at_ms=captured_at_ms,
                        width=item["width"], height=item["height"],
                        confidence=item.get("confidence"), quality_score=item.get("quality_score"),
                        parent_image_id=frame_image_id if item["kind"] == "face" else None)
                    if item["kind"] == "frame":
                        frame_image_id = image_id
                    registered_images.append({"image_id": image_id, "image_type": item["kind"]})
                except Exception:
                    log.exception("image metadata persistence failed camera=%s path=%s", runtime.cfg.camera_id,
                                  item["path"], extra={"category": "STORAGE", "camera_id": runtime.cfg.camera_id})
            if frame_image_id is not None:
                self.event_log.write({
                    "event_type": "capture.saved", "status": "saved",
                    "timestamp_ms": int(time.time() * 1000), "capture_id": capture_id,
                    "camera_id": runtime.cfg.camera_id,
                    "data": {"capture_id": capture_id, "camera_id": runtime.cfg.camera_id,
                             "frame_image_id": frame_image_id, "images": registered_images,
                             "saved_at_ms": int(time.time() * 1000)},
                })
                # Queue cloud uploads only after capture.saved has been committed so
                # downstream upload.complete events cannot overtake the saved event.
                if self.cloud_services is not None:
                    for registered in registered_images:
                        self.cloud_services.enqueue_image(registered["image_id"])

        self.event_log.write({"event_type": "capture.detected", "status": "completed",
                              "capture_id": capture_id, "ts": datetime.now(timezone.utc).isoformat(),
                              "camera_id": runtime.cfg.camera_id, "mode": runtime.mode,
                              "source_frame_ts": source.wall_ts, "face_count": len(faces),
                              "accepted_face_count": accepted_count, "rejected_face_count": rejected_count,
                              "frame_width": source.image.shape[1], "frame_height": source.image.shape[0]})
        runtime.metrics.increment("faces_quality_accepted", accepted_count)
        runtime.metrics.increment("faces_quality_rejected", rejected_count)
        for item in saved_files:
            self.event_log.write({"event_type": "artifact.saved", "status": "success",
                                  "capture_id": capture_id, "ts": datetime.now(timezone.utc).isoformat(),
                                  "camera_id": runtime.cfg.camera_id, "artifact": item["kind"],
                                  "path": str(item["path"]), "width": item["width"], "height": item["height"],
                                  "jpeg_quality": item["quality"]})
        runtime.metrics.increment("captures_saved")
        self.event_log.write({"event_type": "capture.manifest", "capture_id": capture_id,
                              "ts": datetime.now(timezone.utc).isoformat(), "camera_id": runtime.cfg.camera_id,
                              "mode": runtime.mode, "frame": str(original_path), "annotated": str(annotated_path),
                              "faces": face_records, "accepted_face_count": accepted_count,
                              "rejected_face_count": rejected_count})
