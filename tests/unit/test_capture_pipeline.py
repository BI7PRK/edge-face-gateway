import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from core.capture.stream import Frame
from core.runtime.engine import RuntimeEngine


class EventSink:
    def __init__(self):
        self.rows = []

    def write(self, row):
        self.rows.append(row)


class Metrics:
    def __init__(self):
        self.values = {}

    def increment(self, name, amount=1):
        self.values[name] = self.values.get(name, 0) + amount


def test_capture_saves_high_resolution_frame_annotation_and_face_crop(tmp_path: Path):
    engine = object.__new__(RuntimeEngine)
    engine.capture_cfg = {"cooldown_seconds": 0, "jpeg_quality": 96,
                          "face_jpeg_quality": 98, "face_crop_padding_ratio": 0.25,
                          "quality_assessment": {"min_face_width_px": 0, "min_face_height_px": 0,
                                                 "min_sharpness": 0, "min_brightness": 0,
                                                 "max_brightness": 255, "max_noise": 1000, "min_score": 0}}
    engine.output_dir = tmp_path
    engine.event_log = EventSink()

    source_image = np.random.default_rng(7).integers(50, 200, (600, 800, 3), dtype=np.uint8)
    detection_image = np.zeros((300, 400, 3), dtype=np.uint8)
    source = Frame(source_image, 10.0, 1795000000.0)
    detection = Frame(detection_image, 10.0, 1795000000.0)
    runtime = SimpleNamespace(
        cfg=SimpleNamespace(camera_id="cam-01"), mode="single", last_capture_mono=0.0,
        metrics=Metrics(), capture_source=lambda _frame: source,
    )
    face = SimpleNamespace(x1=100, y1=50, x2=200, y2=150, score=0.93, class_id=0)

    engine._maybe_capture(runtime, detection, detection_image, [face])

    files = list((tmp_path / "cam-01").glob("*.jpg"))
    assert len(files) == 3
    frame_path = next(p for p in files if p.name.endswith("_frame.jpg"))
    annotated_path = next(p for p in files if p.name.endswith("_detections.jpg"))
    face_path = next(p for p in files if "_face_" in p.name)
    assert cv2.imread(str(frame_path)).shape[:2] == (600, 800)
    assert cv2.imread(str(annotated_path)).shape[:2] == (600, 800)
    crop = cv2.imread(str(face_path))
    assert crop.shape[:2] == (300, 300)
    assert runtime.metrics.values["captures_saved"] == 1
    assert runtime.metrics.values["faces_quality_accepted"] == 1
    assert runtime.metrics.values["faces_quality_rejected"] == 0
    kinds = [r.get("event_type") for r in engine.event_log.rows]
    assert "capture.detected" in kinds
    assert "quality.face_accepted" in kinds
    assert kinds.count("artifact.saved") == 3


def test_rejected_face_is_not_saved_as_face_crop(tmp_path: Path):
    engine = object.__new__(RuntimeEngine)
    engine.capture_cfg = {"cooldown_seconds": 0, "jpeg_quality": 90, "face_jpeg_quality": 95,
                          "face_crop_padding_ratio": 0.25}
    engine.output_dir = tmp_path
    engine.event_log = EventSink()
    source_image = np.zeros((300, 400, 3), dtype=np.uint8)
    detection_image = np.zeros((300, 400, 3), dtype=np.uint8)
    source = Frame(source_image, 10.0, 1795000000.0)
    detection = Frame(detection_image, 10.0, 1795000000.0)
    runtime = SimpleNamespace(cfg=SimpleNamespace(camera_id="cam-02"), mode="single",
                               last_capture_mono=0.0, metrics=Metrics(),
                               capture_source=lambda _frame: source)
    face = SimpleNamespace(x1=100, y1=80, x2=220, y2=200, score=0.9, class_id=0)

    engine._maybe_capture(runtime, detection, detection_image, [face])

    files = list((tmp_path / "cam-02").glob("*.jpg"))
    assert len(files) == 2  # Diagnostic full frame + annotated frame only.
    assert not any("_face_" in p.name for p in files)
    quality_events = [row for row in engine.event_log.rows if row.get("event_type") == "quality.face_rejected"]
    assert len(quality_events) == 1
    assert quality_events[0]["status"] == "rejected"
    assert "underexposed" in quality_events[0]["reasons"]


def test_capture_score_threshold_rejects_below_default_and_accepts_exactly_078(tmp_path: Path):
    engine = object.__new__(RuntimeEngine)
    engine.capture_cfg = {"cooldown_seconds": 0, "jpeg_quality": 96,
                          "face_jpeg_quality": 98, "face_crop_padding_ratio": 0.25,
                          "quality_assessment": {"min_face_width_px": 0, "min_face_height_px": 0,
                                                 "min_sharpness": 0, "min_brightness": 0,
                                                 "max_brightness": 255, "max_noise": 1000, "min_score": 0}}
    engine.output_dir = tmp_path
    engine.event_log = EventSink()
    source_image = np.random.default_rng(11).integers(50, 200, (300, 400, 3), dtype=np.uint8)
    detection = Frame(np.zeros((150, 200, 3), dtype=np.uint8), 10.0, 1795000000.0)
    runtime = SimpleNamespace(cfg=SimpleNamespace(camera_id="cam-03"), mode="single",
                              last_capture_mono=0.0, metrics=Metrics(),
                              capture_source=lambda _frame: Frame(source_image, 10.0, 1795000000.0))
    faces = [
        SimpleNamespace(x1=20, y1=20, x2=100, y2=100, score=0.779, class_id=0),
        SimpleNamespace(x1=150, y1=20, x2=230, y2=100, score=0.780, class_id=0),
    ]

    engine._maybe_capture(runtime, detection, detection.image, faces)

    face_files = [p for p in (tmp_path / "cam-03").glob("*_face_*.jpg")]
    assert len(face_files) == 1
    assert "_face_002_" in face_files[0].name
    score_events = [row for row in engine.event_log.rows
                    if row.get("event_type") == "capture.score_rejected"]
    assert len(score_events) == 1
    assert score_events[0]["min_capture_score"] == 0.78
    assert score_events[0]["rejected_faces"] == [{"face_index": 1, "confidence": 0.779}]
    assert runtime.metrics.values["faces_quality_rejected"] == 1


def test_capture_with_only_below_threshold_faces_saves_no_images(tmp_path: Path):
    engine = object.__new__(RuntimeEngine)
    engine.capture_cfg = {"cooldown_seconds": 0, "min_capture_score": 0.78}
    engine.output_dir = tmp_path
    engine.event_log = EventSink()
    source_image = np.zeros((300, 400, 3), dtype=np.uint8)
    detection = Frame(np.zeros((150, 200, 3), dtype=np.uint8), 10.0, 1795000000.0)
    runtime = SimpleNamespace(cfg=SimpleNamespace(camera_id="cam-low-score"), mode="single",
                              last_capture_mono=0.0, metrics=Metrics(),
                              capture_source=lambda _frame: Frame(source_image, 10.0, 1795000000.0))
    faces = [
        SimpleNamespace(x1=20, y1=20, x2=100, y2=100, score=0.779, class_id=0),
        SimpleNamespace(x1=100, y1=20, x2=180, y2=100, score=0.5, class_id=0),
    ]

    engine._maybe_capture(runtime, detection, detection.image, faces)

    assert not list(tmp_path.rglob("*.jpg"))
    kinds = [row.get("event_type") for row in engine.event_log.rows]
    assert kinds == ["capture.score_rejected"]
    rejected = engine.event_log.rows[0]
    assert rejected["face_count"] == 2
    assert rejected["min_capture_score"] == 0.78
    assert rejected["rejected_faces"] == [
        {"face_index": 1, "confidence": 0.779},
        {"face_index": 2, "confidence": 0.5},
    ]
    assert runtime.metrics.values["faces_quality_rejected"] == 2
