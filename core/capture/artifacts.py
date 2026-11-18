from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class MappedFace:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    class_id: int


def map_faces(detections: list[Any], detection_shape: tuple[int, ...], source_shape: tuple[int, ...]) -> list[MappedFace]:
    """Map detector-frame boxes to source-frame pixels, clipping to image bounds."""
    detect_h, detect_w = detection_shape[:2]
    source_h, source_w = source_shape[:2]
    if detect_h <= 0 or detect_w <= 0 or source_h <= 0 or source_w <= 0:
        return []
    sx, sy = source_w / detect_w, source_h / detect_h
    result: list[MappedFace] = []
    for det in detections:
        x1 = max(0, min(source_w, int(round(det.x1 * sx))))
        y1 = max(0, min(source_h, int(round(det.y1 * sy))))
        x2 = max(0, min(source_w, int(round(det.x2 * sx))))
        y2 = max(0, min(source_h, int(round(det.y2 * sy))))
        if x2 > x1 and y2 > y1:
            result.append(MappedFace(x1, y1, x2, y2, float(det.score), int(getattr(det, "class_id", 0))))
    return result


def crop_face(image: np.ndarray, face: MappedFace, padding_ratio: float = 0.25) -> np.ndarray | None:
    """Crop from the original high-resolution frame with contextual padding."""
    h, w = image.shape[:2]
    pad_x = int(round((face.x2 - face.x1) * max(0.0, padding_ratio)))
    pad_y = int(round((face.y2 - face.y1) * max(0.0, padding_ratio)))
    x1, y1 = max(0, face.x1 - pad_x), max(0, face.y1 - pad_y)
    x2, y2 = min(w, face.x2 + pad_x), min(h, face.y2 + pad_y)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image[y1:y2, x1:x2]
    return crop.copy() if crop.size else None


def annotate_faces(image: np.ndarray, faces: list[MappedFace], camera_id: str, quality_results: list[Any] | None = None) -> np.ndarray:
    annotated = image.copy()
    h, w = annotated.shape[:2]
    thickness = max(2, round(w / 1280))
    font_scale = max(0.6, w / 1920)
    for index, face in enumerate(faces):
        quality = quality_results[index] if quality_results and index < len(quality_results) else None
        accepted = quality is None or quality.accepted
        color = (40, 220, 40) if accepted else (30, 30, 235)
        label = f"face {face.score:.2f}"
        if quality is not None:
            label += f" Q:{quality.score:.0f}" if accepted else f" REJECT:{quality.reasons[0]}"
        cv2.rectangle(annotated, (face.x1, face.y1), (face.x2, face.y2), color, thickness)
        cv2.putText(annotated, label, (face.x1, max(24, face.y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    cv2.putText(annotated, f"{camera_id} | {len(faces)} faces", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, max(0.65, w / 1920), (0, 220, 255), thickness)
    return annotated
