from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class FaceQuality:
    accepted: bool
    score: float
    reasons: tuple[str, ...]
    face_width: int
    face_height: int
    sharpness: float
    brightness: float
    contrast: float
    noise: float
    dark_pixel_ratio: float
    bright_pixel_ratio: float

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["reasons"] = list(self.reasons)
        return result


def assess_face_quality(image: np.ndarray, face: Any, config: dict[str, Any] | None = None) -> FaceQuality:
    """Assess image quality from the original-resolution face ROI.

    Sharpness is an image-based blur proxy, not a reliable diagnosis of the
    physical cause (motion, focus, optics or compression). Thresholds are
    intentionally configurable and should be calibrated with field samples.
    """
    cfg = config or {}
    height, width = image.shape[:2]
    x1 = max(0, min(width, int(face.x1)))
    y1 = max(0, min(height, int(face.y1)))
    x2 = max(0, min(width, int(face.x2)))
    y2 = max(0, min(height, int(face.y2)))
    roi = image[y1:y2, x1:x2]
    face_width, face_height = max(0, x2 - x1), max(0, y2 - y1)

    if roi.size == 0 or face_width < 1 or face_height < 1:
        return FaceQuality(False, 0.0, ("invalid_face_region",), face_width, face_height,
                           0.0, 0.0, 0.0, 0.0, 1.0, 0.0)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi.astype(np.uint8)
    gray_f = gray.astype(np.float32)
    brightness = float(gray_f.mean())
    contrast = float(gray_f.std())
    # Estimate sensor/compression noise only in locally flat areas so real face
    # edges do not inflate the noise measurement.
    grad_x = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    flat_mask = np.sqrt(grad_x * grad_x + grad_y * grad_y) < 20.0
    residual = gray_f - cv2.GaussianBlur(gray_f, (3, 3), 0.8)
    noise_samples = residual[flat_mask] if int(flat_mask.sum()) >= max(16, gray.size // 20) else residual.reshape(-1)
    noise_median = float(np.median(noise_samples))
    noise = float(np.median(np.abs(noise_samples - noise_median)) * 1.4826)
    # A small bilateral filter suppresses isolated night noise before the
    # sharpness estimate; it is inexpensive and does not change saved images.
    denoise_strength = max(20.0, min(60.0, noise * 3.0))
    denoised = cv2.bilateralFilter(gray, 5, denoise_strength, 3)
    sharpness = float(cv2.Laplacian(denoised, cv2.CV_64F).var())
    dark_ratio = float(np.mean(gray <= 8))
    bright_ratio = float(np.mean(gray >= 247))

    min_width = int(cfg.get("min_face_width_px", 48))
    min_height = int(cfg.get("min_face_height_px", 48))
    min_sharpness = float(cfg.get("min_sharpness", 25.0))
    min_brightness = float(cfg.get("min_brightness", 18.0))
    max_brightness = float(cfg.get("max_brightness", 238.0))
    max_noise = float(cfg.get("max_noise", 14.0))
    min_score = float(cfg.get("min_score", 45.0))

    reasons: list[str] = []
    if face_width < min_width or face_height < min_height:
        reasons.append("face_too_small")
    if sharpness < min_sharpness:
        reasons.append("low_sharpness")
    if brightness < min_brightness or dark_ratio > 0.80:
        reasons.append("underexposed")
    if brightness > max_brightness or bright_ratio > 0.80:
        reasons.append("overexposed")
    if noise > max_noise:
        reasons.append("high_noise")

    # The score helps rank acceptable candidates; hard gates above cannot be
    # offset by strong values in unrelated dimensions.
    sharp_score = min(100.0, sharpness / max(min_sharpness * 3.0, 1.0) * 100.0)
    resolution_score = min(100.0, min(face_width, face_height) / max(min_width * 2.0, 1.0) * 100.0)
    exposure_score = max(0.0, 100.0 - abs(brightness - 110.0) * 1.15)
    contrast_score = min(100.0, contrast / 55.0 * 100.0)
    noise_score = max(0.0, 100.0 - noise * 5.0)
    score = (0.35 * sharp_score + 0.20 * resolution_score + 0.20 * exposure_score
             + 0.10 * contrast_score + 0.15 * noise_score)
    if score < min_score:
        reasons.append("quality_score_too_low")

    return FaceQuality(not reasons, round(score, 2), tuple(reasons), face_width, face_height,
                       round(sharpness, 3), round(brightness, 3), round(contrast, 3),
                       round(noise, 3), round(dark_ratio, 4), round(bright_ratio, 4))
