import cv2
import numpy as np
from types import SimpleNamespace

from core.capture.quality import assess_face_quality


def test_sharp_face_passes_default_quality_gates():
    yy, xx = np.indices((120, 120))
    checker = (((xx // 3 + yy // 3) % 2) * 180 + 35).astype(np.uint8)
    image = cv2.cvtColor(checker, cv2.COLOR_GRAY2BGR)
    face = SimpleNamespace(x1=10, y1=10, x2=110, y2=110)

    result = assess_face_quality(image, face)

    assert result.accepted
    assert result.score >= 45
    assert result.sharpness >= 25
    assert not result.reasons


def test_blurred_face_is_rejected_by_sharpness_gate():
    yy, xx = np.indices((120, 120))
    checker = (((xx // 3 + yy // 3) % 2) * 180 + 35).astype(np.uint8)
    image = cv2.cvtColor(checker, cv2.COLOR_GRAY2BGR)
    image = cv2.GaussianBlur(image, (17, 17), 0)
    face = SimpleNamespace(x1=10, y1=10, x2=110, y2=110)

    result = assess_face_quality(image, face)

    assert not result.accepted
    assert "low_sharpness" in result.reasons


def test_small_face_is_rejected_even_when_other_quality_is_good():
    image = np.full((100, 100, 3), 120, dtype=np.uint8)
    face = SimpleNamespace(x1=10, y1=10, x2=40, y2=40)

    result = assess_face_quality(image, face, {"min_sharpness": 0, "min_score": 0})

    assert not result.accepted
    assert "face_too_small" in result.reasons


def test_underexposed_face_is_rejected():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    face = SimpleNamespace(x1=10, y1=10, x2=90, y2=90)

    result = assess_face_quality(image, face, {"min_sharpness": 0, "min_score": 0})

    assert not result.accepted
    assert "underexposed" in result.reasons


def test_noisy_low_light_face_is_rejected_by_noise_gate():
    yy, xx = np.indices((120, 120))
    checker = (((xx // 3 + yy // 3) % 2) * 180 + 35).astype(np.uint8)
    blurred = cv2.GaussianBlur(checker, (17, 17), 0).astype(np.float32)
    noisy = np.clip(blurred + np.random.default_rng(1).normal(0, 18, checker.shape), 0, 255).astype(np.uint8)
    image = cv2.cvtColor(noisy, cv2.COLOR_GRAY2BGR)
    face = SimpleNamespace(x1=10, y1=10, x2=110, y2=110)

    result = assess_face_quality(image, face)

    assert not result.accepted
    assert "high_noise" in result.reasons
