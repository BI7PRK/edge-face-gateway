from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np
from .backend import Detection, InferenceBackend


class OnnxCpuBackend(InferenceBackend):
    """CPU inference adapter for the inspected fixed NCHW ONNX model."""

    def __init__(self, model_path: str, input_size: int = 640, confidence_threshold: float = 0.45,
                 intra_op_num_threads: int = 2, output_format: str = "xyxy_score_class",
                 normalized_boxes: bool = False):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, int(intra_op_num_threads))
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(model_path, sess_options=options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.output_format = output_format
        self.normalized_boxes = normalized_boxes
        shape = self.session.get_inputs()[0].shape
        if len(shape) != 4 or shape[1] != 3:
            raise ValueError(f"Expected NCHW RGB input, got {shape}")

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, tuple[float, int, int]]:
        h, w = image.shape[:2]
        scale = min(self.input_size / w, self.input_size / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        left, top = (self.input_size - nw) // 2, (self.input_size - nh) // 2
        canvas[top:top + nh, left:left + nw] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))[None, ...]
        return tensor, (scale, left, top)

    def infer_with_timing(self, image: np.ndarray) -> tuple[list[Detection], float]:
        tensor, transform = self._preprocess(image)
        started = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: tensor})
        elapsed_ms = (time.perf_counter() - started) * 1000
        detections = self._decode(outputs[0], image.shape[1], image.shape[0], transform)
        return detections, elapsed_ms

    def infer(self, image: np.ndarray) -> list[Detection]:
        return self.infer_with_timing(image)[0]

    def _decode(self, output: Any, original_w: int, original_h: int,
                transform: tuple[float, int, int]) -> list[Detection]:
        arr = np.asarray(output)
        arr = np.squeeze(arr)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[1] < 5:
            raise ValueError(f"Unsupported model output shape: {np.asarray(output).shape}")
        scale, pad_x, pad_y = transform
        results: list[Detection] = []
        for row in arr:
            if self.output_format == "xyxy_score_class":
                x1, y1, x2, y2, score = map(float, row[:5])
                class_id = int(row[5]) if len(row) > 5 else 0
            elif self.output_format == "xywh_score_class":
                cx, cy, bw, bh, score = map(float, row[:5])
                x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
                class_id = int(row[5]) if len(row) > 5 else 0
            else:
                raise ValueError(f"Unknown output_format: {self.output_format}")
            if not np.isfinite([x1, y1, x2, y2, score]).all() or score < self.confidence_threshold:
                continue
            if self.normalized_boxes:
                x1 *= self.input_size; x2 *= self.input_size
                y1 *= self.input_size; y2 *= self.input_size
            x1 = max(0.0, min(float(original_w), (x1 - pad_x) / scale))
            x2 = max(0.0, min(float(original_w), (x2 - pad_x) / scale))
            y1 = max(0.0, min(float(original_h), (y1 - pad_y) / scale))
            y2 = max(0.0, min(float(original_h), (y2 - pad_y) / scale))
            if x2 > x1 and y2 > y1:
                results.append(Detection(x1, y1, x2, y2, score, class_id))
        return results
