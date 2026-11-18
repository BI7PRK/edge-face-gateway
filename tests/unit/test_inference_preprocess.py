import numpy as np

from core.inference.onnx_cpu import OnnxCpuBackend


def test_preprocess_letterboxes_to_fixed_rgb_nchw():
    backend = object.__new__(OnnxCpuBackend)
    backend.input_size = 640
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    tensor, transform = backend._preprocess(image)
    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert transform == (1.0, 0, 140)
    assert 0.0 <= float(tensor.min()) <= float(tensor.max()) <= 1.0


def test_decode_xyxy_score_class_maps_letterbox_coordinates():
    backend = object.__new__(OnnxCpuBackend)
    backend.input_size = 640
    backend.confidence_threshold = 0.4
    backend.output_format = "xyxy_score_class"
    backend.normalized_boxes = False
    output = np.array([[[100, 150, 300, 350, 0.9, 0]]], dtype=np.float32)
    detections = backend._decode(output, 640, 360, (1.0, 0, 140))
    assert len(detections) == 1
    assert detections[0].x1 == 100
    assert detections[0].y1 == 10
    assert detections[0].x2 == 300
    assert detections[0].y2 == 210
