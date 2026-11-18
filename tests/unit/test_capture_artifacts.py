from types import SimpleNamespace

import numpy as np

from core.capture.artifacts import annotate_faces, crop_face, map_faces


def test_map_faces_scales_and_clips_boxes_to_high_resolution_frame():
    detections = [SimpleNamespace(x1=10, y1=5, x2=30, y2=25, score=0.91, class_id=0),
                  SimpleNamespace(x1=-5, y1=0, x2=100, y2=100, score=0.8, class_id=0)]
    faces = map_faces(detections, (100, 100, 3), (200, 300, 3))
    assert (faces[0].x1, faces[0].y1, faces[0].x2, faces[0].y2) == (30, 10, 90, 50)
    assert (faces[1].x1, faces[1].y1, faces[1].x2, faces[1].y2) == (0, 0, 300, 200)


def test_crop_face_uses_source_resolution_and_padding():
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    face = map_faces([SimpleNamespace(x1=50, y1=40, x2=100, y2=90, score=0.95, class_id=0)],
                     image.shape, image.shape)[0]
    crop = crop_face(image, face, 0.2)
    assert crop.shape == (70, 70, 3)


def test_annotate_faces_returns_full_size_frame():
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    face = map_faces([SimpleNamespace(x1=50, y1=40, x2=100, y2=90, score=0.95, class_id=0)],
                     image.shape, image.shape)[0]
    annotated = annotate_faces(image, [face], "cam-1")
    assert annotated.shape == image.shape
    assert not np.array_equal(annotated, image)
