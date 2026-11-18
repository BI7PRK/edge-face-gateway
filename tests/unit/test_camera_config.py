from core.runtime.config import parse_cameras


def test_dual_stream_camera_is_detect_sub_capture_main():
    cameras = parse_cameras({"cameras": [{"id": "a", "streams": {
        "main": {"url": "rtsp://main", "enabled": True},
        "sub": {"url": "rtsp://sub", "enabled": True},
    }}]})
    assert len(cameras) == 1
    assert cameras[0].mode == "dual"
    assert cameras[0].main_url == "rtsp://main"
    assert cameras[0].sub_url == "rtsp://sub"


def test_single_stream_camera_uses_main_for_detection_and_capture():
    cameras = parse_cameras({"cameras": [{"id": "b", "streams": {
        "main": {"url": "rtsp://main", "enabled": True},
        "sub": {"enabled": False},
    }}]})
    assert len(cameras) == 1
    assert cameras[0].mode == "single"
    assert cameras[0].sub_url is None


def test_missing_main_stream_skips_camera():
    cameras = parse_cameras({"cameras": [{"id": "c", "streams": {
        "main": {"url": "", "enabled": True},
        "sub": {"url": "rtsp://sub", "enabled": True},
    }}]})
    assert cameras == []


def test_duplicate_camera_ids_are_rejected():
    item = {"id": "same", "streams": {"main": {"url": "rtsp://main"}}}
    try:
        parse_cameras({"cameras": [item, item]})
        assert False, "expected duplicate camera ID validation"
    except ValueError as exc:
        assert "unique" in str(exc)
