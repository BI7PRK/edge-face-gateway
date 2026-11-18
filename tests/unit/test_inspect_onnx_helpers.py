from tools.inspect_onnx import build_warnings


def test_warns_for_nms_and_dynamic_shapes():
    warnings = build_warnings(
        inputs=[{"name": "images", "dynamic": True}],
        outputs=[{"name": "output", "dynamic": False}],
        nms_nodes=[{"name": "nms"}],
        custom_domains=[],
        external_tensors=[],
    )
    assert any("Dynamic" in item for item in warnings)
    assert any("NonMaxSuppression" in item for item in warnings)


def test_no_warning_for_static_simple_graph():
    assert build_warnings(
        inputs=[{"name": "images", "dynamic": False}],
        outputs=[{"name": "output", "dynamic": False}],
        nms_nodes=[],
        custom_domains=[],
        external_tensors=[],
    ) == []
