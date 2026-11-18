from types import SimpleNamespace

from core.runtime.engine import RuntimeEngine


def test_camera_online_and_offline_are_emitted_only_on_connection_transitions():
    events = []
    runtime = SimpleNamespace(
        cfg=SimpleNamespace(camera_id="front-door"),
        main=SimpleNamespace(connected=False),
        sub=None,
        mode="single",
    )
    engine = SimpleNamespace(
        runtimes=[runtime],
        _camera_connection_states={"front-door": None},
        event_log=SimpleNamespace(write=events.append),
    )

    RuntimeEngine._poll_camera_connection_events(engine)
    assert events == []  # startup while disconnected is not an offline alert

    runtime.main.connected = True
    RuntimeEngine._poll_camera_connection_events(engine)
    assert [event["event_type"] for event in events] == ["camera.online"]

    RuntimeEngine._poll_camera_connection_events(engine)
    assert len(events) == 1  # no duplicate online message while still connected

    runtime.main.connected = False
    RuntimeEngine._poll_camera_connection_events(engine)
    assert [event["event_type"] for event in events] == ["camera.online", "camera.offline"]
