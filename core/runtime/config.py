from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV.sub(lambda m: os.getenv(m.group(1), ""), value)
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    return value


@dataclass
class CameraConfig:
    camera_id: str
    name: str
    main_url: str
    sub_url: str | None
    enabled: bool = True
    detection_fps: int = 3

    @property
    def mode(self) -> str:
        return "dual" if self.sub_url else "single"


def load_config(path: str) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return _expand_env(raw)


def parse_cameras(config: dict[str, Any]) -> list[CameraConfig]:
    cameras: list[CameraConfig] = []
    for item in config.get("cameras", []):
        if not item.get("enabled", True):
            continue
        streams = item.get("streams") or {}
        main = streams.get("main") or {}
        sub = streams.get("sub") or {}
        main_url = str(main.get("url") or "").strip() if main.get("enabled", True) else ""
        sub_url = str(sub.get("url") or "").strip() if sub.get("enabled", False) else ""
        # A configured substream without a main stream is not a valid capture setup.
        if not main_url:
            continue
        cameras.append(CameraConfig(
            camera_id=str(item["id"]), name=str(item.get("name") or item["id"]),
            main_url=main_url, sub_url=sub_url or None, detection_fps=int(item.get("detection_fps", 3)),
        ))
    ids = [c.camera_id for c in cameras]
    if len(ids) != len(set(ids)):
        raise ValueError("Camera IDs must be unique")
    return cameras
