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
    min_capture_score: float = 0.78

    @property
    def mode(self) -> str:
        return "dual" if self.sub_url else "single"


def load_config(path: str) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return _expand_env(raw)
