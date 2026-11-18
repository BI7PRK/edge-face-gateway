from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int = 0


class InferenceBackend(ABC):
    @abstractmethod
    def infer(self, image: np.ndarray) -> list[Detection]:
        raise NotImplementedError
