from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class FaceDetection:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)


class BaseFaceDetector(ABC):
    @abstractmethod
    def detect(self, image_rgb: np.ndarray) -> list[FaceDetection]:
        raise NotImplementedError

    def select_primary(
        self,
        detections: list[FaceDetection],
        image_shape: tuple[int, ...],
    ) -> FaceDetection | None:
        if not detections:
            return None

        # TODO: support temporal tracking for stable face selection across frames.
        return max(detections, key=lambda d: d.area * max(d.score, 1e-6))
