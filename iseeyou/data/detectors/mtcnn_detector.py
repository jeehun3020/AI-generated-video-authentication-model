from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .base import BaseFaceDetector, FaceDetection


class MTCNNFaceDetector(BaseFaceDetector):
    def __init__(
        self,
        device: str = "auto",
        min_face_size: int = 40,
        keep_all: bool = True,
        thresholds: tuple[float, float, float] = (0.6, 0.7, 0.7),
    ):
        try:
            from facenet_pytorch import MTCNN
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "facenet_pytorch is required for MTCNN detector. "
                "Install dependencies via `pip install -r requirements.txt`."
            ) from exc

        resolved_device = self._resolve_device(device)
        self.model = MTCNN(
            keep_all=keep_all,
            device=resolved_device,
            min_face_size=min_face_size,
            thresholds=thresholds,
            post_process=False,
        )

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device

        if torch.cuda.is_available():
            return "cuda"

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"

        return "cpu"

    def detect(self, image_rgb: np.ndarray) -> list[FaceDetection]:
        boxes, probs = self.model.detect(image_rgb)

        if boxes is None or len(boxes) == 0:
            return []

        detections: list[FaceDetection] = []
        h, w = image_rgb.shape[:2]
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box.tolist()
            score = float(probs[idx]) if probs is not None else 1.0

            x1_i = max(0, min(w - 1, int(round(x1))))
            y1_i = max(0, min(h - 1, int(round(y1))))
            x2_i = max(0, min(w - 1, int(round(x2))))
            y2_i = max(0, min(h - 1, int(round(y2))))

            if x2_i <= x1_i or y2_i <= y1_i:
                continue

            detections.append(
                FaceDetection(x1=x1_i, y1=y1_i, x2=x2_i, y2=y2_i, score=score)
            )

        return detections


class RetinaFaceDetectorPlaceholder(BaseFaceDetector):
    def __init__(self, *_args: Any, **_kwargs: Any):
        # TODO: implement RetinaFace backend and keep API identical to MTCNNFaceDetector.
        raise NotImplementedError(
            "RetinaFace backend is not implemented yet. Use detector.name=mtcnn for now."
        )

    def detect(self, image_rgb: np.ndarray) -> list[FaceDetection]:
        raise NotImplementedError
