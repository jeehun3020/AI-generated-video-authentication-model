from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


def iter_video_frames(
    video_path: str | Path,
    target_fps: float,
    max_frames: int | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps is None or native_fps <= 0:
        native_fps = 30.0

    if target_fps <= 0:
        frame_step = 1
    else:
        frame_step = max(int(round(native_fps / target_fps)), 1)

    frame_idx = 0
    kept = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            yield frame_idx, frame_rgb
            kept += 1
            if max_frames is not None and kept >= max_frames:
                break

        frame_idx += 1

    cap.release()


def resize_image(image: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
