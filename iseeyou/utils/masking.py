from __future__ import annotations

import random
from typing import Any

import numpy as np
from PIL import Image


def apply_band_mask_np(
    image_rgb: np.ndarray,
    *,
    top_ratio: float = 0.0,
    bottom_ratio: float = 0.0,
    left_ratio: float = 0.0,
    right_ratio: float = 0.0,
    fill_mode: str = "median",
) -> np.ndarray:
    if image_rgb.size == 0:
        return image_rgb

    masked = image_rgb.copy()
    h, w = masked.shape[:2]

    top = max(0, min(h, int(round(h * max(0.0, float(top_ratio))))))
    bottom = max(0, min(h, int(round(h * max(0.0, float(bottom_ratio))))))
    left = max(0, min(w, int(round(w * max(0.0, float(left_ratio))))))
    right = max(0, min(w, int(round(w * max(0.0, float(right_ratio))))))

    if fill_mode == "black":
        fill_value = np.zeros((3,), dtype=np.uint8)
    else:
        fill_value = np.median(masked.reshape(-1, 3), axis=0).astype(np.uint8)

    if top > 0:
        masked[:top, :] = fill_value
    if bottom > 0:
        masked[h - bottom :, :] = fill_value
    if left > 0:
        masked[:, :left] = fill_value
    if right > 0:
        masked[:, w - right :] = fill_value

    return masked


def apply_text_mask_np(image_rgb: np.ndarray, text_mask_cfg: dict[str, Any] | None) -> np.ndarray:
    cfg = text_mask_cfg or {}
    if not bool(cfg.get("enabled", False)):
        return image_rgb

    return apply_band_mask_np(
        image_rgb,
        top_ratio=float(cfg.get("top_ratio", 0.0)),
        bottom_ratio=float(cfg.get("bottom_ratio", 0.0)),
        left_ratio=float(cfg.get("left_ratio", 0.0)),
        right_ratio=float(cfg.get("right_ratio", 0.0)),
        fill_mode=str(cfg.get("fill_mode", "median")),
    )


class RandomBandMask:
    def __init__(
        self,
        *,
        p: float = 0.0,
        top_ratio_range: tuple[float, float] = (0.0, 0.0),
        bottom_ratio_range: tuple[float, float] = (0.0, 0.0),
        left_ratio_range: tuple[float, float] = (0.0, 0.0),
        right_ratio_range: tuple[float, float] = (0.0, 0.0),
        fill_mode: str = "median",
    ) -> None:
        self.p = float(p)
        self.top_ratio_range = top_ratio_range
        self.bottom_ratio_range = bottom_ratio_range
        self.left_ratio_range = left_ratio_range
        self.right_ratio_range = right_ratio_range
        self.fill_mode = fill_mode

    def _sample(self, bounds: tuple[float, float]) -> float:
        low, high = float(bounds[0]), float(bounds[1])
        if high <= low:
            return max(0.0, low)
        return random.uniform(low, high)

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.p <= 0.0 or random.random() > self.p:
            return image

        arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
        masked = apply_band_mask_np(
            arr,
            top_ratio=self._sample(self.top_ratio_range),
            bottom_ratio=self._sample(self.bottom_ratio_range),
            left_ratio=self._sample(self.left_ratio_range),
            right_ratio=self._sample(self.right_ratio_range),
            fill_mode=self.fill_mode,
        )
        return Image.fromarray(masked, mode="RGB")
