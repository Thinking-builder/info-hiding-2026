from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def save_rgb(path: str | Path, image: np.ndarray, rgba: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if rgba:
        alpha = np.full((*image.shape[:2], 1), 255, dtype=np.uint8)
        Image.fromarray(np.concatenate([image, alpha], axis=2), "RGBA").save(path)
    else:
        Image.fromarray(image.astype(np.uint8), "RGB").save(path)


def fit_for_block(image: np.ndarray, width: int, height: int, block: int) -> np.ndarray:
    fitted = np.asarray(Image.fromarray(image).resize((width, height), Image.Resampling.LANCZOS))
    return fitted[: height - height % block, : width - width % block]


def resize_to_shape(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    if image.shape[:2] == shape:
        return image.astype(np.uint8)
    return np.asarray(
        Image.fromarray(image.astype(np.uint8)).resize(
            (width, height), Image.Resampling.LANCZOS
        ),
        dtype=np.uint8,
    )


def resize_canonical(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    target = (1024, 768) if height > width else (768, 1024)
    return resize_to_shape(image, target)
