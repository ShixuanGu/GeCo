"""Small image / frame-directory helpers shared by the scripts."""
from __future__ import annotations

import os
from typing import List, Sequence, Tuple

import cv2
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def list_frame_files(frame_dir: str) -> List[str]:
    """Sorted list of image files inside ``frame_dir`` (non-recursive)."""
    if not os.path.isdir(frame_dir):
        raise FileNotFoundError(f"Frame directory does not exist: {frame_dir}")
    files = [
        os.path.join(frame_dir, f)
        for f in os.listdir(frame_dir)
        if os.path.splitext(f.lower())[1] in IMAGE_EXTS
    ]
    return sorted(files)


def load_image_rgb(path: str) -> np.ndarray:
    """Load an image as an RGB ``uint8`` array of shape (H, W, 3)."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not load image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def image_hw(path: str) -> Tuple[int, int]:
    """(H, W) of an image file without decoding the full pixel buffer when possible."""
    try:
        from PIL import Image

        with Image.open(path) as im:
            w, h = im.size
        return int(h), int(w)
    except Exception:  # pragma: no cover - fallback path
        return tuple(load_image_rgb(path).shape[:2])


def check_same_size(paths: Sequence[str]) -> Tuple[int, int]:
    """Assert that all frames share one resolution and return it as (H, W)."""
    hw0 = image_hw(paths[0])
    for p in paths[1:]:
        hw = image_hw(p)
        if hw != hw0:
            raise ValueError(f"All frames must share one resolution; {paths[0]} is {hw0} but {p} is {hw}")
    return hw0
