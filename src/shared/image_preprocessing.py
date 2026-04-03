from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image


def preprocess_pil_image(image: Image.Image, image_size: tuple[int, int]) -> np.ndarray:
    grayscale_image = image.convert("L")
    resized_image = grayscale_image.resize(image_size)
    image_array = np.asarray(resized_image, dtype=np.float32) / 255.0
    return image_array[..., np.newaxis]


def preprocess_image_path(image_path: Path, image_size: tuple[int, int]) -> np.ndarray:
    with Image.open(image_path) as image:
        return preprocess_pil_image(image, image_size)


def preprocess_image_bytes(image_bytes: bytes, image_size: tuple[int, int]) -> np.ndarray:
    with Image.open(BytesIO(image_bytes)) as image:
        return preprocess_pil_image(image, image_size)
