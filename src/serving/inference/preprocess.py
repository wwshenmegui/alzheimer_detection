from __future__ import annotations

import numpy as np

from shared.image_preprocessing import preprocess_image_bytes


def preprocess_uploaded_image(image_bytes: bytes, image_size: tuple[int, int]) -> np.ndarray:
    image_array = preprocess_image_bytes(image_bytes, image_size)
    return image_array.reshape(1, -1)
