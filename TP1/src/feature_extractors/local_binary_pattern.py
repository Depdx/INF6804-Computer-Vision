from typing import Any
import skimage
import numpy as np


class LocalBinaryPattern:
    def __init__(self, n_points: int, radius: float, method: str) -> None:
        self.n_points = n_points
        self.radius = radius
        self.method = method

    def __call__(self, image: np.ndarray):
        if len(image.shape) >= 3:
            image = skimage.color.rgb2gray(image)
            image = skimage.img_as_ubyte(image)
        return skimage.feature.local_binary_pattern(
            image,
            self.n_points,
            self.radius,
            self.method,
        )
