"""
Co-occurrence matrix feature extractor.
"""
from typing import Literal
from skimage import img_as_ubyte
from skimage.color import rgb2gray
from skimage.feature import graycomatrix
from numpy import ndarray


class CoOccurrenceMatrix:
    """
    Class for computing the co-occurrence matrix of an image.

    Args:
        distances (List[int]): List of pixel distances for co-occurrence matrix computation.
        angles (List[float]): List of angles (in radians) for co-occurrence matrix computation.
        levels (int): Number of gray levels for quantization.
        channel (Literal["grey", "red", "green", "blue"]): Channel(s) to compute the co-occurrence matrix on.

    Returns:
        np.ndarray: The computed co-occurrence matrix.

    """

    def __init__(
        self,
        distances: [int],
        angles: [float],
        levels: int,
        channel: Literal[
            "grey",
            "red",
            "green",
            "blue",
        ],
    ) -> None:
        self.distances = distances
        self.angles = angles
        self.levels = levels
        self.channel = channel

    def __call__(self, image: ndarray):
        if "grey" in self.channel:
            if len(image.shape) >= 3:
                image = rgb2gray(image)
                image = img_as_ubyte(image)

            image = image // (256 // self.levels)
            return graycomatrix(
                image,
                self.distances,
                self.angles,
                self.levels,
                normed=True,
            )

        if "red" in self.channel:
            image = image[:, :, 0]

        if "green" in self.channel:
            image = image[:, :, 1]

        if "blue" in self.channel:
            image = image[:, :, 2]

        image = image // (256 // self.levels)

        return graycomatrix(
            image,
            self.distances,
            self.angles,
            self.levels,
            normed=True,
        )
