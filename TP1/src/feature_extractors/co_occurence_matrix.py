import typing
import numpy as np
from skimage.feature import graycomatrix
from skimage import img_as_ubyte
from skimage.color import rgb2gray


class CoOccurenceMatrix:
    def __init__(
            self,
            distances: [int],
            angles: [float],
            levels: int,
            channel: typing.Literal[
                "grey",
                "red",
                "green",
                "blue",
                "every"
            ],
    ) -> None:
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

        self.distances = distances
        self.angles = angles
        self.levels = levels
        self.channel = channel

    def __call__(self, image: np.ndarray):

        if "every" in self.channel:
            assert len(image.shape) >= 3
            assert self.levels % 3 == 0

            image = image // 2 ** (self.levels // 3)

            one_channel_image = 2 ** (2 * self.levels // 3) * image[:, :, 0]  # R
            one_channel_image += 2 ** (self.levels // 3) * image[:, :, 1]  # G
            one_channel_image += image[:, :, 2]  # B

            return graycomatrix(
                one_channel_image,
                self.distances,
                self.angles,
                2**self.levels,
                normed=True,
            )

        elif "grey" in self.channel:

            # If the image is not grey
            if len(image.shape) >= 3:
                image = rgb2gray(image)
                image = img_as_ubyte(image)

            image = image // 2 ** (8 - self.levels)
            return graycomatrix(
                image,
                self.distances,
                self.angles,
                2**self.levels,
                normed=True,
            )
        else:
            channels = {
                "red": 0,
                "green": 1,
                "blue": 2,
            }

            # Select the correct channel
            if len(image.shape) >= 3:
                image = image[:, :, channels[self.channel]]
            image = image // 2 ** (8 - self.levels)

            return graycomatrix(
                image,
                self.distances,
                self.angles,
                2**self.levels,
                normed=True,
            )

