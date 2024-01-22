"""
Co-occurrence matrix feature extractor.
"""
from typing import List
from dataclasses import MISSING
from enum import Enum
import numpy as np
from skimage import img_as_ubyte
from skimage.feature import graycomatrix
from skimage.color import rgb2gray
from src.feature_extractors.feature_extractor import (
    FeatureExtractor,
    FeatureExtractorConfig,
)
from src.decorators.hydra_config_decorator import hydra_config
from src.decorators.register_to_factory import register_to_factory


class ChannelEnum(Enum):
    """
    Enumeration class representing different channels.
    """

    GREY = "grey"
    RED = "red"
    GREEN = "green"
    BLUE = "blue"
    EVERY = "every"


@hydra_config(group="feature_extractor")
class CoOccurrenceMatrixConfig(FeatureExtractorConfig):
    """
    Dataclass for holding co-occurrence matrix configuration.
    """

    distances: List[int] = MISSING
    angles: List[float] = MISSING
    levels: int = MISSING
    channel: str = MISSING


@register_to_factory(FeatureExtractor.factory)
class CoOccurrenceMatrix(FeatureExtractor):
    """
    Class for computing the co-occurrence matrix of an image.

    Args:
        distances (List[int]): List of pixel distances for co-occurrence matrix computation.
        angles (List[float]): List of angles (in radians) for co-occurrence matrix computation.
        levels (int): Number of gray levels for quantization.
        channel (Literal["grey", "red", "green", "blue"]): Channel(s) to compute
            the co-occurrence matrix on.
    """

    def __init__(
        self,
        *,
        distances: [int],
        angles: [float],
        levels: int,
        channel: ChannelEnum,
    ) -> None:
        """
        Class for computing the co-occurrence matrix of an image.

        Args:
            distances (List[int]): List of pixel distances for co-occurrence matrix computation.
            angles (List[float]): List of angles (in radians) for co-occurrence matrix computation.
            levels (int): Number of gray levels for quantization.
            channel (Literal["grey", "red", "green", "blue"]): Channel(s) to compute
                the co-occurrence matrix on.

        Returns:
            np.ndarray: The computed co-occurrence matrix.

        """

        self.distances = distances
        self.angles = angles
        self.levels = levels
        self.channel = channel

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if "every" == self.channel:
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

        if "grey" == self.channel:
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
