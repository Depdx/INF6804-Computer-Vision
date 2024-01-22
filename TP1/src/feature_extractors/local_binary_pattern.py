"""
Local binary pattern feature extractor.
"""

from dataclasses import MISSING
from enum import Enum
import skimage
import numpy as np
from src.feature_extractors.feature_extractor import (
    FeatureExtractor,
    FeatureExtractorConfig,
)
from src.decorators.hydra_config_decorator import hydra_config
from src.decorators.register_to_factory import register_to_factory


class MethodEnum(Enum):
    """
    Enumeration class representing different methods.
    """

    DEFAULT = "default"
    ROR = "ror"
    UNIFORM = "uniform"
    NRI_UNIFORM = "nri_uniform"


@hydra_config(group="feature_extractor")
class LocalBinaryPatternConfig(FeatureExtractorConfig):
    """
    Dataclass for holding local binary pattern configuration.
    """

    n_points: int = MISSING
    radius: float = MISSING
    method: MethodEnum = MISSING


@register_to_factory(FeatureExtractor.factory)
class LocalBinaryPattern(FeatureExtractor):
    """
    Class for computing the local binary pattern of an image.

    Args:
        n_points (int): Number of circularly symmetric neighbor set points.
        radius (float): Radius of circle (spatial resolution).
        method (str): Method to determine the pattern of a pixel,
            either 'default' or 'ror' (rotation invariant).
    """

    def __init__(
        self,
        n_points: int,
        radius: float,
        method: MethodEnum,
    ) -> None:
        self.n_points = n_points
        self.radius = radius
        self.method = method

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) >= 3:
            image = skimage.color.rgb2gray(image)
            image = skimage.img_as_ubyte(image)
        return skimage.feature.local_binary_pattern(
            image,
            self.n_points,
            self.radius,
            self.method,
        )
