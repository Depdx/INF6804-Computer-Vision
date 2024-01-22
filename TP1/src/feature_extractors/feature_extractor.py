"""
This module contains the FeatureExtractor class.
"""

from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from src.utils.factory import Factory


@dataclass
class FeatureExtractorConfig(ABC):
    """
    Dataclass for holding feature extractor configuration.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Returns:
            str: The name of the feature extractor.
        """


class FeatureExtractor(ABC):
    """
    This class represents a feature extractor.

    Args:
        image (np.ndarray): The input image.

    Returns:
        np.ndarray: The extracted features.
    """

    factory = Factory()

    @abstractmethod
    def __call__(self, image: np.ndarray) -> np.ndarray:
        pass
