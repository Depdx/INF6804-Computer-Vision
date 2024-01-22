"""
This module contains the abstract class for descriptors.
"""

from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from src.utils.factory import Factory


@dataclass
class DescriptorConfig(ABC):
    """
    Dataclass for holding descriptor configuration.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Returns:
            str: The name of the descriptor.
        """


class Descriptor(ABC):
    """
    This class represents a descriptor.

    Args:
        features (np.ndarray): The input features.

    Returns:
        np.ndarray: The extracted features.
    """

    factory = Factory()

    @abstractmethod
    def __call__(self, features: np.ndarray) -> np.ndarray:
        pass
