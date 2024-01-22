"""
Abstract base class for computing similarity between two arrays.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from numpy import ndarray
from src.utils.factory import Factory


@dataclass
class SimilarityConfig(ABC):
    """
    Dataclass for holding similarity configuration.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Returns:
            str: The name of the similarity.
        """


class Similarity(ABC):
    """
    Abstract base class for computing similarity between two arrays.

    Subclasses must implement the __call__ method.

    Args:
        x (ndarray): The first array.
        y (ndarray): The second array.

    Returns:
        float: The similarity score between x and y.
    """

    factory = Factory()

    @abstractmethod
    def __call__(self, x: ndarray, y: ndarray) -> float:
        pass
