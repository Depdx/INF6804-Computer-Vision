"""
Abstract base class for computing similarity between two arrays.
"""

from abc import ABC, abstractmethod
from numpy import ndarray


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

    @abstractmethod
    def __call__(self, x: ndarray, y: ndarray) -> float:
        pass
