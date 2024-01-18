"""
IPNormsSimilarity class.
"""

from numpy import ndarray
from src.similarities.similarity import Similarity


class IPNormsSimilarity(Similarity):
    """
    Calculates the similarity between two vectors using the p-norms similarity measure.

    Parameters:
    - p (int): The value of p for the p-norms similarity measure.
        Usually, p = 1 or p = 2.

    Returns:
    - float: The similarity score between the two vectors.
    """

    def __init__(self, p: int):
        self.p = p

    def __call__(self, x: ndarray, y: ndarray) -> float:
        return sum(abs(x - y) ** self.p) ** (1 / self.p)
