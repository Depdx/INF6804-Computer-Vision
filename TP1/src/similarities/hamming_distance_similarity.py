"""
Hamming distance similarity.
"""

from numpy import ndarray
from src.similarities.similarity import Similarity


class HammingDistanceSimilarity(Similarity):
    """
    Calculates the Hamming distance similarity between two arrays.

    The Hamming distance similarity is a measure of similarity between two arrays of equal length.
    It counts the number of positions at which the corresponding elements
    in the two arrays are different.

    Args:
        x (ndarray): The first array.
        y (ndarray): The second array.

    Returns:
        float: The Hamming distance similarity between the two arrays.
    """

    def __call__(self, x: ndarray, y: ndarray) -> float:
        """
        Calculates the Hamming distance similarity between two arrays.

        Parameters:
            x (ndarray): The first array.
            y (ndarray): The second array.

        Returns:
            float: The Hamming distance similarity between x and y.
        """
        return sum(
            [int(x[i] == y[i]) for i in range(len(x))],
        )
