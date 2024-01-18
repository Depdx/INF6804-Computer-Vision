"""
Bhattacharyya distance similarity.
"""

from numpy import ndarray, sqrt, log
from src.similarities.similarity import Similarity


class BhattacharyyaDistanceSimilarity(Similarity):
    """
    Calculates the Bhattacharyya distance similarity between two arrays.
    """

    def __init__(self, dataset):
        super().__init__(dataset)

    def __call__(self, x: ndarray, y: ndarray) -> float:
        """
        Calculates the Bhattacharyya distance similarity between two arrays.

        Parameters:
        x (ndarray): The first array.
        y (ndarray): The second array.

        Returns:
        float: The Bhattacharyya distance similarity between x and y.
        """
        coefficient = sum([sqrt(x[i] * y[i]) for i in range(len(x))])
        return -log(coefficient)
