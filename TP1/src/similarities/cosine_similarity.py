"""
Cosine similarity measure.
"""
from numpy import ndarray, dot
from numpy.linalg import norm
from src.similarities.similarity import Similarity


class CosineSimilarity(Similarity):
    """
    Computes the cosine similarity between two vectors.
    """

    def __init__(self, is_distance=False):
        """
        Initializes the CosineSimilarity object.

        Parameters:
        - is_distance (bool): If True, the cosine similarity will be converted to
            a distance measure.

        Returns:
        - None
        """
        self.is_distance = is_distance

    def __call__(self, x: ndarray, y: ndarray) -> float:
        """
        Calculates the cosine similarity between two vectors.

        Parameters:
        - x (np.ndarray): The first vector.
        - y (np.ndarray): The second vector.

        Returns:
        - similarity (float): The cosine similarity between x and y.
        """
        similarity = dot(x, y) / (norm(x) * norm(y))
        if self.is_distance:
            return 1 - similarity
        return similarity
