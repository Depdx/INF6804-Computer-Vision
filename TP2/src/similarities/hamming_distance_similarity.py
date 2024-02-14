"""
Hamming distance similarity.
"""

from numpy import ndarray
from src.similarities.similarity import Similarity, SimilarityConfig
from src.decorators.hydra_config_decorator import hydra_config
from src.decorators.register_to_factory import register_to_factory


@hydra_config(group="similarity")
class HammingDistanceSimilarityConfig(SimilarityConfig):
    """
    Dataclass for holding HammingDistanceSimilarity configuration.
    """

    name: str = "hamming_distance_similarity"


@register_to_factory(Similarity.factory)
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
