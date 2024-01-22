"""
Bhattacharyya distance similarity.
"""

from numpy import ndarray, sqrt, log
from src.similarities.similarity import Similarity, SimilarityConfig
from src.decorators.hydra_config_decorator import hydra_config
from src.decorators.register_to_factory import register_to_factory


@hydra_config(group="similarity")
class BhattacharyyaDistanceSimilarityConfig(SimilarityConfig):
    """
    Dataclass for holding BhattacharyyaDistanceSimilarity configuration.
    """

    name: str = "bhattacharyya_distance_similarity"


@register_to_factory(Similarity.factory)
class BhattacharyyaDistanceSimilarity(Similarity):
    """
    Calculates the Bhattacharyya distance similarity between two arrays.
    """

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
