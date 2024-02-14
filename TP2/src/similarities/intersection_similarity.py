"""
Intersection similarity.
"""
from numpy import ndarray
from src.similarities.similarity import Similarity, SimilarityConfig
from src.decorators.hydra_config_decorator import hydra_config
from src.decorators.register_to_factory import register_to_factory


@hydra_config(group="similarity")
class IntersectionSimilarityConfig(SimilarityConfig):
    """
    Dataclass for holding IntersectionSimilarity configuration.
    """

    name: str = "intersection_similarity"


@register_to_factory(Similarity.factory)
class IntersectionSimilarity(Similarity):
    """
    Calculates the intersection similarity between two arrays.

    The intersection similarity is computed as the sum of the minimum values
    between corresponding elements of the input arrays, divided by the sum of
    the elements in the first array.

    Args:
        x (ndarray): The first input array.
        y (ndarray): The second input array.

    Returns:
        float: The intersection similarity between the two arrays.
    """

    def __call__(self, x: ndarray, y: ndarray) -> float:
        intersection = sum([min(x[i], y[i]) for i in range(len(x))])
        return intersection / sum(x)
