"""
Maximum difference pair assignments similarity.
"""
from numpy import ndarray
from src.similarities.similarity import Similarity, SimilarityConfig
from src.decorators.hydra_config_decorator import hydra_config
from src.decorators.register_to_factory import register_to_factory


@hydra_config(group="similarity")
class PairAssignmentsSimilarityConfig(SimilarityConfig):
    """
    Dataclass for holding PairAssignmentsSimilarity configuration.
    """

    name: str = "pair_assignments_similarity"


@register_to_factory(Similarity.factory)
class PairAssignmentsSimilarity(Similarity):
    """
    Calculates the similarity between two arrays using
    the maximum difference pair assignments method.
    """

    def __call__(self, x: ndarray, y: ndarray) -> float:
        """
        Calculates the similarity between two arrays using
        the maximum difference pair assignments method.

        Parameters:
            x (ndarray): The first array.
            y (ndarray): The second array.

        Returns:
            float: The similarity between x and y.
        """
        return sum(
            [
                abs(
                    sum(
                        [x[i] - y[i] for i in range(m)],
                    )
                )
            ]
            for m in range(len(x))
        )
