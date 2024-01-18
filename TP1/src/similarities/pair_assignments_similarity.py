"""
Maximum difference pair assignments similarity.
"""
from numpy import ndarray
from src.similarities.similarity import Similarity


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
