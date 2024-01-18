"""
Create a descriptor by flattening a matrix.
"""

import numpy as np


class MatrixFlattener:
    """
    Flattens a matrix into a vector.
    """

    def __call__(self, matrix: np.ndarray):
        """
        Flattens the 2 first dimensions of a given matrix.

        Args:
            matrix (np.ndarray): The input matrix of shape (w, h, d, a).

        Returns:
            np.ndarray: The flattened matrix of shape (w*h, d, a).

        """
        return matrix.reshape(-1, *matrix.shape[2:])
