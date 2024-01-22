"""
Create a descriptor by flattening a matrix.
"""

import numpy as np
from src.descriptors.descriptor import Descriptor, DescriptorConfig
from src.decorators.hydra_config_decorator import hydra_config
from src.decorators.register_to_factory import register_to_factory


@hydra_config(group="descriptor")
class MatrixFlattenerConfig(DescriptorConfig):
    """
    Dataclass for holding matrix flattener configuration.
    """

    name: str = "matrix_flattener"


@register_to_factory(Descriptor.factory)
class MatrixFlattener(Descriptor):
    """
    Flattens a matrix into a vector.
    """

    def __call__(self, matrix: np.ndarray) -> np.ndarray:
        """
        Flattens the 2 first dimensions of a given matrix.

        Args:
            matrix (np.ndarray): The input matrix of shape (w, h, d, a).

        Returns:
            np.ndarray: The flattened matrix of shape (w*h, d, a).

        """
        return matrix.reshape(-1, *matrix.shape[2:])
