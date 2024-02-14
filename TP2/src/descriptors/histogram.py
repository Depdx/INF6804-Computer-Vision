"""
Histogram descriptor.
"""
from dataclasses import dataclass
from numpy import ndarray, arange, histogram
from src.descriptors.descriptor import Descriptor, DescriptorConfig
from src.decorators.hydra_config_decorator import hydra_config
from src.decorators.register_to_factory import register_to_factory


@hydra_config(group="descriptor")
class HistogramConfig(DescriptorConfig):
    """
    Dataclass for holding histogram configuration.
    """

    name: str = "histogram"
    n_bins: int = 10
    density: bool = True


@register_to_factory(Descriptor.factory)
class Histogram(Descriptor):
    """
    Computes the histogram of a given set of features.
    """

    def __init__(
        self,
        n_bins=10,
        density=True,
    ):
        """
        Initializes a Histogram object.

        Args:
            n_bins (int): The number of bins of the histogram.
            density (bool): If True, the histogram is normalized to form a probability density.
        """
        self.bins = arange(0, n_bins)
        self.density = density

    def __call__(self, features: ndarray) -> ndarray:
        """
        Computes the histogram of the given features.

        Args:
            features (np.ndarray): The input features.

        Returns:
            np.ndarray: The computed histogram.

        """
        hist, _ = histogram(features, bins=self.bins, density=self.density)
        return hist
