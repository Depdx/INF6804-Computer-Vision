"""
Histogram descriptor.
"""
from numpy import ndarray, arange, histogram


class Histogram:
    """
    Computes the histogram of a given set of features.
    """

    def __init__(
        self,
        bins=arange(0, 10),
        density=True,
    ):
        """
        Initializes a Histogram object.

        Args:
            bins (np.ndarray): The bin edges for the histogram.
            density (bool): If True, the histogram is normalized to form a probability density.
        """
        self.bins = bins
        self.density = density

    def __call__(self, features: ndarray):
        """
        Computes the histogram of the given features.

        Args:
            features (np.ndarray): The input features.

        Returns:
            np.ndarray: The computed histogram.

        """
        hist, _ = histogram(features, bins=self.bins, density=self.density)
        return hist
