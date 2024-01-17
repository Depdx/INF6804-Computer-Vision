import numpy as np


class Histogram:
    def __init__(
        self,
        bins=np.arange(0, 10),
        density=True,
    ):
        self.bins = bins
        self.density = density

    def __call__(self, features: np.ndarray):
        hist, _ = np.histogram(features, bins=self.bins, density=self.density)
        return hist
