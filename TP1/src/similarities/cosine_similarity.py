import numpy as np


class CosineSimilarity:
    def __init__(self, is_distance=False):
        self.is_distance = is_distance

    def __call__(self, x: np.ndarray, y: np.ndarray):
        similarity = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
        if self.is_distance:
            return 1 - similarity
        return similarity
