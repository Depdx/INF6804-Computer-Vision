import os
import skimage
import functools


class Database:
    def __init__(
        self,
        directory: str,
        feature_extractor: callable,
        descriptor: callable,
    ):
        self.directory = directory
        self.files = os.listdir(directory)
        self.feature_extractor = feature_extractor
        self.descriptor = descriptor

    def query(self, query, similarity_measure: callable, k: int):
        images_features = self.get_images_features()
        query_features = self.descriptor(self.feature_extractor(query))
        similarities = [
            similarity_measure(query_features, image_features)
            for image_features in images_features
        ]

        return sorted(
            zip(similarities, self.get_images(), images_features, self.files),
            key=lambda x: x[0],
            reverse=True,
        )[:k]

    @functools.lru_cache(maxsize=128)
    def get_images_features(self):
        assert self.feature_extractor is not None
        assert self.descriptor is not None
        images_features = []
        for file in self.files:
            image = skimage.io.imread(f"{self.directory}/{file}")
            images_features.append(self.descriptor(self.feature_extractor(image)))
        return images_features

    def get_images(self):
        for file in self.files:
            yield skimage.io.imread(f"{self.directory}/{file}")
