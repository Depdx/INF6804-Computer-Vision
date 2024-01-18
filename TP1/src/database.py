"""
A module for representing a database of images for similarity search.
"""
from functools import lru_cache
from os import listdir
from skimage.io import imread
from tqdm import tqdm

class Database:
    """
    A class representing a database of images for similarity search.

    Args:
        directory (str): The directory path where the images are stored.
        feature_extractor (callable): A function or method used to extract features from an image.
        descriptor (callable): A function or method used to compute descriptors from image features.

    Attributes:
        directory (str): The directory path where the images are stored.
        files (list): A list of file names in the directory.
        feature_extractor (callable): A function or method used to extract features from an image.
        descriptor (callable): A function or method used to compute descriptors from image features.
    """

    def __init__(
        self,
        directory: str,
        feature_extractor: callable,
        descriptor: callable,
    ):
        self.directory = directory
        self.files = listdir(directory)
        self.feature_extractor = feature_extractor
        self.descriptor = descriptor

    def query(self, query, similarity_measure: callable, k: int):
        """
        Perform a similarity search on the database using a query image.

        Args:
            query: The query image.
            similarity_measure (callable): A function or method used to measure similarity
                between images.
            k (int): The number of most similar images to retrieve.

        Returns:
            list: A list of tuples containing the
                similarity score, image, image features, and file name.
        """
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

    @lru_cache(maxsize=128)
    def get_images_features(self):
        """
        Get the features of all images in the database.

        Returns:
            list: A list of image features.
        """
        assert self.feature_extractor is not None
        assert self.descriptor is not None
        images_features = []
        for file in tqdm(self.files, desc="Processing database files"):
            image = imread(f"{self.directory}/{file}")
            images_features.append(self.descriptor(self.feature_extractor(image)))
        return images_features

    def get_images(self):
        """
        Generator function to yield images from the database.

        Yields:
            ndarray: An image from the database.
        """
        for file in self.files:
            yield imread(f"{self.directory}/{file}")
