"""
A module for representing a database of images for similarity search.
"""
from dataclasses import dataclass
from functools import lru_cache
from os import listdir
from typing import List
from skimage.io import imread
from numpy import ndarray
from tqdm.autonotebook import tqdm
from src.decorators.performance_counter_decorator import performance_counter
from src.utils.label_encoder import LabelEncoder


@dataclass
class QueryResult:
    """
    A dataclass representing a query result.

    Args:
        similarity_scores (List[float]): The similarity scores of the images.
        images (List[ndarray]): The images.
        images_features (List[ndarray]): The images features.
        labels (List[str]): The labels of the images.
        query_label (str): The label of the query image.
    """

    similarity_scores: List[float]
    images: List[ndarray]
    images_features: List[ndarray]
    labels: List[int]
    labels_names: List[str]
    query_label: str
    query_label_name: str
    query_image: ndarray


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
        feature_extractor: callable = None,
        descriptor: callable = None,
    ):
        assert feature_extractor is not None
        assert descriptor is not None
        self.directory = directory
        self.files = listdir(directory)
        self.feature_extractor = feature_extractor
        self.descriptor = descriptor
        self.label_encoder = LabelEncoder()

    @performance_counter(metric_name="Processing image")
    def process(self, image):
        """
        Process an image and return its descriptor.
        """
        return self.descriptor(self.feature_extractor(image))

    @performance_counter(metric_name="Query")
    def query(
        self,
        query: ndarray,
        query_file: str,
        similarity_measure: callable,
        k: int,
    ):
        """
        Perform a similarity search on the database using a query image.

        Args:
            query: The query image.
            query_file (str): The query image file name.
            similarity_measure (callable): A function or method used to measure similarity
                between images.
            k (int): The number of most similar images to retrieve.

        Returns:
            list: A list of tuples containing the
                similarity score, image, image features, and file name.
        """
        images_features = self.get_images_features()
        query_features = self.process(query)
        similarities = [
            similarity_measure(query_features, image_features)
            for image_features in images_features
        ]

        results = sorted(
            zip(similarities, self.get_images(), images_features, self.files),
            key=lambda x: x[0],
            reverse=True,
        )[:k]

        similarities, images, images_features, files = zip(*results)
        encoded_query_file = self.label_encoder.encode(query_file)
        encoded_labels = self.label_encoder.encode(files)
        return QueryResult(
            similarity_scores=similarities,
            images=images,
            images_features=images_features,
            labels=encoded_labels,
            labels_names=self.label_encoder.decode(encoded_labels),
            query_label=encoded_query_file,
            query_label_name=self.label_encoder.decode(encoded_query_file),
            query_image=query,
        )

    @lru_cache(maxsize=128)
    @performance_counter(metric_name="Building database")
    def get_images_features(self):
        """
        Get the features of all images in the database.

        Returns:
            list: A list of image features.
        """
        images_features = []
        for file in tqdm(self.files, desc="Processing database files"):
            image = imread(f"{self.directory}/{file}")
            images_features.append(self.process(image))
        return images_features

    def get_images(self):
        """
        Generator function to yield images from the database.

        Yields:
            ndarray: An image from the database.
        """
        for file in self.files:
            yield imread(f"{self.directory}/{file}")
