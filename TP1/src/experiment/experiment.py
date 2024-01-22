"""
This module contains the Experiment class and the ExperimentConfig dataclass.
"""
from dataclasses import MISSING
import os
import skimage
from tqdm.autonotebook import tqdm
from src.feature_extractors.feature_extractor import FeatureExtractorConfig
from src.descriptors.descriptor import DescriptorConfig
from src.similarities.similarity import SimilarityConfig
from src.decorators.hydra_config_decorator import hydra_config
from src.decorators.wandb_run_decorator import wandb_run
from src.database import Database
from src.descriptors.descriptor import Descriptor
from src.feature_extractors.feature_extractor import FeatureExtractor
from src.similarities.similarity import Similarity


@hydra_config(name="base_config")
class ExperimentConfig:
    """
    A dataclass representing the experiment configuration.

    Args:
        feature_extractor (FeatureExtractorConfig): The feature extractor configuration.
        descriptor (DescriptorConfig): The descriptor configuration.
        similarity (SimilarityConfig): The similarity configuration.
    """

    feature_extractor: FeatureExtractorConfig = MISSING
    descriptor: DescriptorConfig = MISSING
    similarity: SimilarityConfig = MISSING
    database_path: str = MISSING
    query_path: str = MISSING
    top_k: int = MISSING


class Experiment:
    """
    This class represents an experiment.

    Args:
        config (ExperimentConfig): The experiment configuration.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config

    @wandb_run()
    def run(self):
        """
        Run the experiment.
        """
        feature_extractor = FeatureExtractor.factory.create(
            self.config.feature_extractor.name,
            **{
                key: value
                for key, value in self.config.feature_extractor.items()
                if key != "name"
            },
        )

        descriptor = Descriptor.factory.create(
            self.config.descriptor.name,
            **{
                key: value
                for key, value in self.config.descriptor.items()
                if key != "name"
            },
        )

        database = Database(
            directory=self.config.database_path,
            feature_extractor=feature_extractor,
            descriptor=descriptor,
        )
        self.evaluate(database)

    def evaluate(self, database: Database):
        """
        Evaluate the experiment.

        Args:
            database (Database): The database.
        """

        similarity_measure = Similarity.factory.create(
            self.config.similarity.name,
            **{
                key: value
                for key, value in self.config.similarity.items()
                if key != "name"
            },
        )

        for root, _, files in tqdm(
            os.walk(self.config.query_path),
            desc="Querying",
        ):
            for file in files:
                query_path = os.path.join(root, file)
                query = skimage.io.imread(query_path)
                results = database.query(
                    query=query,
                    similarity_measure=similarity_measure,
                    k=self.config.top_k,
                )
                for similarity, image, features, file in results:
                    print(f"Similarity: {similarity}")
                    print(f"File: {file}")
                    skimage.io.imshow(image)
                    skimage.io.show()
                # TODO: log the top-k accuracy to wandb
                # TODO: log the query image and the top-k results to wandb
