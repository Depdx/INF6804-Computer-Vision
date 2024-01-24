"""
This module contains the Experiment class and the ExperimentConfig dataclass.
"""
from dataclasses import MISSING
from typing import List
import os
import skimage
from sklearn.metrics import accuracy_score
from tqdm.autonotebook import tqdm
import wandb
from src.feature_extractors.feature_extractor import FeatureExtractorConfig
from src.descriptors.descriptor import DescriptorConfig
from src.similarities.similarity import SimilarityConfig
from src.decorators.hydra_config_decorator import hydra_config
from src.decorators.wandb_run_decorator import wandb_run
from src.database import Database, QueryResult
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

        results: List[QueryResult] = []
        for root, _, files in os.walk(self.config.query_path):
            for file in tqdm(files, desc="Querying", unit="query"):
                query_path = os.path.join(root, file)
                query = skimage.io.imread(query_path)
                result = database.query(
                    query=query,
                    query_file=file,
                    similarity_measure=similarity_measure,
                    k=self.config.top_k,
                )
                results.append(result)

        top_3_accuracy = accuracy_score(
            y_true=[result.query_label for result in results],
            y_pred=[
                result.query_label
                if result.query_label in result.labels[:3]
                else result.labels[0]
                for result in results
            ],
        )

        accuracy = accuracy_score(
            y_true=[result.query_label for result in results],
            y_pred=[result.labels[0] for result in results],
        )

        wandb.log(
            {
                "Metrics/Top-3 Accuracy": top_3_accuracy,
                "Metrics/Accuracy": accuracy,
            }
        )

        for result in results:
            wandb.log(
                {
                    f"Query/{result.query_label}": wandb.Table(
                        allow_mixed_types=True,
                        columns=["Similarity", "Image", "Features", "File"],
                        data=[
                            [
                                similarity,
                                wandb.Image(image),
                                wandb.Histogram(features),
                                file,
                            ]
                            for similarity, image, features, file in zip(
                                result.similarity_scores,
                                result.images,
                                result.images_features,
                                result.labels,
                            )
                        ],
                    )
                }
            )
