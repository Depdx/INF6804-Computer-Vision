"""
This module contains the Experiment class and the ExperimentConfig dataclass.
"""
from dataclasses import dataclass, MISSING
from hydra.core.config_store import ConfigStore
from src.feature_extractors.feature_extractor import FeatureExtractorConfig
from src.descriptors.descriptor import DescriptorConfig
from src.similarities.similarity import SimilarityConfig


@dataclass
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


ConfigStore.instance().store(name="base_config", node=ExperimentConfig)


class Experiment:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        print("heelo")

    def run(self):
        pass
