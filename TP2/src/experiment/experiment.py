"""
This module contains the Experiment class and the ExperimentConfig dataclass.
"""

from dataclasses import MISSING, asdict
from typing import List
import os
import skimage
from sklearn.metrics import accuracy_score
from tqdm.autonotebook import tqdm
from omegaconf import OmegaConf
import wandb
from src.decorators.hydra_config_decorator import hydra_config
from src.decorators.wandb_run_decorator import wandb_run
from src.dataset import Dataset


@hydra_config(name="base_config")
class ExperimentConfig:
    """
    A dataclass representing the experiment configuration.

    Args:
    """


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
        wandb.run.config.update(OmegaConf.to_container(self.config))

        dataset = Dataset()
        print("Dataset loaded")
