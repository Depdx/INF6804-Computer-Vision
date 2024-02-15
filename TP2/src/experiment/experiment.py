"""
This module contains the Experiment class and the ExperimentConfig dataclass.
"""

from dataclasses import MISSING, asdict, dataclass
from typing import List
import os
import skimage
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm.autonotebook import tqdm, trange
from omegaconf import OmegaConf
import wandb
import torch
import torchmetrics
from src.decorators.hydra_config_decorator import hydra_config
from src.decorators.wandb_run_decorator import wandb_run
from src.dataset import CDWDataset, VideoDataset
from src.segmentation_methods.segmentation_method import (
    SegmentationMethod,
    SegmentationMethodConfig,
)


@hydra_config(name="base_config")
@dataclass
class ExperimentConfig:
    """
    A dataclass representing the experiment configuration.

    Args:
        segmentation_method SegmentationMethodConfig: The segmentation methods to use in the experiment.
    """

    segmentation_method: SegmentationMethodConfig = MISSING
    batch_size: int = MISSING


class Experiment:
    """
    This class represents an experiment.

    Args:
        config (ExperimentConfig): The experiment configuration.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.confusion_matrix = torchmetrics.ConfusionMatrix(task="binary")

    @wandb_run()
    def run(self):
        """
        Run the experiment.
        """
        wandb.run.config.update(OmegaConf.to_container(self.config))

        segmentation_method: SegmentationMethod = SegmentationMethod.factory.create(
            self.config.segmentation_method.name, self.config.segmentation_method
        )

        for video_dataset in tqdm(CDWDataset(), desc="Videos", unit="video dataset"):
            video_dataset: VideoDataset
            segmentation_method.fit(video_dataset)
            video_dataloader = DataLoader(
                video_dataset,
                num_workers=os.cpu_count() // 2,
                batch_size=self.config.batch_size,
            )
            for images, ground_truths in tqdm(
                video_dataloader, desc="Segmenting", leave=False
            ):
                ground_truths: torch.Tensor
                images: torch.Tensor
                ground_truths = ground_truths.squeeze(dim=-3)
                # 85 is unknown, 0 is background
                ground_truths = (ground_truths != 85).to(torch.bool) & (
                    ground_truths != 0
                ).to(torch.bool)

                masks = segmentation_method(images)
                self.confusion_matrix.update(masks, ground_truths)
            self.log_metrics(video_dataset.name)
            self.confusion_matrix.reset()

    def log_metrics(self, video_dataset_name: str):
        """
        Log the metrics of the experiment.
        """
        confusion_matrix = self.confusion_matrix.compute()
        true_positive = confusion_matrix[1, 1]
        true_negative = confusion_matrix[0, 0]
        false_positive = confusion_matrix[0, 1]
        false_negative = confusion_matrix[1, 0]
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        specificity = true_negative / (true_negative + false_positive)
        false_positive_rate = false_positive / (false_positive + true_negative)
        false_negative_rate = false_negative / (false_negative + true_positive)
        percentage_of_wrong_classifications = (false_positive + false_negative) / (
            true_positive + true_negative + false_positive + false_negative
        )
        f1_score = 2 * (precision * recall) / (precision + recall)
        average_ranking = (
            recall
            + specificity
            + false_positive_rate
            + false_negative_rate
            + percentage_of_wrong_classifications
            + f1_score
            + precision
        ) / 7
        wandb.log(
            {
                f"Metrics/{video_dataset_name}/True Positive": true_positive,
                f"Metrics/{video_dataset_name}/True Negative": true_negative,
                f"Metrics/{video_dataset_name}/False Positive": false_positive,
                f"Metrics/{video_dataset_name}/False Negative": false_negative,
                f"Metrics/{video_dataset_name}/Recall": recall,
                f"Metrics/{video_dataset_name}/Specificity": specificity,
                f"Metrics/{video_dataset_name}/False Positive Rate": false_positive_rate,
                f"Metrics/{video_dataset_name}/False Negative Rate": false_negative_rate,
                f"Metrics/{video_dataset_name}/Percentage of Wrong Classifications": percentage_of_wrong_classifications,
                f"Metrics/{video_dataset_name}/F1 Score": f1_score,
                f"Metrics/{video_dataset_name}/Precision": precision,
                f"Metrics/{video_dataset_name}/Average ranking": average_ranking,
            }
        )
