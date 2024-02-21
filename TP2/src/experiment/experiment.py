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
from src.metrics.experiment_metrics import ExperimentMetrics


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
        self.per_image_confusion_matrix = torchmetrics.ConfusionMatrix(task="binary")
        self.prediction_table = None
        self.dataset = None

    @wandb_run()
    def run(self):
        """
        Run the experiment.
        """
        wandb.run.config.update(OmegaConf.to_container(self.config))

        segmentation_method: SegmentationMethod = SegmentationMethod.factory.create(
            self.config.segmentation_method.name, self.config.segmentation_method
        )

        self.dataset = CDWDataset()
        for train_video_dataset, test_video_dataset in tqdm(
            self.dataset, desc="Videos", unit="video dataset"
        ):
            train_video_dataset: VideoDataset
            test_video_dataset: VideoDataset
            segmentation_method.fit(train_video_dataset)
            video_dataloader = DataLoader(
                test_video_dataset,
                num_workers=os.cpu_count() // 2,
                batch_size=self.config.batch_size,
            )
            self.__create_prediction_table()

            for images, ground_truths, image_ids in tqdm(
                video_dataloader, desc="Segmenting", leave=False
            ):
                image_min_range = 655
                image_max_range = 670
                if image_ids.min() < image_min_range:
                    continue
                ground_truths: torch.Tensor
                images: torch.Tensor
                ground_truths = ground_truths.squeeze(dim=-3)
                # 85 is unknown, 0 is background
                ground_truths = (ground_truths != 85).to(torch.bool) & (
                    ground_truths != 0
                ).to(torch.bool)

                masks = segmentation_method(images)
                self.confusion_matrix.update(masks, ground_truths)
                self.log_predictions(
                    video_dataset_name=test_video_dataset.name,
                    masks=masks,
                    ground_truths=ground_truths,
                    image_ids=image_ids,
                )
                if image_ids.min() > image_max_range:
                    break
            self.log_metrics(test_video_dataset.name)
            self.confusion_matrix.reset()
            break

    def log_metrics(self, video_dataset_name: str):
        """
        Log the metrics of the experiment.
        """
        confusion_matrix = self.confusion_matrix.compute()
        metrics = ExperimentMetrics(confusion_matrix)
        wandb.log(
            {
                f"Metrics/{video_dataset_name}/True Positive": metrics.true_positive,
                f"Metrics/{video_dataset_name}/True Negative": metrics.true_negative,
                f"Metrics/{video_dataset_name}/False Positive": metrics.false_positive,
                f"Metrics/{video_dataset_name}/False Negative": metrics.false_negative,
                f"Metrics/{video_dataset_name}/Recall": metrics.recall,
                f"Metrics/{video_dataset_name}/Specificity": metrics.specificity,
                f"Metrics/{video_dataset_name}/False Positive Rate": metrics.false_positive_rate,
                f"Metrics/{video_dataset_name}/False Negative Rate": metrics.false_negative_rate,
                f"Metrics/{video_dataset_name}/Percentage of Wrong Classifications": metrics.percentage_of_wrong_classifications,
                f"Metrics/{video_dataset_name}/F1 Score": metrics.f1_score,
                f"Metrics/{video_dataset_name}/Precision": metrics.precision,
                f"Metrics/{video_dataset_name}/Average ranking": metrics.average_ranking,
                f"Predictions/{video_dataset_name}": self.prediction_table,
            }
        )

    def log_predictions(
        self,
        *,
        video_dataset_name: str,
        masks: torch.Tensor,
        ground_truths: torch.Tensor,
        image_ids: torch.Tensor,
    ):
        """
        Log the predictions of the experiment.
        """

        for mask, ground_truth, image_id in zip(
            torch.unbind(masks),
            torch.unbind(ground_truths),
            torch.unbind(image_ids),
        ):
            self.per_image_confusion_matrix.update(mask, ground_truth)
            confusion_matrix = self.per_image_confusion_matrix.compute()
            metrics = ExperimentMetrics(confusion_matrix)
            self.per_image_confusion_matrix.reset()
            image_id = str(image_id.item()).zfill(6)
            self.prediction_table.add_data(
                wandb.Image(
                    os.path.join(
                        self.dataset.dataset_dir,
                        f"{video_dataset_name}/input/in{image_id}.jpg",
                    ),
                    masks={
                        "predictions": {
                            "mask_data": mask.numpy().astype("uint8"),
                            "class_labels": {
                                0: "background",
                                1: "foreground",
                            },
                        },
                    },
                    caption=f"Mask {image_id}",
                ),
                wandb.Image(
                    os.path.join(
                        self.dataset.dataset_dir,
                        f"{video_dataset_name}/groundtruth/gt{image_id}.png",
                    )
                ),
                metrics.true_positive,
                metrics.true_negative,
                metrics.false_positive,
                metrics.false_negative,
                metrics.recall,
                metrics.specificity,
                metrics.false_positive_rate,
                metrics.false_negative_rate,
                metrics.percentage_of_wrong_classifications,
                metrics.f1_score,
                metrics.precision,
                metrics.average_ranking,
            )

    def __create_prediction_table(self):
        """
        Create the prediction table.
        """
        self.prediction_table = wandb.Table(
            columns=[
                "Predicted Mask",
                "Groundtruth",
                "True Positive",
                "True Negative",
                "False Positive",
                "False Negative",
                "Recall",
                "Specificity",
                "False Positive Rate",
                "False Negative Rate",
                "Percentage of Wrong Classifications",
                "F1 Score",
                "Precision",
                "Average ranking",
            ]
        )
