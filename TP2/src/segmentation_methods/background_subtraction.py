from dataclasses import dataclass, MISSING
from enum import Enum
import torch
import wandb
from tqdm.autonotebook import trange
from src.segmentation_methods.segmentation_method import SegmentationMethod
from src.decorators.hydra_config_decorator import hydra_config
from src.segmentation_methods.segmentation_method import SegmentationMethodConfig
from src.decorators.register_to_factory import register_to_factory
from src.dataset import VideoDataset


class MethodEnum(Enum):
    MEAN = "mean"
    MEDIAN = "median"


@hydra_config(group="segmentation_method")
@dataclass
class BackgroundSubtractionConfig(SegmentationMethodConfig):
    threshold: int = MISSING
    method: MethodEnum = MISSING
    name: str = "background_subtraction"


@register_to_factory(SegmentationMethod.factory)
class BackgroundSubtraction(SegmentationMethod):

    def __init__(self, config: BackgroundSubtractionConfig) -> None:
        self.config = config
        self.background = None

    def fit(self, dataset: VideoDataset) -> None:
        if self.config.method == MethodEnum.MEAN.value:
            self.background = None
            for i in trange(
                dataset.start_region_of_interest,
                desc="Fitting background",
                unit="image",
                leave=False,
            ):
                image, _ = dataset[i]
                if self.background is None:
                    self.background = image
                else:
                    self.background += image / dataset.start_region_of_interest
        elif self.config.method == MethodEnum.MEDIAN.value:
            tensors = []
            for i in trange(
                dataset.start_region_of_interest,
                desc="Fitting background",
                unit="image",
                leave=False,
            ):
                image, _ = dataset[i]
                tensors.append(image)
            self.background = torch.stack(tensors)
            self.background = torch.median(self.background, dim=0).values

        wandb.log({f"{dataset.name}/Background": wandb.Image(self.background)})

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """
        Apply the segmentation method to an image.

        Args:
            images (torch.Tensor): The image to segment. The shape is (B, C, H, W).
        Returns:
            torch.Tensor: The segmented mask of the image.
        """
        return (
            torch.sum(torch.pow(images - self.background, 2), dim=-3)
            > self.config.threshold**2
        )
