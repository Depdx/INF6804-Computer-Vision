from dataclasses import dataclass, MISSING
from abc import ABC, abstractmethod
from enum import Enum
import torch
from src.segmentation_methods.segmentation_method import SegmentationMethod
from src.utils.factory import Factory
from src.decorators.hydra_config_decorator import hydra_config
from src.decorators.register_to_factory import register_to_factory
from src.dataset import VideoDataset


class MethodEnum(Enum):
    MEAN = "mean"
    MEDIAN = "median"


@hydra_config(group="segmentation_method")
@dataclass
class BackgroundSubtractionConfig(ABC):
    name: str = "background_subtraction"
    threshold: int = MISSING
    method: MethodEnum = MISSING


@register_to_factory(SegmentationMethod.factory)
class BackgroundSubtraction(SegmentationMethod):

    def __init__(self, config: BackgroundSubtractionConfig) -> None:
        self.config = config
        self.background = None

    def fit(self, dataset: VideoDataset) -> None:
        if self.config.method == MethodEnum.MEAN:
            self.background = torch.zeros_like(dataset[0])
            for i in range(dataset.start_region_of_interest):
                self.background += dataset[i] / dataset.start_region_of_interest
        elif self.config.method == MethodEnum.MEDIAN:
            tensors = []
            for i in range(dataset.start_region_of_interest):
                tensors.append(dataset[i])
            self.background = torch.stack(tensors)
            self.background = torch.median(self.background, dim=0)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return (
            torch.sum(torch.pow(image - self.background, 2), dim=0)
            > self.config.threshold**2
        )
