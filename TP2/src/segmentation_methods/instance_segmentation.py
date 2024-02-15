from dataclasses import dataclass
from torch import Tensor
from src.dataset import VideoDataset
from src.segmentation_methods.segmentation_method import (
    SegmentationMethod,
    SegmentationMethodConfig,
)
from src.decorators.hydra_config_decorator import hydra_config
from src.decorators.register_to_factory import register_to_factory


@hydra_config(group="segmentation_method")
@dataclass
class InstanceSegmentationConfig(SegmentationMethodConfig):
    name: str = "instance_segmentation"


@register_to_factory(SegmentationMethod.factory)
class InstanceSegmentation(SegmentationMethod):

    def __init__(self, config: InstanceSegmentationConfig) -> None:
        self.config = config

    def fit(self, dataset: VideoDataset) -> None:
        return super().fit(dataset)

    def __call__(self, image: Tensor) -> Tensor:
        pass
