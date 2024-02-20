from dataclasses import dataclass, MISSING
from torch import Tensor
import torch
from enum import Enum
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn,
    maskrcnn_resnet50_fpn_v2,
    FasterRCNN,
)
from src.dataset import VideoDataset
from src.segmentation_methods.segmentation_method import (
    SegmentationMethod,
    SegmentationMethodConfig,
)
from src.decorators.hydra_config_decorator import hydra_config
from src.decorators.register_to_factory import register_to_factory


class ModelEnum(Enum):
    MASK_RCNN_RESNET50_FPN = "maskrcnn_resnet50_fpn"
    MASK_RCNN_RESNET50_FPN_V2 = "maskrcnn_resnet50_fpn_v2"


@hydra_config(group="segmentation_method")
@dataclass
class InstanceSegmentationConfig(SegmentationMethodConfig):
    model: ModelEnum = MISSING
    threshold: float = MISSING
    device: str = "cpu"
    name: str = "instance_segmentation"


@register_to_factory(SegmentationMethod.factory)
class InstanceSegmentation(SegmentationMethod):

    def __init__(self, config: InstanceSegmentationConfig) -> None:
        self.config = config
        self.model = self._get_model(config.model)
        self.model = self.model.to(device=self.config.device)

    def fit(self, train_dataset: VideoDataset) -> None:
        return super().fit(train_dataset)

    def __call__(self, images: Tensor) -> Tensor:
        """
        Apply the segmentation method to an image.

        Args:
            images (torch.Tensor): The image to segment. The shape is (B, C, H, W).
        Returns:
            torch.Tensor: The segmented mask of the image.
        """
        images = images.to(device=self.config.device)
        predictions = self.model(images)
        masks = []
        for prediction in predictions:
            mask = None
            for j, score in enumerate(prediction["scores"]):
                if score > self.config.threshold:
                    if mask is None:
                        mask = prediction["masks"][j][0] > self.config.threshold
                    else:
                        mask += prediction["masks"][j][0] > self.config.threshold
            if mask is None:
                masks.append(
                    torch.zeros(
                        (images.shape[-2], images.shape[-1]),
                        device=self.config.device,
                        dtype=torch.bool,
                    )
                )
            else:
                masks.append(mask)
        masks = torch.stack(masks, dim=0)
        masks = masks.to(device="cpu")
        return masks

    def _get_model(self, model: ModelEnum) -> FasterRCNN:
        if model == ModelEnum.MASK_RCNN_RESNET50_FPN.value:
            return maskrcnn_resnet50_fpn(pretrained=True).eval()
        elif model == ModelEnum.MASK_RCNN_RESNET50_FPN_V2.value:
            return maskrcnn_resnet50_fpn_v2(pretrained=True).eval()
        else:
            raise ValueError(f"Model {str(model)} not supported")
