from dataclasses import dataclass, MISSING
from torch import Tensor
import torch
from enum import Enum
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn,
    maskrcnn_resnet50_fpn_v2,
    FasterRCNN,
    MaskRCNN_ResNet50_FPN_Weights,
    MaskRCNN_ResNet50_FPN_V2_Weights,
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
    mask_threshold: float = 0.5
    device: str = "cpu"
    name: str = "instance_segmentation"


@register_to_factory(SegmentationMethod.factory)
class InstanceSegmentation(SegmentationMethod):

    def __init__(self, config: InstanceSegmentationConfig) -> None:
        self.config = config
        self.model, self.images_transform = self._get_model(config.model)
        self.model = self.model.to(device=self.config.device)
        self.coco_labels = [
            0,  # Person
            1,  # Bicycle
            2,  # Car
            3,  # Motorcycle
            5,  # Bus
            7,  # Truck
        ]

    def fit(self, train_dataset: VideoDataset) -> None:
        return super().fit(train_dataset)

    def __call__(self, images: Tensor) -> Tensor:
        """
        Apply the segmentation method to an image.

        Args:
            images (torch.Tensor): The images to segment. The shape is (B, C, H, W).
        Returns:
            torch.Tensor: The segmented mask of the image.
        """
        images = images.to(device=self.config.device)
        images = self.images_transform(images)
        images = images / 255.0  # Normalize the images to [0, 1] range
        predictions = self.model(images)
        masks = []
        for prediction in predictions:
            mask = None
            for j, score in enumerate(prediction["scores"]):
                if prediction["labels"][j] not in self.coco_labels:
                    continue
                if score > self.config.threshold:
                    if mask is None:
                        mask = prediction["masks"][j][0] >= self.config.mask_threshold
                    else:
                        mask += prediction["masks"][j][0] >= self.config.mask_threshold
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
            weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1
            return (
                maskrcnn_resnet50_fpn(weights=weights).eval(),
                weights.transforms(),
            )
        elif model == ModelEnum.MASK_RCNN_RESNET50_FPN_V2.value:
            weights = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1
            return (
                maskrcnn_resnet50_fpn_v2(weights=weights).eval(),
                weights.transforms(),
            )
        else:
            raise ValueError(f"Model {str(model)} not supported")
