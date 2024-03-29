"""
This module contains the abstract class SegmentationMethod and the SegmentationMethodConfig dataclass.
"""

from dataclasses import dataclass
from abc import ABC, abstractmethod
import torch
from src.utils.factory import Factory
from src.dataset import VideoDataset


@dataclass
class SegmentationMethodConfig(ABC):
    """
    A dataclass representing the segmentation method configuration.

    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Returns:
            str: The name of the segmentation method.
        """


class SegmentationMethod(ABC):
    """
    This class represents a segmentation method.
    """

    factory = Factory()

    @abstractmethod
    def fit(self, train_dataset: VideoDataset) -> None:
        pass

    @abstractmethod
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """
        Apply the segmentation method to an image.

        Args:
            images (torch.Tensor): The image to segment. The shape is (B, C, H, W).
        Returns:
            torch.Tensor: The segmented mask of the image.
        """
