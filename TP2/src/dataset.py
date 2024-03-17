import wandb
import os
import torch
from typing import Tuple
from torchvision.io import read_image


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, *, path: str, start: int, end: int):
        self.path = path
        self.name = os.path.basename(path)
        self.start = start
        self.end = end
        self.coco_labels = []
        if "highway" in self.name:
            self.coco_labels = [
                2,  # Car
                4,  # Motorcycle
                6,  # Bus
                8,  # Truck
            ]
        if "office" in self.name:
            self.coco_labels = [
                1,  # Person
                84,  # Book
            ]
        if "pedestrians" in self.name:
            self.coco_labels = [
                1,  # Person
            ]
        if "PETS2006" in self.name:
            self.coco_labels = [
                1,  # Person
            ]

    def __len__(self) -> int:
        return self.end - self.start

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            idx (int): Index of the video to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the input image, the groundtruth and the image id.
        """
        image_id = str(self.start + idx + 1).zfill(6)
        return (
            read_image(os.path.join(self.path, f"input/in{image_id}.jpg")).to(
                torch.float32
            ),
            read_image(os.path.join(self.path, f"groundtruth/gt{image_id}.png")).to(
                torch.float32
            ),
            torch.tensor([int(image_id)], dtype=torch.int32),
        )


class CDWDataset(torch.utils.data.Dataset):
    def __init__(self):
        dataset = wandb.use_artifact("CDW-2012-Baseline:latest")
        dataset_dir = dataset.download()
        dataset_dir = os.path.join(dataset_dir, "CDW-2012-Baseline")
        self.dataset_dir = dataset_dir
        self.videos_directories = os.listdir(dataset_dir)

    def __len__(self) -> int:
        return len(self.videos_directories)

    def __getitem__(self, idx) -> Tuple[VideoDataset, VideoDataset]:
        video_dataset_path = os.path.join(
            self.dataset_dir, self.videos_directories[idx]
        )
        with open(os.path.join(video_dataset_path, "temporalROI.txt"), "r") as file:
            line = file.readline()
            start_region_of_interest, length = [int(x) for x in line.split(sep=" ")]
            train_dataset = VideoDataset(
                path=video_dataset_path,
                start=0,
                end=start_region_of_interest,
            )
            test_dataset = VideoDataset(
                path=video_dataset_path,
                start=start_region_of_interest,
                end=length - start_region_of_interest,
            )
            return train_dataset, test_dataset
