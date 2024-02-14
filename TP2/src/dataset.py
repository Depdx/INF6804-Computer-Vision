import wandb
import os
import torch
from typing import List
from torchvision.io import read_image


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, path: str):
        self.path = path
        self.name = os.path.basename(path)
        with open(os.path.join(path, "temporalROI.txt"), "r") as file:
            line = file.readline()
            self.start_region_of_interest, self.length = [
                int(x) for x in line.split(sep=" ")
            ]
            self.start_region_of_interest -= 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return read_image(os.path.join(self.path, f"input/in{str(idx+1).zfill(6)}.jpg"))


class CDWDataset(torch.utils.data.Dataset):
    def __init__(self):
        dataset = wandb.use_artifact("CDW-2012-Baseline:latest")
        dataset_dir = dataset.download()
        dataset_dir = os.path.join(dataset_dir, "CDW-2012-Baseline")
        self.dataset_dir = dataset_dir
        self.videos_directories = os.listdir(dataset_dir)

    def __len__(self):
        return len(self.videos_directories)

    def __getitem__(self, idx):
        return VideoDataset(
            os.path.join(self.dataset_dir, self.videos_directories[idx])
        )
