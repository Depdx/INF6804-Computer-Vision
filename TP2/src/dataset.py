import wandb
import os
import torch
from typing import List
from torchvision.io import read_image


class DataItem:
    def __init__(self, path: str):
        self.path = path
        self.name = os.path.basename(path)
        with open(os.path.join(path, "temporalROI.txt"), "r") as file:
            line = file.readline()
            self.start, self.end = [int(x) for x in line.split(sep=" ")]

        image_name = f"input/in{str(1).zfill(6)}.jpg"
        self.images = read_image(os.path.join(path, image_name))
        for i in range(2, self.end + 1):
            image_name = f"input/in{str(i).zfill(6)}.jpg"
            self.images = torch.cat(
                (self.images, read_image(os.path.join(path, image_name)))
            )


class Dataset:
    def __init__(self):
        dataset = wandb.use_artifact("CDW-2012-Baseline:latest")
        dataset_dir = dataset.download()
        dataset_dir = os.path.join(dataset_dir, "CDW-2012-Baseline")
        self.dataset_dir = dataset_dir
        self.items = []
        for directory in os.listdir(dataset_dir):
            self.items.append(DataItem(os.path.join(dataset_dir, directory)))
