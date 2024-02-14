"""
This script adds the dataset to wandb.
"""

import wandb
from src.decorators.wandb_run_decorator import wandb_run


@wandb_run(
    job_type="dataset",
    name="add_dataset",
)
def main():
    """
    Main function.
    """
    dataset_artifact = wandb.Artifact(
        "CDW-2012-Baseline",
        type="dataset",
        description="CDW-2012 dataset http://jacarini.dinf.usherbrooke.ca/cdw2012.html",
    )
    dataset_artifact.add_dir("data/dataset/baseline")
    wandb.log_artifact(dataset_artifact)


if __name__ == "__main__":
    main()
