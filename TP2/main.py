"""
Main file for the TP1
"""

from typing import Optional
import hydra
from src.experiment.experiment import Experiment, ExperimentConfig


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(config: ExperimentConfig = None) -> Optional[float]:
    """
    Main function for the TP1.

    Args:
        config (ExperimentConfig): The experiment configuration.
    """
    print(f"Loading config :{config}")

    experiment = Experiment(config)
    return experiment.run()


if __name__ == "__main__":
    main()
