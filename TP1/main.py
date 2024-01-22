import skimage
import wandb
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from src.database import Database
from src.feature_extractors.local_binary_pattern import (
    LocalBinaryPattern,
    LocalBinaryPatternConfig,
)
from src.feature_extractors.co_occurrence_matrix import CoOccurrenceMatrixConfig
from src.descriptors.histogram import Histogram
from src.similarities.cosine_similarity import CosineSimilarity
from src.decorators.wandb_run_decorator import wandb_run, Run
from src.experiment.experiment import Experiment, ExperimentConfig
from src.descriptors.histogram import HistogramConfig
from src.descriptors.matrix_flattener import MatrixFlattenerConfig


@wandb_run(group="test", name="test")
def execute():
    wandb.log({"test": 1})

    db = Database(
        directory="./data/database",
        feature_extractor=LocalBinaryPattern(
            n_points=20,
            radius=10,
            method="uniform",
        ),
        descriptor=Histogram(),
    )

    image_path = "./data/cat_query.jpg"
    query = skimage.io.imread(image_path)

    results = db.query(
        query=query,
        similarity_measure=CosineSimilarity(),
        k=10,
    )


# for similarity, image, features, file in results:
#     print(f"Similarity: {similarity}")
#     print(f"File: {file}")
#     skimage.io.imshow(image)
#     skimage.io.show()


# grid search
# implementation couleur pour co-occurence matrix

# Autmatic topk-accuracy


# add visualisation des résultats
# report


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(config: ExperimentConfig = None) -> None:
    print(f"Loading config :{config.similarity}")

    experiment = Experiment(config)
    experiment.run()


if __name__ == "__main__":
    main()
