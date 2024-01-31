import wandb
from src.utils.environment import Environment


if __name__ == "__main__":
    wandb.login(key=Environment.wanb_api_key)
    runs = wandb.Api(api_key=Environment.wanb_api_key).runs(
        path="depdx/computer-vision-tp1",
        filters={
            "config.similarity.name": {
                "$in": ["cosine_similarity", "intersection_similarity"]
            }
        },
    )
    for finished_run in runs:
        run = wandb.init(
            project="computer-vision-tp1",
            entity="depdx",
            resume=finished_run.id,
        )
        for label in [
            "strawberry",
            "airplane",
            "ball",
            "car",
            "cat",
            "dolphin",
            "face",
            "lotus",
            "pickles",
        ]:
            artifact = run.use_artifact(
                f"depdx/computer-vision-tp1/run-{run.id}-Query{label}Result:latest",
                type="run_table",
            )
            artifact_dir = artifact.download()

            table = artifact.get(f"Query/{label}/Result")
            df = table.get_dataframe()
            is_correct = df["File"].iloc[0] == label
            is_top_k_correct = label in df["File"]
            wandb.log(
                {
                    f"Query/{label}/Is Correct": is_correct,
                    f"Query/{label}/Is Top-k Correct": is_top_k_correct,
                }
            )
            run.finish()
