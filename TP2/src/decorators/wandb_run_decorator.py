"""
Decorator that initializes a wandb run and finishes it after the function
"""

import functools
import wandb
from src.utils.environment import Environment

Run = wandb.wandb_sdk.wandb_run.Run


def wandb_run(group: str = None, name: str = None, **wandb_init_kwargs: dict):
    """
    Decorator that initializes a wandb run and finishes it after the function

    Args:
        group (str, optional): The name of the run group. Defaults to None.
        name (str, optional): The name of the run. Defaults to None.
    """

    def wandb_run_decorator(func: callable):
        """
        Decorator that initializes a wandb run and finishes it after the function
        """

        @functools.wraps(func)
        def wrapper(
            *args,
            **kwargs,
        ):
            if "run" in kwargs:
                raise ValueError("run is a reserved keyword")

            wandb.login(key=Environment.wanb_api_key)
            run = wandb.init(
                project=Environment.wandb_project,
                entity=Environment.wandb_entity,
                group=group,
                name=name,
                **wandb_init_kwargs,
            )
            if run is None:
                raise ValueError("run is None after wandb.init")
            result = func(*args, **kwargs)
            run.finish()
            return result

        return wrapper

    return wandb_run_decorator
