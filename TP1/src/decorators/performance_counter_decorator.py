"""
Decorator that logs the elapsed time of a function
"""
import functools
import time
import wandb


def performance_counter(
    *,
    metric_name: str,
    metric_group: str = "Timers",
):
    """
    Decorator that logs the elapsed time of a function

    Args:
        metric_name (str, optional): The name of the metric. Defaults to the function name.
        metric_group (str, optional): The name of the metric group. Defaults to "Timers".
    """

    def performance_counter_decorator(func: callable):
        """
        Decorator that logs the elapsed time of a function
        """

        @functools.wraps(func)
        def wrapper(
            *args,
            **kwargs,
        ):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()

            wandb.log(
                {
                    f"{metric_group}/{metric_name or func.__name__}": end - start,
                },
            )
            return result

        return wrapper

    return performance_counter_decorator
