"""
A decorator function for registering a class with Hydra's ConfigStore.
"""

from dataclasses import dataclass
from functools import wraps
from typing import Optional
from hydra.core.config_store import ConfigStore
from src.utils.text_utils import TextUtils


def hydra_config(*, group: str, name: Optional[str] = None):
    """
    Decorator function for registering a class with Hydra's ConfigStore.

    Args:
        group (str): The group name for the registered class.
        name (Optional[str]): The name of the registered class.
            If not provided, the class name will be converted from PascalCase to snake_case.

    Returns:
        Callable: The decorator function that registers the class with Hydra's ConfigStore.
    """

    def hydra_config_class(original_class):
        @dataclass
        class HydraConfigClass(original_class):
            """
            A wrapper class for the original class that registers
            the original class with Hydra's ConfigStore.
            """

        class_name = original_class.__name__.replace("Config", "")
        store_name = name or f"base_{TextUtils.pascal_to_snake(class_name)}"
        ConfigStore.instance().store(
            group=group, name=store_name, node=HydraConfigClass
        )

        return HydraConfigClass

    return hydra_config_class
