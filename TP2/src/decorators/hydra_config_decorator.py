"""
A decorator function for registering a class with Hydra's ConfigStore.
"""

from typing import Optional
from hydra.core.config_store import ConfigStore
from src.utils.text_utils import TextUtils


def hydra_config(*, group: Optional[str] = None, name: Optional[str] = None):
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

        class_name = original_class.__name__.replace("Config", "")
        store_name = name or f"base_{TextUtils.pascal_to_snake(class_name)}"
        ConfigStore.instance().store(group=group, name=store_name, node=original_class)

        return original_class

    return hydra_config_class
