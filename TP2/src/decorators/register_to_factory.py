"""
Decorator function that registers a class to a factory.
"""

from src.utils.factory import Factory


def register_to_factory(factory: Factory):
    """
    Decorator function that registers a class to a factory.

    Args:
        factory (Factory): The factory object to register the class to.

    Returns:
        Callable: The decorator function that registers the class to the factory.
    """

    def register_to_factory_class(original_class):
        factory.register(original_class)
        return original_class

    return register_to_factory_class
