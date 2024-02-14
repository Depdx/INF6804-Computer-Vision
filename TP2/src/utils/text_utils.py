"""
A module containing text utilities.
"""
from re import sub


class TextUtils:
    """
    A class containing text utilities.
    """

    @staticmethod
    def pascal_to_snake(name: str) -> str:
        """
        Convert PascalCase to snake_case.
        """
        s1 = sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
        return sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
