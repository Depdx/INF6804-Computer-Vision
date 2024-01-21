"""
Environment configuration.
"""
from os import getenv
from dotenv import load_dotenv


class _Environment:
    """
    Represents the environment configuration.

    Attributes:
        wanb_api_key (str): The API key for Weights & Biases.
    """

    def __init__(self):
        load_dotenv(".env")
        self.wanb_api_key = getenv("WANDB_API_KEY")


Environment = _Environment()
