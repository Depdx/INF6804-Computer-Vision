"""
Module for encoding labels based on file names.
"""
from typing import List, Union
from re import findall


class LabelEncoder:
    """
    Class for encoding labels based on file names.
    """

    def __init__(self):
        self.labels_encoding = {}

    def encode(self, files_name: Union[List[str], str]) -> Union[List[str], str]:
        """
        Encodes the labels based on the given file names.

        Args:
            files_name (Union[List[str], str]): The file names to encode.

        Returns:
            Union[List[str], str]: The encoded labels.
        """
        is_string = isinstance(files_name, str)
        if is_string:
            files_name = [files_name]

        encoded_labels = []
        for file_name in files_name:
            finds = findall(r"([a-zA-Z]+)_", file_name)
            if len(finds) == 0:
                continue
            label = finds[0]
            encoded_labels.append(
                self.labels_encoding.setdefault(label, len(self.labels_encoding))
            )

        return encoded_labels if not is_string else encoded_labels[0]
