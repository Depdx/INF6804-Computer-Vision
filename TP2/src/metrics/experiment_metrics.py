"""
Module to compute the metrics of an experiment.
"""

from torch import Tensor


class ExperimentMetrics:
    """
    Class to compute the metrics of an experiment.

    Args:
        confusion_matrix (Tensor): The confusion matrix of the experiment.
    """

    def __init__(self, confusion_matrix: Tensor) -> None:
        self.true_positive = confusion_matrix[1, 1]
        self.true_negative = confusion_matrix[0, 0]
        self.false_positive = confusion_matrix[0, 1]
        self.false_negative = confusion_matrix[1, 0]
        self.precision = self.true_positive / (self.true_positive + self.false_positive)
        self.recall = self.true_positive / (self.true_positive + self.false_negative)
        self.specificity = self.true_negative / (
            self.true_negative + self.false_positive
        )
        self.false_positive_rate = self.false_positive / (
            self.false_positive + self.true_negative
        )
        self.false_negative_rate = self.false_negative / (
            self.false_negative + self.true_positive
        )
        self.percentage_of_wrong_classifications = (
            self.false_positive + self.false_negative
        ) / (
            self.true_positive
            + self.true_negative
            + self.false_positive
            + self.false_negative
        )
        self.f1_score = (
            2 * (self.precision * self.recall) / (self.precision + self.recall)
        )
        self.average_ranking = (
            self.recall
            + self.specificity
            + self.false_positive_rate
            + self.false_negative_rate
            + self.percentage_of_wrong_classifications
            + self.f1_score
            + self.precision
        ) / 7
