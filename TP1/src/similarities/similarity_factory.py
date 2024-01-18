"""
Similarity factory module.
"""

from dataclasses import dataclass
from src.utils.factory import Factory
from src.similarities.bhattacharyya_distance_similarity import (
    BhattacharyyaDistanceSimilarity,
)
from src.similarities.cosine_similarity import CosineSimilarity
from src.similarities.hamming_distance_similarity import HammingDistanceSimilarity
from src.similarities.intersection_similarity import IntersectionSimilarity
from src.similarities.ip_norms_similarity import IPNormsSimilarity
from similarities.pair_assignments_similarity import (
    PairAssignmentsSimilarity,
)


@dataclass
class SimilarityEnum:
    """
    Enumeration class representing different similarity measures.
    """

    BHATTACHARYYA_DISTANCE = "bhattacharyya_distance"
    COSINE = "cosine"
    HAMMING_DISTANCE = "hamming_distance"
    INTERSECTION = "intersection"
    IP_NORMS = "ip_norms"
    PAIR_ASSIGNMENTS = "maximum_difference_pair_assignments"


similarity_factory = Factory(
    registry={
        SimilarityEnum.BHATTACHARYYA_DISTANCE: BhattacharyyaDistanceSimilarity,
        SimilarityEnum.COSINE: CosineSimilarity,
        SimilarityEnum.HAMMING_DISTANCE: HammingDistanceSimilarity,
        SimilarityEnum.INTERSECTION: IntersectionSimilarity,
        SimilarityEnum.IP_NORMS: IPNormsSimilarity,
        SimilarityEnum.PAIR_ASSIGNMENTS: PairAssignmentsSimilarity,
    }
)
