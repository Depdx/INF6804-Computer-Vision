"""
Similarities module.
"""

from src.similarities.similarity import Similarity
from src.similarities.cosine_similarity import CosineSimilarity
from src.similarities.bhattacharyya_distance_similarity import (
    BhattacharyyaDistanceSimilarity,
)
from src.similarities.ip_norms_similarity import IPNormsSimilarity
from src.similarities.hamming_distance_similarity import HammingDistanceSimilarity
from src.similarities.intersection_similarity import IntersectionSimilarity
from src.similarities.pair_assignments_similarity import PairAssignmentsSimilarity
