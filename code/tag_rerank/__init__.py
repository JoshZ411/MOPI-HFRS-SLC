"""
tag_rerank — Tag-Augmented Inference Re-scoring for MOPI-HFRS
Author: Harshit
Branch: harshit-develop

Addresses the tag-blind inference gap in SGSL:
  training uses Jaccard health loss → inference ignores tags entirely.
  
Formula:
  score(u, i) = (1 - α) * norm(u_emb · i_emb) + α * Jaccard(tags_u, tags_i)
"""

from .scorer import jaccard_matrix, tag_augmented_scores
from .evaluate import tars_get_metrics, tars_eval, assert_baseline_parity

__all__ = [
    "jaccard_matrix",
    "tag_augmented_scores",
    "tars_get_metrics",
    "tars_eval",
    "assert_baseline_parity",
]
