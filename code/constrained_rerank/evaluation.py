"""
evaluation.py — Parity-safe evaluation wrappers and acceptance gates.

Computes the same 6 metrics as get_metrics() in RCSYS_utils.py, but operates
on a pre-computed reranked top-K tensor instead of generating it from embeddings.
"""

import torch
import numpy as np
import sys
import os

# Add parent code/ directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from RCSYS_utils import (
    get_user_positive_items,
    RecallPrecision_ATk,
    NDCGatK_r,
    calculate_health_score,
    calculate_average_health_tags,
    calculate_percentage_recommended_foods,
)


def evaluate_topk_list(top_K_items, edge_index, user_tags, food_tags, K, num_foods):
    """
    Compute all 6 metrics on a given top-K item tensor.
    Replicates the metric computation from get_metrics() (RCSYS_utils.py:353-380)
    but uses a pre-built top_K_items tensor.

    NOTE on parity: The baseline eval() in RCSYS_utils subsets user embeddings via
    structured_negative_sampling before computing ratings, producing a [num_edges, num_items]
    rating matrix. We use full [num_users, num_items] embeddings instead, which is more
    correct (user IDs always map to the right rows). Absolute metric values may differ
    slightly from main.py eval output, but the internal baseline-vs-reranked comparison
    is consistent since both use this same function.

    Args:
        top_K_items: [num_users, K] tensor of recommended item indices
        edge_index: eval split edge_index (defines ground truth positives)
        user_tags: [num_users, num_tags] user health tags
        food_tags: [num_foods, num_tags] food health tags
        K: number of recommendations
        num_foods: total number of food items

    Returns:
        dict with keys: recall, precision, ndcg, health_score,
                       avg_health_tags_ratio, percentage_recommended_foods
    """
    # Get all unique users in evaluated split
    users = edge_index[0].unique()
    test_user_pos_items = get_user_positive_items(edge_index)

    # Convert test user pos items dictionary into a list
    test_user_pos_items_list = [
        test_user_pos_items[user.item()] for user in users
    ]

    # Determine the correctness of topk predictions
    r = []
    for user in users:
        ground_truth_items = test_user_pos_items[user.item()]
        label = list(map(lambda x: x in ground_truth_items, top_K_items[user].tolist()))
        r.append(label)
    r = torch.Tensor(np.array(r).astype('float'))

    recall, precision = RecallPrecision_ATk(test_user_pos_items_list, r, K)
    ndcg = NDCGatK_r(test_user_pos_items_list, r, K)
    health_score = calculate_health_score(users, top_K_items, user_tags, food_tags)
    avg_health_tags_ratio = calculate_average_health_tags(users, top_K_items, food_tags)
    percentage_recommended_foods = calculate_percentage_recommended_foods(users, top_K_items, num_foods)

    return {
        'recall': recall,
        'precision': precision,
        'ndcg': ndcg,
        'health_score': health_score,
        'avg_health_tags_ratio': avg_health_tags_ratio,
        'percentage_recommended_foods': percentage_recommended_foods,
    }


def compute_acceptance_gates(baseline_metrics, reranked_metrics, ndcg_floor=0.07):
    """
    Check acceptance gates as defined in auto_implement_plan.md Phase 6.

    Primary hard gate: test ndcg drop fraction must be <= configured floor.
    Secondary utility gate: non-negative trend in secondary metrics.

    Args:
        baseline_metrics: dict from evaluate_topk_list
        reranked_metrics: dict from evaluate_topk_list
        ndcg_floor: maximum allowed ndcg drop fraction (default 0.07)

    Returns:
        dict with:
            hard_gate_pass: bool
            ndcg_drop_fraction: float
            secondary_gate_pass: bool
            details: dict of per-metric deltas
    """
    baseline_ndcg = baseline_metrics['ndcg']
    reranked_ndcg = reranked_metrics['ndcg']

    if baseline_ndcg > 0:
        ndcg_drop_fraction = (baseline_ndcg - reranked_ndcg) / baseline_ndcg
    else:
        ndcg_drop_fraction = 0.0

    hard_gate_pass = ndcg_drop_fraction <= ndcg_floor

    # Secondary metrics: health_score, avg_health_tags_ratio, percentage_recommended_foods
    secondary_keys = ['health_score', 'avg_health_tags_ratio', 'percentage_recommended_foods']
    details = {}
    secondary_improvements = 0
    for key in list(baseline_metrics.keys()):
        delta = reranked_metrics[key] - baseline_metrics[key]
        details[f'{key}_delta'] = delta
        details[f'{key}_baseline'] = baseline_metrics[key]
        details[f'{key}_reranked'] = reranked_metrics[key]
        if key in secondary_keys and delta >= 0:
            secondary_improvements += 1

    secondary_gate_pass = secondary_improvements >= len(secondary_keys)

    return {
        'hard_gate_pass': hard_gate_pass,
        'ndcg_drop_fraction': ndcg_drop_fraction,
        'secondary_gate_pass': secondary_gate_pass,
        'details': details,
    }


def generate_comparison_table(baseline_metrics, reranked_metrics, diagnostics_agg):
    """
    Generate a formatted comparison table for logging.

    Args:
        baseline_metrics: dict from evaluate_topk_list
        reranked_metrics: dict from evaluate_topk_list
        diagnostics_agg: dict from aggregate_diagnostics

    Returns:
        str: formatted comparison table
    """
    lines = []
    lines.append("=" * 70)
    lines.append("BASELINE vs CONSTRAINED RERANK COMPARISON")
    lines.append("=" * 70)
    lines.append(f"{'Metric':<35} {'Baseline':>12} {'Reranked':>12} {'Delta':>10}")
    lines.append("-" * 70)

    for key in baseline_metrics:
        b = baseline_metrics[key]
        r = reranked_metrics[key]
        delta = r - b
        sign = '+' if delta >= 0 else ''
        lines.append(f"{key:<35} {b:>12.5f} {r:>12.5f} {sign}{delta:>9.5f}")

    lines.append("-" * 70)
    lines.append("CONSTRAINED RERANK DIAGNOSTICS")
    lines.append("-" * 70)

    for key, val in diagnostics_agg.items():
        if isinstance(val, float):
            lines.append(f"{key:<35} {val:>12.4f}")
        else:
            lines.append(f"{key:<35} {val:>12}")

    lines.append("=" * 70)
    return '\n'.join(lines)
