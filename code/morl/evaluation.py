"""
MORL Evaluation — Trade-off selection via validation metrics (Phase 7).

Metrics computed for each weight vector w on a recommendation list:
    - NDCG@K          : ranking quality (preference objective)
    - Health score    : Jaccard-based tag-coverage alignment
    - Diversity score : mean pairwise cosine dissimilarity within list

Functions:
    evaluate_policy_on_split — generate Top-K lists and compute all metrics
    select_operating_weight  — grid search over w vectors → pick w*
"""

import math
import itertools
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .environment import RecommendationEnv, build_candidate_pools
from .policy import MORLPolicy


# -----------------------------------------------------------------------
# List generation
# -----------------------------------------------------------------------

def generate_recommendations(
    env: RecommendationEnv,
    policy: MORLPolicy,
    user_ids: List[int],
    weight: torch.Tensor,
    K: int = 20,
    greedy: bool = True,
) -> Dict[int, List[int]]:
    """Generate Top-K recommendation lists using the trained policy.

    Parameters
    ----------
    env      : RecommendationEnv (candidate pools must already exclude test edges)
    policy   : trained MORLPolicy (eval mode)
    user_ids : users to generate for
    weight   : Tensor (3,) — fixed preference weight vector
    K        : list length
    greedy   : if True, use argmax (deterministic); if False, sample from distribution

    Returns
    -------
    recommendations : dict  user_id → list of K item_ids
    """
    policy.eval()
    recommendations = {}

    with torch.no_grad():
        for uid in user_ids:
            state = env.reset(uid)
            selected = []
            done = False

            while not done and len(selected) < K:
                cand_embs = env.get_candidate_embeddings()
                if cand_embs.shape[0] == 0:
                    break
                log_probs = policy.forward(state, weight, cand_embs)
                if greedy:
                    action_idx = log_probs.argmax().item()
                else:
                    action_idx = torch.multinomial(log_probs.exp(), num_samples=1).item()
                next_state, _, done = env.step(action_idx)
                selected.append(env._selected[-1])
                state = next_state

            recommendations[uid] = selected

    return recommendations


# -----------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------

def _dcg(relevances: List[float]) -> float:
    """Compute Discounted Cumulative Gain."""
    return sum(rel / math.log2(rank + 2) for rank, rel in enumerate(relevances))


def ndcg_at_k(recommended: List[int], ground_truth: set, k: int) -> float:
    """NDCG@K for a single user."""
    relevances = [1.0 if item in ground_truth else 0.0 for item in recommended[:k]]
    ideal = sorted(relevances, reverse=True)
    dcg = _dcg(relevances)
    idcg = _dcg(ideal)
    return dcg / idcg if idcg > 0 else 0.0


def health_score_at_k(
    recommended: List[int],
    user_tag: torch.Tensor,
    item_tags: torch.Tensor,
) -> float:
    """Jaccard similarity between union of recommended items' tags and user tags."""
    if not recommended:
        return 0.0
    rec_tags = item_tags[torch.tensor(recommended)]  # (K, tag_dim)
    coverage = rec_tags.float().sum(dim=0).clamp(max=1.0)  # (tag_dim,) binary union
    user_t = user_tag.float()
    intersection = (coverage * user_t).sum().item()
    union = (coverage + user_t).clamp(max=1.0).sum().item()
    return intersection / (union + 1e-8)


def diversity_score_at_k(recommended: List[int], item_emb: torch.Tensor) -> float:
    """Mean pairwise cosine dissimilarity within the recommended list."""
    if len(recommended) < 2:
        return 0.0
    embs = item_emb[torch.tensor(recommended)]  # (K, d)
    embs_norm = F.normalize(embs, dim=-1)
    cos_matrix = embs_norm @ embs_norm.T  # (K, K)
    K = embs.shape[0]
    # Mean of off-diagonal elements
    mask = ~torch.eye(K, dtype=torch.bool, device=embs.device)
    mean_cos_sim = cos_matrix[mask].mean().item()
    return 1.0 - mean_cos_sim  # dissimilarity


# -----------------------------------------------------------------------
# Full split evaluation
# -----------------------------------------------------------------------

def evaluate_policy_on_split(
    env: RecommendationEnv,
    policy: MORLPolicy,
    user_ids: List[int],
    weight: torch.Tensor,
    ground_truth: Dict[int, set],
    K: int = 20,
) -> Dict[str, float]:
    """Evaluate policy for a fixed weight vector on a user split.

    Parameters
    ----------
    env          : RecommendationEnv with eval-time candidate pools
    policy       : trained MORLPolicy
    user_ids     : evaluation users
    weight       : Tensor (3,) preference weight
    ground_truth : dict  user_id → set of positive item_ids
    K            : list length

    Returns
    -------
    metrics : dict with keys ndcg, health, diversity, recall
    """
    recs = generate_recommendations(env, policy, user_ids, weight, K=K)

    ndcg_vals, health_vals, div_vals, recall_vals = [], [], [], []

    for uid in user_ids:
        rec_list = recs.get(uid, [])
        gt = ground_truth.get(uid, set())

        ndcg_vals.append(ndcg_at_k(rec_list, gt, K))
        health_vals.append(health_score_at_k(rec_list, env.user_tags[uid], env.item_tags))
        div_vals.append(diversity_score_at_k(rec_list, env.item_emb))

        # Recall@K
        hits = len(set(rec_list[:K]) & gt)
        recall_vals.append(hits / max(len(gt), 1))

    return {
        'ndcg': sum(ndcg_vals) / max(len(ndcg_vals), 1),
        'health': sum(health_vals) / max(len(health_vals), 1),
        'diversity': sum(div_vals) / max(len(div_vals), 1),
        'recall': sum(recall_vals) / max(len(recall_vals), 1),
    }


# -----------------------------------------------------------------------
# Weight selection (Phase 7)
# -----------------------------------------------------------------------

def _simplex_grid(n_points: int = 10) -> List[Tuple[float, float, float]]:
    """Generate evenly spaced weight vectors on the 3-simplex, including corners."""
    weights = []
    # Corner points
    for i in range(3):
        w = [0.0, 0.0, 0.0]
        w[i] = 1.0
        weights.append(tuple(w))
    # Uniform
    weights.append((1 / 3, 1 / 3, 1 / 3))
    # Grid interior
    step = 1.0 / n_points
    for i in range(n_points + 1):
        for j in range(n_points + 1 - i):
            k = n_points - i - j
            w = (i * step, j * step, k * step)
            if w not in weights:
                weights.append(w)
    return weights


def select_operating_weight(
    env: RecommendationEnv,
    policy: MORLPolicy,
    val_user_ids: List[int],
    ground_truth: Dict[int, set],
    K: int = 20,
    n_grid: int = 10,
    alpha: float = 0.7,
    beta: float = 0.3,
    diversity_threshold: float = 0.3,
    device=None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Grid search over weight vectors to find the best operating point w*.

    Selection criterion (Option A from plan):
        Maximize α·NDCG + β·Health   subject to  diversity ≥ diversity_threshold

    Parameters
    ----------
    env                  : eval RecommendationEnv
    policy               : trained MORLPolicy
    val_user_ids         : validation users
    ground_truth         : dict user_id → set of positive item_ids
    K                    : list length
    n_grid               : grid resolution for simplex search
    alpha, beta          : NDCG and Health weights in objective
    diversity_threshold  : minimum acceptable diversity
    device               : torch.device

    Returns
    -------
    w_star   : Tensor (3,) — selected weight vector
    best_metrics : dict — validation metrics under w_star
    """
    device = device or torch.device('cpu')
    grid = _simplex_grid(n_grid)
    # Pre-convert all grid points to tensors to avoid repeated allocation inside the loop
    grid_tensors = [
        torch.tensor(list(w_tuple), dtype=torch.float32, device=device)
        for w_tuple in grid
    ]

    best_score = -float('inf')
    best_w = torch.tensor([1 / 3, 1 / 3, 1 / 3], device=device)
    best_metrics = {}

    print(f"\nSearching over {len(grid_tensors)} weight vectors on the simplex...")
    for w in grid_tensors:
        metrics = evaluate_policy_on_split(env, policy, val_user_ids, w, ground_truth, K=K)

        # Option A: maximize α·NDCG + β·Health  s.t. diversity ≥ threshold
        if metrics['diversity'] >= diversity_threshold:
            score = alpha * metrics['ndcg'] + beta * metrics['health']
        else:
            score = -float('inf')

        if score > best_score:
            best_score = score
            best_w = w
            best_metrics = metrics

    if best_score == -float('inf'):
        # Fallback: relax diversity constraint — pick highest NDCG
        print("Diversity constraint never met; falling back to best NDCG.")
        for w in grid_tensors:
            metrics = evaluate_policy_on_split(env, policy, val_user_ids, w, ground_truth, K=K)
            score = alpha * metrics['ndcg'] + beta * metrics['health']
            if score > best_score:
                best_score = score
                best_w = w
                best_metrics = metrics

    print(f"Selected w* = {best_w.tolist()}")
    print(f"Validation metrics under w*: {best_metrics}")
    return best_w, best_metrics
