"""Evaluation for the sequential MORL policy.

Provides:
  evaluate_sequential — run a greedy single-policy rollout on a user split and
                         compute recommendation metrics.
  compare_baselines   — print a side-by-side table of one-shot vs sequential
                         metrics on the same frozen embeddings.
"""

import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_HERE)
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

from seqmorl.environment import SequentialRecEnv
from seqmorl.policy import SequentialPolicy


# ---------------------------------------------------------------------------
# Greedy policy rollout
# ---------------------------------------------------------------------------

def _greedy_rollout(policy: SequentialPolicy,
                    env: SequentialRecEnv,
                    user_ids: list[int],
                    excluded_items_by_user: dict[int, set[int]] | None = None) -> dict[int, list[int]]:
    """Return per-user recommended item lists (greedy policy)."""
    policy.eval()
    recommendations: dict[int, list[int]] = {}
    with torch.no_grad():
        for uid in user_ids:
            state = env.reset(uid)
            items: list[int] = []
            done = False
            excluded_items = set()
            if excluded_items_by_user is not None:
                excluded_items = excluded_items_by_user.get(uid, set())

            excluded_action_mask = torch.zeros(env.action_dim, dtype=torch.bool, device=env.device)
            if excluded_items:
                candidate_items = env.candidate_pools[uid]
                excluded_tensor = torch.tensor(
                    sorted(excluded_items), dtype=torch.long, device=env.device
                )
                excluded_action_mask = torch.isin(candidate_items, excluded_tensor)

            while not done:
                mask = env.get_action_mask()
                if excluded_items:
                    mask = mask | excluded_action_mask
                if bool(mask.all()):
                    break
                action, _, _ = policy.select_action(state, mask, greedy=True)
                next_state, _, done, item_idx = env.step(action)
                items.append(item_idx)
                state = next_state
            recommendations[uid] = items
    return recommendations


# ---------------------------------------------------------------------------
# Metric helpers (aligned with RCSYS_utils.py conventions)
# ---------------------------------------------------------------------------

def _ndcg_at_k(recommended: list[int], ground_truth: set[int], k: int) -> float:
    hits = [1.0 if item in ground_truth else 0.0 for item in recommended[:k]]
    dcg = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(ground_truth))))
    return dcg / ideal if ideal > 0 else 0.0


def _recall_precision_at_k(recommended: list[int],
                            ground_truth: set[int],
                            k: int):
    hits = len(set(recommended[:k]) & ground_truth)
    recall = hits / len(ground_truth) if ground_truth else 0.0
    precision = hits / k if k > 0 else 0.0
    return recall, precision


def _health_score(recommended: list[int],
                  user_id: int,
                  user_tags: torch.Tensor,
                  food_tags: torch.Tensor) -> float:
    """Proportion of recommended items sharing at least one tag with the user."""
    if not recommended:
        return 0.0
    u_tags = user_tags[user_id].bool()
    i_tags = food_tags[torch.tensor(recommended)].bool()
    match = (i_tags & u_tags.unsqueeze(0)).any(dim=1).float()
    return float(match.mean().item())


def _diversity_score(recommended: list[int], item_emb: torch.Tensor) -> float:
    """1 - mean pairwise cosine similarity among recommended items."""
    if len(recommended) < 2:
        return 0.0
    embs = F.normalize(item_emb[torch.tensor(recommended)], dim=1)
    sim_matrix = embs @ embs.T
    n = len(recommended)
    # Upper triangle (excluding diagonal).
    upper = sim_matrix.triu(diagonal=1)
    num_pairs = n * (n - 1) / 2
    mean_sim = float(upper.sum().item() / (num_pairs + 1e-8))
    return max(0.0, 1.0 - mean_sim)


# ---------------------------------------------------------------------------
# Public evaluation API
# ---------------------------------------------------------------------------

def evaluate_sequential(policy: SequentialPolicy,
                        env: SequentialRecEnv,
                        user_ids: list[int],
                        pos_edge_index: torch.Tensor,
                        user_tags: torch.Tensor,
                        food_tags: torch.Tensor,
                        K: int = 20,
                        exclude_edge_indices: list[torch.Tensor] | None = None) -> dict:
    """Evaluate *policy* on *user_ids* and return metric dict.

    Args:
        policy              : Trained SequentialPolicy.
        env                 : SequentialRecEnv (same frozen embeddings).
        user_ids            : User IDs to evaluate.
        pos_edge_index      : Ground-truth positive edges for this split [2, N].
        user_tags           : [num_users, num_tags]
        food_tags           : [num_items, num_tags]
        K                   : Evaluation cutoff.
        exclude_edge_indices: Additional edge indices to mask (e.g. train edges).

    Returns:
        Dict with ndcg, recall, precision, health, diversity, coverage.
    """
    # Build ground-truth positive sets per user.
    gt: dict[int, set[int]] = defaultdict(set)
    for u, i in pos_edge_index.T.tolist():
        gt[int(u)].add(int(i))

    excluded_items_by_user: dict[int, set[int]] = defaultdict(set)
    if exclude_edge_indices is not None:
        for edge_index in exclude_edge_indices:
            if edge_index is None or edge_index.numel() == 0:
                continue
            for u, i in edge_index.T.tolist():
                excluded_items_by_user[int(u)].add(int(i))

    # Greedy rollout.
    recs = _greedy_rollout(policy, env, user_ids, excluded_items_by_user)

    ndcg_vals, rec_vals, prec_vals, health_vals, div_vals = [], [], [], [], []
    all_recommended: set[int] = set()

    for uid in user_ids:
        recommended = recs.get(uid, [])
        if uid in excluded_items_by_user:
            excluded = excluded_items_by_user[uid]
            recommended = [i for i in recommended if i not in excluded]
        ground_truth = gt.get(uid, set())
        if not ground_truth:
            continue

        ndcg_vals.append(_ndcg_at_k(recommended, ground_truth, K))
        r, p = _recall_precision_at_k(recommended, ground_truth, K)
        rec_vals.append(r)
        prec_vals.append(p)
        health_vals.append(_health_score(recommended, uid, user_tags, food_tags))
        div_vals.append(_diversity_score(recommended, env.item_emb))
        all_recommended.update(recommended[:K])

    num_items = env.num_items
    coverage = len(all_recommended) / num_items if num_items > 0 else 0.0

    return {
        'ndcg': float(np.mean(ndcg_vals)) if ndcg_vals else 0.0,
        'recall': float(np.mean(rec_vals)) if rec_vals else 0.0,
        'precision': float(np.mean(prec_vals)) if prec_vals else 0.0,
        'health': float(np.mean(health_vals)) if health_vals else 0.0,
        'diversity': float(np.mean(div_vals)) if div_vals else 0.0,
        'coverage': coverage,
    }


def compare_baselines(baseline_metrics: dict,
                      sequential_metrics: dict,
                      split: str = 'test'):
    """Print a side-by-side comparison table."""
    print(f"\n{'='*60}")
    print(f"  Evaluation comparison ({split} split)")
    print(f"{'='*60}")
    print(f"{'Metric':<22}{'One-shot Baseline':>18}{'Sequential MORL':>18}")
    print(f"{'-'*60}")
    keys = ['ndcg', 'recall', 'precision', 'health', 'diversity', 'coverage']
    for k in keys:
        b = baseline_metrics.get(k, float('nan'))
        s = sequential_metrics.get(k, float('nan'))
        delta = s - b
        sign = '+' if delta >= 0 else ''
        print(f"  {k:<20}{b:>18.5f}{s:>18.5f}  ({sign}{delta:.5f})")
    print(f"{'='*60}\n")
