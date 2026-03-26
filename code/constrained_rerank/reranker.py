"""
reranker.py — Constrained edit executor.

Takes baseline anchor lists and proposes bounded edits to improve secondary
metrics (health/diversity) while enforcing hard ranking floor constraints.
"""

import torch

from .constraints import FeasibilityChecker, ListDiagnostics, aggregate_diagnostics


def compute_health_overlap(user_tag, food_tags_candidates):
    """
    Compute Jaccard-like health tag overlap between a user and candidate foods.

    Args:
        user_tag: [num_tags] binary user health tag vector
        food_tags_candidates: [num_candidates, num_tags] binary food tag matrix

    Returns:
        overlap_scores: [num_candidates] health overlap score per candidate
    """
    user_expanded = user_tag.unsqueeze(0).expand_as(food_tags_candidates)
    intersection = torch.logical_and(user_expanded, food_tags_candidates).sum(dim=1).float()
    union = torch.logical_or(user_expanded, food_tags_candidates).sum(dim=1).float()
    return intersection / (union + 1e-8)


def constrained_rerank(anchor_topk, anchor_scores, candidate_pool, candidate_scores,
                       checker, user_tags, food_tags, users=None):
    """
    Perform constrained reranking for all users.

    For each user, iterates over editable positions (after lock_positions),
    finds the best candidate from the pool that improves health tag overlap,
    checks feasibility, and swaps if allowed. Falls back to anchor on rejection.

    Args:
        anchor_topk: [num_users, K] baseline top-K item indices
        anchor_scores: [num_users, K] scores for anchor items
        candidate_pool: [num_users, M] top-M candidate item indices
        candidate_scores: [num_users, M] scores for candidate pool
        checker: FeasibilityChecker instance
        user_tags: [num_users, num_tags] user health tags
        food_tags: [num_foods, num_tags] food health tags
        users: optional tensor of user indices to rerank (default: all)

    Returns:
        reranked_topk: [num_users, K] reranked item indices
        all_diagnostics: list of ListDiagnostics per user
    """
    num_users, K = anchor_topk.shape
    reranked_topk = anchor_topk.clone()
    all_diagnostics = []

    user_indices = users if users is not None else range(num_users)

    for u_idx, u in enumerate(user_indices):
        u = int(u)
        diag = ListDiagnostics()
        current_list = reranked_topk[u].tolist()
        swap_count = 0

        # Get this user's health tags
        u_tags = user_tags[u]

        # Get candidate items not already in anchor list
        pool_items = candidate_pool[u].tolist()
        pool_scores_list = candidate_scores[u].tolist()

        # Compute health overlap for all candidates
        pool_items_tensor = candidate_pool[u].long()
        pool_food_tags = food_tags[pool_items_tensor]
        pool_health_scores = compute_health_overlap(u_tags, pool_food_tags)

        # Sort candidates by health overlap (descending) for greedy selection
        sorted_indices = torch.argsort(pool_health_scores, descending=True)

        # Iterate over editable positions
        for pos in range(K):
            if pos < checker.lock_positions:
                continue

            anchor_item = current_list[pos]
            anchor_score_at_pos = anchor_scores[u, pos].item()

            # Try to find the best health-improving candidate for this position
            swapped = False
            for cand_idx in sorted_indices:
                cand_idx = cand_idx.item()
                cand_item = pool_items[cand_idx]
                cand_score = pool_scores_list[cand_idx]

                # Skip if candidate is same as anchor at this position
                if cand_item == anchor_item:
                    continue

                # Check health improvement: candidate should have better overlap
                anchor_food_tags = food_tags[anchor_item]
                anchor_health = compute_health_overlap(u_tags, anchor_food_tags.unsqueeze(0)).item()
                cand_health = pool_health_scores[cand_idx].item()

                if cand_health <= anchor_health:
                    continue

                # Check feasibility
                is_feasible, reason = checker.check_swap(
                    position=pos,
                    anchor_score=anchor_score_at_pos,
                    candidate_score=cand_score,
                    current_list=current_list,
                    candidate_item=cand_item,
                    swap_count=swap_count,
                )

                diag.record_attempt(is_feasible, reason)

                if is_feasible:
                    current_list[pos] = cand_item
                    swap_count += 1
                    swapped = True
                    break

            # If no candidate was feasible or improving, keep anchor (fallback)
            if not swapped:
                pass  # anchor item remains

        reranked_topk[u] = torch.tensor(current_list, dtype=reranked_topk.dtype,
                                        device=reranked_topk.device)
        all_diagnostics.append(diag)

    return reranked_topk, all_diagnostics
