"""
scorer.py — Core scoring logic for Tag-Augmented Re-scoring (TARS)
Author: Harshit | Branch: harshit-develop

Motivation
----------
SGSL trains with a Jaccard-based health loss, teaching the model's embedding
space to align with health tag structure. However, at inference time,
get_metrics() in RCSYS_utils.py scores items using ONLY:

    rating = user_emb @ food_emb.T

Health tags are completely ignored at inference. A food with perfect
nutritional alignment (Jaccard = 1.0) but a slightly lower embedding
score will always rank below a tag-mismatched food. TARS fixes this.

Formula
-------
    score(u, i) = (1 - α) * norm(u_emb · i_emb)  +  α * J(tags_u, tags_i)

Where:
    - norm(·) = per-user min-max normalization, putting embedding scores
                in [0, 1] to match the Jaccard scale
    - J(a, b) = |a ∩ b| / |a ∪ b|  (Jaccard similarity, already in [0,1])
    - α ∈ [0, 1] is the tag-blending coefficient
        α = 0.0 → pure embedding baseline (identical to SGSL)
        α = 1.0 → pure tag-based ranking
        α ∈ (0, 1) → the interpolated TARS model
"""

import torch
import torch.nn.functional as F


# ─── Jaccard Matrix ──────────────────────────────────────────────────────────

def jaccard_matrix(user_tags: torch.Tensor, food_tags: torch.Tensor) -> torch.Tensor:
    """
    Compute the full Jaccard similarity matrix between all users and all foods.

    For binary tag vectors, Jaccard is computed efficiently via matrix multiply:
        intersection(u, i) = tags_u · tags_i   (dot product counts shared active tags)
        union(u, i)        = ||tags_u||₁ + ||tags_i||₁ - intersection(u, i)
        jaccard(u, i)      = intersection(u, i) / (union(u, i) + ε)

    Parameters
    ----------
    user_tags : Tensor, shape (N_u, T)
        Binary health tag vectors for all users.
    food_tags : Tensor, shape (N_i, T)
        Binary nutritional tag vectors for all food items.

    Returns
    -------
    Tensor, shape (N_u, N_i), values in [0, 1]
        jaccard[u, i] = Jaccard similarity between user u's tags and food i's tags.
        Value is 0 if both vectors are all-zero (no tags).
    """
    u = user_tags.float().cpu()   # (N_u, T) — always on CPU (tags are small)
    f = food_tags.float().cpu()   # (N_i, T)

    # Intersection: dot product of binary vectors = count of shared active tags
    # Shape: (N_u, N_i)
    inter = torch.matmul(u, f.T)

    # Per-vector L1 norms (= number of active tags)
    u_sum = u.sum(dim=1, keepdim=True)   # (N_u, 1)
    f_sum = f.sum(dim=1, keepdim=True)   # (N_i, 1)

    # Union = |A| + |B| - |A ∩ B|
    union = u_sum + f_sum.T - inter      # (N_u, N_i)

    # Jaccard with epsilon for stability when both users and foods have zero tags
    jac = inter / (union + 1e-8)         # (N_u, N_i), values in [0, 1]

    return jac


# ─── Tag-Augmented Score Matrix ──────────────────────────────────────────────

def tag_augmented_scores(
    user_emb: torch.Tensor,
    food_emb: torch.Tensor,
    user_tags: torch.Tensor,
    food_tags: torch.Tensor,
    alpha: float = 0.1,
) -> torch.Tensor:
    """
    Compute tag-augmented recommendation scores for all (user, food) pairs.

    score(u, i) = (1 - alpha) * norm(u_emb · i_emb)  +  alpha * J(tags_u, tags_i)

    The embedding score is per-user min-max normalized to [0, 1] so that it
    is on the same scale as the Jaccard score before blending.

    Parameters
    ----------
    user_emb : Tensor, shape (N_u, D)
        Learned user embeddings from SGSL forward pass.
    food_emb : Tensor, shape (N_i, D)
        Learned food item embeddings from SGSL forward pass.
    user_tags : Tensor, shape (N_u, T)
        Binary health tag vectors for users (from graph['user'].tags).
    food_tags : Tensor, shape (N_i, T)
        Binary nutritional tag vectors for foods (from graph['food'].tags).
    alpha : float, default 0.1
        Tag blending coefficient (0 = pure embedding, 1 = pure Jaccard).

    Returns
    -------
    Tensor, shape (N_u, N_i)
        Blended recommendation scores. Higher = more recommended.

    Notes
    -----
    When alpha=0.0, the function skips normalization entirely and returns the
    raw dot-product scores, guaranteeing identical ranking to the SGSL baseline.
    """
    num_users, _ = user_emb.shape
    num_foods, _ = food_emb.shape

    # ── Step 1: Embedding dot-product scores ──────────────────────────────────
    emb_scores = torch.matmul(user_emb, food_emb.T)   # (N_u, N_i)

    assert emb_scores.shape == (num_users, num_foods), (
        f"Score matrix shape mismatch: expected ({num_users}, {num_foods}), "
        f"got {emb_scores.shape}"
    )

    # ── Alpha = 0: return raw scores, identical to baseline ───────────────────
    if alpha == 0.0:
        return emb_scores

    # ── Step 2: Per-user min-max normalize embedding scores to [0, 1] ─────────
    # Per-user normalization preserves relative ordering within each user's list
    # while making the scale compatible with Jaccard scores
    emb_min = emb_scores.min(dim=1, keepdim=True)[0]   # (N_u, 1)
    emb_max = emb_scores.max(dim=1, keepdim=True)[0]   # (N_u, 1)
    emb_norm = (emb_scores - emb_min) / (emb_max - emb_min + 1e-8)   # (N_u, N_i), in [0,1]

    # ── Step 3: Full Jaccard matrix ───────────────────────────────────────────
    # Computed on CPU (tags are small: ~8170 × T and ~6769 × T)
    jac = jaccard_matrix(user_tags, food_tags).to(user_emb.device)    # (N_u, N_i)

    # ── Step 4: Convex blend ──────────────────────────────────────────────────
    scores = (1.0 - alpha) * emb_norm + alpha * jac     # (N_u, N_i), in [0,1]

    return scores


# ─── Per-User Score Diagnostics ──────────────────────────────────────────────

def score_diagnostics(
    user_emb: torch.Tensor,
    food_emb: torch.Tensor,
    user_tags: torch.Tensor,
    food_tags: torch.Tensor,
    alpha: float,
    sample_users: int = 5,
) -> None:
    """
    Print diagnostic information about score distributions for a few users.
    Useful for sanity-checking that alpha shifts scores in the expected direction.
    """
    n = min(sample_users, user_emb.shape[0])
    emb_scores = torch.matmul(user_emb[:n], food_emb.T)   # (n, N_i)
    jac_scores = jaccard_matrix(user_tags[:n], food_tags).to(user_emb.device)

    emb_top1 = emb_scores.argmax(dim=1)
    jac_top1 = jac_scores.argmax(dim=1)

    print(f"\n{'─'*60}")
    print(f"TARS Score Diagnostics (alpha={alpha}, first {n} users)")
    print(f"{'─'*60}")
    print(f"{'User':>6} | {'Emb Top-1':>12} | {'Jac Top-1':>12} | {'Emb scores [min,max]':>24} | {'Jac[user,Emb-Top1]':>20}")
    print(f"{'─'*6}-+-{'─'*12}-+-{'─'*12}-+-{'─'*24}-+-{'─'*20}")
    for u in range(n):
        e_min = emb_scores[u].min().item()
        e_max = emb_scores[u].max().item()
        jac_at_emb_top = jac_scores[u, emb_top1[u]].item()
        print(
            f"{u:>6} | {emb_top1[u].item():>12} | {jac_top1[u].item():>12} | "
            f"  [{e_min:>8.4f}, {e_max:>8.4f}]   | {jac_at_emb_top:>20.4f}"
        )
    print(f"{'─'*60}\n")
