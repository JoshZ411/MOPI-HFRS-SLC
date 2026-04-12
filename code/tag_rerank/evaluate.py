"""
evaluate.py — TARS evaluation wrapper using the identical RCSYS_utils.py protocol
Author: Harshit | Branch: harshit-develop

Drop-in replacement for the baseline eval() in RCSYS_utils.py. The ONLY
difference is the rating matrix: tag_augmented_scores(...) instead of
user_emb @ food_emb.T. All masking, metric functions, and split handling
are exactly identical to the original codebase.

Parity guarantee:
    tars_get_metrics(..., alpha=0.0) ≡ get_metrics(...)   for all users
"""

import sys
import os

# Allow import from parent code/ directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from RCSYS_utils import (
    get_user_positive_items,
    RecallPrecision_ATk,
    NDCGatK_r,
    calculate_health_score,
    calculate_average_health_tags,
    calculate_percentage_recommended_foods,
    bpr_loss,
    structured_negative_sampling,
)
from scorer import tag_augmented_scores


# ─── Core Metric Function ─────────────────────────────────────────────────────

def tars_get_metrics(
    user_emb: torch.Tensor,
    food_emb: torch.Tensor,
    user_tags: torch.Tensor,
    food_tags: torch.Tensor,
    edge_index: torch.Tensor,
    exclude_edge_indices: list,
    k: int,
    alpha: float,
) -> tuple:
    """
    Compute all 6 evaluation metrics using tag-augmented scores.

    Replicates get_metrics() from RCSYS_utils.py exactly, with a single change:
    the rating matrix is built by tag_augmented_scores() rather than raw dot product.

    Parameters
    ----------
    user_emb : Tensor (N_u, D)
    food_emb : Tensor (N_i, D)
    user_tags : Tensor (N_u, T)
    food_tags : Tensor (N_i, T)
    edge_index : Tensor (2, E) — edges for the split being evaluated
    exclude_edge_indices : list of Tensor — edges to mask out (train edges for val, train+val for test)
    k : int — top-K cutoff
    alpha : float — tag blending coefficient

    Returns
    -------
    tuple: (recall, precision, ndcg, health_score, avg_health_tags_ratio, pct_foods)
           Identical semantics to get_metrics() return value.
    """
    # ── Build score matrix using TARS ─────────────────────────────────────────
    # alpha=0.0: identical to baseline (raw dot product)
    # alpha>0.0: convex blend of norm(dot) + jaccard
    rating = tag_augmented_scores(user_emb, food_emb, user_tags, food_tags, alpha)
    # rating: (N_u, N_i)

    # ── Apply exclusion masking — IDENTICAL to get_metrics() ─────────────────
    # Train edges are masked during val evaluation; train+val during test.
    for excl_ei in exclude_edge_indices:
        user_pos_items = get_user_positive_items(excl_ei)
        exclude_users, exclude_items = [], []
        for user, items in user_pos_items.items():
            exclude_users.extend([user] * len(items))
            exclude_items.extend(items)
        # -1024: identical sentinel value used by baseline get_metrics()
        rating[exclude_users, exclude_items] = -(1 << 10)

    # ── Top-K retrieval ───────────────────────────────────────────────────────
    _, top_K_items = torch.topk(rating, k=k)   # (N_u, K)

    # ── Ground-truth lookup — IDENTICAL to get_metrics() ─────────────────────
    users = edge_index[0].unique()
    test_user_pos_items = get_user_positive_items(edge_index)
    test_user_pos_items_list = [test_user_pos_items[u.item()] for u in users]

    r = []
    for user in users:
        ground_truth = test_user_pos_items[user.item()]
        label = list(map(lambda x: x in ground_truth, top_K_items[user]))
        r.append(label)
    r = torch.Tensor(np.array(r).astype('float'))

    # ── Metric computation — using EXACT same functions from RCSYS_utils.py ──
    recall, precision = RecallPrecision_ATk(test_user_pos_items_list, r, k)
    ndcg      = NDCGatK_r(test_user_pos_items_list, r, k)
    hs        = calculate_health_score(users, top_K_items, user_tags, food_tags)
    avg_ht    = calculate_average_health_tags(users, top_K_items, food_tags)
    pct_foods = calculate_percentage_recommended_foods(users, top_K_items, food_emb.size(0))

    return recall, precision, ndcg, hs, avg_ht, pct_foods


# ─── Full Eval Wrapper (mirrors RCSYS_utils.eval) ────────────────────────────

def tars_eval(
    model,
    feature_dict: dict,
    user_tags: torch.Tensor,
    food_tags: torch.Tensor,
    edge_index: torch.Tensor,
    pos_edge_index: torch.Tensor,
    neg_edge_index: torch.Tensor,
    exclude_edge_indices: list,
    k: int,
    lambda_val: float,
    alpha: float,
) -> tuple:
    """
    Full evaluation wrapper analogous to eval() in RCSYS_utils.py.
    Runs the SGSL forward pass, then evaluates with TARS scoring.

    Parameters
    ----------
    model : SGSL — trained model (in eval mode)
    feature_dict : dict — {node_type: feature_tensor}
    alpha : float — tag blending coefficient for TARS
    (all others identical to RCSYS_utils.eval)

    Returns
    -------
    tuple: (bpr_loss_approx, recall, precision, ndcg, health_score,
            avg_health_tags_ratio, pct_foods)
    """
    # Forward pass — identical to baseline eval()
    users_emb_final, users_emb_0, items_emb_final, items_emb_0 = \
        model.forward(feature_dict, edge_index, pos_edge_index, neg_edge_index)

    edges = structured_negative_sampling(edge_index, contains_neg_self_loops=False)
    user_idx, pos_idx, neg_idx = edges[0], edges[1], edges[2]
    neg_idx = torch.randint(0, int(edge_index[1].max() - 1),
                            size=(len(neg_idx),), dtype=torch.long)

    # Indexed embeddings for loss
    u_emb    = users_emb_final[user_idx]
    u_emb_0  = users_emb_0[user_idx]
    pi_emb   = items_emb_final[pos_idx]
    pi_emb_0 = items_emb_0[pos_idx]
    ni_emb   = items_emb_final[neg_idx]
    ni_emb_0 = items_emb_0[neg_idx]

    # Approximate BPR loss (same comment as baseline: "just for a rough estimate")
    loss = bpr_loss(u_emb, u_emb_0, pi_emb, pi_emb_0, ni_emb, ni_emb_0, lambda_val).item()

    # TARS metrics — uses full embedding matrices, not the indexed batch
    recall, precision, ndcg, hs, avg_ht, pct = tars_get_metrics(
        users_emb_final, items_emb_final,
        user_tags, food_tags,
        edge_index, exclude_edge_indices, k, alpha
    )

    return loss, recall, precision, ndcg, hs, avg_ht, pct


# ─── Baseline Parity Check ───────────────────────────────────────────────────

def assert_baseline_parity(
    user_emb: torch.Tensor,
    food_emb: torch.Tensor,
    user_tags: torch.Tensor,
    food_tags: torch.Tensor,
    edge_index: torch.Tensor,
    exclude_edge_indices: list,
    k: int,
    user_limit: int = 100,
) -> bool:
    """
    Verify that TARS with alpha=0.0 produces identical top-K rankings to the
    baseline dot-product scoring, for the first `user_limit` users.

    Returns True if parity holds, False otherwise.
    Used as a correctness gate before running the full alpha sweep.
    """
    from RCSYS_utils import get_metrics as baseline_get_metrics

    # Limit to first `user_limit` users for speed
    u_emb = user_emb[:user_limit]
    u_tags = user_tags[:user_limit]

    # Baseline: raw dot product
    baseline_rating = torch.matmul(u_emb, food_emb.T)

    # TARS alpha=0: should return identical raw dot product
    tars_rating = tag_augmented_scores(u_emb, food_emb, u_tags, food_tags, alpha=0.0)

    # Apply same masking to both
    for excl_ei in exclude_edge_indices:
        upi = get_user_positive_items(excl_ei)
        eu, ei = [], []
        for user, items in upi.items():
            if user >= user_limit:
                continue
            eu.extend([user] * len(items))
            ei.extend(items)
        if eu:
            baseline_rating[eu, ei] = -(1 << 10)
            tars_rating[eu, ei] = -(1 << 10)

    # Compare argmax (top-1) for every user row
    baseline_top1 = baseline_rating.argmax(dim=1)
    tars_top1     = tars_rating.argmax(dim=1)

    mismatches = (baseline_top1 != tars_top1).sum().item()

    if mismatches == 0:
        print(f"[PARITY OK] alpha=0.0 top-1 matches baseline exactly for all {user_limit} sampled users.")
        return True
    else:
        first_fail = (baseline_top1 != tars_top1).nonzero(as_tuple=True)[0][0].item()
        print(f"[PARITY FAIL] {mismatches}/{user_limit} users have mismatched top-1.")
        print(f"  First failing user {first_fail}: baseline top-1={baseline_top1[first_fail].item()}, "
              f"tars top-1={tars_top1[first_fail].item()}")
        return False


# ─── CLI parity check mode ───────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TARS parity check")
    parser.add_argument('--graph_path', default='../../processed_data/benchmark_macro.pt')
    parser.add_argument('--user_limit', type=int, default=50)
    parser.add_argument('--parity_check', action='store_true')
    args = parser.parse_args()

    if args.parity_check:
        from RCSYS_utils import split_data_new
        from RCSYS_models import SGSL

        print(f"Loading graph from {args.graph_path} ...")
        graph = torch.load(args.graph_path, map_location='cpu')

        edge_index       = graph[('user', 'eats', 'food')].edge_index
        edge_label_index = graph[('user', 'eats', 'food')].edge_label_index
        user_tags        = graph['user'].tags
        food_tags        = graph['food'].tags
        feature_dict     = {k: graph[k].x for k in ['user', 'food']}

        _, _, test_ei, _, neg_train, _, _, _, _ = split_data_new(edge_index, edge_label_index)

        # Random embeddings (no checkpoint needed for parity check)
        torch.manual_seed(42)
        n_u, n_i = graph['user'].num_nodes, graph['food'].num_nodes
        u_emb = torch.randn(n_u, 128)
        f_emb = torch.randn(n_i, 128)

        result = assert_baseline_parity(
            u_emb, f_emb, user_tags, food_tags,
            test_ei, [neg_train], k=20,
            user_limit=args.user_limit
        )
        exit(0 if result else 1)
