"""
anchor.py — Baseline anchor list generation from frozen SGSL embeddings.

Loads a trained SGSL checkpoint, reconstructs the model, runs forward pass
on eval split edges (matching baseline eval exactly), and generates anchor
top-K lists with the same exclusion masking as get_metrics().
"""

import sys
import os
import torch

# Add parent code/ directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from RCSYS_models import SGSL
from RCSYS_utils import get_user_positive_items


def load_model_and_data(checkpoint_path, graph_path, device):
    """
    Load checkpoint and reconstruct SGSL model from state_dict.

    Args:
        checkpoint_path: Path to sgsl_checkpoint.pt
        graph_path: Path to benchmark_macro.pt (needed for SGSL constructor)
        device: torch device

    Returns:
        model: Loaded SGSL model in eval mode
        ckpt: Full checkpoint dict with all tensors and metadata
    """
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    graph = torch.load(graph_path, map_location='cpu')

    model = SGSL(
        graph,
        embedding_dim=ckpt['hidden_dim'],
        feature_threshold=ckpt['feature_threshold'],
        num_layer=ckpt['layers'],
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, ckpt


def get_embeddings_for_split(model, feature_dict, edge_index, pos_edge_index, neg_edge_index, device):
    """
    Run model.forward() on given split's edges to produce embeddings.
    This matches what eval() does at RCSYS_utils.py:400-401.

    Args:
        model: SGSL model in eval mode
        feature_dict: dict of node features
        edge_index: eval split edge_index
        pos_edge_index: positive edges for this split
        neg_edge_index: negative edges for this split
        device: torch device

    Returns:
        users_emb_final: [num_users, embedding_dim]
        items_emb_final: [num_items, embedding_dim]
    """
    feature_dict_dev = {k: v.to(device) for k, v in feature_dict.items()}
    edge_index_dev = edge_index.to(device)
    pos_edge_dev = pos_edge_index.to(device)
    neg_edge_dev = neg_edge_index.to(device)

    with torch.no_grad():
        users_emb_final, _, items_emb_final, _ = model.forward(
            feature_dict_dev, edge_index_dev, pos_edge_dev, neg_edge_dev
        )

    return users_emb_final, items_emb_final


def get_anchor_list_and_scores(users_emb, items_emb, K, M, exclude_edge_indices):
    """
    Generate baseline anchor top-K lists and candidate pools.
    Replicates exact masking logic from get_metrics() (RCSYS_utils.py:337-348).

    Args:
        users_emb: [num_users, dim] user embeddings
        items_emb: [num_items, dim] item embeddings
        K: top-K for anchor list
        M: top-M for candidate pool (M > K)
        exclude_edge_indices: list of edge_index tensors to mask out

    Returns:
        anchor_topk: [num_users, K] baseline top-K item indices
        anchor_scores: [num_users, K] dot-product scores for anchor items
        candidate_pool: [num_users, M] top-M candidate item indices
        candidate_scores: [num_users, M] scores for candidate pool
        rating_matrix: [num_users, num_items] full masked rating matrix
    """
    # Compute full rating matrix
    rating = torch.matmul(users_emb, items_emb.T)

    # Apply exclusion masking (same as get_metrics lines 337-348)
    for exclude_edge_index in exclude_edge_indices:
        user_pos_items = get_user_positive_items(exclude_edge_index)
        exclude_users = []
        exclude_items = []
        for user, items in user_pos_items.items():
            exclude_users.extend([user] * len(items))
            exclude_items.extend(items)
        rating[exclude_users, exclude_items] = -(1 << 10)

    # Get top-K anchor list
    anchor_scores, anchor_topk = torch.topk(rating, k=K)

    # Get top-M candidate pool
    candidate_scores, candidate_pool = torch.topk(rating, k=M)

    return anchor_topk, anchor_scores, candidate_pool, candidate_scores, rating
