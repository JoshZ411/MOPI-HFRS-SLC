"""Simple linear re-rank baseline.

Re-scores all items with:

    combined_score = alpha * gnn_score + (1 - alpha) * health_hit

where ``health_hit`` is 1 if the item shares at least one health tag with the
user, 0 otherwise.

Evaluation **exactly replicates** main.py's eval() / get_metrics() so that
alpha=1.0 produces the published GNN baseline (ndcg≈0.10252).

There are **three interacting bugs** in main.py's evaluation:

  Bug 1 — Edge-indexed rating matrix
      eval() slices users_emb_final[user_indices] where user_indices =
      test_edge_index[0].  The resulting matrix has shape (E_test, d);
      get_metrics then builds rating = (E_test, num_items).

  Bug 2 — Global user ID used as edge-position index
      get_metrics fetches top_K_items[u] for each eval user u.  Since u is a
      global user ID (0..8169) but top_K_items is indexed by edge position
      (0..E_test-1=62844), user u gets the recommendations of whichever user
      happens to occupy edge position u in test_edge_index.

  Bug 3 — Exclusion applied to the edge-indexed matrix with global user IDs
      neg_train_edge_index (train edges NOT in edge_label_index, ~115K edges)
      is passed as exclude_edge_indices.  get_metrics does
      rating[global_user_id, item] = -(1<<10), using global user IDs as row
      indices into the edge-indexed rating matrix.  This corrupts rows
      0..max_user_id of the (E_test, num_items) matrix, dramatically changing
      the top-K items for users whose global IDs fall in that range.

This script builds gnn_scores_per_user and health_hits_per_user of shape
(num_users, num_items), then for each alpha:
  1.  combined_per_user = alpha * gnn + (1-alpha) * health
  2.  combined_edge      = combined_per_user[test_ei[0]]   # edge-indexed (Bug 1)
  3.  combined_edge[excl_users, excl_items] = -inf          # buggy exclusion (Bug 3)
  4.  top_K_edge         = topk(combined_edge, K)           # (E_test, K)
  5.  for user u: recs   = top_K_edge[u]                   # global-ID lookup (Bug 2)

Usage
-----
    cd code/
    python -m morl.rerank_baseline \\
        --checkpoint embeddings_checkpoint.pt \\
        --graph_path ../processed_data/benchmark_macro.pt \\
        --device cuda --K 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.model_selection import train_test_split

try:
    from .logging_utils import setup_logger
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from morl.logging_utils import setup_logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_user_positive_items(edge_index: torch.Tensor) -> dict[int, list[int]]:
    pos: dict[int, list[int]] = {}
    for u, i in edge_index.T.tolist():
        pos.setdefault(int(u), []).append(int(i))
    return pos


# ---------------------------------------------------------------------------
# Core evaluation  —  replicates main.py's eval() / get_metrics() exactly
# ---------------------------------------------------------------------------

def evaluate_mainy_style(
    *,
    top_K_edge: torch.Tensor,        # (E_test, K) — EDGE-indexed; row u = recs for the u-th test edge
    eval_user_ids: list[int],        # unique user IDs in the eval split
    pos_items_per_user: dict[int, list[int]],
    user_tags_gpu: torch.Tensor,     # (num_total_users, tag_dim)
    item_tags_gpu: torch.Tensor,     # (num_items, tag_dim)
    item_emb_gpu: torch.Tensor,      # (num_items, d)
    K: int,
    device: torch.device,
) -> dict[str, float]:
    """Replicates main.py's exact per-user evaluation.

    For each eval user u (global ID):
      - recs   = top_K_edge[u]          — Bug 2: global user ID used as edge-position index
      - GT     = pos_items_per_user[u]  — correct ground truth for user u
      - health = user_tags_gpu[u] paired with recs
    """
    positions = torch.arange(1, K + 1, dtype=torch.float32, device=device)
    discounts = 1.0 / torch.log2(positions + 1.0)  # (K,)

    ndcg_list, recall_list, health_list, div_list = [], [], [], []

    for u_global in eval_user_ids:
        # Bug 2: use global user ID as edge-position index
        rec_tensor = top_K_edge[u_global]  # (K,)

        ground_truth = set(pos_items_per_user.get(u_global, []))
        gt_size = len(ground_truth)

        # NDCG@K and Recall@K — ground truth is correct for u_global
        if gt_size > 0:
            labels = torch.tensor(
                [1.0 if item in ground_truth else 0.0 for item in rec_tensor.tolist()],
                dtype=torch.float32, device=device,
            )
            dcg = float((labels * discounts).sum().item())
            ideal_k = min(gt_size, K)
            idcg = float(discounts[:ideal_k].sum().item())
            ndcg_list.append(dcg / idcg if idcg > 0.0 else 0.0)
            hits = int((labels > 0.5).sum().item())
            recall_list.append(hits / gt_size)
        else:
            ndcg_list.append(0.0)
            recall_list.append(0.0)

        # Health — correct user tags, buggy item assignment (edge-position indexed)
        user_tag_v = user_tags_gpu[u_global]
        rec_tags = item_tags_gpu[rec_tensor]  # (K, tag_dim)
        healthy_count = int(
            (rec_tags.bool() & user_tag_v.bool()).any(dim=1).sum().item()
        )
        health_list.append(healthy_count / K)

        # Diversity (1 - mean pairwise cosine similarity)
        rec_embs = item_emb_gpu[rec_tensor]  # (K, d)
        rec_embs_n = torch.nn.functional.normalize(rec_embs, dim=1)
        sim_mat = rec_embs_n @ rec_embs_n.T
        triu = torch.triu_indices(K, K, offset=1)
        div_list.append(1.0 - sim_mat[triu[0], triu[1]].mean().item())

    def _m(lst: list[float]) -> float:
        return float(np.mean(lst)) if lst else 0.0

    return {
        'ndcg':         _m(ndcg_list),
        'recall':       _m(recall_list),
        'health_score': _m(health_list),
        'diversity':    _m(div_list),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description='Simple linear re-rank baseline (main.py-compatible)')
    parser.add_argument('--checkpoint', type=str, default='embeddings_checkpoint.pt')
    parser.add_argument('--graph_path',  type=str, default='../processed_data/benchmark_macro.pt')
    parser.add_argument('--K',    type=int, default=20,  help='Recommendation list length.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--alpha_steps', type=int, default=11,
                        help='Number of alpha values to sweep from 0.0 to 1.0 inclusive.')
    parser.add_argument('--output', type=str, default='rerank_baseline_results.json',
                        help='Path to write JSON results.')
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'],
                        help='Which split to evaluate on.')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = (torch.device('cuda') if args.device == 'auto' and torch.cuda.is_available()
              else torch.device(args.device if args.device != 'auto' else 'cpu'))

    logger = setup_logger('.')
    logger.info('Device: %s', device)

    # ------------------------------------------------------------------
    # Load embeddings + graph
    # ------------------------------------------------------------------
    logger.info('Loading checkpoint: %s', args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    user_emb = ckpt['user_emb']  # (U, d)
    item_emb = ckpt['item_emb']  # (I, d)
    num_total_users = user_emb.size(0)
    num_items = item_emb.size(0)
    logger.info('user_emb: %s  item_emb: %s', tuple(user_emb.shape), tuple(item_emb.shape))

    logger.info('Loading graph: %s', args.graph_path)
    graph = torch.load(args.graph_path, map_location='cpu')
    user_tags = graph['user'].tags  # (U, tag_dim)
    food_tags = graph['food'].tags  # (I, tag_dim)
    edge_index = graph[('user', 'eats', 'food')].edge_index
    # edge_label_index: labeled (positive) subset of interactions
    eli = graph[('user', 'eats', 'food')].edge_label_index  # (2, E_labeled)

    # 60/20/20 split — identical to main.py / morl_main.py
    edges = edge_index.numpy().T
    train_edges_arr, test_edges_arr = train_test_split(edges, test_size=0.2, random_state=args.seed)
    train_edges_arr, val_edges_arr  = train_test_split(train_edges_arr, test_size=0.25, random_state=args.seed)

    train_ei = torch.LongTensor(train_edges_arr).T
    val_ei   = torch.LongTensor(val_edges_arr).T
    test_ei  = torch.LongTensor(test_edges_arr).T

    # ------------------------------------------------------------------
    # Build neg_train_edge_index (Bug 3 ingredient):
    # train edges NOT in edge_label_index — passed as exclude_edge_indices
    # in main.py's final test eval call.
    # ------------------------------------------------------------------
    eli_set = set(tuple(eli[:, i].tolist()) for i in range(eli.size(1)))
    neg_train_edges = [e for e in train_edges_arr.tolist() if tuple(e) not in eli_set]
    logger.info('neg_train_edge_index: %d edges', len(neg_train_edges))

    if neg_train_edges:
        neg_train_ei = torch.LongTensor(neg_train_edges).T
    else:
        neg_train_ei = torch.zeros(2, 0, dtype=torch.long)

    # Flatten exclusion to (excl_users, excl_items) — global user IDs
    neg_user_pos: dict[int, list[int]] = {}
    for i in range(neg_train_ei.size(1)):
        u  = int(neg_train_ei[0][i].item())
        it = int(neg_train_ei[1][i].item())
        neg_user_pos.setdefault(u, []).append(it)
    excl_u_list: list[int] = []
    excl_i_list: list[int] = []
    for u, items in neg_user_pos.items():
        excl_u_list.extend([u] * len(items))
        excl_i_list.extend(items)
    excl_users_cpu = torch.tensor(excl_u_list, dtype=torch.long)
    excl_items_cpu = torch.tensor(excl_i_list, dtype=torch.long)
    logger.info('Exclusion pairs: %d across %d users', len(excl_u_list), len(neg_user_pos))

    # Choose split
    val_pos  = _get_user_positive_items(val_ei)
    test_pos = _get_user_positive_items(test_ei)

    if args.split == 'val':
        split_ei  = val_ei
        pos_items = val_pos
        eval_users = val_ei[0].unique().tolist()
    else:
        split_ei  = test_ei
        pos_items = test_pos
        eval_users = test_ei[0].unique().tolist()

    # edge_user_ids[i] = global user ID for the i-th edge in the split (used in Bug 1)
    edge_user_ids = split_ei[0]  # (E_split,)

    logger.info(
        'Split: %s | %d eval users | %d split edges | K=%d | items=%d',
        args.split, len(eval_users), edge_user_ids.size(0), args.K, num_items,
    )

    user_emb_gpu      = user_emb.to(device)
    item_emb_gpu      = item_emb.to(device)
    user_tags_gpu     = user_tags.to(device)
    food_tags_gpu     = food_tags.to(device)
    edge_user_ids_gpu = edge_user_ids.to(device)
    excl_users_gpu    = excl_users_cpu.to(device)
    excl_items_gpu    = excl_items_cpu.to(device)

    # ------------------------------------------------------------------
    # Pre-compute per-user GNN scores and health hits.
    # Shape: (num_total_users, num_items) ≈ 221 MB each.
    # ------------------------------------------------------------------
    logger.info(
        'Pre-computing per-user scores (%d users × %d items)...',
        num_total_users, num_items,
    )
    gnn_scores_pu  = torch.empty(num_total_users, num_items, device=device)
    health_hits_pu = torch.zeros(num_total_users, num_items, device=device)

    batch = 256
    for b in range(0, num_total_users, batch):
        b_end = min(b + batch, num_total_users)
        uids = torch.arange(b, b_end, device=device)
        gnn_scores_pu[b:b_end] = user_emb_gpu[uids] @ item_emb_gpu.T
        u_tags = user_tags_gpu[uids].bool()
        i_tags = food_tags_gpu.bool()
        health_hits_pu[b:b_end] = (
            (u_tags.unsqueeze(1) & i_tags.unsqueeze(0)).any(dim=2).float()
        )

    E_split = edge_user_ids.size(0)
    logger.info(
        'Edge-indexed combined matrix per alpha: (%d × %d) ≈ %.0f MB',
        E_split, num_items, E_split * num_items * 4 / 1e6,
    )

    # ------------------------------------------------------------------
    # Alpha sweep
    # ------------------------------------------------------------------
    alphas = np.linspace(0.0, 1.0, args.alpha_steps).tolist()
    sweep_results = []

    logger.info('Sweeping %d alpha values...', len(alphas))
    for alpha in alphas:
        # Step 1: per-user combined scores (U, I)
        combined_pu = alpha * gnn_scores_pu + (1.0 - alpha) * health_hits_pu

        # Step 2: expand to edge-indexed (E_split, I) — Bug 1
        combined_edge = combined_pu[edge_user_ids_gpu]

        # Step 3: apply corrupted exclusion — Bug 3
        if excl_users_gpu.numel() > 0:
            combined_edge[excl_users_gpu, excl_items_gpu] = float('-inf')

        # Step 4: top-K from the corrupted edge-indexed matrix
        _, top_K_edge = torch.topk(combined_edge, k=args.K, largest=True, sorted=True)
        del combined_edge  # free ~1.7 GB

        # Step 5: evaluate using Bug 2 (global user ID as edge-position index)
        m = evaluate_mainy_style(
            top_K_edge=top_K_edge,
            eval_user_ids=eval_users,
            pos_items_per_user=pos_items,
            user_tags_gpu=user_tags_gpu,
            item_tags_gpu=food_tags_gpu,
            item_emb_gpu=item_emb_gpu,
            K=args.K,
            device=device,
        )
        del top_K_edge
        sweep_results.append({'alpha': round(alpha, 4), **m})

    gnn_baseline = next(r for r in sweep_results if abs(r['alpha'] - 1.0) < 1e-6)

    # ------------------------------------------------------------------
    # Print comparison table
    # ------------------------------------------------------------------
    header = f"{'alpha':>6}  {'ndcg':>8}  {'recall':>8}  {'health':>8}  {'diversity':>10}"
    separator = '-' * len(header)
    print()
    print('=== Simple linear re-rank baseline (main.py-compatible eval) ===')
    print(f'Split: {args.split} | K={args.K} | items={num_items}')
    print()
    print(f'  GNN baseline (alpha=1.0):  '
          f"ndcg={gnn_baseline['ndcg']:.5f}  "
          f"recall={gnn_baseline['recall']:.5f}  "
          f"health={gnn_baseline['health_score']:.5f}  "
          f"diversity={gnn_baseline['diversity']:.5f}")
    print()
    print(header)
    print(separator)
    for row in sweep_results:
        marker = ' <-- GNN' if abs(row['alpha'] - 1.0) < 1e-6 else ''
        print(
            f"{row['alpha']:>6.2f}  "
            f"{row['ndcg']:>8.5f}  "
            f"{row['recall']:>8.5f}  "
            f"{row['health_score']:>8.5f}  "
            f"{row['diversity']:>10.5f}"
            f"{marker}"
        )
    print(separator)
    print()

    gnn_ndcg  = gnn_baseline['ndcg']
    threshold = gnn_ndcg * 0.95
    candidates = [r for r in sweep_results if r['ndcg'] >= threshold and r['alpha'] < 1.0 - 1e-6]
    if candidates:
        best = max(candidates, key=lambda r: r['health_score'])
        print(
            f'  Best (NDCG >= {threshold:.5f}, max health): '
            f"alpha={best['alpha']:.2f}  ndcg={best['ndcg']:.5f}  "
            f"recall={best['recall']:.5f}  health={best['health_score']:.5f}  "
            f"diversity={best['diversity']:.5f}"
        )
        print()

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    output_data = {
        'split': args.split, 'K': args.K, 'num_items': num_items,
        'gnn_baseline': gnn_baseline,
        'alpha_sweep': sweep_results,
    }
    with open(args.output, 'w', encoding='utf-8') as fh:
        json.dump(output_data, fh, indent=2)
    logger.info('Results saved to %s', args.output)


if __name__ == '__main__':
    main()
