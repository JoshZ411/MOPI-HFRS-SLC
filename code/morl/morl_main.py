"""
MORL entry point.

Usage (from the code/ directory, after running main.py):

    python -m morl.morl_main \\
        --checkpoint embeddings_checkpoint.pt \\
        --graph_path ../processed_data/benchmark_macro.pt \\
        --epochs 200 \\
        --batch_size 64 \\
        --K 20 \\
        --M 200 \\
        --output_dir morl_output

This script:
1. Loads frozen embeddings from embeddings_checkpoint.pt (Phase 1).
2. Builds candidate pools from the frozen embeddings (Phase 2).
3. Trains the conditional MORL policy (Phases 3–6).
4. Selects the best trade-off weight w* via validation metrics (Phase 7).
5. Evaluates on the test split and reports metrics vs. baseline.
"""

import argparse
import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split

from .environment import build_candidate_pools
from .policy import ConditionalPolicy, sample_weight_vector
from .training import train_morl, evaluate_morl


def get_user_positive_items(edge_index):
    """Build a dict mapping user_id → list of positive item indices."""
    pos_items: dict = {}
    for u, i in edge_index.T.tolist():
        pos_items.setdefault(u, []).append(i)
    return pos_items


def build_exclude_dict(edge_index):
    """Build a dict mapping user_id → set of item indices (for masking pools)."""
    excl: dict = {}
    for u, i in edge_index.T.tolist():
        excl.setdefault(u, set()).add(i)
    return excl


def simplex_grid(n_points: int = 15) -> list:
    """Return *n_points* weight vectors sampled from the probability simplex in ℝ³.

    Includes corner points (1,0,0), (0,1,0), (0,0,1) and uniform (1/3,1/3,1/3).
    """
    corners = [
        torch.tensor([1.0, 0.0, 0.0]),
        torch.tensor([0.0, 1.0, 0.0]),
        torch.tensor([0.0, 0.0, 1.0]),
        torch.tensor([1 / 3, 1 / 3, 1 / 3]),
    ]
    random_pts = [sample_weight_vector(batch_size=1) for _ in range(n_points - len(corners))]
    return corners + random_pts


def main():
    parser = argparse.ArgumentParser(description='MORL sequential recommendation')
    parser.add_argument('--checkpoint', type=str, default='embeddings_checkpoint.pt',
                        help='Path to frozen embeddings checkpoint produced by main.py.')
    parser.add_argument('--graph_path', type=str, default='../processed_data/benchmark_macro.pt',
                        help='Path to the processed heterogeneous graph (.pt file).')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--K', type=int, default=20, help='Recommendation list length.')
    parser.add_argument('--M', type=int, default=200, help='Candidate pool size.')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='morl_output')
    parser.add_argument('--val_weight_alpha', type=float, default=0.7,
                        help='Weight for NDCG in w* selection (1-alpha → health).')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Phase 1: Load frozen embeddings
    # ------------------------------------------------------------------
    print(f"\n[Phase 1] Loading embeddings from {args.checkpoint} ...")
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    user_emb = ckpt['user_emb']   # (num_users, d)
    item_emb = ckpt['item_emb']   # (num_items, d)
    print(f"  user_emb: {tuple(user_emb.shape)}")
    print(f"  item_emb: {tuple(item_emb.shape)}")

    # ------------------------------------------------------------------
    # Load graph for tags and edge splits
    # ------------------------------------------------------------------
    print(f"\nLoading graph from {args.graph_path} ...")
    graph = torch.load(args.graph_path, map_location='cpu')
    user_tags = graph['user'].tags  # (num_users, tag_dim)
    food_tags = graph['food'].tags  # (num_items, tag_dim)
    edge_index = graph[('user', 'eats', 'food')].edge_index  # (2, E)

    # Reproduce same 60/20/20 split as main.py
    edges = edge_index.numpy().T
    train_edges, test_edges = train_test_split(edges, test_size=0.2, random_state=args.seed)
    train_edges, val_edges = train_test_split(train_edges, test_size=0.25, random_state=args.seed)

    train_edge_index = torch.LongTensor(train_edges).T
    val_edge_index = torch.LongTensor(val_edges).T
    test_edge_index = torch.LongTensor(test_edges).T

    train_pos = get_user_positive_items(train_edge_index)
    val_pos = get_user_positive_items(val_edge_index)
    test_pos = get_user_positive_items(test_edge_index)

    # Users active in each split
    train_users = train_edge_index[0].unique().tolist()
    val_users = val_edge_index[0].unique().tolist()
    test_users = test_edge_index[0].unique().tolist()

    # Exclusion dicts (only used at eval time)
    exclude_val = build_exclude_dict(train_edge_index)
    exclude_test = build_exclude_dict(train_edge_index)
    for u, items in build_exclude_dict(val_edge_index).items():
        exclude_test.setdefault(u, set()).update(items)

    # ------------------------------------------------------------------
    # Phases 3–6: Train MORL policy
    # ------------------------------------------------------------------
    print("\n[Phases 3-6] Training MORL policy ...")
    policy = train_morl(
        user_emb=user_emb,
        item_emb=item_emb,
        user_tags=user_tags,
        item_tags=food_tags,
        train_user_ids=train_users,
        val_user_ids=val_users,
        exclude_per_user_train=None,   # no exclusion during RL training
        exclude_per_user_val=exclude_val,
        K=args.K,
        M=args.M,
        hidden_dim=args.hidden_dim,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_dir=args.output_dir,
        device=device,
    )

    # ------------------------------------------------------------------
    # Phase 7: Trade-off selection via validation metrics
    # ------------------------------------------------------------------
    print("\n[Phase 7] Selecting best trade-off weight w* via validation metrics ...")
    weight_grid = simplex_grid(n_points=15)
    best_score = -1.0
    best_w = weight_grid[3]  # uniform default

    print(f"{'w_pref':>8} {'w_health':>9} {'w_div':>7} | "
          f"{'NDCG':>8} {'Health':>8} {'Div':>8} | {'score':>8}")
    print('-' * 68)

    median_div = None
    all_val_results = []
    for w in weight_grid:
        metrics = evaluate_morl(
            policy=policy,
            user_emb=user_emb,
            item_emb=item_emb,
            user_tags=user_tags,
            item_tags=food_tags,
            eval_user_ids=val_users,
            pos_items_per_user=val_pos,
            weight=w,
            exclude_per_user=exclude_val,
            K=args.K,
            M=args.M,
            device=device,
        )
        all_val_results.append((w, metrics))

    # Compute median diversity for diversity threshold
    div_vals = [m['diversity'] for _, m in all_val_results]
    median_div = float(np.median(div_vals))

    for w, metrics in all_val_results:
        wp, wh, wd = w[0].item(), w[1].item(), w[2].item()
        # Option A: maximise α·NDCG + (1-α)·Health  s.t. Diversity ≥ median_div
        if metrics['diversity'] >= median_div:
            score = args.val_weight_alpha * metrics['ndcg'] + \
                    (1 - args.val_weight_alpha) * metrics['health_score']
        else:
            score = -1.0
        print(f"{wp:8.3f} {wh:9.3f} {wd:7.3f} | "
              f"{metrics['ndcg']:8.4f} {metrics['health_score']:8.4f} "
              f"{metrics['diversity']:8.4f} | {score:8.4f}")
        if score > best_score:
            best_score = score
            best_w = w

    print(f"\nSelected w* = [{best_w[0]:.3f}, {best_w[1]:.3f}, {best_w[2]:.3f}]"
          f"  (val score={best_score:.4f})")

    # ------------------------------------------------------------------
    # Final evaluation on test split
    # ------------------------------------------------------------------
    print("\n[Final] Evaluating MORL on test split ...")
    test_metrics = evaluate_morl(
        policy=policy,
        user_emb=user_emb,
        item_emb=item_emb,
        user_tags=user_tags,
        item_tags=food_tags,
        eval_user_ids=test_users,
        pos_items_per_user=test_pos,
        weight=best_w,
        exclude_per_user=exclude_test,
        K=args.K,
        M=args.M,
        device=device,
    )

    print("\n=== Test Results (MORL, sequential, w*) ===")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.5f}")

    # Save results
    results_path = os.path.join(args.output_dir, 'test_results.pt')
    torch.save({
        'best_w': best_w,
        'test_metrics': test_metrics,
        'val_grid_results': [(w.tolist(), m) for w, m in all_val_results],
    }, results_path)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
