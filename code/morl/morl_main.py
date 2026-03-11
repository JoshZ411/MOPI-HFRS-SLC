"""
MORL Entry Point — morl_main.py

Usage:
    cd code
    python -m morl.morl_main [options]

Workflow:
    1. Load frozen embeddings from embeddings_checkpoint.pt   (Phase 1)
    2. Load user/item health tags from the graph checkpoint.
    3. Build candidate pools (top-M per user).                (Phase 2)
    4. Construct MDP environment.                             (Phase 3 & 4)
    5. Instantiate conditional MORL policy.                   (Phase 5)
    6. Train policy with REINFORCE.                           (Phase 6)
    7. Select operating weight w* via validation grid search. (Phase 7)
    8. Evaluate MORL policy and baseline on test set.
"""

import argparse
import os
import sys
import torch
import numpy as np
import random

# Allow running as `python -m morl.morl_main` from code/ directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from morl.environment import RecommendationEnv, build_candidate_pools
from morl.policy import MORLPolicy, sample_weight_vector
from morl.training import train_morl
from morl.evaluation import evaluate_policy_on_split, select_operating_weight


# -----------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_user_positive_items(edge_index: torch.Tensor):
    """Map each user id to the set of positive item ids from an edge_index."""
    pos_items = {}
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        pos_items.setdefault(u, set()).add(v)
    return pos_items


def split_users(all_user_ids, train_frac=0.6, val_frac=0.2, seed=42):
    """Split users into train/val/test sets."""
    rng = np.random.default_rng(seed)
    ids = np.array(all_user_ids)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return (ids[:n_train].tolist(),
            ids[n_train:n_train + n_val].tolist(),
            ids[n_train + n_val:].tolist())


# -----------------------------------------------------------------------
# Baseline evaluation (one-shot Pareto/dot-product ranking)
# -----------------------------------------------------------------------

def evaluate_baseline(user_emb, item_emb, user_ids, user_tags, item_tags,
                       ground_truth, exclude_edges=None, K=20, device=None):
    """Evaluate the one-shot dot-product ranking baseline.

    Returns dict of mean metrics across *user_ids*.
    """
    from morl.evaluation import ndcg_at_k, health_score_at_k, diversity_score_at_k

    device = device or torch.device('cpu')
    user_emb = user_emb.to(device)
    item_emb = item_emb.to(device)
    user_tags = user_tags.to(device)
    item_tags = item_tags.to(device)

    scores = torch.matmul(user_emb, item_emb.T)  # (num_users, num_items)

    if exclude_edges is not None:
        users_ex = exclude_edges[0].to(device)
        items_ex = exclude_edges[1].to(device)
        scores[users_ex, items_ex] = float('-inf')

    K_actual = min(K, item_emb.shape[0])
    _, top_K = torch.topk(scores, k=K_actual, dim=1)  # (num_users, K)

    ndcg_vals, health_vals, div_vals, recall_vals = [], [], [], []
    for uid in user_ids:
        rec_list = top_K[uid].tolist()
        gt = ground_truth.get(uid, set())

        ndcg_vals.append(ndcg_at_k(rec_list, gt, K))
        health_vals.append(health_score_at_k(rec_list, user_tags[uid], item_tags))
        div_vals.append(diversity_score_at_k(rec_list, item_emb))
        hits = len(set(rec_list[:K]) & gt)
        recall_vals.append(hits / max(len(gt), 1))

    return {
        'ndcg': sum(ndcg_vals) / max(len(ndcg_vals), 1),
        'health': sum(health_vals) / max(len(health_vals), 1),
        'diversity': sum(div_vals) / max(len(div_vals), 1),
        'recall': sum(recall_vals) / max(len(recall_vals), 1),
    }


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ---- Phase 1: Load frozen embeddings ----
    print(f"\n[Phase 1] Loading frozen embeddings from {args.embeddings_path}")
    ckpt = torch.load(args.embeddings_path, map_location='cpu')
    user_emb = ckpt['user_emb']   # (num_users, d)
    item_emb = ckpt['item_emb']   # (num_items, d)
    print(f"  user_emb: {user_emb.shape},  item_emb: {item_emb.shape}")

    # ---- Load graph for tags and edge splits ----
    print(f"\n[Data] Loading graph from {args.graph_path}")
    graph = torch.load(args.graph_path, map_location='cpu')
    user_tags = graph['user'].tags.float()   # (num_users, tag_dim)
    item_tags = graph['food'].tags.float()   # (num_items, tag_dim)
    edge_index = graph[('user', 'eats', 'food')].edge_index  # (2, E)
    print(f"  user_tags: {user_tags.shape},  item_tags: {item_tags.shape}")

    num_users = user_emb.shape[0]

    # ---- Train/val/test user split ----
    all_user_ids = list(range(num_users))
    train_user_ids, val_user_ids, test_user_ids = split_users(
        all_user_ids, train_frac=0.6, val_frac=0.2, seed=args.seed)
    print(f"  Users — train: {len(train_user_ids)}, val: {len(val_user_ids)}, test: {len(test_user_ids)}")

    # ---- Ground-truth positive items (from full edge_index) ----
    ground_truth_all = get_user_positive_items(edge_index)

    # ---- Phase 2: Candidate pools ----
    print(f"\n[Phase 2] Building candidate pools (M={args.M})...")
    # During RL training: do NOT exclude training positives
    train_pools = build_candidate_pools(user_emb, item_emb, M=args.M, device=device)

    # For evaluation: exclude seen interactions
    train_edges_set = edge_index  # full edge_index used as exclusion set at test time
    eval_pools = build_candidate_pools(user_emb, item_emb, M=args.M,
                                       exclude_edges=train_edges_set, device=device)

    # ---- Phase 3 & 4: Environments ----
    print("\n[Phase 3/4] Constructing MDP environments...")
    train_env = RecommendationEnv(user_emb, item_emb, user_tags, item_tags,
                                  train_pools, K=args.K, device=device)
    eval_env = RecommendationEnv(user_emb, item_emb, user_tags, item_tags,
                                 eval_pools, K=args.K, device=device)

    state_dim = train_env.state_dim()
    item_dim = item_emb.shape[1]
    print(f"  state_dim: {state_dim},  item_dim: {item_dim}")

    # ---- Phase 5: Policy ----
    print("\n[Phase 5] Instantiating conditional MORL policy...")
    policy = MORLPolicy(
        state_dim=state_dim,
        item_dim=item_dim,
        hidden_dim=args.hidden_dim,
        weight_dim=3,
    ).to(device)
    num_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"  Policy parameters: {num_params:,}")

    # ---- Phase 6: Training ----
    print("\n[Phase 6] Training MORL policy...")
    stats = train_morl(
        env=train_env,
        policy=policy,
        train_user_ids=train_user_ids,
        num_episodes=args.num_episodes,
        batch_size=args.batch_size,
        lr=args.lr,
        dirichlet_alpha=args.dirichlet_alpha,
        grad_clip=args.grad_clip,
        log_interval=args.log_interval,
        checkpoint_path=args.policy_checkpoint,
        device=device,
    )

    # Load best checkpoint
    best_ckpt = torch.load(args.policy_checkpoint, map_location=device)
    policy.load_state_dict(best_ckpt['policy_state_dict'])
    policy.eval()
    print(f"  Loaded best policy from episode {best_ckpt['episode']} "
          f"(return={best_ckpt['best_return']:.4f})")

    # ---- Phase 7: Trade-off selection ----
    print("\n[Phase 7] Selecting operating weight w* via validation grid search...")
    val_ground_truth = {uid: ground_truth_all.get(uid, set()) for uid in val_user_ids}

    w_star, val_metrics = select_operating_weight(
        env=eval_env,
        policy=policy,
        val_user_ids=val_user_ids,
        ground_truth=val_ground_truth,
        K=args.K,
        n_grid=args.n_grid,
        alpha=args.alpha,
        beta=args.beta,
        diversity_threshold=args.diversity_threshold,
        device=device,
    )

    # ---- Test evaluation: MORL vs Baseline ----
    print("\n[Evaluation] Comparing MORL policy vs one-shot baseline on test set...")
    test_ground_truth = {uid: ground_truth_all.get(uid, set()) for uid in test_user_ids}

    morl_metrics = evaluate_policy_on_split(
        env=eval_env,
        policy=policy,
        user_ids=test_user_ids,
        weight=w_star,
        ground_truth=test_ground_truth,
        K=args.K,
    )

    baseline_metrics = evaluate_baseline(
        user_emb=user_emb,
        item_emb=item_emb,
        user_ids=test_user_ids,
        user_tags=user_tags,
        item_tags=item_tags,
        ground_truth=test_ground_truth,
        exclude_edges=train_edges_set,
        K=args.K,
        device=device,
    )

    print("\n===== TEST SET RESULTS =====")
    print(f"{'Metric':<15} {'Baseline':>12} {'MORL':>12}")
    print("-" * 41)
    for metric in ['ndcg', 'health', 'diversity', 'recall']:
        print(f"{metric:<15} {baseline_metrics[metric]:>12.4f} {morl_metrics[metric]:>12.4f}")
    print(f"\nMORL w*: {w_star.tolist()}")

    # Save final results summary
    results = {
        'w_star': w_star.cpu().tolist(),
        'val_metrics': val_metrics,
        'morl_test_metrics': morl_metrics,
        'baseline_test_metrics': baseline_metrics,
        'training_stats': {k: v[-1] if v else None for k, v in stats.items()},
    }
    torch.save(results, args.results_path)
    print(f"\nResults saved to {args.results_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MORL Sequential Recommendation')

    # Paths
    parser.add_argument('--embeddings_path', type=str, default='embeddings_checkpoint.pt',
                        help='Path to frozen embeddings checkpoint from main.py')
    parser.add_argument('--graph_path', type=str, default='../processed_data/benchmark_macro.pt',
                        help='Path to processed graph with tags')
    parser.add_argument('--policy_checkpoint', type=str, default='morl_policy.pt',
                        help='Where to save the best policy checkpoint')
    parser.add_argument('--results_path', type=str, default='morl_results.pt',
                        help='Where to save final evaluation results')

    # MDP
    parser.add_argument('--K', type=int, default=20,
                        help='Recommendation list length (episode horizon)')
    parser.add_argument('--M', type=int, default=200,
                        help='Candidate pool size per user')

    # Policy
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Policy encoder hidden dimension')

    # Training
    parser.add_argument('--num_episodes', type=int, default=2000,
                        help='Number of training gradient steps')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Users sampled per gradient step')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Adam learning rate')
    parser.add_argument('--dirichlet_alpha', type=float, default=1.0,
                        help='Dirichlet concentration for weight sampling')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Max gradient norm')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Print stats every N episodes')
    parser.add_argument('--seed', type=int, default=42)

    # Weight selection
    parser.add_argument('--n_grid', type=int, default=10,
                        help='Simplex grid resolution for weight search')
    parser.add_argument('--alpha', type=float, default=0.7,
                        help='NDCG weight in w* selection objective')
    parser.add_argument('--beta', type=float, default=0.3,
                        help='Health weight in w* selection objective')
    parser.add_argument('--diversity_threshold', type=float, default=0.3,
                        help='Minimum diversity score for w* selection constraint')

    args = parser.parse_args()
    main(args)
