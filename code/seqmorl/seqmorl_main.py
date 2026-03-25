"""Sequential MORL entry point.

Run after ``python main.py`` has produced ``embeddings_checkpoint.pt``.

Canonical commands:
  # CPU (laptop / development)
  python seqmorl/seqmorl_main.py --device cpu --epochs 500

  # GPU cluster
  python seqmorl/seqmorl_main.py --device cuda --epochs 500

  # Auto-detect (recommended default)
  python seqmorl/seqmorl_main.py --device auto --epochs 500

All heavy computation happens in ``code/seqmorl/``.
SGSL training / MGDA mechanics remain entirely untouched.
"""

import argparse
import os
import sys
import torch

# Allow imports from code/ directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_HERE)
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

from seqmorl.environment import SequentialRecEnv
from seqmorl.policy import SequentialPolicy
from seqmorl.training import train_implicit_morl
from seqmorl.evaluation import evaluate_sequential, compare_baselines, compute_svs, compute_car
from seqmorl.constrained_diagnostics import (
    extract_constraint_stats_from_training_log,
    print_constraint_diagnostics,
    add_constraint_diagnostics_to_results,
)
from seqmorl.logging_utils import WandbTracker, save_json
from RCSYS_utils import split_data_new


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------

def resolve_device(choice: str) -> torch.device:
    """Resolve ``--device`` flag to a torch.device.

    'auto' : CUDA if available, else CPU.
    'cpu'  : force CPU.
    'cuda' : require CUDA; fail with a clear error if unavailable.
    """
    if choice == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if choice == 'cpu':
        return torch.device('cpu')
    if choice == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda requested but CUDA is not available on this machine."
            )
        return torch.device('cuda')
    raise ValueError(f"Unknown --device choice: {choice!r}. Use cpu | cuda | auto.")


# ---------------------------------------------------------------------------
# One-shot baseline metrics (from frozen embeddings only, no policy)
# ---------------------------------------------------------------------------

_EXCLUDED_RATING = -(1 << 10)  # Large negative value to mask seen interactions from ranking.


def _oneshot_metrics(user_emb: torch.Tensor,
                     item_emb: torch.Tensor,
                     user_tags: torch.Tensor,
                     food_tags: torch.Tensor,
                     pos_edge_index: torch.Tensor,
                     exclude_edge_indices: list,
                     user_ids: list[int],
                     K: int,
                     device: torch.device) -> dict:
    """Compute one-shot Pareto-ranking metrics using frozen embeddings."""
    import numpy as np
    from collections import defaultdict

    user_emb = user_emb.to(device)
    item_emb = item_emb.to(device)

    rating = torch.matmul(user_emb, item_emb.T)

    # Exclude training interactions.
    for excl in exclude_edge_indices:
        for u, i in excl.T.tolist():
            if 0 <= u < rating.size(0) and 0 <= i < rating.size(1):
                rating[int(u), int(i)] = _EXCLUDED_RATING

    _, top_K = torch.topk(rating, k=K)

    gt: dict[int, set[int]] = defaultdict(set)
    for u, i in pos_edge_index.T.tolist():
        gt[int(u)].add(int(i))

    ndcg_vals, rec_vals, prec_vals, health_vals, div_vals = [], [], [], [], []
    all_recommended: set[int] = set()

    for uid in user_ids:
        ground_truth = gt.get(uid, set())
        if not ground_truth:
            continue
        recs = top_K[uid].tolist()

        from seqmorl.evaluation import (
            _ndcg_at_k, _recall_precision_at_k, _health_score, _diversity_score
        )
        ndcg_vals.append(_ndcg_at_k(recs, ground_truth, K))
        r, p = _recall_precision_at_k(recs, ground_truth, K)
        rec_vals.append(r)
        prec_vals.append(p)
        health_vals.append(_health_score(recs, uid, user_tags.cpu(), food_tags.cpu()))
        div_vals.append(_diversity_score(recs, item_emb.cpu()))
        all_recommended.update(recs[:K])

    coverage = len(all_recommended) / item_emb.size(0)

    return {
        'ndcg': float(np.mean(ndcg_vals)) if ndcg_vals else 0.0,
        'recall': float(np.mean(rec_vals)) if rec_vals else 0.0,
        'precision': float(np.mean(prec_vals)) if prec_vals else 0.0,
        'health': float(np.mean(health_vals)) if health_vals else 0.0,
        'diversity': float(np.mean(div_vals)) if div_vals else 0.0,
        'coverage': coverage,
        'car': compute_car(
            float(np.mean(rec_vals)) if rec_vals else 0.0,
            float(np.mean(div_vals)) if div_vals else 0.0,
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Implicit sequential MORL for MOPI-HFRS.'
    )
    parser.add_argument('--device', type=str, default='auto',
                        choices=['cpu', 'cuda', 'auto'],
                        help='Execution device (default: auto).')
    parser.add_argument('--checkpoint', type=str,
                        default='embeddings_checkpoint.pt',
                        help='Path to frozen embeddings checkpoint from main.py.')
    parser.add_argument('--graph', type=str,
                        default='../processed_data/benchmark_macro.pt',
                        help='Path to graph data (for tags and split info).')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate.')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor.')
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                        help='GAE lambda for per-objective advantage estimation.')
    parser.add_argument('--K', type=int, default=20,
                        help='Recommendation list length.')
    parser.add_argument('--M', type=int, default=200,
                        help='Candidate pool size per user.')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Policy hidden layer width.')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping norm.')
    parser.add_argument('--log_interval', type=int, default=50,
                        help='Epochs between console log lines.')
    parser.add_argument('--val_eval_interval', type=int, default=5,
                        help='Epoch interval for validation ranking evaluation during training (0 disables).')
    parser.add_argument('--early_stop_on_val_ndcg', type=int, default=1, choices=[0, 1],
                        help='Enable early stopping when val_ndcg plateaus (1=enabled, 0=disabled).')
    parser.add_argument('--val_ndcg_patience', type=int, default=4,
                        help='Number of consecutive val evaluations without improvement before stopping (0 disables).')
    parser.add_argument('--val_ndcg_min_delta', type=float, default=1e-4,
                        help='Minimum val_ndcg improvement required to reset patience.')
    parser.add_argument('--val_ndcg_warmup_epochs', type=int, default=8,
                        help='Number of initial epochs before val_ndcg early-stop checks are allowed.')
    parser.add_argument('--output_dir', type=str, default='seqmorl_output',
                        help='Directory for checkpoints and metrics.')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Enable W&B offline logging.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed.')
    parser.add_argument('--train_user_limit', type=int, default=0,
                        help='Optional cap on number of train users per epoch (0 = all).')
    parser.add_argument('--val_user_limit', type=int, default=0,
                        help='Optional cap on number of val users used during training diagnostics (0 = all).')
    parser.add_argument('--train_batch_users', type=int, default=256,
                        help='Number of users per training chunk inside each epoch (0 = all users at once).')
    parser.add_argument('--users_per_epoch', type=int, default=0,
                        help='Optional number of randomly shuffled train users used per epoch (0 = all train users).')
    parser.add_argument('--exclude_seen_candidates', type=int, default=1, choices=[0, 1],
                        help='Exclude seen train interactions when building candidate pools (1=enabled, 0=disabled).')
    parser.add_argument('--test_ndcg_drop_floor', type=float, default=0.07,
                        help='Maximum allowed fractional test NDCG drop vs one-shot baseline before flagging failure.')
    parser.add_argument('--constrained_mode', type=int, default=0, choices=[0, 1],
                        help='Enable anchor-based constrained reranking (1=enabled, 0=disabled).')
    parser.add_argument('--anchor_epsilon', type=float, default=0.05,
                        help='Score margin gate: candidate must score >= anchor_score - epsilon.')
    parser.add_argument('--max_swaps_per_list', type=int, default=4,
                        help='Maximum number of swaps allowed per top-K list in constrained mode.')
    parser.add_argument('--anchor_lock_positions', type=int, default=6,
                        help='Number of top positions (1-indexed) to keep locked to anchor in constrained mode.')
    args = parser.parse_args()

    # --- Device setup ---
    device = resolve_device(args.device)
    print(f"[seqmorl] Using device: {device}")

    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load frozen embeddings ---
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    user_emb = ckpt['user_emb'].to(device)
    item_emb = ckpt['item_emb'].to(device)
    print(f"[seqmorl] Loaded embeddings: users={user_emb.shape}, items={item_emb.shape}")

    # --- Load graph for tags and edge splits ---
    graph = torch.load(args.graph, map_location='cpu')
    user_tags = graph['user'].tags
    food_tags = graph['food'].tags
    edge_index = graph[('user', 'eats', 'food')].edge_index
    edge_label_index = graph[('user', 'eats', 'food')].edge_label_index

    (train_edge_index, val_edge_index, test_edge_index,
     _pos_train_edge_index, _neg_train_edge_index,
     pos_val_edge_index, _neg_val_edge_index,
     pos_test_edge_index, _neg_test_edge_index) = split_data_new(
        edge_index, edge_label_index
    )

    train_users = train_edge_index[0].unique().tolist()
    val_users = val_edge_index[0].unique().tolist()
    test_users = test_edge_index[0].unique().tolist()

    if args.train_user_limit and args.train_user_limit > 0:
        train_users = train_users[:args.train_user_limit]
    if args.val_user_limit and args.val_user_limit > 0:
        val_users = val_users[:args.val_user_limit]

    print(
        f"[seqmorl] User split sizes: train={len(train_users)} "
        f"val={len(val_users)} test={len(test_users)}"
    )

    # --- Build environment ---
    env = SequentialRecEnv(
        user_emb=user_emb,
        item_emb=item_emb,
        food_tags=food_tags.to(device),
        user_tags=user_tags.to(device),
        M=args.M,
        K=args.K,
        device=str(device),
        exclude_edge_indices=[train_edge_index] if args.exclude_seen_candidates == 1 else None,
    )
    print(f"[seqmorl] State dim={env.state_dim}, Action dim={env.action_dim}")

    # --- Build policy ---
    policy = SequentialPolicy.build(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    # --- W&B tracker ---
    tracker = WandbTracker(
        enabled=args.use_wandb,
        project='seqmorl-implicit',
        mode='offline',
        base_dir=args.output_dir,
        config=vars(args),
    )

    # --- One-shot baseline (val) ---
    print("[seqmorl] Computing one-shot baseline on val split …")
    val_exclusions = [train_edge_index]
    baseline_val = _oneshot_metrics(
        user_emb.cpu(), item_emb.cpu(),
        user_tags, food_tags,
        pos_val_edge_index,
        val_exclusions,
        val_users, args.K, torch.device('cpu'),
    )
    print("[seqmorl] One-shot val metrics:", baseline_val)

    # --- Train sequential policy ---
    print("[seqmorl] Starting implicit MORL training …")
    train_user_ids = [int(u) for u in train_users]
    val_user_ids = [int(u) for u in val_users]

    policy = train_implicit_morl(
        policy=policy,
        env=env,
        train_user_ids=train_user_ids,
        val_user_ids=val_user_ids,
        args=args,
        tracker=tracker,
        output_dir=args.output_dir,
        val_eval_context={
            'pos_edge_index': pos_val_edge_index,
            'user_tags': user_tags,
            'food_tags': food_tags,
            'K': args.K,
            'exclude_edge_indices': val_exclusions,
        },
    )

    # --- Sequential evaluation (val) ---
    print("[seqmorl] Evaluating sequential policy on val split …")
    seq_val = evaluate_sequential(
        policy=policy,
        env=env,
        user_ids=val_user_ids,
        pos_edge_index=pos_val_edge_index,
        user_tags=user_tags,
        food_tags=food_tags,
        K=args.K,
        exclude_edge_indices=val_exclusions,
    )

    compare_baselines(baseline_val, seq_val, split='val')
    svs_val = compute_svs(baseline_val, seq_val)
    print(
        "[seqmorl] SVS (val): "
        f"{svs_val['svs']:+.6f} "
        f"[ndcg={svs_val['delta_ndcg']:+.6f}, "
        f"health={svs_val['delta_health']:+.6f}, "
        f"div={svs_val['delta_diversity']:+.6f}]"
    )

    # --- One-shot baseline (test) ---
    print("[seqmorl] Computing one-shot baseline on test split …")
    test_exclusions = [train_edge_index, val_edge_index]
    baseline_test = _oneshot_metrics(
        user_emb.cpu(), item_emb.cpu(),
        user_tags, food_tags,
        pos_test_edge_index,
        test_exclusions,
        [int(u) for u in test_users], args.K, torch.device('cpu'),
    )

    # --- Sequential evaluation (test) ---
    print("[seqmorl] Evaluating sequential policy on test split …")
    test_user_ids = [int(u) for u in test_users]
    seq_test = evaluate_sequential(
        policy=policy,
        env=env,
        user_ids=test_user_ids,
        pos_edge_index=pos_test_edge_index,
        user_tags=user_tags,
        food_tags=food_tags,
        K=args.K,
        exclude_edge_indices=test_exclusions,
    )

    compare_baselines(baseline_test, seq_test, split='test')
    svs_test = compute_svs(baseline_test, seq_test)
    print(
        "[seqmorl] SVS (test): "
        f"{svs_test['svs']:+.6f} "
        f"[ndcg={svs_test['delta_ndcg']:+.6f}, "
        f"health={svs_test['delta_health']:+.6f}, "
        f"div={svs_test['delta_diversity']:+.6f}]"
    )

    baseline_test_ndcg = float(baseline_test.get('ndcg', 0.0))
    seq_test_ndcg = float(seq_test.get('ndcg', 0.0))
    ndcg_drop_fraction = (
        max(0.0, baseline_test_ndcg - seq_test_ndcg) / max(1e-8, baseline_test_ndcg)
    )
    ranking_floor_pass = ndcg_drop_fraction <= float(args.test_ndcg_drop_floor)
    floor_status = "PASS" if ranking_floor_pass else "FAIL"
    print(
        "[seqmorl] Ranking floor check (test): "
        f"{floor_status} | ndcg_drop_fraction={ndcg_drop_fraction:.4f} "
        f"(limit={float(args.test_ndcg_drop_floor):.4f})"
    )
    
    # Extract and report constraint diagnostics if running in constrained mode
    constraint_stats = {}
    if args.constrained_mode:
        train_metrics_path = os.path.join(args.output_dir, 'train_metrics.jsonl')
        constraint_stats = extract_constraint_stats_from_training_log(train_metrics_path)
        if constraint_stats:
            print_constraint_diagnostics(constraint_stats)

    # --- Save results ---
    results = {
        'args': vars(args),
        'val': {
            'baseline': baseline_val,
            'sequential': seq_val,
            'svs': svs_val,
        },
        'test': {
            'baseline': baseline_test,
            'sequential': seq_test,
            'svs': svs_test,
            'ranking_floor': {
                'ndcg_drop_fraction': ndcg_drop_fraction,
                'max_allowed_drop_fraction': float(args.test_ndcg_drop_floor),
                'passed': bool(ranking_floor_pass),
            },
        },
    }
    
    # Add constraint diagnostics to results if available
    results = add_constraint_diagnostics_to_results(results, constraint_stats)
    
    results_path = os.path.join(args.output_dir, 'results.json')
    save_json(results_path, results)
    print(f"[seqmorl] Results saved to {results_path}")

    tracker.finish()


if __name__ == '__main__':
    main()
