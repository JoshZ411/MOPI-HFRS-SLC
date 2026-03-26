"""
main.py — CLI entrypoint for constrained reranking.

Loads a trained SGSL checkpoint, generates baseline anchor lists,
runs constrained reranking, evaluates both, checks acceptance gates,
and saves all artifacts.
"""

import sys
import os
import argparse
import torch

# Add parent code/ directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch_geometric

from .anchor import load_model_and_data, get_embeddings_for_split, get_anchor_list_and_scores
from .constraints import FeasibilityChecker, aggregate_diagnostics
from .reranker import constrained_rerank
from .evaluation import evaluate_topk_list, compute_acceptance_gates, generate_comparison_table
from .logging_utils import (
    save_run_config,
    save_results_json,
    init_wandb_offline,
    log_wandb_metrics,
    finish_wandb,
    save_wandb_leet_command,
    append_auto_logs,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Constrained Reranker for MOPI-HFRS-SLC'
    )
    # Required
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results and artifacts')
    # Paths
    parser.add_argument('--checkpoint_path', type=str,
                        default='../checkpoints/sgsl_checkpoint.pt',
                        help='Path to SGSL checkpoint')
    parser.add_argument('--graph_path', type=str,
                        default='../processed_data/benchmark_macro.pt',
                        help='Path to benchmark graph data')
    # Device
    parser.add_argument('--device', type=str, default='auto',
                        choices=['cpu', 'cuda', 'auto'],
                        help='Device: cpu, cuda, or auto')
    # Recommendation params
    parser.add_argument('--K', type=int, default=20,
                        help='Top-K for recommendation list')
    parser.add_argument('--M', type=int, default=200,
                        help='Top-M candidate pool size')
    # Constraint params
    parser.add_argument('--anchor_lock_positions', type=int, default=6,
                        help='Number of top positions locked (immutable)')
    parser.add_argument('--anchor_epsilon', type=float, default=0.05,
                        help='Score margin tolerance for swaps')
    parser.add_argument('--max_swaps_per_list', type=int, default=4,
                        help='Maximum number of swaps per user list')
    # Gates
    parser.add_argument('--ndcg_floor', type=float, default=0.07,
                        help='Maximum allowed NDCG drop fraction')
    # Optional
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--split', type=str, default='test',
                        choices=['val', 'test'],
                        help='Which split to evaluate on')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Enable W&B offline logging')

    return parser.parse_args()


def resolve_device(device_str):
    if device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif device_str == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def run(args):
    print(f"Constrained Reranker — split={args.split}, K={args.K}, M={args.M}")
    print(f"Constraints: lock={args.anchor_lock_positions}, "
          f"epsilon={args.anchor_epsilon}, max_swaps={args.max_swaps_per_list}")
    print(f"NDCG floor: {args.ndcg_floor}")
    print()

    # Resolve device
    device = resolve_device(args.device)
    print(f"Device: {device}")

    # Set seed
    torch_geometric.seed_everything(args.seed)

    # Save run config
    config = vars(args)
    config['device_resolved'] = str(device)
    save_run_config(args.output_dir, config)

    # Init W&B if requested
    wandb_run = None
    if args.use_wandb:
        run_name = (f"rerank_lock{args.anchor_lock_positions}_"
                    f"eps{args.anchor_epsilon}_swaps{args.max_swaps_per_list}")
        wandb_run = init_wandb_offline(run_name, config)

    # ──────────────────────────────────────────────
    # Step 1: Load model and data
    # ──────────────────────────────────────────────
    print("\n[1/6] Loading model and checkpoint...")
    model, ckpt = load_model_and_data(args.checkpoint_path, args.graph_path, device)

    # Select split edges
    if args.split == 'val':
        edge_index = ckpt['val_edge_index']
        pos_edge_index = ckpt['pos_val_edge_index']
        neg_edge_index = ckpt['neg_val_edge_index']
    else:
        edge_index = ckpt['test_edge_index']
        pos_edge_index = ckpt['pos_test_edge_index']
        neg_edge_index = ckpt['neg_test_edge_index']

    feature_dict = ckpt['feature_dict']
    user_tags = ckpt['user_tags'].to(device)
    food_tags = ckpt['food_tags'].to(device)
    neg_train_edge_index = ckpt['neg_train_edge_index']
    num_foods = ckpt['num_foods']

    # ──────────────────────────────────────────────
    # Step 2: Generate embeddings (matching baseline eval)
    # ──────────────────────────────────────────────
    print("[2/6] Generating embeddings (forward pass on eval edges)...")
    users_emb, items_emb = get_embeddings_for_split(
        model, feature_dict, edge_index, pos_edge_index, neg_edge_index, device
    )

    # ──────────────────────────────────────────────
    # Step 3: Generate anchor lists
    # ──────────────────────────────────────────────
    print("[3/6] Generating anchor lists and candidate pool...")
    anchor_topk, anchor_scores, candidate_pool, candidate_scores, _ = get_anchor_list_and_scores(
        users_emb, items_emb, args.K, args.M, [neg_train_edge_index.to(device)]
    )

    # ──────────────────────────────────────────────
    # Step 4: Compute baseline metrics
    # ──────────────────────────────────────────────
    print("[4/6] Computing baseline metrics...")
    baseline_metrics = evaluate_topk_list(
        anchor_topk, edge_index.to(device), user_tags, food_tags, args.K, num_foods
    )
    print(f"  Baseline: ndcg={baseline_metrics['ndcg']:.5f}, "
          f"recall={baseline_metrics['recall']:.5f}, "
          f"health={baseline_metrics['health_score']:.5f}")

    # ──────────────────────────────────────────────
    # Step 5: Run constrained reranking
    # ──────────────────────────────────────────────
    print("[5/6] Running constrained reranking...")
    checker = FeasibilityChecker(
        lock_positions=args.anchor_lock_positions,
        epsilon=args.anchor_epsilon,
        max_swaps=args.max_swaps_per_list,
    )
    reranked_topk, all_diagnostics = constrained_rerank(
        anchor_topk, anchor_scores, candidate_pool, candidate_scores,
        checker, user_tags, food_tags,
    )
    diagnostics_agg = aggregate_diagnostics(all_diagnostics)

    # ──────────────────────────────────────────────
    # Step 6: Evaluate reranked list and check gates
    # ──────────────────────────────────────────────
    print("[6/6] Evaluating reranked list and checking acceptance gates...")
    reranked_metrics = evaluate_topk_list(
        reranked_topk, edge_index.to(device), user_tags, food_tags, args.K, num_foods
    )
    print(f"  Reranked: ndcg={reranked_metrics['ndcg']:.5f}, "
          f"recall={reranked_metrics['recall']:.5f}, "
          f"health={reranked_metrics['health_score']:.5f}")

    gates = compute_acceptance_gates(baseline_metrics, reranked_metrics, args.ndcg_floor)

    # Print comparison table
    table = generate_comparison_table(baseline_metrics, reranked_metrics, diagnostics_agg)
    print()
    print(table)

    # Gate results
    print()
    if gates['hard_gate_pass']:
        print(f"HARD GATE: PASS (ndcg_drop_fraction={gates['ndcg_drop_fraction']:.4f} <= {args.ndcg_floor})")
    else:
        print(f"HARD GATE: FAIL (ndcg_drop_fraction={gates['ndcg_drop_fraction']:.4f} > {args.ndcg_floor})")

    if gates['secondary_gate_pass']:
        print("SECONDARY GATE: PASS (all secondary metrics non-negative)")
    else:
        print("SECONDARY GATE: FAIL (some secondary metrics degraded)")

    # Save artifacts
    save_results_json(args.output_dir, baseline_metrics, reranked_metrics, diagnostics_agg, gates)

    # Save comparison table
    table_path = os.path.join(args.output_dir, 'comparison_table.txt')
    with open(table_path, 'w') as f:
        f.write(table)

    # W&B logging
    if wandb_run is not None:
        log_wandb_metrics(wandb_run, {
            **{f'baseline_{k}': v for k, v in baseline_metrics.items()},
            **{f'reranked_{k}': v for k, v in reranked_metrics.items()},
            'ndcg_drop_fraction': gates['ndcg_drop_fraction'],
            'hard_gate_pass': int(gates['hard_gate_pass']),
            **{f'diag_{k}': v for k, v in diagnostics_agg.items()},
        })
        save_wandb_leet_command(args.output_dir)
        finish_wandb(wandb_run)

    # Append auto_logs
    append_auto_logs(
        phase=f"Constrained Rerank ({args.split})",
        files_changed=[],
        commands=[f"python -m constrained_rerank.main --output_dir {args.output_dir} "
                  f"--K {args.K} --M {args.M} "
                  f"--anchor_lock_positions {args.anchor_lock_positions} "
                  f"--anchor_epsilon {args.anchor_epsilon} "
                  f"--max_swaps_per_list {args.max_swaps_per_list}"],
        metrics={**baseline_metrics, **{f'reranked_{k}': v for k, v in reranked_metrics.items()}},
        gate_status='PASS' if gates['hard_gate_pass'] else 'FAIL',
        blockers=[],
        next_phase='Experiment matrix' if gates['hard_gate_pass'] else 'Tighten constraints',
    )

    print(f"\nAll artifacts saved to {args.output_dir}")
    return gates['hard_gate_pass']


if __name__ == '__main__':
    args = parse_args()
    success = run(args)
    sys.exit(0 if success else 1)
