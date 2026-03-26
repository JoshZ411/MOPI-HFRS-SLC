"""
MORL entry point.

Usage (from the code/ directory, after running main.py):

    python -m morl.morl_main \\
        --checkpoint embeddings_checkpoint.pt \\
        --graph_path ../processed_data/benchmark_macro.pt \\
        --device cuda \
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
import csv
import os
import sys
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from .logging_utils import WandbTracker, append_jsonl, build_run_config, save_json, setup_logger
from .policy import sample_weight_vector
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


def resolve_device(device_arg: str) -> torch.device:
    """Resolve the requested device string into a torch.device."""
    if device_arg == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device_arg.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError(
            'CUDA was requested, but torch.cuda.is_available() is False. '
            'Use a CUDA-enabled PyTorch build and launch the script from that environment, '
            'or rerun with --device cpu.'
        )

    return torch.device(device_arg)


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
    parser.add_argument('--entropy_coef', type=float, default=0.01,
                        help='Coefficient for normalized entropy regularization in the REINFORCE loss.')
    parser.add_argument('--disable_failfast', action='store_true',
                        help='Disable probe-based fail-fast guardrail in training.')
    parser.add_argument('--failfast_patience', type=int, default=3,
                        help='Number of consecutive stagnant probe checks before early stopping.')
    parser.add_argument('--failfast_span_threshold', type=float, default=0.5,
                        help='Fail-fast span threshold: first_action_span must exceed this to count as progress.')
    parser.add_argument('--failfast_jaccard_threshold', type=float, default=0.95,
                        help='Fail-fast jaccard threshold: pairwise_jaccard must drop below this to count as progress.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='morl_output')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to run on: auto, cpu, cuda, or cuda:N.')
    parser.add_argument('--val_weight_alpha', type=float, default=0.7,
                        help='Weight for NDCG in w* selection (1-alpha → health).')
    parser.add_argument(
        '--val_selection_mode',
        type=str,
        default='unconstrained',
        choices=['constrained', 'unconstrained', 'soft-constrained'],
        help='Validation weight selection mode: constrained uses diversity threshold; unconstrained maximizes alpha*ndcg + (1-alpha)*health directly; soft-constrained applies a diversity shortfall penalty.',
    )
    parser.add_argument(
        '--val_diversity_penalty_lambda',
        type=float,
        default=0.5,
        help='Penalty weight for soft-constrained mode: score -= lambda * max(0, target_diversity - diversity).',
    )
    parser.add_argument('--log_every', type=int, default=10,
                        help='Epoch interval for training log summaries.')
    parser.add_argument('--probe_every', type=int, default=10,
                        help='Epoch interval for fixed-user weight sensitivity probes.')
    parser.add_argument('--num_probe_users', type=int, default=4,
                        help='Number of train users to use for weight-sensitivity probes.')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Enable Weights & Biases logging for MORL runs.')
    parser.add_argument('--wandb_project', type=str, default='mopi-morl',
                        help='W&B project name used when --use_wandb is set.')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='Optional W&B entity/team.')
    parser.add_argument('--wandb_mode', type=str, default='offline', choices=['online', 'offline', 'disabled'],
                        help='W&B mode. Use offline for terminal-only workflows and wandb beta leet.')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Optional run name for logging outputs and W&B.')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(args.output_dir)
    metrics_path = os.path.join(args.output_dir, 'train_metrics.jsonl')
    eval_path = os.path.join(args.output_dir, 'eval_metrics.jsonl')
    config_path = os.path.join(args.output_dir, 'run_config.json')
    wandb_dir = os.path.join(args.output_dir, 'wandb')
    leet_hint_path = os.path.join(args.output_dir, 'wandb_leet_command.txt')

    tracker = WandbTracker(
        enabled=args.use_wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.run_name,
        mode=args.wandb_mode,
        base_dir=wandb_dir,
        config=build_run_config(args),
        logger=logger,
    )
    save_json(config_path, build_run_config(args, extra={'device': str(device)}))
    logger.info('Device: %s', device)
    logger.info('Run configuration saved to %s', config_path)
    if tracker.enabled:
        leet_command = tracker.leet_command(sys.executable)
        if leet_command is not None:
            with open(leet_hint_path, 'w', encoding='utf-8') as handle:
                handle.write(leet_command)
                handle.write('\n')
            logger.info('To view live local W&B plots during training, run: %s', leet_command)
            logger.info('LEET command saved to %s', leet_hint_path)
            logger.info('LEET charts begin populating after the first epoch is logged.')
    else:
        logger.info('W&B tracking is disabled. Pass --use_wandb to enable local live plots and metric history.')

    # ------------------------------------------------------------------
    # Phase 1: Load frozen embeddings
    # ------------------------------------------------------------------
    logger.info('[Phase 1] Loading embeddings from %s', args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    user_emb = ckpt['user_emb']   # (num_users, d)
    item_emb = ckpt['item_emb']   # (num_items, d)
    logger.info('Embedding shapes: user_emb=%s item_emb=%s', tuple(user_emb.shape), tuple(item_emb.shape))

    # ------------------------------------------------------------------
    # Load graph for tags and edge splits
    # ------------------------------------------------------------------
    logger.info('Loading graph from %s', args.graph_path)
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
    probe_users = train_users[:args.num_probe_users]

    # Exclusion dicts (only used at eval time)
    exclude_val = build_exclude_dict(train_edge_index)
    exclude_test = build_exclude_dict(train_edge_index)
    for u, items in build_exclude_dict(val_edge_index).items():
        exclude_test.setdefault(u, set()).update(items)

    logger.info(
        'Graph stats: users=%d items=%d tags=%d edges=%d',
        user_emb.size(0),
        item_emb.size(0),
        user_tags.size(1),
        edge_index.size(1),
    )
    logger.info(
        'Split stats: train_edges=%d val_edges=%d test_edges=%d | train_users=%d val_users=%d test_users=%d',
        train_edge_index.size(1),
        val_edge_index.size(1),
        test_edge_index.size(1),
        len(train_users),
        len(val_users),
        len(test_users),
    )
    logger.info('Probe users for fixed-weight diagnostics: %s', probe_users)

    # ------------------------------------------------------------------
    # Phases 3–6: Train MORL policy
    # ------------------------------------------------------------------
    logger.info('[Phases 3-6] Training MORL policy')
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
        entropy_coef=args.entropy_coef,
        checkpoint_dir=args.output_dir,
        log_every=args.log_every,
        metrics_path=metrics_path,
        logger=logger,
        tracker=tracker,
        probe_user_ids=probe_users,
        probe_every=args.probe_every,
        failfast_enabled=not args.disable_failfast,
        failfast_patience=args.failfast_patience,
        failfast_span_threshold=args.failfast_span_threshold,
        failfast_jaccard_threshold=args.failfast_jaccard_threshold,
        device=device,
    )

    # ------------------------------------------------------------------
    # Phase 7: Trade-off selection via validation metrics.
    # ------------------------------------------------------------------
    logger.info('[Phase 7] Selecting best trade-off weight w* via validation metrics')
    weight_grid = simplex_grid(n_points=15)
    best_score = -1.0
    best_w = weight_grid[3]  # uniform default
    best_unconstrained_score = -1.0
    best_unconstrained_w = weight_grid[3]
    best_constrained_score = -1.0
    best_constrained_w = weight_grid[3]

    logger.info('%8s %9s %7s | %8s %8s %8s | %8s', 'w_pref', 'w_health', 'w_div', 'NDCG', 'Health', 'Div', 'score')
    logger.info('%s', '-' * 68)

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
    val_rows = []

    for w, metrics in all_val_results:
        wp, wh, wd = w[0].item(), w[1].item(), w[2].item()
        unconstrained_score = args.val_weight_alpha * metrics['ndcg'] + \
                     (1 - args.val_weight_alpha) * metrics['health_score']
        diversity_shortfall = max(0.0, median_div - metrics['diversity'])
        soft_constrained_score = unconstrained_score - args.val_diversity_penalty_lambda * diversity_shortfall

        # Option A: maximise α·NDCG + (1-α)·Health  s.t. Diversity ≥ median_div
        if metrics['diversity'] >= median_div:
            constrained_score = unconstrained_score
        else:
            constrained_score = -1.0

        if args.val_selection_mode == 'constrained':
            score = constrained_score
        elif args.val_selection_mode == 'unconstrained':
            score = unconstrained_score
        else:
            score = soft_constrained_score
        row = {
            'w_pref': wp,
            'w_health': wh,
            'w_div': wd,
            'ndcg': metrics['ndcg'],
            'health_score': metrics['health_score'],
            'diversity': metrics['diversity'],
            'recall': metrics['recall'],
            'unconstrained_score': unconstrained_score,
            'constrained_score': constrained_score,
            'soft_constrained_score': soft_constrained_score,
            'diversity_shortfall': diversity_shortfall,
            'score': score,
            'passes_diversity_constraint': metrics['diversity'] >= median_div,
        }
        val_rows.append(row)
        append_jsonl(eval_path, {'type': 'val_weight', **row})
        logger.info(
            '%8.3f %9.3f %7.3f | %8.4f %8.4f %8.4f | %8.4f',
            wp,
            wh,
            wd,
            metrics['ndcg'],
            metrics['health_score'],
            metrics['diversity'],
            score,
        )

        if unconstrained_score > best_unconstrained_score:
            best_unconstrained_score = unconstrained_score
            best_unconstrained_w = w

        if constrained_score > best_constrained_score:
            best_constrained_score = constrained_score
            best_constrained_w = w

        if score > best_score:
            best_score = score
            best_w = w

    ndcg_span = max(row['ndcg'] for row in val_rows) - min(row['ndcg'] for row in val_rows)
    health_span = max(row['health_score'] for row in val_rows) - min(row['health_score'] for row in val_rows)
    div_span = max(row['diversity'] for row in val_rows) - min(row['diversity'] for row in val_rows)
    if max(ndcg_span, health_span, div_span) < 1e-4:
        logger.warning('Validation metrics are nearly identical across weight vectors; conditional policy may not be responding to w.')

    val_csv_path = os.path.join(args.output_dir, 'validation_weight_grid.csv')
    with open(val_csv_path, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(val_rows[0].keys()))
        writer.writeheader()
        writer.writerows(val_rows)
    logger.info('Validation grid saved to %s', val_csv_path)

    if tracker is not None:
        tracker.log(
            {
                'val/ndcg_span': ndcg_span,
                'val/health_span': health_span,
                'val/diversity_span': div_span,
                'val/median_diversity': median_div,
            }
        )
        tracker.log_table(
            name='val/weight_grid',
            rows=[[row[key] for key in val_rows[0].keys()] for row in val_rows],
            columns=list(val_rows[0].keys()),
        )

    logger.info(
        'Selection mode=%s | constrained best=[%.3f, %.3f, %.3f] score=%.4f | unconstrained best=[%.3f, %.3f, %.3f] score=%.4f',
        args.val_selection_mode,
        best_constrained_w[0].item(),
        best_constrained_w[1].item(),
        best_constrained_w[2].item(),
        best_constrained_score,
        best_unconstrained_w[0].item(),
        best_unconstrained_w[1].item(),
        best_unconstrained_w[2].item(),
        best_unconstrained_score,
    )
    logger.info(
        'Selected w* = [%.3f, %.3f, %.3f] (active mode score=%.4f)',
        best_w[0].item(),
        best_w[1].item(),
        best_w[2].item(),
        best_score,
    )

    # ------------------------------------------------------------------
    # Final evaluation on test split
    # ------------------------------------------------------------------
    logger.info('[Final] Evaluating MORL on test split')
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

    logger.info('=== Test Results (MORL, sequential, w*) ===')
    for k, v in test_metrics.items():
        logger.info('  %s: %.5f', k, v)
        append_jsonl(eval_path, {'type': 'test_metric', 'metric': k, 'value': float(v)})

    if tracker is not None:
        tracker.log({f'test/{key}': value for key, value in test_metrics.items()})
        tracker.log({'test/best_w_pref': best_w[0].item(), 'test/best_w_health': best_w[1].item(), 'test/best_w_div': best_w[2].item()})

    # Save results
    results_path = os.path.join(args.output_dir, 'test_results.pt')
    torch.save({
        'best_w': best_w,
        'test_metrics': test_metrics,
        'val_grid_results': [(w.tolist(), m) for w, m in all_val_results],
    }, results_path)
    logger.info('Results saved to %s', results_path)
    tracker.finish()


if __name__ == '__main__':
    main()
