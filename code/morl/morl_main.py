"""MORL entry point."""

import argparse
import os
import sys
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from .logging_utils import WandbTracker, append_jsonl, build_run_config, save_json, setup_logger
from .training import train_morl, evaluate_morl, measure_candidate_pool_ceiling


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
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='Discount factor for per-objective reward-to-go returns.')
    parser.add_argument('--entropy_coef', type=float, default=0.01,
                        help='Coefficient for normalized entropy regularization during training.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='morl_output')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to run on: auto, cpu, cuda, or cuda:N.')
    parser.add_argument('--log_every', type=int, default=10,
                        help='Epoch interval for training log summaries.')
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

    val_pool_ceiling = measure_candidate_pool_ceiling(
        user_emb=user_emb,
        item_emb=item_emb,
        eval_user_ids=val_users,
        pos_items_per_user=val_pos,
        exclude_per_user=exclude_val,
        K=args.K,
        M=args.M,
        device=device,
    )
    test_pool_ceiling = measure_candidate_pool_ceiling(
        user_emb=user_emb,
        item_emb=item_emb,
        eval_user_ids=test_users,
        pos_items_per_user=test_pos,
        exclude_per_user=exclude_test,
        K=args.K,
        M=args.M,
        device=device,
    )

    logger.info('=== Candidate Pool Ceiling (validation) ===')
    for key, value in val_pool_ceiling.items():
        logger.info('  %s: %.5f', key, value)
        append_jsonl(eval_path, {'type': 'val_pool_ceiling', 'metric': key, 'value': float(value)})

    logger.info('=== Candidate Pool Ceiling (test) ===')
    for key, value in test_pool_ceiling.items():
        logger.info('  %s: %.5f', key, value)
        append_jsonl(eval_path, {'type': 'test_pool_ceiling', 'metric': key, 'value': float(value)})

    if tracker is not None:
        tracker.log({f'val_pool/{key}': value for key, value in val_pool_ceiling.items()})
        tracker.log({f'test_pool/{key}': value for key, value in test_pool_ceiling.items()})

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
        exclude_per_user_train=exclude_val,
        exclude_per_user_val=exclude_val,
        K=args.K,
        M=args.M,
        hidden_dim=args.hidden_dim,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        gamma=args.gamma,
        entropy_coef=args.entropy_coef,
        checkpoint_dir=args.output_dir,
        log_every=args.log_every,
        metrics_path=metrics_path,
        logger=logger,
        tracker=tracker,
        device=device,
    )

    # ------------------------------------------------------------------
    # Phase 7: Validation metrics for the trained policy.
    # ------------------------------------------------------------------
    logger.info('[Phase 7] Evaluating MORL on validation split')
    val_metrics = evaluate_morl(
        policy=policy,
        user_emb=user_emb,
        item_emb=item_emb,
        user_tags=user_tags,
        item_tags=food_tags,
        eval_user_ids=val_users,
        pos_items_per_user=val_pos,
        exclude_per_user=exclude_val,
        K=args.K,
        M=args.M,
        device=device,
    )

    logger.info('=== Validation Results (MORL, sequential) ===')
    for key, value in val_metrics.items():
        logger.info('  %s: %.5f', key, value)
        append_jsonl(eval_path, {'type': 'val_metric', 'metric': key, 'value': float(value)})

    if tracker is not None:
        tracker.log({f'val/{key}': value for key, value in val_metrics.items()})

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
        exclude_per_user=exclude_test,
        K=args.K,
        M=args.M,
        device=device,
    )

    logger.info('=== Test Results (MORL, sequential) ===')
    for k, v in test_metrics.items():
        logger.info('  %s: %.5f', k, v)
        append_jsonl(eval_path, {'type': 'test_metric', 'metric': k, 'value': float(v)})

    if tracker is not None:
        tracker.log({f'test/{key}': value for key, value in test_metrics.items()})

    # Save results
    results_path = os.path.join(args.output_dir, 'test_results.pt')
    torch.save({
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
    }, results_path)
    logger.info('Results saved to %s', results_path)
    tracker.finish()


if __name__ == '__main__':
    main()
