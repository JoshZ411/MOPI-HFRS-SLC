"""Generate per-user top-K recommendation lists from a trained MORL policy.

Loads a saved policy checkpoint (``morl_policy_final.pt`` or any epoch
checkpoint) and outputs a JSON file mapping each eval user to their
recommended item IDs and associated food names.

Usage
-----
    cd code/
    python -m morl.morl_infer \\
        --policy  ../morl_output_a2c_beta20_5k/morl_policy_final.pt \\
        --checkpoint embeddings_checkpoint.pt \\
        --graph_path ../processed_data/benchmark_macro.pt \\
        --device cuda --K 20 --split test \\
        --output  ../morl_output_a2c_beta20_5k/recommendations.json

Output schema
-------------
{
  "meta": {
    "split": "test",
    "K": 20,
    "num_users": 1632,
    "num_items": 6769
  },
  "recommendations": {
    "<user_id>": {
      "items": [item_id, ...],       // top-K item indices in recommendation order
      "ground_truth": [item_id, ...] // held-out positive items for this user
    },
    ...
  }
}
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
    from .policy import ConditionalPolicy
    from .environment import RecommendationEnv, build_candidate_pools
    from .training import get_recommendations
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from morl.logging_utils import setup_logger
    from morl.policy import ConditionalPolicy
    from morl.environment import RecommendationEnv, build_candidate_pools
    from morl.training import get_recommendations


def _get_user_positive_items(edge_index: torch.Tensor) -> dict[int, list[int]]:
    pos: dict[int, list[int]] = {}
    for u, i in edge_index.T.tolist():
        pos.setdefault(int(u), []).append(int(i))
    return pos


def main() -> None:
    parser = argparse.ArgumentParser(description='MORL inference — generate top-K recommendation lists')
    parser.add_argument('--policy',      type=str, required=True,
                        help='Path to saved policy checkpoint (morl_policy_final.pt or epoch checkpoint).')
    parser.add_argument('--checkpoint',  type=str, default='embeddings_checkpoint.pt',
                        help='Path to frozen GNN embeddings checkpoint.')
    parser.add_argument('--graph_path',  type=str, default='../processed_data/benchmark_macro.pt',
                        help='Path to the processed heterogeneous graph (.pt file).')
    parser.add_argument('--K',           type=int, default=20,
                        help='Recommendation list length.')
    parser.add_argument('--hidden_dim',  type=int, default=256,
                        help='Policy hidden dimension (must match training).')
    parser.add_argument('--split',       type=str, default='test', choices=['val', 'test', 'all'],
                        help='Which user split to generate recommendations for.')
    parser.add_argument('--seed',        type=int, default=42)
    parser.add_argument('--device',      type=str, default='auto')
    parser.add_argument('--output',      type=str, default='recommendations.json',
                        help='Path to write the output JSON file.')
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
    logger.info('Loading embeddings: %s', args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    user_emb = ckpt['user_emb']  # (U, d)
    item_emb = ckpt['item_emb']  # (I, d)
    num_items = item_emb.size(0)
    d = user_emb.size(1)
    logger.info('user_emb: %s  item_emb: %s', tuple(user_emb.shape), tuple(item_emb.shape))

    logger.info('Loading graph: %s', args.graph_path)
    graph = torch.load(args.graph_path, map_location='cpu')
    user_tags = graph['user'].tags   # (U, tag_dim)
    food_tags = graph['food'].tags   # (I, tag_dim)
    tag_dim = user_tags.size(1)

    # Optionally load food names for human-readable output
    food_names: list[str] | None = None
    if hasattr(graph['food'], 'node_id'):
        raw = graph['food'].node_id
        if isinstance(raw, list):
            food_names = [str(n) for n in raw]
        elif isinstance(raw, torch.Tensor):
            food_names = [str(int(n)) for n in raw.tolist()]

    edge_index = graph[('user', 'eats', 'food')].edge_index

    # Reproduce the same 60/20/20 split used in morl_main.py
    edges = edge_index.numpy().T
    train_edges_arr, test_edges_arr = train_test_split(edges, test_size=0.2, random_state=args.seed)
    train_edges_arr, val_edges_arr  = train_test_split(train_edges_arr, test_size=0.25, random_state=args.seed)

    train_ei = torch.LongTensor(train_edges_arr).T
    val_ei   = torch.LongTensor(val_edges_arr).T
    test_ei  = torch.LongTensor(test_edges_arr).T

    val_pos  = _get_user_positive_items(val_ei)
    test_pos = _get_user_positive_items(test_ei)

    if args.split == 'val':
        eval_users = val_ei[0].unique().tolist()
        pos_items  = val_pos
    elif args.split == 'test':
        eval_users = test_ei[0].unique().tolist()
        pos_items  = test_pos
    else:  # 'all'
        eval_users = list(range(user_emb.size(0)))
        pos_items  = {**val_pos, **test_pos}

    logger.info('Split: %s | %d eval users | K=%d', args.split, len(eval_users), args.K)

    # ------------------------------------------------------------------
    # Rebuild the policy (architecture must match training)
    # ------------------------------------------------------------------
    state_dim     = 2 * d + tag_dim + 1
    candidate_dim = d

    policy = ConditionalPolicy(
        state_dim=state_dim,
        candidate_dim=candidate_dim,
        hidden_dim=args.hidden_dim,
    )
    logger.info('Loading policy weights: %s', args.policy)
    ckpt_policy = torch.load(args.policy, map_location='cpu')
    # Checkpoints may be a bare state_dict or a full training snapshot
    if isinstance(ckpt_policy, dict) and 'policy_state_dict' in ckpt_policy:
        state_dict = ckpt_policy['policy_state_dict']
    else:
        state_dict = ckpt_policy
    policy.load_state_dict(state_dict)
    policy.to(device)
    policy.eval()
    logger.info('Policy loaded (%d parameters)', sum(p.numel() for p in policy.parameters()))

    # ------------------------------------------------------------------
    # Build full-item-space candidate pools (one list per user = all items)
    # Matches the eval protocol used in evaluate_morl with M=num_items.
    # ------------------------------------------------------------------
    logger.info('Building full-item-space candidate pools (%d items)...', num_items)
    pools = build_candidate_pools(
        user_emb, item_emb,
        M=num_items,
        exclude_per_user=None,
        device=device,
    )

    env = RecommendationEnv(
        user_emb=user_emb,
        item_emb=item_emb,
        user_tags=user_tags,
        item_tags=food_tags,
        candidate_pools=pools,
        K=args.K,
        device=device,
    )

    # ------------------------------------------------------------------
    # Generate recommendations
    # ------------------------------------------------------------------
    logger.info('Generating recommendations for %d users...', len(eval_users))
    recommendations: dict[str, dict] = {}

    with torch.no_grad():
        for i, user_id in enumerate(eval_users):
            if user_id not in pools:
                continue
            rec_list = get_recommendations(policy, env, user_id, args.K, device=device)
            entry: dict = {'items': rec_list}
            if food_names is not None:
                entry['item_names'] = [food_names[idx] for idx in rec_list if idx < len(food_names)]
            entry['ground_truth'] = pos_items.get(user_id, [])
            recommendations[str(user_id)] = entry

            if (i + 1) % 500 == 0:
                logger.info('  %d / %d users done', i + 1, len(eval_users))

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    output_data = {
        'meta': {
            'split':      args.split,
            'K':          args.K,
            'num_users':  len(recommendations),
            'num_items':  num_items,
            'policy':     os.path.abspath(args.policy),
        },
        'recommendations': recommendations,
    }
    with open(args.output, 'w', encoding='utf-8') as fh:
        json.dump(output_data, fh, indent=2)
    logger.info('Recommendations saved to %s  (%d users)', args.output, len(recommendations))

    # ------------------------------------------------------------------
    # Quick summary stats
    # ------------------------------------------------------------------
    ndcg_list, recall_list, health_list = [], [], []
    user_tags_gpu = user_tags.to(device)
    food_tags_gpu = food_tags.to(device)
    item_emb_gpu  = item_emb.to(device)

    K = args.K
    positions = torch.arange(1, K + 1, dtype=torch.float32, device=device)
    discounts = 1.0 / torch.log2(positions + 1.0)

    with torch.no_grad():
        for user_id, entry in ((int(u), v) for u, v in recommendations.items()):
            rec = entry['items']
            gt  = set(entry['ground_truth'])
            k_actual = min(len(rec), K)
            rec_t = torch.tensor(rec[:k_actual], dtype=torch.long, device=device)

            if gt:
                labels = torch.tensor(
                    [1.0 if it in gt else 0.0 for it in rec[:k_actual]],
                    dtype=torch.float32, device=device,
                )
                dcg  = float((labels * discounts[:k_actual]).sum().item())
                idcg = float(discounts[:min(len(gt), K)].sum().item())
                ndcg_list.append(dcg / idcg if idcg > 0 else 0.0)
                recall_list.append(int((labels > 0.5).sum().item()) / len(gt))
            else:
                ndcg_list.append(0.0)
                recall_list.append(0.0)

            user_tag_v = user_tags_gpu[user_id]
            rec_tags   = food_tags_gpu[rec_t]
            health_list.append(
                int((rec_tags.bool() & user_tag_v.bool()).any(dim=1).sum().item()) / K
            )

    def _m(lst: list) -> float:
        return float(np.mean(lst)) if lst else 0.0

    ndcg     = _m(ndcg_list)
    recall   = _m(recall_list)
    health   = _m(health_list)

    header    = f"{'':>4}{'ndcg':>8}  {'recall':>8}  {'health':>8}"
    separator = '-' * len(header)
    print()
    print(f'=== MORL Inference — {args.split.capitalize()} Results ===')
    print(f'Split: {args.split} | K={K} | items={num_items} | users={len(recommendations)}')
    print()
    print(f'  MORL policy:  ndcg={ndcg:.5f}  recall={recall:.5f}  health={health:.5f}')
    print()
    print(header)
    print(separator)
    print(f"{'':>4}{ndcg:>8.5f}  {recall:>8.5f}  {health:>8.5f}  <-- MORL policy")
    print(separator)
    print()


if __name__ == '__main__':
    main()
