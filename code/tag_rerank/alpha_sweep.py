"""
alpha_sweep.py — Sweep alpha ∈ [0, 1] to find the optimal TARS trade-off
Author: Harshit | Branch: harshit-develop

Runs TARS evaluation at multiple alpha values on the val split and prints
a comparison table showing how each metric changes relative to baseline (alpha=0).

Usage
-----
    # Fast smoke test (50 users, CPU)
    python alpha_sweep.py --user_limit 50

    # Full val split evaluation
    python alpha_sweep.py --graph_path ../../processed_data/benchmark_macro.pt

    # Save results to JSON
    python alpha_sweep.py --output_json sweep_results.json

Expected findings
-----------------
    - health_score    : non-decreasing as alpha increases (tag alignment improves)
    - ndcg            : degraded at high alpha (embedding signal lost)
    - pct_foods       : likely improves at moderate alpha (Jaccard promotes diversity)
    - Optimal alpha   : the value where health_score improves WITHOUT significant ndcg drop
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import torch
import numpy as np

from RCSYS_utils import split_data_new
from RCSYS_models import SGSL
from scorer import tag_augmented_scores, score_diagnostics
from evaluate import tars_get_metrics, assert_baseline_parity


# ─── Alpha values to sweep ───────────────────────────────────────────────────
DEFAULT_ALPHAS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]

METRIC_NAMES = [
    "recall", "precision", "ndcg",
    "health_score", "avg_health_tags", "pct_foods"
]


def run_sweep(
    user_emb: torch.Tensor,
    food_emb: torch.Tensor,
    user_tags: torch.Tensor,
    food_tags: torch.Tensor,
    edge_index: torch.Tensor,
    exclude_edge_indices: list,
    k: int,
    alphas: list,
    user_limit: int = None,
) -> dict:
    """
    Evaluate all alpha values and collect metric results.

    Parameters
    ----------
    user_limit : int or None
        If set, only evaluate on the first `user_limit` unique users in the split.
        Useful for fast smoke tests.

    Returns
    -------
    dict: {alpha_str: {metric_name: value}} for all alphas
    """
    # Optionally limit to a user subset for fast testing
    if user_limit is not None:
        unique_users = edge_index[0].unique()[:user_limit]
        mask = torch.isin(edge_index[0], unique_users)
        edge_index = edge_index[:, mask]
        user_emb   = user_emb[:unique_users.max().item() + 1]
        food_emb   = food_emb
        user_tags  = user_tags[:unique_users.max().item() + 1]
        print(f"[INFO] User limit active: evaluating on {len(unique_users)} users.")

    results = {}

    for alpha in alphas:
        metrics = tars_get_metrics(
            user_emb, food_emb, user_tags, food_tags,
            edge_index, exclude_edge_indices, k, alpha
        )
        results[str(alpha)] = dict(zip(METRIC_NAMES, metrics))
        print(f"  alpha={alpha:.2f} | "
              f"recall={metrics[0]:.5f} | ndcg={metrics[2]:.5f} | "
              f"health={metrics[3]:.5f} | coverage={metrics[5]:.5f}")

    return results


def print_sweep_table(results: dict, alphas: list) -> None:
    """Print a formatted comparison table, showing deltas from alpha=0 (baseline)."""
    baseline = results["0.0"]

    col_w = 10
    header = (
        f"{'alpha':>6} | "
        f"{'recall':>{col_w}} | {'Δrecall':>{col_w}} | "
        f"{'ndcg':>{col_w}} | {'Δndcg':>{col_w}} | "
        f"{'health':>{col_w}} | {'Δhealth':>{col_w}} | "
        f"{'coverage':>{col_w}} | {'Δcoverage':>{col_w}}"
    )
    sep = "─" * len(header)
    print(f"\n{sep}")
    print("TARS Alpha Sweep Results (val split)")
    print(sep)
    print(header)
    print(sep)

    for alpha in alphas:
        r = results[str(alpha)]
        d_recall   = r["recall"]       - baseline["recall"]
        d_ndcg     = r["ndcg"]         - baseline["ndcg"]
        d_health   = r["health_score"] - baseline["health_score"]
        d_coverage = r["pct_foods"]    - baseline["pct_foods"]

        def fmt_delta(v):
            return f"+{v:.5f}" if v >= 0 else f"{v:.5f}"

        print(
            f"{alpha:>6.2f} | "
            f"{r['recall']:>{col_w}.5f} | {fmt_delta(d_recall):>{col_w}} | "
            f"{r['ndcg']:>{col_w}.5f} | {fmt_delta(d_ndcg):>{col_w}} | "
            f"{r['health_score']:>{col_w}.5f} | {fmt_delta(d_health):>{col_w}} | "
            f"{r['pct_foods']:>{col_w}.5f} | {fmt_delta(d_coverage):>{col_w}}"
        )

    print(sep)

    # ── Auto-select best alpha ────────────────────────────────────────────────
    # "Best": max health_score gain where ndcg drop < 5% of baseline ndcg
    ndcg_floor = baseline["ndcg"] * 0.95   # allow up to 5% relative NDCG drop
    best_alpha = None
    best_health_gain = -1.0

    for alpha in alphas:
        if alpha == 0.0:
            continue
        r = results[str(alpha)]
        if r["ndcg"] >= ndcg_floor:
            gain = r["health_score"] - baseline["health_score"]
            if gain > best_health_gain:
                best_health_gain = gain
                best_alpha = alpha

    if best_alpha is not None:
        print(f"\n[AUTO-SELECT] Best alpha = {best_alpha}"
              f" (health_score +{best_health_gain:+.5f}, "
              f"ndcg within 5% of baseline)")
    else:
        print("\n[AUTO-SELECT] No alpha found where ndcg stays within 5% of baseline."
              " Consider tighter alpha range.")
    print()


def main():
    parser = argparse.ArgumentParser(description="TARS alpha sweep")
    parser.add_argument('--graph_path', type=str,
                        default='../../processed_data/benchmark_macro.pt',
                        help='Path to benchmark_macro.pt')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to saved SGSL model state_dict (.pt). '
                             'If not provided, runs with random init embeddings '
                             '(for smoke testing only).')
    parser.add_argument('--alphas', type=float, nargs='+', default=DEFAULT_ALPHAS,
                        help='Alpha values to sweep (space-separated, e.g. 0 0.1 0.2)')
    parser.add_argument('--K', type=int, default=20,
                        help='Top-K cutoff for evaluation (must match baseline K)')
    parser.add_argument('--split', choices=['val', 'test'], default='val',
                        help='Which split to evaluate on')
    parser.add_argument('--user_limit', type=int, default=None,
                        help='Limit evaluation to first N users (for fast smoke tests)')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto',
                        help='Device to use for forward pass')
    parser.add_argument('--output_json', type=str, default='tag_rerank_sweep.json',
                        help='Save sweep results to this JSON file')
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--feature_threshold', type=float, default=0.3)
    parser.add_argument('--run_diagnostics', action='store_true',
                        help='Print score diagnostics for first 5 users at alpha=0.1')
    args = parser.parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif args.device == 'cuda':
        assert torch.cuda.is_available(), "CUDA requested but not available."
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"[INFO] Device: {device}")

    # ── Load graph ────────────────────────────────────────────────────────────
    print(f"[INFO] Loading graph: {args.graph_path}")
    graph = torch.load(args.graph_path, map_location='cpu')

    edge_index       = graph[('user', 'eats', 'food')].edge_index
    edge_label_index = graph[('user', 'eats', 'food')].edge_label_index
    user_tags        = graph['user'].tags
    food_tags        = graph['food'].tags
    feature_dict     = {k: graph[k].x.to(device) for k in ['user', 'food']}

    num_users = graph['user'].num_nodes
    num_foods = graph['food'].num_nodes
    print(f"[INFO] Graph: {num_users} users × {num_foods} foods")

    # ── Data splits ───────────────────────────────────────────────────────────
    (train_ei, val_ei, test_ei,
     pos_train, neg_train, pos_val, neg_val,
     pos_test, neg_test) = split_data_new(edge_index, edge_label_index)

    eval_ei      = val_ei  if args.split == 'val' else test_ei
    pos_eval_ei  = pos_val if args.split == 'val' else pos_test
    neg_eval_ei  = neg_val if args.split == 'val' else neg_test
    exclude_eids = [neg_train] if args.split == 'val' else [neg_train, neg_val]

    eval_ei.to(device)
    exclude_eids = [e.to(device) for e in exclude_eids]

    # ── SGSL model ────────────────────────────────────────────────────────────
    model = SGSL(graph, embedding_dim=args.hidden_dim,
                 feature_threshold=args.feature_threshold, num_layer=args.layers)

    if args.checkpoint is not None:
        print(f"[INFO] Loading checkpoint: {args.checkpoint}")
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state)
        print("[INFO] Checkpoint loaded successfully.")
    else:
        print("[WARN] No checkpoint provided. Using RANDOMLY INITIALIZED embeddings.")
        print("       Results are for pipeline verification only, not meaningful metrics.")

    model = model.to(device)
    model.eval()

    # ── Get embeddings via forward pass ───────────────────────────────────────
    with torch.no_grad():
        u_emb, _, f_emb, _ = model.forward(
            feature_dict,
            train_ei.to(device),
            pos_train.to(device),
            neg_train.to(device),
        )

    print(f"[INFO] Embeddings: users {u_emb.shape}, foods {f_emb.shape}")

    # ── Parity check BEFORE sweep ─────────────────────────────────────────────
    print("\n[STEP 1] Running alpha=0.0 parity check ...")
    parity_ok = assert_baseline_parity(
        u_emb.cpu(), f_emb.cpu(),
        user_tags.cpu(), food_tags.cpu(),
        eval_ei.cpu(), [e.cpu() for e in exclude_eids],
        k=args.K, user_limit=min(100, num_users)
    )
    if not parity_ok:
        print("[ERROR] Parity check failed. Aborting sweep.")
        return

    # ── Optional diagnostics ──────────────────────────────────────────────────
    if args.run_diagnostics:
        score_diagnostics(u_emb.cpu(), f_emb.cpu(), user_tags.cpu(), food_tags.cpu(), alpha=0.1)

    # ── Run sweep ─────────────────────────────────────────────────────────────
    print(f"\n[STEP 2] Running alpha sweep: {args.alphas} on {args.split} split ...")
    results = run_sweep(
        u_emb.cpu(), f_emb.cpu(),
        user_tags.cpu(), food_tags.cpu(),
        eval_ei.cpu(), [e.cpu() for e in exclude_eids],
        k=args.K,
        alphas=args.alphas,
        user_limit=args.user_limit,
    )

    # ── Print table ───────────────────────────────────────────────────────────
    print_sweep_table(results, args.alphas)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    output = {
        "config": {
            "graph_path": args.graph_path,
            "checkpoint": args.checkpoint,
            "alphas": args.alphas,
            "K": args.K,
            "split": args.split,
            "user_limit": args.user_limit,
            "hidden_dim": args.hidden_dim,
            "layers": args.layers,
            "feature_threshold": args.feature_threshold,
        },
        "results": results,
    }
    with open(args.output_json, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"[INFO] Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
