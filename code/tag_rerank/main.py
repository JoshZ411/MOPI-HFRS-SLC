"""
main.py — TARS CLI entrypoint
Author: Harshit | Branch: harshit-develop

Trains (or loads) the SGSL model, evaluates with tag-augmented re-scoring at a
specified alpha, and reports baseline vs. TARS metrics side-by-side.

Usage examples
--------------
    # Evaluate at alpha=0.1 (train fresh — requires data)
    python main.py --alpha 0.1 --epochs 200

    # Load a pre-trained checkpoint and evaluate
    python main.py --alpha 0.15 --checkpoint ../../processed_data/sgsl_checkpoint.pt

    # Quick dry-run with random embeddings
    python main.py --alpha 0.1 --skip_training --user_limit 50

    # Run full alpha sweep after training
    python main.py --run_sweep --epochs 200
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import torch
import torch.optim as optim

from RCSYS_utils import (
    split_data_new,
    get_metrics,
    eval as baseline_eval,
    pareto_loss,
    sample_mini_batch,
)
from RCSYS_models import SGSL
from evaluate import tars_get_metrics, assert_baseline_parity
from alpha_sweep import run_sweep, print_sweep_table, DEFAULT_ALPHAS


# ─── Training loop (mirrors code/main.py exactly) ────────────────────────────

def train_sgsl(model, graph, train_ei, val_ei, test_ei,
               pos_train, neg_train, pos_val, neg_val, pos_test, neg_test,
               feature_dict, user_tags, food_tags, device,
               epochs, batch_size, lr, lambda_val, iters_per_eval,
               iters_per_lr_decay, K):
    """
    Train the SGSL model with the same loop as code/main.py.
    Returns the trained model.
    """
    model = model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    user_features = graph['user'].x.to(device)
    food_features = graph['food'].x.to(device)

    print(f"\n[TRAIN] Starting SGSL training for {epochs} epochs ...")

    for epoch in range(epochs):
        u_emb_f, u_emb_0, i_emb_f, i_emb_0 = model.forward(
            feature_dict, train_ei, pos_train, neg_train
        )

        user_idx, pos_idx, neg_idx = sample_mini_batch(batch_size, train_ei)

        u_f  = u_emb_f[user_idx];   u_0  = u_emb_0[user_idx]
        pi_f = i_emb_f[pos_idx];    pi_0 = i_emb_0[pos_idx]
        ni_f = i_emb_f[neg_idx];    ni_0 = i_emb_0[neg_idx]

        u_tags_b  = user_tags[user_idx]
        pi_tags_b = food_tags[pos_idx]
        ni_tags_b = food_tags[neg_idx]

        u_feat_b  = user_features[user_idx]
        u_feat_b  = torch.nn.functional.pad(
            u_feat_b, (0, food_features.size(1) - u_feat_b.size(1))
        )
        pi_feat_b = food_features[pos_idx]
        ni_feat_b = food_features[neg_idx]

        train_loss, _, _ = pareto_loss(
            model, u_f, u_0, pi_f, pi_0, ni_f, ni_0,
            u_feat_b, pi_feat_b, ni_feat_b,
            u_tags_b, pi_tags_b, ni_tags_b, lambda_val
        )

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if epoch % iters_per_eval == 0 and epoch != 0:
            model.eval()
            with torch.no_grad():
                _, recall, precision, ndcg, hs, ht, pct = baseline_eval(
                    model, feature_dict, user_tags, food_tags,
                    val_ei, pos_val, neg_val, [neg_train], K, lambda_val
                )
            print(f"  Epoch {epoch:>4} | loss={train_loss.item():.5f} | "
                  f"recall={recall:.5f} | ndcg={ndcg:.5f} | health={hs:.5f}")
            model.train()

        if epoch % iters_per_lr_decay == 0 and epoch != 0:
            scheduler.step()

    print("[TRAIN] Done.\n")
    return model


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="TARS: Tag-Augmented Inference Re-scoring for MOPI-HFRS"
    )
    # Data
    parser.add_argument('--graph_path', default='../../processed_data/benchmark_macro.pt')
    parser.add_argument('--checkpoint', default=None,
                        help='Load pre-trained SGSL state_dict (.pt). Skips training.')
    parser.add_argument('--save_checkpoint', default=None,
                        help='Save trained model state_dict to this path.')

    # TARS
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Tag blending coefficient (0=baseline, 1=pure Jaccard)')
    parser.add_argument('--run_sweep', action='store_true',
                        help='Run full alpha sweep after evaluation')
    parser.add_argument('--sweep_alphas', type=float, nargs='+', default=DEFAULT_ALPHAS)

    # Evaluation
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--user_limit', type=int, default=None,
                        help='Limit to first N users for fast smoke test')
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip training (use random or loaded embeddings only)')

    # Model / training
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--LAMBDA', type=float, default=1e-6)
    parser.add_argument('--iters_per_eval', type=int, default=50)
    parser.add_argument('--iters_per_lr_decay', type=int, default=200)
    parser.add_argument('--feature_threshold', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto')

    # Output
    parser.add_argument('--output_json', default='tars_results.json')

    args = parser.parse_args()

    # ── Seed & device ─────────────────────────────────────────────────────────
    import torch_geometric
    torch_geometric.seed_everything(args.seed)

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif args.device == 'cuda':
        assert torch.cuda.is_available()
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"[INFO] Device: {device}")

    # ── Load graph ────────────────────────────────────────────────────────────
    print(f"[INFO] Loading: {args.graph_path}")
    graph = torch.load(args.graph_path, map_location='cpu')

    edge_index       = graph[('user', 'eats', 'food')].edge_index
    edge_label_index = graph[('user', 'eats', 'food')].edge_label_index
    user_tags        = graph['user'].tags.to(device)
    food_tags        = graph['food'].tags.to(device)
    feature_dict     = {k: graph[k].x.to(device) for k in ['user', 'food']}

    num_users = graph['user'].num_nodes
    num_foods = graph['food'].num_nodes
    print(f"[INFO] {num_users} users × {num_foods} foods | "
          f"tag dims: {user_tags.shape[1]}")

    # ── Splits ────────────────────────────────────────────────────────────────
    (train_ei, val_ei, test_ei,
     pos_train, neg_train, pos_val, neg_val,
     pos_test, neg_test) = split_data_new(edge_index, edge_label_index)

    # Move to device
    for t in [train_ei, val_ei, test_ei,
              pos_train, neg_train, pos_val, neg_val, pos_test, neg_test]:
        t.to(device)

    eval_ei     = val_ei  if args.split == 'val' else test_ei
    pos_eval_ei = pos_val if args.split == 'val' else pos_test
    neg_eval_ei = neg_val if args.split == 'val' else neg_test
    excl_eids   = [neg_train] if args.split == 'val' else [neg_train, neg_val]

    # ── Model ─────────────────────────────────────────────────────────────────
    model = SGSL(graph, embedding_dim=args.hidden_dim,
                 feature_threshold=args.feature_threshold, num_layer=args.layers)

    if args.checkpoint:
        print(f"[INFO] Loading checkpoint: {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print("[INFO] Checkpoint loaded.")
    elif not args.skip_training:
        model = train_sgsl(
            model, graph, train_ei, val_ei, test_ei,
            pos_train, neg_train, pos_val, neg_val, pos_test, neg_test,
            feature_dict, user_tags, food_tags, device,
            epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, lambda_val=args.LAMBDA,
            iters_per_eval=args.iters_per_eval,
            iters_per_lr_decay=args.iters_per_lr_decay,
            K=args.K
        )
    else:
        print("[WARN] skip_training=True with no checkpoint → random embeddings.")

    if args.save_checkpoint:
        torch.save(model.state_dict(), args.save_checkpoint)
        print(f"[INFO] Checkpoint saved to: {args.save_checkpoint}")

    # ── Get embeddings ────────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        u_emb, _, f_emb, _ = model.forward(
            feature_dict, train_ei, pos_train, neg_train
        )

    # ── Parity check ─────────────────────────────────────────────────────────
    print("\n[STEP 1] Parity check (alpha=0.0 must match baseline) ...")
    assert_baseline_parity(
        u_emb.cpu(), f_emb.cpu(),
        user_tags.cpu(), food_tags.cpu(),
        eval_ei.cpu(), [e.cpu() for e in excl_eids],
        k=args.K, user_limit=min(200, num_users)
    )

    # ── Baseline evaluation ───────────────────────────────────────────────────
    print(f"\n[STEP 2] Baseline (alpha=0.0) evaluation on {args.split} ...")
    with torch.no_grad():
        b_metrics = tars_get_metrics(
            u_emb, f_emb, user_tags, food_tags,
            eval_ei, excl_eids, args.K, alpha=0.0
        )
    b_recall, b_prec, b_ndcg, b_hs, b_ht, b_pct = b_metrics

    print(f"  [BASELINE] recall={b_recall:.5f} | prec={b_prec:.5f} | "
          f"ndcg={b_ndcg:.5f} | health={b_hs:.5f} | "
          f"avg_tags={b_ht:.5f} | coverage={b_pct:.5f}")

    # ── TARS evaluation ───────────────────────────────────────────────────────
    print(f"\n[STEP 3] TARS evaluation (alpha={args.alpha}) on {args.split} ...")
    with torch.no_grad():
        t_metrics = tars_get_metrics(
            u_emb, f_emb, user_tags, food_tags,
            eval_ei, excl_eids, args.K, alpha=args.alpha
        )
    t_recall, t_prec, t_ndcg, t_hs, t_ht, t_pct = t_metrics

    print(f"  [TARS-{args.alpha}] recall={t_recall:.5f} | prec={t_prec:.5f} | "
          f"ndcg={t_ndcg:.5f} | health={t_hs:.5f} | "
          f"avg_tags={t_ht:.5f} | coverage={t_pct:.5f}")

    # ── Side-by-side delta ────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Δ (TARS - Baseline) at alpha={args.alpha}")
    print(f"{'─'*60}")
    for name, bv, tv in zip(
        ["recall", "precision", "ndcg", "health_score", "avg_health_tags", "coverage"],
        b_metrics, t_metrics
    ):
        delta = tv - bv
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
        print(f"  {name:<22}: {bv:.5f} → {tv:.5f}  ({arrow} {delta:+.5f})")

    # ── Optional alpha sweep ──────────────────────────────────────────────────
    if args.run_sweep:
        print(f"\n[STEP 4] Running full alpha sweep ...")
        sweep_results = run_sweep(
            u_emb.cpu(), f_emb.cpu(),
            user_tags.cpu(), food_tags.cpu(),
            eval_ei.cpu(), [e.cpu() for e in excl_eids],
            k=args.K, alphas=args.sweep_alphas,
            user_limit=args.user_limit
        )
        print_sweep_table(sweep_results, args.sweep_alphas)
    else:
        sweep_results = None

    # ── Save results ──────────────────────────────────────────────────────────
    metric_names = ["recall", "precision", "ndcg", "health_score", "avg_health_tags", "pct_foods"]
    output = {
        "config": vars(args),
        "baseline": dict(zip(metric_names, [round(v, 6) for v in b_metrics])),
        f"tars_alpha_{args.alpha}": dict(zip(metric_names, [round(v, 6) for v in t_metrics])),
        "delta": dict(zip(metric_names, [round(tv - bv, 6) for bv, tv in zip(b_metrics, t_metrics)])),
    }
    if sweep_results:
        output["sweep"] = sweep_results

    with open(args.output_json, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n[INFO] Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
