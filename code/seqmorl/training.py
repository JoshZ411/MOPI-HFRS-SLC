"""Implicit sequential MORL training loop.

Design:
  - One shared policy network (no explicit preference vector input).
  - Three objectives: preference, health, diversity.
  - MGDA-style gradient combination: per-objective policy-gradient losses are
    computed, their gradients are L2-normalised, and the min-norm convex
    combination is applied — mirroring ``pareto_loss`` in RCSYS_utils.py.
  - Objective-specific moving-average baselines reduce variance.
  - Comprehensive W&B metrics expose collapse-detection diagnostics.

References:
  code/RCSYS_utils.py  — pareto_loss MGDA pattern (adapted for RL objectives).
  code/min_norm_solvers.py — MinNormSolver, gradient_normalizers.
"""

import os
import sys
import math
import torch

# Allow importing from the parent code/ directory when run as a script.
_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_HERE)
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

from min_norm_solvers import MinNormSolver, gradient_normalizers
from seqmorl.environment import SequentialRecEnv
from seqmorl.policy import SequentialPolicy
from seqmorl.evaluation import evaluate_sequential
from seqmorl.logging_utils import WandbTracker, save_jsonl, save_json

OBJECTIVES = ['pref', 'health', 'div']

# Guardrail thresholds (derived from prior MORL experiment analysis).
DOMINANCE_THRESHOLD = 10    # max(coeffs) / min(coeffs); sustained > this → pause
MIN_ENTROPY_THRESHOLD = 0.05  # action entropy below this → mode collapse risk
DOMINANCE_HIGH_EPOCHS = 50  # consecutive epochs above threshold before alert


def _chunk_users(user_ids: list[int], chunk_size: int) -> list[list[int]]:
    """Split user IDs into contiguous chunks."""
    if chunk_size <= 0:
        return [user_ids]
    return [user_ids[i:i + chunk_size] for i in range(0, len(user_ids), chunk_size)]


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------

def collect_rollouts(policy: SequentialPolicy,
                     env: SequentialRecEnv,
                     user_ids: list[int],
                     gamma: float = 0.99):
    """Run one episode per user and return trajectory data.

    Returns:
        log_probs  : Tensor [total_steps]
        returns    : Tensor [total_steps, 3]  — per-objective discounted returns.
        entropies  : Tensor [total_steps]
        actions    : list[int]
        first_actions : list[int]  — first action index per user (for probe).
    """
    all_log_probs = []
    all_returns = []
    all_entropies = []
    all_actions = []
    first_actions = []

    policy.train()

    for uid in user_ids:
        state = env.reset(uid)
        rewards_buf: list[torch.Tensor] = []
        log_probs_buf: list[torch.Tensor] = []
        entropies_buf: list[torch.Tensor] = []
        actions_buf: list[int] = []
        done = False

        while not done:
            mask = env.get_action_mask()
            if bool(mask.all()):
                break
            action, log_prob, entropy = policy.select_action(state, mask, greedy=False)
            next_state, reward, done, _ = env.step(action)

            log_probs_buf.append(log_prob)
            rewards_buf.append(reward)
            # Entropy is only for diagnostics; detach to avoid graph growth.
            entropies_buf.append(entropy.detach())
            actions_buf.append(action)

            state = next_state

        if not rewards_buf:
            continue

        # Compute discounted returns per objective.
        T = len(rewards_buf)
        rewards_tensor = torch.stack(rewards_buf)  # [T, 3]
        returns = torch.zeros_like(rewards_tensor)
        G = torch.zeros(3, device=rewards_tensor.device)
        for t in reversed(range(T)):
            G = rewards_tensor[t] + gamma * G
            returns[t] = G

        all_log_probs.append(torch.stack(log_probs_buf))
        all_returns.append(returns)
        all_entropies.append(torch.stack(entropies_buf))
        all_actions.extend(actions_buf)
        if actions_buf:
            first_actions.append(actions_buf[0])

    if not all_log_probs:
        return None

    return {
        'log_probs': torch.cat(all_log_probs),
        'returns': torch.cat(all_returns, dim=0),
        'entropies': torch.cat(all_entropies),
        'actions': all_actions,
        'first_actions': first_actions,
    }


# ---------------------------------------------------------------------------
# MGDA-style policy gradient update
# ---------------------------------------------------------------------------

def _compute_objective_loss(log_probs: torch.Tensor,
                             advantages: torch.Tensor) -> torch.Tensor:
    """REINFORCE loss: -E[log_prob * advantage]."""
    return -(log_probs * advantages.detach()).mean()


def mgda_policy_update(policy: SequentialPolicy,
                       optimizer: torch.optim.Optimizer,
                       rollout: dict,
                       baselines: dict[str, float],
                       grad_clip: float = 1.0) -> dict:
    """One MGDA update step.

    Mirrors pareto_loss in RCSYS_utils.py, adapted for three RL objectives.

    Args:
        policy    : Policy network.
        optimizer : Optimiser (already zeroed).
        rollout   : Output from collect_rollouts.
        baselines : Running mean returns per objective (for advantage computation).
        grad_clip : Gradient clipping norm after MGDA combination.

    Returns:
        Dict of scalar statistics for logging.
    """
    log_probs = rollout['log_probs']   # [N]
    returns = rollout['returns']       # [N, 3]

    grads: dict[str, list] = {}
    loss_data: dict[str, torch.Tensor] = {}
    raw_grad_norms: dict[str, float] = {}

    for i, obj in enumerate(OBJECTIVES):
        adv = returns[:, i] - baselines[obj]
        loss = _compute_objective_loss(log_probs, adv)
        loss_data[obj] = loss
        grads[obj] = []

        optimizer.zero_grad()
        retain = i < (len(OBJECTIVES) - 1)
        loss.backward(retain_graph=retain)
        raw_norm = 0.0
        for param in policy.parameters():
            if param.grad is not None:
                raw_norm += param.grad.data.norm(2).item() ** 2
                grads[obj].append(param.grad.data.detach().clone())
        raw_grad_norms[obj] = math.sqrt(raw_norm)
        optimizer.zero_grad()

    # L2 gradient normalisation (mirrors gradient_normalizers 'l2').
    gn = gradient_normalizers(grads, loss_data, 'l2')
    for obj in OBJECTIVES:
        for gi in range(len(grads[obj])):
            grads[obj][gi] = grads[obj][gi] / gn[obj].to(grads[obj][gi].device)

    # Min-norm convex combination (MGDA).
    sol_array, _ = MinNormSolver.find_min_norm_element_FW(
        [grads[obj] for obj in OBJECTIVES]
    )
    sol = {obj: float(sol_array[i]) for i, obj in enumerate(OBJECTIVES)}

    # Form combined gradient direction directly from stored objective grads.
    # This is equivalent to backward() on the weighted loss but avoids an extra pass.
    optimizer.zero_grad()
    params = [p for p in policy.parameters() if p.requires_grad]
    for pi, param in enumerate(params):
        combined_grad = None
        for obj in OBJECTIVES:
            if pi >= len(grads[obj]):
                continue
            g = float(sol[obj]) * grads[obj][pi]
            combined_grad = g if combined_grad is None else (combined_grad + g)
        if combined_grad is not None:
            param.grad = combined_grad

    combined_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
    optimizer.step()

    combined_loss = sum(float(sol[obj]) * float(loss_data[obj].item()) for obj in OBJECTIVES)

    # Objective dominance ratio: max coeff / (min coeff + eps).
    coeffs = [sol[obj] for obj in OBJECTIVES]
    dominance_ratio = max(coeffs) / (min(coeffs) + 1e-8)

    stats = {
        'policy_loss': float(combined_loss),
        'grad_norm': float(combined_norm),
        'objective_dominance_ratio': dominance_ratio,
        'num_steps': int(log_probs.numel()),
        'num_users': int(len(rollout.get('first_actions', []))),
    }
    for obj in OBJECTIVES:
        stats[f'mgda_coeff_{obj}'] = sol[obj]
        stats[f'objective_grad_norm_{obj}'] = raw_grad_norms[obj]
        stats[f'mean_reward_{obj}'] = float(returns[:, OBJECTIVES.index(obj)].mean().item())
        stats[f'std_reward_{obj}'] = float(returns[:, OBJECTIVES.index(obj)].std().item())

    return stats


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_implicit_morl(policy: SequentialPolicy,
                        env: SequentialRecEnv,
                        train_user_ids: list[int],
                        val_user_ids: list[int],
                        args,
                        tracker: WandbTracker,
                        output_dir: str,
                        val_eval_context: dict | None = None) -> SequentialPolicy:
    """Implicit multi-objective sequential MORL training.

    Args:
        policy        : Initialised SequentialPolicy (on correct device).
        env           : SequentialRecEnv (on correct device).
        train_user_ids: User IDs to train on each epoch.
        val_user_ids  : User IDs for probe diagnostics.
        args          : Namespace from argparse (see seqmorl_main.py).
        tracker       : WandbTracker instance.
        output_dir    : Directory for checkpoints and metric JSONL.

    Returns:
        Trained policy.
    """
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    # Per-objective exponential moving average baseline.
    baselines = {obj: 0.0 for obj in OBJECTIVES}
    baseline_alpha = 0.05  # EMA decay for advantage baselines.

    # Cumulative advantage tracking.
    cumulative_adv = {obj: 0.0 for obj in OBJECTIVES}

    # Dominance guardrail state.
    dominance_high_count = 0

    metrics_path = os.path.join(output_dir, 'train_metrics.jsonl')
    best_ndcg_policy_path = os.path.join(output_dir, 'policy_best.pt')
    best_reward_policy_path = os.path.join(output_dir, 'policy_best_reward.pt')
    best_mean_reward = -float('inf')
    best_val_ndcg = -float('inf')
    no_improve_evals = 0
    train_batch_users = int(getattr(args, 'train_batch_users', 0))
    if train_batch_users <= 0:
        train_batch_users = len(train_user_ids)
    users_per_epoch = int(getattr(args, 'users_per_epoch', 0))
    if users_per_epoch <= 0:
        users_per_epoch = len(train_user_ids)
    val_eval_interval = int(getattr(args, 'val_eval_interval', 5))
    val_ndcg_patience = int(getattr(args, 'val_ndcg_patience', 0))
    val_ndcg_min_delta = float(getattr(args, 'val_ndcg_min_delta', 0.0))
    early_stop_on_val_ndcg = int(getattr(args, 'early_stop_on_val_ndcg', 1)) == 1
    val_ndcg_warmup_epochs = int(getattr(args, 'val_ndcg_warmup_epochs', 0))
    for epoch in range(args.epochs):
        policy.train()
        # Shuffle per epoch and train in user mini-batches to bound graph size.
        shuffled = train_user_ids[:]
        if len(shuffled) > 1:
            perm = torch.randperm(len(shuffled)).tolist()
            shuffled = [shuffled[i] for i in perm]
        epoch_users = shuffled[:users_per_epoch]
        user_chunks = _chunk_users(epoch_users, train_batch_users)

        print(
            f"[seqmorl][epoch {epoch}] collecting rollouts for {len(epoch_users)} users "
            f"in {len(user_chunks)} chunks (chunk_size={train_batch_users}) ..."
        )

        weight_sum = 0.0
        weighted: dict[str, float] = {}
        all_actions_epoch: list[int] = []
        first_actions_epoch: list[int] = []
        entropy_sum = 0.0
        total_steps = 0

        for chunk_idx, chunk_users in enumerate(user_chunks):
            rollout = collect_rollouts(
                policy,
                env,
                chunk_users,
                args.gamma,
            )
            if rollout is None:
                continue

            stats_chunk = mgda_policy_update(
                policy,
                optimizer,
                rollout,
                baselines,
                grad_clip=args.grad_clip,
            )

            step_weight = float(max(1, stats_chunk.get('num_steps', 1)))
            weight_sum += step_weight
            for key, value in stats_chunk.items():
                if key in {'num_steps', 'num_users'}:
                    continue
                weighted[key] = weighted.get(key, 0.0) + float(value) * step_weight

            all_actions_epoch.extend(rollout['actions'])
            first_actions_epoch.extend(rollout['first_actions'])
            entropy_sum += float(rollout['entropies'].sum().item())
            total_steps += int(rollout['entropies'].numel())

            if len(user_chunks) > 1 and (
                chunk_idx == 0
                or chunk_idx == len(user_chunks) - 1
                or chunk_idx % 10 == 0
            ):
                print(
                    f"[seqmorl][epoch {epoch}] chunk {chunk_idx + 1}/{len(user_chunks)} "
                    f"steps={stats_chunk['num_steps']} loss={stats_chunk['policy_loss']:.4f}"
                )

        if weight_sum == 0.0:
            print(f"[Epoch {epoch}] No rollout data across chunks — skipping.")
            continue

        stats = {k: (v / weight_sum) for k, v in weighted.items()}
        stats['mean_entropy'] = (entropy_sum / max(1, total_steps))
        stats['mean_action_position'] = (
            float(sum(all_actions_epoch) / len(all_actions_epoch))
            if all_actions_epoch else 0.0
        )
        stats['num_steps'] = int(total_steps)
        stats['num_users'] = int(len(first_actions_epoch))

        coeffs_epoch = [stats[f'mgda_coeff_{o}'] for o in OBJECTIVES]
        stats['objective_dominance_ratio'] = max(coeffs_epoch) / (min(coeffs_epoch) + 1e-8)

        # Update baselines via EMA.
        for obj in OBJECTIVES:
            baselines[obj] = (
                (1 - baseline_alpha) * baselines[obj]
                + baseline_alpha * stats[f'mean_reward_{obj}']
            )

        # Cumulative advantage.
        for obj in OBJECTIVES:
            cumulative_adv[obj] += stats[f'mean_reward_{obj}'] - baselines[obj]
            stats[f'cumulative_advantage_{obj}'] = cumulative_adv[obj]

        # Probing diagnostics every 25 epochs.
        if epoch % 25 == 0 and first_actions_epoch:
            first = first_actions_epoch
            stats['probe_first_action_span'] = max(first) - min(first)
            # Pairwise Jaccard on top-K item sets not available without full
            # rollout replay; approximate with action-list diversity.
            unique_actions = len(set(all_actions_epoch))
            total_actions = len(all_actions_epoch)
            stats['probe_pairwise_jaccard'] = (
                unique_actions / total_actions if total_actions else 0.0
            )

        # Guardrail checks.
        dom = stats['objective_dominance_ratio']
        if dom > DOMINANCE_THRESHOLD:
            dominance_high_count += 1
        else:
            dominance_high_count = 0

        if dominance_high_count >= DOMINANCE_HIGH_EPOCHS:
            print(
                f"[GUARDRAIL] Epoch {epoch}: objective_dominance_ratio > {DOMINANCE_THRESHOLD} "
                f"for {dominance_high_count} consecutive epochs. "
                "Prior MORL failure mode detected — investigate before continuing."
            )

        if stats['mean_entropy'] < MIN_ENTROPY_THRESHOLD:
            print(
                f"[GUARDRAIL] Epoch {epoch}: mean_entropy={stats['mean_entropy']:.4f} "
                f"< {MIN_ENTROPY_THRESHOLD} — possible mode collapse."
            )

        if epoch % 25 == 0 and stats.get('probe_first_action_span', 1) < 1:
            print(
                f"[GUARDRAIL] Epoch {epoch}: probe_first_action_span ≈ 0 "
                "— prior MORL failure symptom (all users choosing same first action)."
            )

        # Save checkpoint if mean reward improved.
        mean_r = sum(stats[f'mean_reward_{o}'] for o in OBJECTIVES) / 3
        if mean_r > best_mean_reward:
            best_mean_reward = mean_r
            torch.save(policy.state_dict(), best_reward_policy_path)

        # Periodic validation for ranking-aligned model selection.
        run_val_eval = (
            val_eval_context is not None
            and val_eval_interval > 0
            and (epoch % val_eval_interval == 0 or epoch == args.epochs - 1)
        )
        should_stop_early = False
        if run_val_eval:
            assert val_eval_context is not None
            eval_ctx = val_eval_context
            val_metrics = evaluate_sequential(
                policy=policy,
                env=env,
                user_ids=val_user_ids,
                pos_edge_index=eval_ctx['pos_edge_index'],
                user_tags=eval_ctx['user_tags'],
                food_tags=eval_ctx['food_tags'],
                K=int(eval_ctx['K']),
                exclude_edge_indices=eval_ctx.get('exclude_edge_indices'),
            )
            val_ndcg = float(val_metrics.get('ndcg', 0.0))
            stats['val_ndcg'] = val_ndcg
            stats['val_recall'] = float(val_metrics.get('recall', 0.0))
            stats['val_precision'] = float(val_metrics.get('precision', 0.0))

            if val_ndcg > (best_val_ndcg + val_ndcg_min_delta):
                best_val_ndcg = val_ndcg
                if epoch >= val_ndcg_warmup_epochs:
                    no_improve_evals = 0
                torch.save(policy.state_dict(), best_ndcg_policy_path)
            elif epoch >= val_ndcg_warmup_epochs:
                no_improve_evals += 1

            if (
                early_stop_on_val_ndcg
                and val_ndcg_patience > 0
                and epoch >= val_ndcg_warmup_epochs
                and no_improve_evals >= val_ndcg_patience
            ):
                should_stop_early = True
                print(
                    f"[EARLY STOP] Epoch {epoch}: val_ndcg did not improve by > {val_ndcg_min_delta:.6f} "
                    f"for {no_improve_evals} evaluation checkpoints (patience={val_ndcg_patience})."
                )

        # Logging.
        epoch_log = {'epoch': epoch, **stats}
        save_jsonl(metrics_path, [epoch_log])
        tracker.log({f'train/{k}': v for k, v in stats.items()}, step=epoch)

        if epoch % args.log_interval == 0:
            coeff_str = ', '.join(
                f"{o}={stats[f'mgda_coeff_{o}']:.3f}" for o in OBJECTIVES
            )
            print(
                f"Epoch {epoch:4d} | loss={stats['policy_loss']:.4f} | "
                f"dom={dom:.2f} | ent={stats['mean_entropy']:.3f} | "
                f"coeffs=[{coeff_str}]"
            )
            if 'val_ndcg' in stats:
                print(
                    f"[seqmorl][epoch {epoch}] val_ndcg={stats['val_ndcg']:.5f} "
                    f"val_recall={stats['val_recall']:.5f} "
                    f"val_precision={stats['val_precision']:.5f}"
                )

        if should_stop_early:
            break

    # Save final policy.
    final_path = os.path.join(output_dir, 'policy_final.pt')
    torch.save(policy.state_dict(), final_path)
    print(f"Training complete. Final policy saved to {final_path}")
    print(f"Best reward policy saved to {best_reward_policy_path}")

    if os.path.exists(best_ndcg_policy_path):
        policy.load_state_dict(torch.load(best_ndcg_policy_path, map_location=env.device))
        print(
            f"Best ranking policy saved to {best_ndcg_policy_path} "
            f"(best val_ndcg={best_val_ndcg:.5f})"
        )
    else:
        torch.save(policy.state_dict(), best_ndcg_policy_path)
        print(
            f"No val-eval checkpoint created; wrote fallback best policy to {best_ndcg_policy_path}"
        )

    # Save leet command.
    leet_cmd = tracker.leet_command()
    save_json(os.path.join(output_dir, 'wandb_leet_command.txt'), {'command': leet_cmd})
    print(f"W&B leet command: {leet_cmd}")

    return policy
