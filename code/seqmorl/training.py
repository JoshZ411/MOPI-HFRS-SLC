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
import json
import math
import torch
import torch.nn.functional as F

# Allow importing from the parent code/ directory when run as a script.
_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_HERE)
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)

from min_norm_solvers import MinNormSolver, gradient_normalizers
from seqmorl.environment import SequentialRecEnv
from seqmorl.policy import SequentialPolicy
from seqmorl.logging_utils import WandbTracker, save_jsonl, save_json

OBJECTIVES = ['pref', 'health', 'div']

# Guardrail thresholds (derived from prior MORL experiment analysis).
DOMINANCE_THRESHOLD = 10    # max(coeffs) / min(coeffs); sustained > this → pause
MIN_ENTROPY_THRESHOLD = 0.05  # action entropy below this → mode collapse risk
DOMINANCE_HIGH_EPOCHS = 50  # consecutive epochs above threshold before alert


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------

def collect_rollouts(policy: SequentialPolicy,
                     env: SequentialRecEnv,
                     user_ids: list[int],
                     gamma: float = 0.99,
                     device: torch.device | None = None):
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
            action, log_prob, entropy = policy.select_action(state, mask, greedy=False)
            next_state, reward, done, _ = env.step(action)

            log_probs_buf.append(log_prob)
            rewards_buf.append(reward)
            entropies_buf.append(entropy)
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
        loss.backward(retain_graph=True)
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

    # Form combined loss and apply update.
    optimizer.zero_grad()
    combined_loss = sum(sol[obj] * loss_data[obj] for obj in OBJECTIVES)
    combined_loss.backward()
    combined_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
    optimizer.step()

    # Objective dominance ratio: max coeff / (min coeff + eps).
    coeffs = [sol[obj] for obj in OBJECTIVES]
    dominance_ratio = max(coeffs) / (min(coeffs) + 1e-8)

    stats = {
        'policy_loss': float(combined_loss.item()),
        'grad_norm': float(combined_norm),
        'objective_dominance_ratio': dominance_ratio,
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
                        output_dir: str) -> SequentialPolicy:
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
    best_policy_path = os.path.join(output_dir, 'policy_best.pt')
    best_mean_reward = -float('inf')

    for epoch in range(args.epochs):
        policy.train()
        rollout = collect_rollouts(policy, env, train_user_ids, args.gamma)
        if rollout is None:
            print(f"[Epoch {epoch}] No rollout data — skipping.")
            continue

        stats = mgda_policy_update(policy, optimizer, rollout, baselines,
                                   grad_clip=args.grad_clip)

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

        # Entropy and action position diagnostics.
        entropies = rollout['entropies']
        stats['mean_entropy'] = float(entropies.mean().item())
        stats['mean_action_position'] = (
            float(sum(rollout['actions']) / len(rollout['actions']))
            if rollout['actions'] else 0.0
        )

        # Probing diagnostics every 25 epochs.
        if epoch % 25 == 0 and rollout['first_actions']:
            first = rollout['first_actions']
            stats['probe_first_action_span'] = max(first) - min(first)
            # Pairwise Jaccard on top-K item sets not available without full
            # rollout replay; approximate with action-list diversity.
            unique_actions = len(set(rollout['actions']))
            total_actions = len(rollout['actions'])
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

        # Save checkpoint if mean preference reward improved.
        mean_r = sum(stats[f'mean_reward_{o}'] for o in OBJECTIVES) / 3
        if mean_r > best_mean_reward:
            best_mean_reward = mean_r
            torch.save(policy.state_dict(), best_policy_path)

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

    # Save final policy.
    final_path = os.path.join(output_dir, 'policy_final.pt')
    torch.save(policy.state_dict(), final_path)
    print(f"Training complete. Final policy saved to {final_path}")
    print(f"Best policy saved to {best_policy_path}")

    # Save leet command.
    leet_cmd = tracker.leet_command()
    save_json(os.path.join(output_dir, 'wandb_leet_command.txt'), {'command': leet_cmd})
    print(f"W&B leet command: {leet_cmd}")

    return policy
