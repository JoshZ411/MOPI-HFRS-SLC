"""
MORL training loop using objective-wise REINFORCE with MGDA aggregation.

No gradients flow into the frozen GNN embeddings.
Only the ConditionalPolicy parameters are updated.
"""

import logging
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import torch
import torch.optim as optim

from .environment import RecommendationEnv, build_candidate_pools
from .logging_utils import append_jsonl
from .policy import ConditionalPolicy
from min_norm_solvers import MinNormSolver, gradient_normalizers


def run_episode(
    env: RecommendationEnv,
    policy: ConditionalPolicy,
    user_id: int,
    device: torch.device,
):
    """Roll out one K-step episode for a single user.

    Returns
    -------
    log_probs : List[torch.Tensor]   per-step log π(a_t | s_t, w)
    rewards   : Dict[str, torch.Tensor] objective-wise reward sequences
    entropy_terms : List[torch.Tensor] per-step normalized entropy values
    diagnostics : dict
    """
    state = env.reset(user_id).to(device)
    log_probs: List[torch.Tensor] = []
    reward_pref: List[torch.Tensor] = []
    reward_health: List[torch.Tensor] = []
    reward_div: List[torch.Tensor] = []
    entropy_terms: List[torch.Tensor] = []
    entropies: List[float] = []
    normalized_entropies: List[float] = []
    selected_positions: List[int] = []
    selected_probs: List[float] = []
    max_probs: List[float] = []

    while True:
        remaining = env.remaining
        if not remaining:
            break

        candidate_embeddings = env.item_emb[remaining]

        action, log_prob, normalized_entropy, info = cast(
            Tuple[int, torch.Tensor, torch.Tensor, Dict[str, float]],
            policy.select_action(state, candidate_embeddings, return_info=True),
        )
        state, reward, done = env.step(action)
        state = state.to(device)

        log_probs.append(log_prob)
        reward_pref.append(reward[0].to(device))
        reward_health.append(reward[1].to(device))
        reward_div.append(reward[2].to(device))
        entropy_terms.append(normalized_entropy)
        entropies.append(info['entropy'])
        normalized_entropies.append(info['normalized_entropy'])
        selected_positions.append(action)
        selected_probs.append(info['selected_prob'])
        max_probs.append(info['max_prob'])

        if done:
            break

    rewards = {
        'pref': torch.stack(reward_pref),
        'health': torch.stack(reward_health),
        'div': torch.stack(reward_div),
    }
    diagnostics = {
        'episode_length': len(reward_pref),
        'mean_entropy': sum(entropies) / len(entropies) if entropies else 0.0,
        'mean_normalized_entropy': sum(normalized_entropies) / len(normalized_entropies) if normalized_entropies else 0.0,
        'selected_positions': selected_positions,
        'mean_selected_prob': sum(selected_probs) / len(selected_probs) if selected_probs else 0.0,
        'mean_max_prob': sum(max_probs) / len(max_probs) if max_probs else 0.0,
    }
    return log_probs, rewards, entropy_terms, diagnostics


def _safe_mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _safe_std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean_value = _safe_mean(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def _grad_norm(policy: ConditionalPolicy) -> float:
    total = 0.0
    for param in policy.parameters():
        if param.grad is None:
            continue
        grad_value = param.grad.detach().data.norm(2).item()
        total += grad_value ** 2
    return math.sqrt(total)


def summarize_candidate_pools(candidate_pools: Dict[int, List[int]], K: int) -> Dict[str, float]:
    sizes = [len(items) for items in candidate_pools.values()]
    if not sizes:
        return {
            'pool_users': 0.0,
            'pool_size_mean': 0.0,
            'pool_size_min': 0.0,
            'pool_size_max': 0.0,
            'pool_users_below_k': 0.0,
        }
    below_k = sum(size < K for size in sizes)
    return {
        'pool_users': float(len(sizes)),
        'pool_size_mean': _safe_mean(sizes),
        'pool_size_min': float(min(sizes)),
        'pool_size_max': float(max(sizes)),
        'pool_users_below_k': float(below_k),
    }


def _discounted_returns(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    returns = torch.zeros_like(rewards)
    running = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    for idx in range(rewards.size(0) - 1, -1, -1):
        running = rewards[idx] + gamma * running
        returns[idx] = running
    return returns


def _normalize_returns(returns: torch.Tensor) -> torch.Tensor:
    if returns.numel() <= 1:
        return torch.zeros_like(returns)
    mean = returns.mean()
    std = returns.std(unbiased=False)
    return (returns - mean) / std.clamp_min(1e-8)


def _collect_flat_gradients(policy: ConditionalPolicy) -> torch.Tensor:
    grads: List[torch.Tensor] = []
    for param in policy.parameters():
        if param.grad is None:
            grads.append(torch.zeros_like(param, memory_format=torch.contiguous_format).view(-1).cpu())
        else:
            grads.append(param.grad.detach().view(-1).cpu())
    return torch.cat(grads)


def train_morl(
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    user_tags: torch.Tensor,
    item_tags: torch.Tensor,
    train_user_ids: List[int],
    val_user_ids: List[int],
    exclude_per_user_train: Optional[Dict[int, set]] = None,
    exclude_per_user_val: Optional[Dict[int, set]] = None,
    K: int = 20,
    M: int = 200,
    hidden_dim: int = 256,
    num_epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    gamma: float = 1.0,
    checkpoint_dir: str = '.',
    checkpoint_every: int = 10,
    log_every: int = 10,
    metrics_path: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    tracker: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> ConditionalPolicy:
    """Train the MORL conditional policy.

    Parameters
    ----------
    user_emb, item_emb : frozen embeddings from SGSL training.
    user_tags, item_tags : binary health-tag tensors.
    train_user_ids : user indices used for RL training episodes.
    val_user_ids : user indices used for validation (held out from RL training).
    exclude_per_user_train : dict mapping user_id → set of item indices to mask
        from candidate pools during RL training (optional; if None, full top-M
        is used without exclusion as per design choice).
    exclude_per_user_val : dict mapping user_id → set of item indices to mask
        during validation ranking (training + val positives excluded).
    K : recommendation list length.
    M : candidate pool size.
    hidden_dim : policy hidden layer width.
    num_epochs : number of training epochs.
    batch_size : users per gradient step.
    lr : Adam learning rate.
    gamma : discount factor used to build reward-to-go returns.
    checkpoint_dir : directory to save policy checkpoints.
    checkpoint_every : save every N epochs.
    device : compute device.

    Returns
    -------
    policy : trained ConditionalPolicy
    """
    dev = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = logger or logging.getLogger(__name__)

    # ----- Build candidate pools (fixed for entire RL training) -----
    logger.info('Building candidate pools for MORL training ...')
    pools = build_candidate_pools(
        user_emb, item_emb, M=M,
        exclude_per_user=exclude_per_user_train,
        device=dev,
    )
    # Restrict to train users only
    train_pools = {u: pools[u] for u in train_user_ids if u in pools}
    pool_stats = summarize_candidate_pools(train_pools, K=K)
    logger.info(
        'Candidate pools ready: users=%d mean_size=%.2f min=%d max=%d below_K=%d',
        int(pool_stats['pool_users']),
        pool_stats['pool_size_mean'],
        int(pool_stats['pool_size_min']),
        int(pool_stats['pool_size_max']),
        int(pool_stats['pool_users_below_k']),
    )
    if pool_stats['pool_users_below_k'] > 0:
        logger.warning(
            '%d users have candidate pools smaller than K=%d; those episodes may terminate early.',
            int(pool_stats['pool_users_below_k']),
            K,
        )

    # ----- Instantiate environment -----
    env = RecommendationEnv(
        user_emb=user_emb,
        item_emb=item_emb,
        user_tags=user_tags,
        item_tags=item_tags,
        candidate_pools=train_pools,
        K=K,
        device=dev,
    )

    # ----- Build policy -----
    d = user_emb.size(1)
    tag_dim = user_tags.size(1)
    state_dim = 2 * d + tag_dim + 1
    candidate_dim = item_emb.size(1)
    policy = ConditionalPolicy(
        state_dim=state_dim,
        candidate_dim=candidate_dim,
        hidden_dim=hidden_dim,
    ).to(dev)

    optimizer = optim.Adam(policy.parameters(), lr=lr)

    stats: List[Dict[str, Any]] = []

    logger.info(
        'Starting MORL training: epochs=%d batch_size=%d K=%d M=%d lr=%.4g hidden_dim=%d gamma=%.4f',
        num_epochs,
        batch_size,
        K,
        M,
        lr,
        hidden_dim,
        gamma,
    )

    final_epoch = 0

    for epoch in range(1, num_epochs + 1):
        policy.train()

        # Sample a random batch of users
        batch_users = torch.randperm(len(train_user_ids))[:batch_size].tolist()
        batch_users = [train_user_ids[i] for i in batch_users]

        all_log_probs: List[List[torch.Tensor]] = []
        all_returns_pref: List[float] = []
        all_returns_health: List[float] = []
        all_returns_div: List[float] = []
        reward_component_means: List[List[float]] = []
        episode_lengths: List[float] = []
        entropies: List[float] = []
        normalized_entropies: List[float] = []
        selected_positions: List[int] = []
        selected_probs: List[float] = []
        max_probs: List[float] = []
        loss_pref_terms: List[torch.Tensor] = []
        loss_health_terms: List[torch.Tensor] = []
        loss_div_terms: List[torch.Tensor] = []

        for user_id in batch_users:
            log_probs, rewards, entropy_terms, episode_diag = run_episode(
                env, policy, user_id, dev
            )
            all_log_probs.append(log_probs)

            returns_pref = _normalize_returns(_discounted_returns(rewards['pref'], gamma))
            returns_health = _normalize_returns(_discounted_returns(rewards['health'], gamma))
            returns_div = _normalize_returns(_discounted_returns(rewards['div'], gamma))

            log_prob_tensor = torch.stack(log_probs)
            loss_pref_terms.append(-(log_prob_tensor * returns_pref.detach()).sum())
            loss_health_terms.append(-(log_prob_tensor * returns_health.detach()).sum())
            loss_div_terms.append(-(log_prob_tensor * returns_div.detach()).sum())

            all_returns_pref.append(float(returns_pref.mean().detach().cpu()))
            all_returns_health.append(float(returns_health.mean().detach().cpu()))
            all_returns_div.append(float(returns_div.mean().detach().cpu()))
            reward_component_means.append([
                float(rewards['pref'].mean().detach().cpu()),
                float(rewards['health'].mean().detach().cpu()),
                float(rewards['div'].mean().detach().cpu()),
            ])
            episode_lengths.append(float(episode_diag['episode_length']))
            entropies.append(episode_diag['mean_entropy'])
            normalized_entropies.append(episode_diag['mean_normalized_entropy'])
            selected_positions.extend(episode_diag['selected_positions'])
            selected_probs.append(episode_diag['mean_selected_prob'])
            max_probs.append(episode_diag['mean_max_prob'])

        loss_pref = torch.stack(loss_pref_terms).mean()
        loss_health = torch.stack(loss_health_terms).mean()
        loss_div = torch.stack(loss_div_terms).mean()

        losses = {
            'pref': loss_pref,
            'health': loss_health,
            'div': loss_div,
        }
        grads: Dict[str, List[torch.Tensor]] = {}
        optimizer.zero_grad()
        for task_name, task_loss in losses.items():
            task_loss.backward(retain_graph=True)
            grads[task_name] = [_collect_flat_gradients(policy)]
            policy.zero_grad()

        gn = gradient_normalizers(grads, losses, 'l2')
        for task_name in grads:
            grads[task_name][0] = grads[task_name][0] / gn[task_name].to(grads[task_name][0].device)

        solver = getattr(MinNormSolver, 'find_min_norm_element_FW')
        alpha_array, _ = solver([
            grads['pref'],
            grads['health'],
            grads['div'],
        ])
        alpha_pref, alpha_health, alpha_div = [float(value) for value in alpha_array]
        policy_loss = alpha_pref * loss_pref + alpha_health * loss_health + alpha_div * loss_div

        optimizer.zero_grad()
        policy_loss.backward()
        grad_norm = _grad_norm(policy)
        optimizer.step()

        reward_pref = [reward[0] for reward in reward_component_means]
        reward_health = [reward[1] for reward in reward_component_means]
        reward_div = [reward[2] for reward in reward_component_means]

        epoch_stats = {
            'epoch': epoch,
            'policy_loss': policy_loss.item(),
            'loss_pref': loss_pref.item(),
            'loss_health': loss_health.item(),
            'loss_div': loss_div.item(),
            'mean_return': _safe_mean(all_returns_pref + all_returns_health + all_returns_div),
            'std_return': _safe_std(all_returns_pref + all_returns_health + all_returns_div),
            'mean_return_pref': _safe_mean(all_returns_pref),
            'mean_return_health': _safe_mean(all_returns_health),
            'mean_return_div': _safe_mean(all_returns_div),
            'std_return_pref': _safe_std(all_returns_pref),
            'std_return_health': _safe_std(all_returns_health),
            'std_return_div': _safe_std(all_returns_div),
            'mean_episode_length': _safe_mean(episode_lengths),
            'mean_reward_pref': _safe_mean(reward_pref),
            'mean_reward_health': _safe_mean(reward_health),
            'mean_reward_div': _safe_mean(reward_div),
            'std_reward_pref': _safe_std(reward_pref),
            'std_reward_health': _safe_std(reward_health),
            'std_reward_div': _safe_std(reward_div),
            'mean_entropy': _safe_mean(entropies),
            'mean_normalized_entropy': _safe_mean(normalized_entropies),
            'mean_selected_prob': _safe_mean(selected_probs),
            'mean_max_prob': _safe_mean(max_probs),
            'mean_action_position': _safe_mean([float(position) for position in selected_positions]),
            'std_action_position': _safe_std([float(position) for position in selected_positions]),
            'grad_norm': grad_norm,
            'alpha_pref_mean': alpha_pref,
            'alpha_health_mean': alpha_health,
            'alpha_div_mean': alpha_div,
        }

        if selected_positions:
            top_positions = {f'action_pos_{idx}_rate': 0.0 for idx in range(min(10, M))}
            total_positions = len(selected_positions)
            for idx in range(min(10, M)):
                top_positions[f'action_pos_{idx}_rate'] = (
                    sum(position == idx for position in selected_positions) / total_positions
                )
            epoch_stats.update(top_positions)

        stats.append(epoch_stats)

        if metrics_path is not None:
            append_jsonl(metrics_path, {'type': 'train_epoch', **epoch_stats})

        if tracker is not None:
            tracker.log({f'train/{key}': value for key, value in epoch_stats.items()}, step=epoch)

        if epoch % log_every == 0:
            logger.info(
                'Epoch %4d | loss=%.4f [pref=%.4f health=%.4f div=%.4f] | '
                'return[p/h/d]=%.4f/%.4f/%.4f | reward[p/h/d]=%.4f/%.4f/%.4f | '
                'entropy=%.4f entropy_norm=%.4f grad=%.4f alpha[p/h/d]=%.3f/%.3f/%.3f',
                epoch,
                epoch_stats['policy_loss'],
                epoch_stats['loss_pref'],
                epoch_stats['loss_health'],
                epoch_stats['loss_div'],
                epoch_stats['mean_return_pref'],
                epoch_stats['mean_return_health'],
                epoch_stats['mean_return_div'],
                epoch_stats['mean_reward_pref'],
                epoch_stats['mean_reward_health'],
                epoch_stats['mean_reward_div'],
                epoch_stats['mean_entropy'],
                epoch_stats['mean_normalized_entropy'],
                epoch_stats['grad_norm'],
                epoch_stats['alpha_pref_mean'],
                epoch_stats['alpha_health_mean'],
                epoch_stats['alpha_div_mean'],
            )

        if epoch_stats['mean_entropy'] < 0.05:
            logger.warning('Epoch %d action entropy is very low (%.4f); exploration may have collapsed.', epoch, epoch_stats['mean_entropy'])
        if epoch_stats['mean_normalized_entropy'] < 0.1:
            logger.warning(
                'Epoch %d normalized action entropy is very low (%.4f); the policy is becoming too deterministic over the active action set.',
                epoch,
                epoch_stats['mean_normalized_entropy'],
            )
        if epoch_stats['std_reward_pref'] < 1e-4 and epoch_stats['std_reward_health'] < 1e-4 and epoch_stats['std_reward_div'] < 1e-4:
            logger.warning('Epoch %d reward variance is nearly zero across the batch.', epoch)

        if epoch % checkpoint_every == 0:
            ckpt_path = os.path.join(checkpoint_dir, f'morl_policy_epoch{epoch}.pt')
            torch.save(
                {'epoch': epoch,
                 'policy_state_dict': policy.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'stats': stats},
                ckpt_path,
            )
            logger.info('Saved checkpoint: %s', ckpt_path)

        final_epoch = epoch

    # Final checkpoint
    final_path = os.path.join(checkpoint_dir, 'morl_policy_final.pt')
    torch.save(
        {'epoch': final_epoch if final_epoch > 0 else num_epochs,
         'policy_state_dict': policy.state_dict(),
         'optimizer_state_dict': optimizer.state_dict(),
         'stats': stats},
        final_path,
    )
    logger.info('Training complete. Final checkpoint saved to %s', final_path)
    return policy


# ------------------------------------------------------------------
# Evaluation helpers
# ------------------------------------------------------------------

def evaluate_morl(
    policy: ConditionalPolicy,
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    user_tags: torch.Tensor,
    item_tags: torch.Tensor,
    eval_user_ids: List[int],
    pos_items_per_user: Dict[int, List[int]],
    exclude_per_user: Optional[Dict[int, set]] = None,
    K: int = 20,
    M: int = 200,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Evaluate a trained MORL policy on *eval_user_ids*.

    Returns
    -------
    metrics : dict with keys ndcg, health_score, diversity, recall
    """
    dev = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy.eval()

    pools = build_candidate_pools(
        user_emb, item_emb, M=M,
        exclude_per_user=exclude_per_user,
        device=dev,
    )
    eval_pools = {u: pools[u] for u in eval_user_ids if u in pools}

    env = RecommendationEnv(
        user_emb=user_emb,
        item_emb=item_emb,
        user_tags=user_tags,
        item_tags=item_tags,
        candidate_pools=eval_pools,
        K=K,
        device=dev,
    )

    ndcg_list, health_list, div_list, recall_list = [], [], [], []

    with torch.no_grad():
        for user_id in eval_user_ids:
            if user_id not in eval_pools:
                continue
            rec_list = get_recommendations(policy, env, user_id, K, device=dev)

            # ---- NDCG@K ----
            ground_truth = set(pos_items_per_user.get(user_id, []))
            dcg, idcg = 0.0, 0.0
            for rank, item in enumerate(rec_list[:K], start=1):
                rel = 1.0 if item in ground_truth else 0.0
                dcg += rel / math.log2(rank + 1)
                if rank <= len(ground_truth):
                    idcg += 1.0 / math.log2(rank + 1)
            ndcg_list.append(dcg / idcg if idcg > 0 else 0.0)

            # ---- Recall@K ----
            hits = len(set(rec_list[:K]) & ground_truth)
            recall_list.append(hits / len(ground_truth) if ground_truth else 0.0)

            # ---- Health score ----
            user_tag_vec = user_tags[user_id].float()
            healthy_count = 0
            for item in rec_list[:K]:
                if torch.any(torch.logical_and(user_tag_vec.bool(),
                                               item_tags[item].bool())).item():
                    healthy_count += 1
            health_list.append(healthy_count / K)

            # ---- Diversity (mean pairwise 1 - cosine_sim) ----
            if len(rec_list) >= 2:
                rec_embs = item_emb[rec_list[:K]].to(dev)  # (k, d)
                rec_embs_norm = torch.nn.functional.normalize(rec_embs, dim=1)
                sim_mat = rec_embs_norm @ rec_embs_norm.T  # (k, k)
                k = rec_embs.size(0)
                idx = torch.triu_indices(k, k, offset=1)
                mean_sim = sim_mat[idx[0], idx[1]].mean().item()
                div_list.append(1.0 - mean_sim)
            else:
                div_list.append(0.0)

    def _mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    return {
        'ndcg': _mean(ndcg_list),
        'health_score': _mean(health_list),
        'diversity': _mean(div_list),
        'recall': _mean(recall_list),
    }


def get_recommendations(
    policy: ConditionalPolicy,
    env: RecommendationEnv,
    user_id: int,
    K: int,
    device: Optional[torch.device] = None,
) -> List[int]:
    """Greedily decode a recommendation list for one user."""
    dev = device or env.device
    state = env.reset(user_id).to(dev)

    while len(env.selected) < K:
        remaining = env.remaining
        if not remaining:
            break
        candidate_embeddings = env.item_emb[remaining]
        action, _ = cast(
            Tuple[int, torch.Tensor],
            policy.select_action(state, candidate_embeddings, greedy=True),
        )
        state, _, done = env.step(action)
        state = state.to(dev)
        if done:
            break

    return list(env.selected)
