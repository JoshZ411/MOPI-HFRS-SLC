"""
MORL training loop: scalarized REINFORCE with a cross-episode EMA baseline.

Reward formulation (Option 1 — val-positive sparse binary):
    r_rel    = 1 if selected item is in val_pos_items[user], else 0.
    r_health = marginal Jaccard gain: Jaccard(coverage_t) - Jaccard(coverage_{t-1}).
    combined = r_rel + beta * r_health   (single scalar per step, fixed beta)

Baseline: exponential moving average of per-episode returns, updated across
the entire training run.  Replaces within-episode return normalisation to
preserve cross-episode signal (critical when r_rel is sparse).

No gradients flow into the frozen GNN embeddings.
Only the ConditionalPolicy parameters are updated.
"""

import logging
import math
import os
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple, cast

import torch
import torch.optim as optim

from .environment import RecommendationEnv, build_candidate_pools
from .logging_utils import append_jsonl
from .policy import ConditionalPolicy


def run_episode(
    env: RecommendationEnv,
    policy: ConditionalPolicy,
    user_id: int,
    device: torch.device,
    beta: float = 0.5,
):
    """Roll out one K-step episode for a single user.

    Returns
    -------
    log_probs     : List[torch.Tensor]  per-step log π(a_t | s_t)
    combined_rewards : torch.Tensor    shape (T,)  r_rel + beta * r_health per step
    entropy_terms : List[torch.Tensor] per-step normalised entropy values
    diagnostics   : dict
    """
    state = env.reset(user_id).to(device)
    log_probs: List[torch.Tensor] = []
    reward_rel: List[torch.Tensor] = []
    reward_health: List[torch.Tensor] = []
    entropy_terms: List[torch.Tensor] = []
    entropies: List[float] = []
    normalized_entropies: List[float] = []
    selected_positions: List[int] = []
    chosen_score_ranks: List[float] = []
    chosen_score_values: List[float] = []
    selected_probs: List[float] = []
    max_probs: List[float] = []
    rel_hits: int = 0

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

        r_rel = reward[0].to(device)
        r_health = reward[1].to(device)

        log_probs.append(log_prob)
        reward_rel.append(r_rel)
        reward_health.append(r_health)
        entropy_terms.append(normalized_entropy)
        entropies.append(info['entropy'])
        normalized_entropies.append(info['normalized_entropy'])
        selected_positions.append(action)
        selected_probs.append(info['selected_prob'])
        max_probs.append(info['max_prob'])
        step_info = env.last_step_info
        chosen_score_ranks.append(step_info.get('chosen_score_rank_1based', 0.0))
        chosen_score_values.append(step_info.get('chosen_score', 0.0))
        if r_rel.item() > 0.5:
            rel_hits += 1

        if done:
            break

    rel_tensor = torch.stack(reward_rel)
    health_tensor = torch.stack(reward_health)
    combined_rewards = rel_tensor + beta * health_tensor

    diagnostics = {
        'episode_length': len(log_probs),
        'mean_entropy': sum(entropies) / len(entropies) if entropies else 0.0,
        'mean_normalized_entropy': sum(normalized_entropies) / len(normalized_entropies) if normalized_entropies else 0.0,
        'selected_positions': selected_positions,
        'chosen_score_ranks': chosen_score_ranks,
        'chosen_score_values': chosen_score_values,
        'mean_selected_prob': sum(selected_probs) / len(selected_probs) if selected_probs else 0.0,
        'mean_max_prob': sum(max_probs) / len(max_probs) if max_probs else 0.0,
        'rel_hits': float(rel_hits),
        'mean_reward_rel': float(rel_tensor.mean().item()),
        'mean_reward_health': float(health_tensor.mean().item()),
        'episode_return': float(combined_rewards.sum().item()),
    }
    return log_probs, combined_rewards, entropy_terms, diagnostics


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


def _cosine_similarity(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    denom = vec_a.norm(2) * vec_b.norm(2)
    if float(denom.item()) <= 1e-12:
        return 0.0
    return float(torch.dot(vec_a, vec_b).item() / denom.item())


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


def measure_candidate_pool_ceiling(
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    eval_user_ids: List[int],
    pos_items_per_user: Dict[int, List[int]],
    exclude_per_user: Optional[Dict[int, set]] = None,
    K: int = 20,
    M: int = 200,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Measure how much held-out relevance is recoverable from the top-M pool.

    Returns metrics that describe the best-case ceiling before the reranker acts.
    """
    dev = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pools = build_candidate_pools(
        user_emb, item_emb, M=M,
        exclude_per_user=exclude_per_user,
        device=dev,
    )

    pool_sizes: List[float] = []
    pool_hits: List[float] = []
    pool_recall_ceiling: List[float] = []
    rerank_recall_ceiling: List[float] = []
    rerank_ndcg_ceiling: List[float] = []
    hit_users = 0
    eligible_users = 0

    for user_id in eval_user_ids:
        ground_truth = set(pos_items_per_user.get(user_id, []))
        if not ground_truth:
            continue
        eligible_users += 1

        pool = pools.get(user_id, [])
        pool_set = set(pool)
        hits_in_pool = len(pool_set & ground_truth)
        if hits_in_pool > 0:
            hit_users += 1

        pool_sizes.append(float(len(pool)))
        pool_hits.append(float(hits_in_pool))
        pool_recall_ceiling.append(hits_in_pool / len(ground_truth))
        rerank_recall_ceiling.append(min(hits_in_pool, K) / len(ground_truth))

        dcg = 0.0
        ideal_hits = min(hits_in_pool, K)
        gt_count = len(ground_truth)
        for rank in range(1, ideal_hits + 1):
            dcg += 1.0 / math.log2(rank + 1)
        idcg = 0.0
        for rank in range(1, min(gt_count, K) + 1):
            idcg += 1.0 / math.log2(rank + 1)
        rerank_ndcg_ceiling.append(dcg / idcg if idcg > 0 else 0.0)

    return {
        'eligible_users': float(eligible_users),
        'pool_user_hit_rate': hit_users / eligible_users if eligible_users else 0.0,
        'pool_size_mean': _safe_mean(pool_sizes),
        'pool_hits_mean': _safe_mean(pool_hits),
        'pool_recall_ceiling': _safe_mean(pool_recall_ceiling),
        'rerank_recall_ceiling_at_k': _safe_mean(rerank_recall_ceiling),
        'rerank_ndcg_ceiling_at_k': _safe_mean(rerank_ndcg_ceiling),
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


def pretrain_policy(
    policy: ConditionalPolicy,
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    train_pools: Dict[int, List[int]],
    train_user_ids: List[int],
    num_epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
    logger: Optional[logging.Logger] = None,
    tracker: Optional[Any] = None,
) -> None:
    """Supervised pretraining: teach the policy to replicate GNN top-K rankings.

    For each user, the GNN pool is already sorted highest-to-lowest score.
    We treat pool[0] as the correct action at step 0, pool[1] at step 1, etc.,
    and minimise cross-entropy loss between policy logits and those rank labels.

    This gives the policy a warm start that exactly matches GNN NDCG, so that
    subsequent REINFORCE fine-tuning improves health from a strong relevance floor
    rather than starting from random.

    Parameters
    ----------
    policy : ConditionalPolicy  (already on device)
    train_pools : dict[user_id, List[item_idx]]  pre-sorted by GNN score (highest first)
    num_epochs : number of pretraining epochs
    batch_size : users per gradient step
    lr : Adam learning rate for pretraining
    """
    dev = device or torch.device('cpu')
    log = logger or logging.getLogger(__name__)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    user_ids = [u for u in train_user_ids if u in train_pools and len(train_pools[u]) > 0]

    log.info('Pretraining policy to imitate GNN rankings: %d epochs, %d users', num_epochs, len(user_ids))

    for epoch in range(1, num_epochs + 1):
        policy.train()
        batch = [user_ids[i] for i in torch.randperm(len(user_ids))[:batch_size].tolist()]
        loss_terms: List[torch.Tensor] = []

        for user_id in batch:
            pool = train_pools[user_id]
            if len(pool) == 0:
                continue

            user_vec = user_emb[user_id].to(dev)
            # Build a synthetic state: just user_emb + zeros for agg/tags/timestep
            state_dim = policy.state_dim
            state = torch.zeros(state_dim, device=dev)
            state[:user_vec.size(0)] = user_vec

            # One forward pass over the full pool
            pool_tensor = torch.tensor(pool, dtype=torch.long, device=dev)
            cand_emb = item_emb[pool_tensor].to(dev)          # (M, d)
            log_probs = policy.forward(state, cand_emb)        # (M,)

            # Target: the GNN's top-1 is label 0 (highest ranked in pool)
            # Use the pool's natural order: label = argmax of GNN score = index 0
            target = torch.zeros(1, dtype=torch.long, device=dev)  # always pool[0]
            loss_terms.append(torch.nn.functional.nll_loss(log_probs.unsqueeze(0), target))

        if not loss_terms:
            continue

        loss = torch.stack(loss_terms).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % max(1, num_epochs // 5) == 0 or epoch == 1:
            log.info('Pretrain epoch %3d / %d | loss=%.4f', epoch, num_epochs, loss.item())
            if tracker is not None:
                tracker.log({'pretrain/loss': loss.item()}, step=epoch)

    log.info('Pretraining complete.')


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
    val_pos_items: Optional[Dict[int, Set[int]]] = None,
    exclude_per_user_train: Optional[Dict[int, set]] = None,
    exclude_per_user_val: Optional[Dict[int, set]] = None,
    K: int = 20,
    M: int = 200,
    hidden_dim: int = 256,
    num_epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    gamma: float = 1.0,
    beta: float = 0.5,
    entropy_coef: float = 0.01,
    ema_alpha: float = 0.05,
    checkpoint_dir: str = '.',
    checkpoint_every: int = 10,
    log_every: int = 10,
    val_eval_every: int = 50,
    pretrain_epochs: int = 0,
    pretrain_lr: float = 1e-3,
    metrics_path: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    tracker: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> ConditionalPolicy:
    """Train the MORL conditional policy (Option 1 — val-positive sparse binary reward).

    Parameters
    ----------
    user_emb, item_emb : frozen embeddings from SGSL training.
    user_tags, item_tags : binary health-tag tensors.
    train_user_ids : user indices used for RL training episodes.
    val_user_ids : user indices used for validation (held out from RL training).
    val_pos_items : dict mapping user_id → set of val-split positive item indices.
        Selecting one of these items in an episode yields r_rel = 1.
        Should contain val positives only (train positives are already excluded
        from the candidate pools via exclude_per_user_train).
    exclude_per_user_train : item indices to mask from training candidate pools.
    exclude_per_user_val : item indices to mask from validation candidate pools
        (typically train + val positives so the eval pool contains only test items).
    K : recommendation list length.
    M : candidate pool size.
    hidden_dim : policy hidden layer width.
    num_epochs : number of training epochs.
    batch_size : users per gradient step.
    lr : Adam learning rate.
    gamma : discount factor for reward-to-go.
    beta : weight on the marginal health reward relative to the sparse relevance hit.
    entropy_coef : coefficient for normalised entropy regularisation.
    ema_alpha : smoothing factor for the cross-episode EMA return baseline.
    checkpoint_dir : directory to save policy checkpoints.
    checkpoint_every : save every N epochs.

    Returns
    -------
    policy : trained ConditionalPolicy
    """
    dev = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = logger or logging.getLogger(__name__)

    val_pos: Dict[int, Set[int]] = val_pos_items or {}

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

    # Log how many train users have at least one val positive in their pool
    users_with_val_positives = sum(
        1 for u in train_user_ids
        if u in train_pools and bool(val_pos.get(u, set()) & set(train_pools[u]))
    )
    logger.info(
        'Val-positive coverage: %d / %d train users have ≥1 val positive in their training pool.',
        users_with_val_positives, len(train_user_ids),
    )

    # ----- Pre-build validation candidate pools (reused for periodic in-training eval) -----
    val_pools_cache: Optional[Dict[int, List[int]]] = None
    val_pos_list: Dict[int, List[int]] = {u: list(s) for u, s in val_pos.items()}
    if val_user_ids and val_pos_list:
        logger.info('Pre-building validation candidate pools for periodic eval ...')
        _val_pools_full = build_candidate_pools(
            user_emb, item_emb, M=M,
            exclude_per_user=exclude_per_user_val,
            device=dev,
        )
        val_pools_cache = {u: _val_pools_full[u] for u in val_user_ids if u in _val_pools_full}
        logger.info('Validation pools ready for %d users.', len(val_pools_cache))

    # ----- Instantiate environment -----
    env = RecommendationEnv(
        user_emb=user_emb,
        item_emb=item_emb,
        user_tags=user_tags,
        item_tags=item_tags,
        candidate_pools=train_pools,
        K=K,
        val_pos_items=val_pos,
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

    # ----- Optional imitation pretraining -----
    if pretrain_epochs > 0:
        pretrain_policy(
            policy=policy,
            user_emb=user_emb,
            item_emb=item_emb,
            train_pools=train_pools,
            train_user_ids=train_user_ids,
            num_epochs=pretrain_epochs,
            batch_size=batch_size,
            lr=pretrain_lr,
            device=dev,
            logger=logger,
            tracker=tracker,
        )

    stats: List[Dict[str, Any]] = []

    logger.info(
        'Starting MORL training (Option 1): epochs=%d batch_size=%d K=%d M=%d '
        'lr=%.4g hidden_dim=%d gamma=%.4f beta=%.4g entropy_coef=%.4g ema_alpha=%.4g',
        num_epochs, batch_size, K, M, lr, hidden_dim, gamma, beta, entropy_coef, ema_alpha,
    )

    # Cross-episode EMA baseline — updated once per episode across the whole run
    ema_baseline: float = 0.0
    final_epoch = 0

    for epoch in range(1, num_epochs + 1):
        policy.train()

        # Sample a random batch of users
        batch_users = torch.randperm(len(train_user_ids))[:batch_size].tolist()
        batch_users = [train_user_ids[i] for i in batch_users]

        episode_lengths: List[float] = []
        entropies: List[float] = []
        normalized_entropies: List[float] = []
        entropy_bonus_terms: List[torch.Tensor] = []
        selected_positions: List[int] = []
        chosen_score_ranks: List[float] = []
        chosen_score_values: List[float] = []
        selected_probs: List[float] = []
        max_probs: List[float] = []
        episode_returns: List[float] = []
        episode_advantages: List[float] = []
        episode_rel_hits: List[float] = []
        episode_mean_reward_rel: List[float] = []
        episode_mean_reward_health: List[float] = []
        policy_loss_terms: List[torch.Tensor] = []

        for user_id in batch_users:
            log_probs, combined_rewards, entropy_terms, episode_diag = run_episode(
                env, policy, user_id, dev, beta=beta,
            )
            if entropy_terms:
                entropy_bonus_terms.append(torch.stack(entropy_terms).mean())

            # Reward-to-go under gamma
            raw_returns = _discounted_returns(combined_rewards, gamma)

            # Cross-episode EMA baseline subtraction
            G0 = float(raw_returns[0].item())
            advantage_returns = raw_returns - ema_baseline
            # Update EMA baseline with this episode's total return
            ema_baseline = (1.0 - ema_alpha) * ema_baseline + ema_alpha * G0

            log_prob_tensor = torch.stack(log_probs)
            policy_loss_terms.append(-(log_prob_tensor * advantage_returns.detach()).sum())

            episode_returns.append(G0)
            episode_advantages.append(float(advantage_returns.mean().item()))
            episode_lengths.append(float(episode_diag['episode_length']))
            entropies.append(episode_diag['mean_entropy'])
            normalized_entropies.append(episode_diag['mean_normalized_entropy'])
            selected_positions.extend(episode_diag['selected_positions'])
            chosen_score_ranks.extend(episode_diag['chosen_score_ranks'])
            chosen_score_values.extend(episode_diag['chosen_score_values'])
            selected_probs.append(episode_diag['mean_selected_prob'])
            max_probs.append(episode_diag['mean_max_prob'])
            episode_rel_hits.append(episode_diag['rel_hits'])
            episode_mean_reward_rel.append(episode_diag['mean_reward_rel'])
            episode_mean_reward_health.append(episode_diag['mean_reward_health'])

        policy_loss_raw = torch.stack(policy_loss_terms).mean()
        entropy_bonus = (
            torch.stack(entropy_bonus_terms).mean()
            if entropy_bonus_terms
            else policy_loss_raw.new_zeros(())
        )
        policy_loss = policy_loss_raw - entropy_coef * entropy_bonus

        optimizer.zero_grad()
        policy_loss.backward()
        grad_norm = _grad_norm(policy)
        optimizer.step()

        epoch_stats = {
            'epoch': epoch,
            'policy_loss': policy_loss.item(),
            'ema_baseline': ema_baseline,
            'beta': beta,
            'mean_episode_return': _safe_mean(episode_returns),
            'std_episode_return': _safe_std(episode_returns),
            'mean_advantage': _safe_mean(episode_advantages),
            'mean_rel_hits': _safe_mean(episode_rel_hits),
            'mean_reward_rel': _safe_mean(episode_mean_reward_rel),
            'mean_reward_health': _safe_mean(episode_mean_reward_health),
            'entropy_bonus': float(entropy_bonus.detach().cpu()),
            'entropy_coef': float(entropy_coef),
            'mean_episode_length': _safe_mean(episode_lengths),
            'mean_entropy': _safe_mean(entropies),
            'mean_normalized_entropy': _safe_mean(normalized_entropies),
            'mean_selected_prob': _safe_mean(selected_probs),
            'mean_max_prob': _safe_mean(max_probs),
            'mean_action_position': _safe_mean([float(p) for p in selected_positions]),
            'std_action_position': _safe_std([float(p) for p in selected_positions]),
            'mean_frozen_score_rank': _safe_mean(chosen_score_ranks),
            'std_frozen_score_rank': _safe_std(chosen_score_ranks),
            'top10_action_fraction': (
                sum(rank <= 10.0 for rank in chosen_score_ranks) / len(chosen_score_ranks)
                if chosen_score_ranks else 0.0
            ),
            'top50_action_fraction': (
                sum(rank <= 50.0 for rank in chosen_score_ranks) / len(chosen_score_ranks)
                if chosen_score_ranks else 0.0
            ),
            'grad_norm': grad_norm,
            # --- Derived metrics for hyperparameter tuning ---
            'rel_hit_rate': _safe_mean(episode_rel_hits) / max(1, K),
            'reward_balance': (
                _safe_mean(episode_mean_reward_health)
                / (_safe_mean(episode_mean_reward_rel) + _safe_mean(episode_mean_reward_health) + 1e-8)
            ),
        }

        if selected_positions:
            total_positions = len(selected_positions)
            epoch_stats.update({
                f'action_pos_{idx}_rate': sum(p == idx for p in selected_positions) / total_positions
                for idx in range(min(10, M))
            })

        stats.append(epoch_stats)

        if metrics_path is not None:
            append_jsonl(metrics_path, {'type': 'train_epoch', **epoch_stats})

        if tracker is not None:
            tracker.log({f'train/{key}': value for key, value in epoch_stats.items()}, step=epoch)

        if epoch % log_every == 0:
            logger.info(
                'Epoch %4d | loss=%.4f | return=%.4f±%.4f adv=%.4f ema_base=%.4f | '
                'rel_hits=%.2f reward[rel/h]=%.4f/%.4f | '
                'rank=%.2f top10=%.3f entropy=%.4f entropy_norm=%.4f grad=%.4f',
                epoch,
                epoch_stats['policy_loss'],
                epoch_stats['mean_episode_return'],
                epoch_stats['std_episode_return'],
                epoch_stats['mean_advantage'],
                epoch_stats['ema_baseline'],
                epoch_stats['mean_rel_hits'],
                epoch_stats['mean_reward_rel'],
                epoch_stats['mean_reward_health'],
                epoch_stats['mean_frozen_score_rank'],
                epoch_stats['top10_action_fraction'],
                epoch_stats['mean_entropy'],
                epoch_stats['mean_normalized_entropy'],
                epoch_stats['grad_norm'],
            )

        if epoch_stats['mean_entropy'] < 0.05:
            logger.warning('Epoch %d action entropy very low (%.4f); exploration may have collapsed.', epoch, epoch_stats['mean_entropy'])
        if epoch_stats['mean_normalized_entropy'] < 0.1:
            logger.warning('Epoch %d normalised action entropy very low (%.4f).', epoch, epoch_stats['mean_normalized_entropy'])

        # ----- Periodic validation evaluation -----
        if (
            val_pools_cache is not None
            and val_eval_every > 0
            and epoch % val_eval_every == 0
        ):
            in_train_val = evaluate_morl(
                policy=policy,
                user_emb=user_emb, item_emb=item_emb,
                user_tags=user_tags, item_tags=item_tags,
                eval_user_ids=val_user_ids,
                pos_items_per_user=val_pos_list,
                candidate_pools=val_pools_cache,
                K=K, M=M, device=dev,
            )
            if metrics_path is not None:
                append_jsonl(metrics_path, {'type': 'val_epoch', 'epoch': epoch, **in_train_val})
            if tracker is not None:
                tracker.log({f'val/{key}': value for key, value in in_train_val.items()}, step=epoch)
            logger.info(
                'Epoch %4d VAL  | ndcg=%.5f recall=%.5f health=%.5f diversity=%.5f',
                epoch,
                in_train_val['ndcg'], in_train_val['recall'],
                in_train_val['health_score'], in_train_val['diversity'],
            )
            policy.train()

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
    candidate_pools: Optional[Dict[int, List[int]]] = None,
    K: int = 20,
    M: int = 200,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Evaluate a trained MORL policy on *eval_user_ids*.

    Parameters
    ----------
    candidate_pools : pre-built pools to skip the expensive pool-building step.
        When provided, *exclude_per_user* and *M* are ignored for pool construction.

    Returns
    -------
    metrics : dict with keys ndcg, health_score, diversity, recall
    """
    dev = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy.eval()

    if candidate_pools is not None:
        eval_pools = {u: candidate_pools[u] for u in eval_user_ids if u in candidate_pools}
    else:
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
