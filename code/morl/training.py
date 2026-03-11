"""
MORL training loop using REINFORCE with a per-batch mean-return baseline.

No gradients flow into the frozen GNN embeddings.
Only the ConditionalPolicy parameters are updated.
"""

import math
import torch
import torch.optim as optim
import os
from typing import List, Optional, Dict, Any

from .environment import RecommendationEnv, build_candidate_pools
from .policy import ConditionalPolicy, sample_weight_vector


def run_episode(
    env: RecommendationEnv,
    policy: ConditionalPolicy,
    user_id: int,
    weight: torch.Tensor,
    num_candidates: int,
    device: torch.device,
):
    """Roll out one K-step episode for a single user conditioned on *weight*.

    Returns
    -------
    log_probs : List[torch.Tensor]   per-step log π(a_t | s_t, w)
    rewards   : torch.Tensor  shape (K, 3)  multi-objective reward vectors
    """
    state = env.reset(user_id).to(device)
    log_probs: List[torch.Tensor] = []
    reward_list: List[torch.Tensor] = []

    while True:
        remaining = env.remaining
        if not remaining:
            break

        action, log_prob = policy.select_action(
            state, weight, remaining, num_candidates
        )
        state, reward, done = env.step(action)
        state = state.to(device)

        log_probs.append(log_prob)
        reward_list.append(reward.to(device))

        if done:
            break

    rewards = torch.stack(reward_list)  # (T, 3)
    return log_probs, rewards


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
    checkpoint_dir: str = '.',
    checkpoint_every: int = 10,
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
    checkpoint_dir : directory to save policy checkpoints.
    checkpoint_every : save every N epochs.
    device : compute device.

    Returns
    -------
    policy : trained ConditionalPolicy
    """
    dev = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ----- Build candidate pools (fixed for entire RL training) -----
    print("Building candidate pools ...")
    pools = build_candidate_pools(
        user_emb, item_emb, M=M,
        exclude_per_user=exclude_per_user_train,
        device=dev,
    )
    # Restrict to train users only
    train_pools = {u: pools[u] for u in train_user_ids if u in pools}

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
    policy = ConditionalPolicy(
        state_dim=state_dim,
        num_candidates=M,
        hidden_dim=hidden_dim,
    ).to(dev)

    optimizer = optim.Adam(policy.parameters(), lr=lr)

    stats: List[Dict[str, Any]] = []

    print(f"Starting MORL training: {num_epochs} epochs, batch_size={batch_size}, K={K}, M={M}")

    for epoch in range(1, num_epochs + 1):
        policy.train()

        # Sample a random batch of users
        batch_users = torch.randperm(len(train_user_ids))[:batch_size].tolist()
        batch_users = [train_user_ids[i] for i in batch_users]

        all_log_probs: List[List[torch.Tensor]] = []
        all_returns: List[float] = []

        for user_id in batch_users:
            w = sample_weight_vector(batch_size=1, device=dev)  # (3,)
            log_probs, rewards = run_episode(
                env, policy, user_id, w, M, dev
            )
            # Scalar episodic return: R = Σ_t  w · r_t
            R = torch.sum(rewards @ w).item()
            all_log_probs.append(log_probs)
            all_returns.append(R)

        # REINFORCE baseline: mean return over the batch
        baseline = sum(all_returns) / len(all_returns)

        policy_loss_terms: List[torch.Tensor] = []
        for log_probs, R in zip(all_log_probs, all_returns):
            advantage = R - baseline
            episode_loss = -sum(log_probs) * advantage
            policy_loss_terms.append(episode_loss)

        policy_loss = sum(policy_loss_terms) / len(batch_users)
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        epoch_stats = {
            'epoch': epoch,
            'policy_loss': policy_loss.item(),
            'mean_return': baseline,
        }
        stats.append(epoch_stats)

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:4d} | loss={policy_loss.item():.4f} | "
                f"mean_return={baseline:.4f}"
            )

        if epoch % checkpoint_every == 0:
            ckpt_path = os.path.join(checkpoint_dir, f'morl_policy_epoch{epoch}.pt')
            torch.save(
                {'epoch': epoch,
                 'policy_state_dict': policy.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'stats': stats},
                ckpt_path,
            )

    # Final checkpoint
    final_path = os.path.join(checkpoint_dir, 'morl_policy_final.pt')
    torch.save(
        {'epoch': num_epochs,
         'policy_state_dict': policy.state_dict(),
         'optimizer_state_dict': optimizer.state_dict(),
         'stats': stats},
        final_path,
    )
    print(f"Training complete. Final checkpoint saved to {final_path}")
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
    weight: torch.Tensor,
    exclude_per_user: Optional[Dict[int, set]] = None,
    K: int = 20,
    M: int = 200,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Evaluate a trained MORL policy on *eval_user_ids* using weight *w*.

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

    w = weight.to(dev)

    ndcg_list, health_list, div_list, recall_list = [], [], [], []

    with torch.no_grad():
        for user_id in eval_user_ids:
            if user_id not in eval_pools:
                continue
            state = env.reset(user_id).to(dev)

            while True:
                remaining = env.remaining
                if not remaining:
                    break
                action, _ = policy.select_action(
                    state, w, remaining, M, greedy=True
                )
                state, _, done = env.step(action)
                state = state.to(dev)
                if done:
                    break

            rec_list = env.selected  # list of item indices

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
