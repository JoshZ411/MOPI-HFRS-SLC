"""
MORL Training Loop — REINFORCE with baseline for conditional policy π(a | s, w).

Algorithm (per gradient step):
    1. Sample batch of B users.
    2. For each user:
        a. Sample weight vector w ~ Dirichlet(1,1,1).
        b. Reset environment → initial state s_0.
        c. Run K-step episode: collect {s_t, a_t, log_π(a_t|s_t,w), r_t}.
        d. Compute scalar episodic return: R = Σ_t  w · r_t.
    3. Compute REINFORCE loss:
           L = -Σ_{user} Σ_t log_π(a_t | s_t, w) * (R - baseline)
       where baseline = running mean of R.
    4. Back-propagate and update policy parameters.

No gradients ever flow into frozen GNN embeddings.
"""

import torch
import torch.optim as optim
import random
from typing import List, Dict, Optional
import math

from .environment import RecommendationEnv
from .policy import MORLPolicy, sample_weight_vector


class RunningMean:
    """Simple scalar running mean used as REINFORCE baseline."""

    def __init__(self, momentum: float = 0.99):
        self.momentum = momentum
        self.value = 0.0
        self._initialized = False

    def update(self, x: float):
        if not self._initialized:
            self.value = x
            self._initialized = True
        else:
            self.value = self.momentum * self.value + (1 - self.momentum) * x

    def get(self) -> float:
        return self.value


def run_episode(env: RecommendationEnv, policy: MORLPolicy,
                user_id: int, weight: torch.Tensor, device=None):
    """Run one K-step episode for a single user with a fixed weight vector.

    Returns
    -------
    log_probs   : List[Tensor scalar]  — log π(a_t | s_t, w) per step
    rewards     : List[Tensor (3,)]    — reward vector per step
    """
    state = env.reset(user_id)
    log_probs = []
    rewards = []
    done = False

    while not done:
        cand_embs = env.get_candidate_embeddings()  # (|pool|, d)
        if cand_embs.shape[0] == 0:
            break

        action_idx, log_prob = policy.select_action(state, weight, cand_embs)
        next_state, reward_vec, done = env.step(action_idx)

        log_probs.append(log_prob)
        rewards.append(reward_vec)
        state = next_state

    return log_probs, rewards


def compute_scalar_return(rewards: List[torch.Tensor], weight: torch.Tensor) -> torch.Tensor:
    """Compute scalarized episodic return R = Σ_t w · r_t."""
    total = torch.zeros(1, device=weight.device)
    for r in rewards:
        total = total + (weight * r).sum()
    return total.squeeze()


def train_morl(
    env: RecommendationEnv,
    policy: MORLPolicy,
    train_user_ids: List[int],
    num_episodes: int = 2000,
    batch_size: int = 32,
    lr: float = 1e-3,
    dirichlet_alpha: float = 1.0,
    grad_clip: float = 1.0,
    log_interval: int = 100,
    checkpoint_path: str = 'morl_policy.pt',
    device=None,
) -> Dict:
    """Train conditional MORL policy using REINFORCE.

    Parameters
    ----------
    env             : RecommendationEnv
    policy          : MORLPolicy (on *device*)
    train_user_ids  : list of user ids used for training
    num_episodes    : total gradient-update steps
    batch_size      : users sampled per gradient step
    lr              : Adam learning rate
    dirichlet_alpha : Dirichlet concentration parameter for weight sampling
    grad_clip       : max gradient norm
    log_interval    : print stats every N episodes
    checkpoint_path : where to save best policy checkpoint
    device          : torch.device

    Returns
    -------
    stats : dict with training history
    """
    device = device or torch.device('cpu')
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    baseline = RunningMean(momentum=0.99)

    stats = {
        'policy_loss': [],
        'mean_return': [],
        'mean_r_pref': [],
        'mean_r_health': [],
        'mean_r_div': [],
    }

    best_return = -math.inf

    for episode in range(1, num_episodes + 1):
        # Sample batch of users
        batch_users = random.choices(train_user_ids, k=batch_size)

        # Sample one weight vector per user
        weights = sample_weight_vector(batch_size=batch_size, alpha=dirichlet_alpha, device=device)

        episode_returns = []
        episode_log_probs_list = []
        per_obj_rewards = [[], [], []]

        for i, uid in enumerate(batch_users):
            w = weights[i]  # (3,)
            log_probs, rewards = run_episode(env, policy, uid, w, device=device)
            if not log_probs:
                continue
            R = compute_scalar_return(rewards, w)
            episode_returns.append(R)
            episode_log_probs_list.append(log_probs)

            # Per-objective accumulation for logging
            for r in rewards:
                for obj in range(3):
                    per_obj_rewards[obj].append(r[obj].item())

        if not episode_returns:
            continue

        mean_R = torch.stack(episode_returns).mean()
        baseline.update(mean_R.item())
        b = baseline.get()

        # REINFORCE loss: L = -Σ_{user} Σ_t log_π(a_t|s_t,w) * (R - b)
        loss_terms = []
        for R, lp_list in zip(episode_returns, episode_log_probs_list):
            advantage = (R - b).detach()
            for lp in lp_list:
                loss_terms.append(-lp * advantage)
        loss = torch.stack(loss_terms).sum() / max(len(episode_returns), 1)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
        optimizer.step()

        # Logging
        mean_R_val = mean_R.item()
        stats['policy_loss'].append(loss.item())
        stats['mean_return'].append(mean_R_val)
        stats['mean_r_pref'].append(
            sum(per_obj_rewards[0]) / max(len(per_obj_rewards[0]), 1))
        stats['mean_r_health'].append(
            sum(per_obj_rewards[1]) / max(len(per_obj_rewards[1]), 1))
        stats['mean_r_div'].append(
            sum(per_obj_rewards[2]) / max(len(per_obj_rewards[2]), 1))

        if mean_R_val > best_return:
            best_return = mean_R_val
            torch.save({
                'episode': episode,
                'policy_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_return': best_return,
            }, checkpoint_path)

        if episode % log_interval == 0:
            print(
                f"Episode {episode:5d} | loss: {loss.item():8.4f} | "
                f"mean_R: {mean_R_val:8.4f} | baseline: {b:8.4f} | "
                f"r_pref: {stats['mean_r_pref'][-1]:6.4f} | "
                f"r_health: {stats['mean_r_health'][-1]:6.4f} | "
                f"r_div: {stats['mean_r_div'][-1]:6.4f}"
            )

    print(f"\nTraining complete. Best return: {best_return:.4f}")
    print(f"Policy checkpoint saved to {checkpoint_path}")
    return stats
