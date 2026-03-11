"""
MORL Policy Network — Conditional policy π(a | s, w).

The policy scores each remaining candidate item given:
    - extended state s'_t = concat(s_t, w)   where w ∈ Δ^2 (3-dim weight simplex)
    - candidate item embeddings (|pool| × d)

Architecture:
    1. State encoder: MLP maps s'_t → hidden vector h_s
    2. Score: dot product between h_s and each candidate embedding (after linear projection)
    3. Output: log-softmax over |pool| actions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MORLPolicy(nn.Module):
    """Conditional policy π(a | s, w).

    Parameters
    ----------
    state_dim   : int — dimensionality of s_t (output of env.state_dim())
    item_dim    : int — dimensionality of item embeddings
    hidden_dim  : int — hidden layer size for state encoder
    weight_dim  : int — length of preference weight vector w (default 3)
    dropout     : float — dropout probability in state encoder
    """

    def __init__(self, state_dim: int, item_dim: int, hidden_dim: int = 256,
                 weight_dim: int = 3, dropout: float = 0.1):
        super().__init__()
        self.weight_dim = weight_dim
        input_dim = state_dim + weight_dim  # s'_t = concat(s_t, w)

        # State encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Project item embeddings to same space as encoded state
        self.item_proj = nn.Linear(item_dim, hidden_dim, bias=False)

    def forward(self, state: torch.Tensor, weight: torch.Tensor,
                candidate_embs: torch.Tensor) -> torch.Tensor:
        """Compute log-probabilities over candidate actions.

        Parameters
        ----------
        state         : Tensor (state_dim,) or (B, state_dim)
        weight        : Tensor (weight_dim,) or (B, weight_dim)
        candidate_embs: Tensor (M, item_dim) or (B, M, item_dim)

        Returns
        -------
        log_probs : Tensor (M,) or (B, M)
        """
        batched = state.dim() == 2
        if not batched:
            state = state.unsqueeze(0)          # (1, state_dim)
            weight = weight.unsqueeze(0)        # (1, weight_dim)
            candidate_embs = candidate_embs.unsqueeze(0)  # (1, M, item_dim)

        # s'_t = concat(s_t, w)
        extended_state = torch.cat([state, weight], dim=-1)   # (B, state_dim + weight_dim)
        h_s = self.encoder(extended_state)                     # (B, hidden_dim)

        # Project candidates
        cand_proj = self.item_proj(candidate_embs)             # (B, M, hidden_dim)

        # Score: h_s · cand_proj^T  →  (B, M)
        scores = torch.bmm(cand_proj, h_s.unsqueeze(-1)).squeeze(-1)  # (B, M)
        log_probs = F.log_softmax(scores, dim=-1)              # (B, M)

        if not batched:
            log_probs = log_probs.squeeze(0)                   # (M,)
        return log_probs

    def select_action(self, state: torch.Tensor, weight: torch.Tensor,
                      candidate_embs: torch.Tensor):
        """Sample one action and return (action_idx, log_prob).

        Parameters
        ----------
        state         : Tensor (state_dim,)
        weight        : Tensor (weight_dim,)
        candidate_embs: Tensor (M, item_dim)

        Returns
        -------
        action_idx : int  — index into current candidate pool
        log_prob   : Tensor scalar
        """
        log_probs = self.forward(state, weight, candidate_embs)  # (M,)
        probs = log_probs.exp()
        action_idx = torch.multinomial(probs, num_samples=1).item()
        return action_idx, log_probs[action_idx]


def sample_weight_vector(batch_size: int = 1, alpha: float = 1.0,
                         device=None) -> torch.Tensor:
    """Sample preference weight vectors from Dirichlet(α, α, α).

    Parameters
    ----------
    batch_size : int
    alpha      : float — Dirichlet concentration (1.0 = uniform simplex)
    device     : torch.device

    Returns
    -------
    w : Tensor (batch_size, 3)  — rows sum to 1
    """
    device = device or torch.device('cpu')
    concentration = torch.full((batch_size, 3), alpha, device=device)
    dist = torch.distributions.Dirichlet(concentration)
    return dist.sample()
