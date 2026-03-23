"""Implicit sequential recommendation policy.

A single MLP maps the current state vector to logits over the M-size candidate
pool.  Already-selected items are masked to -inf before sampling/argmax so they
are never re-selected.

No explicit preference/tradeoff vector is consumed at inference time; objective
balancing is handled entirely at training time via MGDA gradient combination.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class SequentialPolicy(nn.Module):
    """MLP policy: state -> logits over candidate pool.

    Args:
        state_dim  : Dimension of the input state vector.
        action_dim : Candidate pool size M (output logit dimension).
        hidden_dim : Hidden layer width.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor,
                action_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return logits over actions.

        Args:
            state       : Tensor [state_dim] or [batch, state_dim].
            action_mask : Boolean tensor matching action dimension; True entries
                          are set to -inf (invalid / already selected).
        Returns:
            logits: Tensor matching the action dimension.
        """
        logits = self.net(state)
        if action_mask is not None:
            logits = logits.masked_fill(action_mask, float('-inf'))
        return logits

    def select_action(self, state: torch.Tensor,
                      action_mask: torch.Tensor | None = None,
                      greedy: bool = False):
        """Sample (train) or argmax (eval) an action.

        Args:
            state       : Tensor [state_dim].
            action_mask : Boolean mask over actions (True = invalid).
            greedy      : If True, select argmax instead of sampling.

        Returns:
            action   (int)          : Selected action index.
            log_prob (torch.Tensor) : Log-probability of the selected action.
            entropy  (torch.Tensor) : Distribution entropy.
        """
        logits = self.forward(state, action_mask)
        # This distribution is built at every rollout step; skip repeated
        # argument validation for a substantial speed-up in CPU smoke runs.
        dist = Categorical(logits=logits, validate_args=False)
        if greedy:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return int(action.item()), log_prob, entropy
