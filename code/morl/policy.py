"""
Conditional policy network π(a | s, w) for MORL sequential recommendation.

The policy encodes the environment state s_t and builds separate objective
logits for preference, health, and diversity. The final action logits are a
weighted sum of those objective-specific logits using w ∈ Δ².

Architecture:
    state encoder: s_t → Linear → ReLU → Linear → ReLU
    objective heads:
        logits_pref(s_t), logits_health(s_t), logits_div(s_t)
    composition:
        logits = w_pref*logits_pref + w_health*logits_health + w_div*logits_div
    → mask already-selected items (set to −∞)
    → Softmax → action probabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


class ConditionalPolicy(nn.Module):
    """Conditional policy π(a | s, w) for trade-off-aware sequential recommendation.

    Parameters
    ----------
    state_dim : int
        Dimensionality of the environment state (2*d + tag_dim + 1).
    num_candidates : int
        Maximum candidate pool size M.  The output head is fixed to this
        size; invalid (already-selected) actions are masked to −∞.
    weight_dim : int
        Dimensionality of the preference weight vector (default 3 for
        [w_pref, w_health, w_div]).
    hidden_dim : int
        Width of hidden layers (default 256).
    """

    def __init__(
        self,
        state_dim: int,
        num_candidates: int,
        weight_dim: int = 3,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.num_candidates = num_candidates
        self.weight_dim = weight_dim

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.pref_head = nn.Linear(hidden_dim, num_candidates)
        self.health_head = nn.Linear(hidden_dim, num_candidates)
        self.div_head = nn.Linear(hidden_dim, num_candidates)

    def forward(
        self,
        state: torch.Tensor,
        weight: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute action log-probabilities.

        Parameters
        ----------
        state : torch.Tensor
            Shape (state_dim,) or (batch, state_dim).
        weight : torch.Tensor
            Shape (weight_dim,) or (batch, weight_dim).
        mask : torch.Tensor, optional
            Boolean tensor of shape (num_candidates,) or (batch, num_candidates).
            ``True`` means the action is *valid* (not yet selected).
            Actions with mask==False are set to −∞ before softmax.

        Returns
        -------
        log_probs : torch.Tensor
            Shape (num_candidates,) or (batch, num_candidates).
        """
        batched = state.dim() == 2
        if not batched:
            state = state.unsqueeze(0)
            weight = weight.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)

        state_hidden = self.state_encoder(state)  # (batch, hidden_dim)

        logits_pref = self.pref_head(state_hidden)
        logits_health = self.health_head(state_hidden)
        logits_div = self.div_head(state_hidden)

        # Enforce weight-conditioning by construction at the logit level.
        w_pref = weight[:, 0].unsqueeze(-1)
        w_health = weight[:, 1].unsqueeze(-1)
        w_div = weight[:, 2].unsqueeze(-1)
        logits = w_pref * logits_pref + w_health * logits_health + w_div * logits_div

        if mask is not None:
            logits = logits.masked_fill(~mask, float('-inf'))

        log_probs = F.log_softmax(logits, dim=-1)

        if not batched:
            log_probs = log_probs.squeeze(0)

        return log_probs

    def select_action(
        self,
        state: torch.Tensor,
        weight: torch.Tensor,
        remaining_items: List[int],
        num_candidates: int,
        greedy: bool = False,
        return_info: bool = False,
    ):
        """Sample (or greedily select) an action from the remaining candidate pool.

        Parameters
        ----------
        state : torch.Tensor  shape (state_dim,)
        weight : torch.Tensor  shape (weight_dim,)
        remaining_items : List[int]
            Global item IDs still available in the current candidate pool.
            The policy head operates over local candidate-pool positions, so
            only the current pool length is used here.
        num_candidates : int
            Total pool size M (used to build the mask).
        greedy : bool
            If True, select the highest-probability action (for evaluation).

        Returns
        -------
        action : int
            Position within ``remaining_items`` (i.e. index into the
            *remaining* subpool, consistent with ``env.step(action)``).
        log_prob : torch.Tensor  scalar
        info : dict, optional
            Policy diagnostics for the chosen action.
        """
        mask = torch.zeros(num_candidates, dtype=torch.bool, device=state.device)
        active_count = min(len(remaining_items), num_candidates)
        if active_count == 0:
            raise ValueError('select_action called with an empty candidate pool')

        # The policy head is defined over candidate-pool slots [0, M). The
        # environment keeps global item IDs in its remaining list, so valid
        # actions are the active local positions [0, len(remaining_items)).
        mask[:active_count] = True

        log_probs = self.forward(state, weight, mask=mask)  # (num_candidates,)
        valid_log_probs = log_probs[:active_count]

        if greedy:
            local_action = valid_log_probs.argmax().item()
        else:
            probs = valid_log_probs.exp()
            local_action = torch.multinomial(probs, num_samples=1).item()

        log_prob = valid_log_probs[local_action]
        if not return_info:
            return local_action, log_prob

        probs = valid_log_probs.exp()
        entropy = -(probs * valid_log_probs).sum()
        info: Dict[str, float] = {
            'entropy': entropy.item(),
            'selected_prob': probs[local_action].item(),
            'max_prob': probs.max().item(),
            'active_count': float(active_count),
        }
        return local_action, log_prob, info


def sample_weight_vector(
    batch_size: int = 1,
    weight_dim: int = 3,
    alpha: float = 1.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Sample preference weight vectors from a symmetric Dirichlet distribution.

    Parameters
    ----------
    batch_size : int
    weight_dim : int
        Dimension of the simplex (default 3 for pref / health / div).
    alpha : float
        Dirichlet concentration parameter.  alpha=1 gives a uniform
        distribution over the simplex.
    device : torch.device

    Returns
    -------
    w : torch.Tensor  shape (batch_size, weight_dim) or (weight_dim,) if batch_size==1
    """
    concentration = torch.full((weight_dim,), alpha)
    dist = torch.distributions.Dirichlet(concentration)
    w = dist.sample((batch_size,))  # (batch_size, weight_dim)
    if device is not None:
        w = w.to(device)
    if batch_size == 1:
        w = w.squeeze(0)
    return w
