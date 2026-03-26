"""
Policy network π(a | s, C) for MORL sequential recommendation.

The policy encodes the environment state s_t and scores each candidate item
embedding directly. There is no preference-weight conditioning.

Architecture:
    state encoder: s_t → Linear → ReLU → Linear → ReLU
    candidate encoder: c_i → Linear → ReLU
    scorer:
        logits_i = <state_hidden, candidate_hidden_i>
    → Softmax → action probabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class ConditionalPolicy(nn.Module):
    """Policy π(a | s, C) for sequential recommendation.

    Parameters
    ----------
    state_dim : int
        Dimensionality of the environment state (2*d + tag_dim + 1).
    candidate_dim : int
        Dimensionality of each candidate item embedding.
    hidden_dim : int
        Width of hidden layers (default 256).
    """

    def __init__(
        self,
        state_dim: int,
        candidate_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.candidate_dim = candidate_dim
        self.hidden_dim = hidden_dim

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        state: torch.Tensor,
        candidate_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute action log-probabilities over the active candidate set.

        Parameters
        ----------
        state : torch.Tensor
            Shape (state_dim,) or (batch, state_dim).
        candidate_embeddings : torch.Tensor
            Shape (num_candidates, candidate_dim) or
            (batch, num_candidates, candidate_dim).

        Returns
        -------
        log_probs : torch.Tensor
            Shape (num_candidates,) or (batch, num_candidates).
        """
        batched = state.dim() == 2
        if not batched:
            state = state.unsqueeze(0)
            candidate_embeddings = candidate_embeddings.unsqueeze(0)

        state_hidden = self.state_encoder(state)  # (batch, hidden_dim)
        candidate_hidden = self.candidate_encoder(candidate_embeddings)
        logits = torch.einsum('bh,bnh->bn', state_hidden, candidate_hidden)

        log_probs = F.log_softmax(logits, dim=-1)

        if not batched:
            log_probs = log_probs.squeeze(0)

        return log_probs

    def select_action(
        self,
        state: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        greedy: bool = False,
        return_info: bool = False,
    ):
        """Sample (or greedily select) an action from the active candidate set.

        Parameters
        ----------
        state : torch.Tensor  shape (state_dim,)
        candidate_embeddings : torch.Tensor
            Shape (num_candidates, candidate_dim) for the remaining local pool.
        greedy : bool
            If True, select the highest-probability action (for evaluation).

        Returns
        -------
        action : int
            Position within the current local candidate set, consistent with
            ``env.step(action)``.
        log_prob : torch.Tensor  scalar
        normalized_entropy : torch.Tensor  scalar, optional
            Entropy of the valid-action distribution normalized by
            ``log(num_candidates)`` so values remain comparable as the candidate
            set shrinks during an episode.
        info : dict, optional
            Policy diagnostics for the chosen action.
        """
        active_count = int(candidate_embeddings.size(0))
        if active_count == 0:
            raise ValueError('select_action called with an empty candidate pool')

        log_probs = self.forward(state, candidate_embeddings)
        valid_log_probs = log_probs
        local_action: int

        if greedy:
            local_action = int(valid_log_probs.argmax().item())
        else:
            probs = valid_log_probs.exp()
            local_action = int(torch.multinomial(probs, num_samples=1).item())

        log_prob = valid_log_probs[local_action]
        if not return_info:
            return local_action, log_prob

        probs = valid_log_probs.exp()
        entropy = -(probs * valid_log_probs).sum()
        if active_count > 1:
            normalizer = torch.log(valid_log_probs.new_tensor(float(active_count)))
            normalized_entropy = entropy / normalizer.clamp_min(1e-8)
        else:
            normalized_entropy = torch.zeros_like(entropy)
        info: Dict[str, float] = {
            'entropy': entropy.item(),
            'normalized_entropy': normalized_entropy.item(),
            'selected_prob': probs[local_action].item(),
            'max_prob': probs.max().item(),
            'active_count': float(active_count),
        }
        return local_action, log_prob, normalized_entropy, info
