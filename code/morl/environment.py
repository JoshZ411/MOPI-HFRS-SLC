"""
Deterministic K-step MDP environment for sequential food-list construction.

State:
    s_t = concat(user_emb, agg_emb, tag_coverage, [t / K])

Action:
    index into the candidate list (items not yet selected in this episode).
    In the full-item-space A2C formulation the candidate list is all items not
    yet selected, with no pool ceiling.

Reward (2-component scalar, summed in training with a fixed beta weight):
    r_rel    = 1.0 if the selected item is in train_pos_items[user], else 0.0.
               Fires every step (dense signal).  Uses only train-split labels so
               there is no val/test leakage — the same supervision signal the GNN
               was trained on.
    r_health = 1.0 if the selected item's tags overlap with the user's health tags,
               else 0.0.  Binary per-item signal, fires independently at every step.
               Directly matches the health_score eval metric (fraction of recommended
               items that individually match the user's health profile).
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Set, Tuple


class RecommendationEnv:
    """Deterministic K-step sequential recommendation environment.

    Parameters
    ----------
    user_emb : torch.Tensor
        Frozen user embeddings, shape (num_users, d).
    item_emb : torch.Tensor
        Frozen item embeddings, shape (num_items, d).
    user_tags : torch.Tensor
        Binary health-tag vectors for users, shape (num_users, tag_dim).
    item_tags : torch.Tensor
        Binary health-tag vectors for items, shape (num_items, tag_dim).
    candidate_pools : dict[int, List[int]]
        Pre-computed candidate item indices per user.  Pass
        {u: list(range(num_items))} for the full-item-space formulation.
    K : int
        Episode length (recommendation list length).
    train_pos_items : dict[int, set[int]], optional
        Train-split positive item indices per user.  Selecting one of these
        items yields r_rel = 1.0 every step (dense, no leakage).
    device : torch.device
    """

    def __init__(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        user_tags: torch.Tensor,
        item_tags: torch.Tensor,
        candidate_pools: dict,
        K: int = 20,
        train_pos_items: Optional[Dict[int, Set[int]]] = None,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device('cpu')
        self.user_emb = user_emb.to(self.device)
        self.item_emb = item_emb.to(self.device)
        self.user_tags = user_tags.float().to(self.device)
        self.item_tags = item_tags.float().to(self.device)
        self.candidate_pools = candidate_pools
        self.K = K
        self.train_pos_items: Dict[int, Set[int]] = train_pos_items or {}

        self.d = user_emb.size(1)
        self.tag_dim = user_tags.size(1)
        self.state_dim = 2 * self.d + self.tag_dim + 1

        # Episode state (reset per user)
        self._user_id: int = -1
        self._selected: List[int] = []
        self._remaining: List[int] = []
        self._agg_emb = torch.zeros(self.d, device=self.device)
        self._tag_coverage = torch.zeros(self.tag_dim, device=self.device)
        self._t: int = 0
        self.last_step_info: Dict[str, float] = {}
        # Set to False during training to skip the O(remaining×d) score-rank matmul.
        # Only needed for diagnostic logging; set True on log epochs.
        self.compute_rank: bool = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, user_id: int) -> torch.Tensor:
        """Start a new episode for *user_id*.

        Returns
        -------
        state : torch.Tensor  shape (state_dim,)
        """
        self._user_id = user_id
        self._selected = []
        self._remaining = list(self.candidate_pools[user_id])
        self._agg_emb = torch.zeros(self.d, device=self.device)
        self._tag_coverage = torch.zeros(self.tag_dim, device=self.device)
        self._t = 0
        self.last_step_info = {}
        return self._build_state()

    def step(self, action: int) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """Select item at position *action* in the remaining candidate list.

        Parameters
        ----------
        action : int
            Index into ``self.remaining`` (the current unmasked pool).

        Returns
        -------
        next_state : torch.Tensor  shape (state_dim,)
        reward     : torch.Tensor  shape (2,)  [r_rel, r_health]
        done       : bool
        """
        assert 0 <= action < len(self._remaining), \
            f"action {action} out of range (remaining={len(self._remaining)})"

        user_vec = self.user_emb[self._user_id]

        # ---- score-rank diagnostic (O(remaining×d) matmul — skipped when compute_rank=False) ----
        if self.compute_rank:
            rem_t = torch.tensor(self._remaining, dtype=torch.long, device=self.device)
            rem_scores = self.item_emb[rem_t] @ user_vec
            chosen_score_t = rem_scores[action]
            chosen_rank_1based = int((rem_scores > chosen_score_t).sum().item()) + 1
            chosen_score_val = chosen_score_t.item()
        else:
            # Single dot product for the selected item only — O(d) instead of O(remaining×d)
            chosen_score_val = (self.item_emb[self._remaining[action]] * user_vec).sum().item()
            chosen_rank_1based = 0

        item_idx = self._remaining.pop(action)
        self._selected.append(item_idx)

        item_vec = self.item_emb[item_idx]

        # ---- update aggregated embedding (incremental mean) ----
        t = len(self._selected)
        self._agg_emb = (self._agg_emb * (t - 1) + item_vec) / t

        self._t += 1
        done = (self._t >= self.K) or (len(self._remaining) == 0)

        # ---- r_rel: per-step binary hit on train positives (no leakage) ----
        r_rel = 1.0 if item_idx in self.train_pos_items.get(self._user_id, set()) else 0.0

        # ---- r_health: binary per-item tag overlap with user's health profile ----
        # Fires 1.0 whenever the selected item individually matches the user's health tags.
        # This directly matches the health_score eval metric (healthy_count / K).
        # Unlike the previous Jaccard-delta formulation, this signal never saturates:
        # the policy receives health feedback at every step throughout the full episode.
        user_tag_vec = self.user_tags[self._user_id]
        new_item_tags = self.item_tags[item_idx]
        has_health_overlap = (new_item_tags.bool() & user_tag_vec.bool()).any()
        r_health_t = torch.ones((), device=self.device) if has_health_overlap else torch.zeros((), device=self.device)

        # Still update tag_coverage for the state representation (_build_state uses it)
        self._tag_coverage = torch.clamp(self._tag_coverage + new_item_tags, max=1.0)

        reward = torch.stack([r_health_t.new_tensor(r_rel), r_health_t])
        self.last_step_info = {
            'chosen_score': chosen_score_val,
            'chosen_score_rank_1based': float(chosen_rank_1based),
            'list_rank': self._t,
            'r_rel': r_rel,
            'r_health': r_health_t.item(),  # 1.0 = item matched user health tags, 0.0 = no match
            'terminal': done,
        }

        return self._build_state(), reward, done

    @property
    def remaining(self) -> List[int]:
        """Indices of items still available for selection in the current episode."""
        return self._remaining

    @property
    def selected(self) -> List[int]:
        """Indices of items already selected in the current episode."""
        return self._selected

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_state(self) -> torch.Tensor:
        user_vec = self.user_emb[self._user_id]
        agg_emb = self._agg_emb
        tag_coverage = self._tag_coverage
        timestep = torch.tensor([self._t / self.K], device=self.device)
        return torch.cat([user_vec, agg_emb, tag_coverage, timestep])



# ------------------------------------------------------------------
# Candidate pool construction (Phase 2)
# ------------------------------------------------------------------

def build_candidate_pools(
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    M: int = 200,
    exclude_per_user: Optional[dict] = None,
    batch_size: int = 512,
    device: Optional[torch.device] = None,
) -> dict:
    """Compute top-M item candidates for each user using dot-product scoring.

    Parameters
    ----------
    user_emb : torch.Tensor  shape (num_users, d)
    item_emb : torch.Tensor  shape (num_items, d)
    M : int
        Number of candidates to retain per user.
    exclude_per_user : dict[int, set], optional
        Training/validation positive item indices to mask out (used at eval time).
    batch_size : int
        Number of users to process per GPU batch.
    device : torch.device

    Returns
    -------
    pools : dict[int, List[int]]
        Mapping from user index to list of top-M item indices.
    """
    dev = device or torch.device('cpu')
    user_emb = user_emb.to(dev)
    item_emb = item_emb.to(dev)

    num_users = user_emb.size(0)
    num_items = item_emb.size(0)
    pools: dict = {}

    for start in range(0, num_users, batch_size):
        end = min(start + batch_size, num_users)
        scores = torch.matmul(user_emb[start:end], item_emb.T)  # (batch, num_items)

        if exclude_per_user is not None:
            for local_u, global_u in enumerate(range(start, end)):
                excl = exclude_per_user.get(global_u, set())
                if excl:
                    excl_t = torch.tensor(list(excl), dtype=torch.long, device=dev)
                    scores[local_u, excl_t] = float('-inf')

        k = min(M, num_items)
        _, top_indices = torch.topk(scores, k=k, dim=1)  # (batch, k)
        top_indices = top_indices.cpu()

        for local_u in range(end - start):
            global_u = start + local_u
            pools[global_u] = top_indices[local_u].tolist()

    return pools
