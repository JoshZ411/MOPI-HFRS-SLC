"""
Deterministic K-step MDP environment for sequential food-list construction.

State:
    s_t = concat(user_emb,          # d-dim frozen user embedding
                 agg_emb,           # d-dim mean of items selected so far (zeros at t=0)
                 tag_coverage,      # tag_dim binary union of health tags selected so far
                 [t / K])           # scalar normalised timestep

Action:
    index into the candidate pool (items not yet selected in this episode).

Reward (per step, multi-objective vector):
    r_pref   = mean log-sigmoid margin between the selected item and sampled
               unchosen candidates under the frozen dot-product scorer
    r_health = Jaccard(new_tag_coverage, user_tags)
    r_div    = -mean_cosine_sim(item_emb, selected_embs)   (0 when selecting the first item)
"""

import torch
import torch.nn.functional as F
from typing import List, Literal, Optional, Tuple


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
        Pre-computed top-M item indices per user.
    K : int
        Episode length (recommendation list length).
    pref_negative_samples : int
        Number of unchosen candidates sampled to form the BPR-style
        preference reward at each step.
    pref_negative_sampling : {'hard', 'random', 'mixed'}
        Strategy used to choose unselected comparison items.
    pref_hard_negative_ratio : float
        Fraction of comparison items drawn from the highest-scoring unchosen
        candidates when ``pref_negative_sampling='mixed'``.
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
        pref_negative_samples: int = 10,
        pref_negative_sampling: Literal['hard', 'random', 'mixed'] = 'mixed',
        pref_hard_negative_ratio: float = 0.7,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device('cpu')
        self.user_emb = user_emb.to(self.device)
        self.item_emb = item_emb.to(self.device)
        self.user_tags = user_tags.float().to(self.device)
        self.item_tags = item_tags.float().to(self.device)
        self.candidate_pools = candidate_pools
        self.K = K
        self.pref_negative_samples = max(1, pref_negative_samples)
        self.pref_negative_sampling = pref_negative_sampling
        self.pref_hard_negative_ratio = min(max(pref_hard_negative_ratio, 0.0), 1.0)

        self.d = user_emb.size(1)
        self.tag_dim = user_tags.size(1)
        self.state_dim = 2 * self.d + self.tag_dim + 1

        self._item_emb_norm = F.normalize(self.item_emb, dim=1)

        # Episode state (reset per user)
        self._user_id: int = -1
        self._selected: List[int] = []
        self._remaining: List[int] = []
        self._agg_emb = torch.zeros(self.d, device=self.device)
        self._tag_coverage = torch.zeros(self.tag_dim, device=self.device)
        self._t: int = 0

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
        reward     : torch.Tensor  shape (3,)  [r_pref, r_health, r_div]
        done       : bool
        """
        assert 0 <= action < len(self._remaining), \
            f"action {action} out of range (remaining={len(self._remaining)})"

        item_idx = self._remaining.pop(action)
        self._selected.append(item_idx)

        item_vec = self.item_emb[item_idx]
        # ---- r_pref: sampled BPR reward using the frozen dot-product scorer ----
        user_vec = self.user_emb[self._user_id]
        if self._remaining:
            negative_indices = self._sample_negative_indices(user_vec)
            chosen_score = torch.dot(user_vec, item_vec)
            negative_scores = self.item_emb[negative_indices] @ user_vec
            r_pref = F.logsigmoid(chosen_score - negative_scores).mean().item()
        else:
            r_pref = 0.0

        # ---- update aggregated embedding (incremental mean) ----
        t = len(self._selected)
        agg_emb = self._agg_emb
        self._agg_emb = (agg_emb * (t - 1) + item_vec) / t

        # ---- r_div: negative mean cosine-sim with previously selected items ----
        if len(self._selected) > 1:
            prev_embs = self.item_emb[self._selected[:-1]]  # (t-1, d)
            sims = F.cosine_similarity(item_vec.unsqueeze(0), prev_embs, dim=1)
            r_div = -sims.mean().item()
        else:
            r_div = 0.0

        # ---- update tag coverage and compute r_health ----
        new_item_tags = self.item_tags[item_idx]
        tag_coverage = self._tag_coverage
        self._tag_coverage = torch.clamp(tag_coverage + new_item_tags, max=1.0)

        user_tag_vec = self.user_tags[self._user_id]
        intersection = torch.sum(torch.min(self._tag_coverage, user_tag_vec))
        union = torch.sum(torch.max(self._tag_coverage, user_tag_vec))
        r_health = (intersection / (union + 1e-8)).item()

        self._t += 1
        done = (self._t >= self.K) or (len(self._remaining) == 0)
        reward = torch.tensor([r_pref, r_health, r_div], dtype=torch.float32, device=self.device)

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

    def _sample_negative_indices(self, user_vec: torch.Tensor) -> torch.Tensor:
        remaining_count = len(self._remaining)
        negative_count = min(self.pref_negative_samples, remaining_count)
        remaining_indices = torch.tensor(self._remaining, dtype=torch.long, device=self.device)

        if self.pref_negative_sampling == 'random' or negative_count >= remaining_count:
            sampled_positions = torch.randperm(remaining_count, device=self.device)[:negative_count]
            return remaining_indices[sampled_positions]

        remaining_scores = self.item_emb[remaining_indices] @ user_vec
        if self.pref_negative_sampling == 'hard':
            hard_positions = torch.topk(remaining_scores, k=negative_count, largest=True).indices
            return remaining_indices[hard_positions]

        hard_count = min(
            negative_count,
            max(1, int(round(negative_count * self.pref_hard_negative_ratio))),
        )
        random_count = negative_count - hard_count

        hard_positions = torch.topk(remaining_scores, k=hard_count, largest=True).indices
        if random_count <= 0:
            return remaining_indices[hard_positions]

        chosen_mask = torch.zeros(remaining_count, dtype=torch.bool, device=self.device)
        chosen_mask[hard_positions] = True
        random_pool_positions = torch.nonzero(~chosen_mask, as_tuple=False).flatten()
        if random_pool_positions.numel() == 0:
            return remaining_indices[hard_positions]

        random_count = min(random_count, int(random_pool_positions.numel()))
        random_pick_order = torch.randperm(random_pool_positions.numel(), device=self.device)[:random_count]
        random_positions = random_pool_positions[random_pick_order]
        return remaining_indices[torch.cat([hard_positions, random_positions], dim=0)]

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
