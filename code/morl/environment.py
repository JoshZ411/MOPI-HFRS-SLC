"""
MORL Environment — Sequential list-construction MDP.

State s_t:
    - user_emb:            d-dim frozen user embedding
    - aggregated_item_emb: d-dim mean of items selected so far (zero at t=0)
    - tag_coverage:        tag_dim-dim binary union of selected items' health tags
    - normalized_t:        scalar t/K

Action:
    - index into candidate pool (masked to prevent duplicates)

Transition:
    - append item to list
    - update aggregated embedding via incremental mean
    - update tag coverage via bitwise OR
    - increment t

Episode ends after K selections.

Reward vector per step: [r_pref, r_health, r_div]
"""

import torch
import torch.nn.functional as F


class RecommendationEnv:
    """Deterministic K-step sequential recommendation environment.

    Parameters
    ----------
    user_emb : Tensor (num_users, d)
        Frozen user embeddings.
    item_emb : Tensor (num_items, d)
        Frozen item embeddings.
    user_tags : Tensor (num_users, tag_dim)  — int/float binary
        Health-tag vectors for users.
    item_tags : Tensor (num_items, tag_dim)  — int/float binary
        Health-tag vectors for items.
    candidate_pools : dict[int → list[int]]
        Top-M item indices per user.
    K : int
        Recommendation list length (episode horizon).
    device : torch.device
    """

    def __init__(self, user_emb, item_emb, user_tags, item_tags, candidate_pools, K=20, device=None):
        self.device = device or torch.device('cpu')
        self.user_emb = user_emb.to(self.device)   # (num_users, d)
        self.item_emb = item_emb.to(self.device)   # (num_items, d)
        self.user_tags = user_tags.float().to(self.device)  # (num_users, tag_dim)
        self.item_tags = item_tags.float().to(self.device)  # (num_items, tag_dim)
        self.candidate_pools = candidate_pools      # dict user_id → list of item_ids
        self.K = K
        self.d = user_emb.shape[1]
        self.tag_dim = item_tags.shape[1]

        # Episode state (reset per user)
        self._user_id = None
        self._selected = []
        self._t = 0
        self._aggregated_emb = None  # (d,)
        self._tag_coverage = None    # (tag_dim,)
        self._pool = None            # remaining candidate item ids (list)
        self._pool_tensor = None     # (M_remaining, d)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, user_id: int):
        """Start a new episode for *user_id*. Returns initial state tensor."""
        self._user_id = user_id
        self._selected = []
        self._t = 0
        self._aggregated_emb = torch.zeros(self.d, device=self.device)
        self._tag_coverage = torch.zeros(self.tag_dim, device=self.device)
        self._pool = list(self.candidate_pools[user_id])  # copy
        self._selected_tensor = torch.empty(0, dtype=torch.long, device=self.device)
        return self._get_state()

    def step(self, action_idx: int):
        """Execute one step.

        Parameters
        ----------
        action_idx : int
            Index into the *current* remaining pool list.

        Returns
        -------
        next_state : Tensor (state_dim,)
        reward_vec : Tensor (3,)  [r_pref, r_health, r_div]
        done       : bool
        """
        assert self._user_id is not None, "Call reset() before step()."
        item_id = self._pool[action_idx]
        item_emb = self.item_emb[item_id]        # (d,)
        item_tag = self.item_tags[item_id]        # (tag_dim,)

        # ---- Reward ----
        reward_vec = self._compute_reward(item_emb, item_tag)

        # ---- Transition ----
        t = self._t
        # Incremental mean: mean_{t+1} = (t * mean_t + new) / (t+1)
        self._aggregated_emb = (t * self._aggregated_emb + item_emb) / (t + 1)
        self._tag_coverage = torch.clamp(self._tag_coverage + item_tag, max=1.0)
        self._selected.append(item_id)
        self._selected_tensor = torch.cat([
            self._selected_tensor,
            torch.tensor([item_id], dtype=torch.long, device=self.device)
        ])
        self._pool.pop(action_idx)
        self._t += 1

        done = (self._t >= self.K) or (len(self._pool) == 0)
        next_state = self._get_state()
        return next_state, reward_vec, done

    def state_dim(self) -> int:
        """Dimensionality of the state vector (without weight conditioning)."""
        # user_emb (d) + aggregated_emb (d) + tag_coverage (tag_dim) + normalized_t (1)
        return 2 * self.d + self.tag_dim + 1

    def current_pool(self):
        """Return list of remaining candidate item ids for current episode."""
        return list(self._pool)

    def get_candidate_embeddings(self):
        """Return embedding matrix for remaining candidates. Shape (|pool|, d)."""
        if not self._pool:
            return torch.empty(0, self.d, device=self.device)
        return self.item_emb[torch.tensor(self._pool, device=self.device)]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_state(self) -> torch.Tensor:
        """Concatenate state components into a single vector."""
        user_emb = self.user_emb[self._user_id]          # (d,)
        norm_t = torch.tensor([self._t / self.K], device=self.device)  # (1,)
        return torch.cat([user_emb, self._aggregated_emb, self._tag_coverage, norm_t])

    def _compute_reward(self, item_emb: torch.Tensor, item_tag: torch.Tensor) -> torch.Tensor:
        """Compute per-step 3-dimensional reward vector.

        r_pref  : dot-product preference score between user and selected item.
        r_health: Jaccard similarity between cumulative tag coverage (after adding
                  item) and user profile tags.
        r_div   : negative mean cosine similarity of new item to already-selected
                  items (encourages diversity; 0 if list is empty).
        """
        user_emb = self.user_emb[self._user_id]

        # Preference reward
        r_pref = torch.dot(user_emb, item_emb)

        # Health reward — Jaccard of (coverage ∪ item_tag) vs user tags
        new_coverage = torch.clamp(self._tag_coverage + item_tag, max=1.0)
        user_t = self.user_tags[self._user_id]
        intersection = (new_coverage * user_t).sum()
        union = torch.clamp(new_coverage + user_t, max=1.0).sum()
        r_health = intersection / (union + 1e-8)

        # Diversity reward
        if self._selected_tensor.numel() == 0:
            r_div = torch.tensor(0.0, device=self.device)
        else:
            selected_embs = self.item_emb[self._selected_tensor]               # (t, d)
            cos_sims = F.cosine_similarity(item_emb.unsqueeze(0), selected_embs, dim=1)  # (t,)
            r_div = -cos_sims.mean()

        return torch.stack([r_pref, r_health, r_div])


# ------------------------------------------------------------------
# Candidate pool construction (Phase 2)
# ------------------------------------------------------------------

def build_candidate_pools(user_emb: torch.Tensor, item_emb: torch.Tensor, M: int = 200,
                           exclude_edges=None, device=None) -> dict:
    """Compute top-M candidate items per user using dot-product scores.

    Parameters
    ----------
    user_emb   : Tensor (num_users, d)
    item_emb   : Tensor (num_items, d)
    M          : candidate pool size
    exclude_edges : optional Tensor (2, E) — user/item edges to mask out
                    (used at eval/test time to exclude seen interactions).
    device     : torch.device

    Returns
    -------
    candidate_pools : dict  user_id (int) → list of M item_ids (int)
    """
    device = device or torch.device('cpu')
    user_emb = user_emb.to(device)
    item_emb = item_emb.to(device)

    # scores: (num_users, num_items)
    scores = torch.matmul(user_emb, item_emb.T)

    if exclude_edges is not None:
        users_ex = exclude_edges[0].to(device)
        items_ex = exclude_edges[1].to(device)
        scores[users_ex, items_ex] = float('-inf')

    M_actual = min(M, item_emb.shape[0])
    _, top_indices = torch.topk(scores, k=M_actual, dim=1)  # (num_users, M)

    candidate_pools = {}
    for uid in range(user_emb.shape[0]):
        candidate_pools[uid] = top_indices[uid].tolist()

    return candidate_pools
