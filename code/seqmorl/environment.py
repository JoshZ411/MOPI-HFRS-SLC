"""Sequential recommendation MDP environment.

State at step t:
    concat(user_emb [dim], item_agg_emb [dim], tag_coverage [num_tags], t/K [1])
    => state_dim = 2*dim + num_tags + 1

Action:
    Index into the per-user candidate pool of size M (already-selected slots masked out).

Rewards (3-vector per step):
    r_pref  : cosine similarity between user_emb and selected item_emb
    r_health: Jaccard similarity between user tags and selected item tags
    r_div   : 1 - cosine similarity of selected item to aggregate of already-selected items
              (0 on the first step since no prior items exist)
"""

import torch
import torch.nn.functional as F

EPS = 1e-8  # Numerical stability constant for division operations.


class SequentialRecEnv:
    """Sequential top-K list construction environment.

    Args:
        user_emb   : Tensor [num_users, dim]  — frozen user embeddings.
        item_emb   : Tensor [num_items, dim]  — frozen item embeddings.
        food_tags  : Tensor [num_items, num_tags] — binary item health tags.
        user_tags  : Tensor [num_users, num_tags] — binary user health tags.
        M          : Candidate pool size per user.
        K          : List length (episode horizon).
        device     : torch device.
    """

    def __init__(self, user_emb, item_emb, food_tags, user_tags,
                 M: int = 200, K: int = 20, device: str = 'cpu',
                 exclude_edge_indices: list[torch.Tensor] | None = None):
        self.device = torch.device(device)
        self.user_emb = user_emb.to(self.device)
        self.item_emb = item_emb.to(self.device)
        self.food_tags = food_tags.float().to(self.device)
        self.user_tags = user_tags.float().to(self.device)

        self.M = min(M, item_emb.size(0))
        self.K = K

        self.num_users = user_emb.size(0)
        self.num_items = item_emb.size(0)
        self.dim = user_emb.size(1)
        self.num_tags = food_tags.size(1)

        self.exclude_edge_indices = exclude_edge_indices or []
        self.excluded_items_by_user = self._build_excluded_items_by_user()

        # Precompute top-M candidate pools once (no dynamic reranking).
        self._precompute_candidate_pools()

        # Episode state (reset per user).
        self.current_user: int = -1
        self.step_count: int = 0
        self.selected_mask: torch.Tensor = torch.zeros(self.M, dtype=torch.bool,
                                                       device=self.device)
        self.item_agg_emb: torch.Tensor = torch.zeros(self.dim, device=self.device)
        self.tag_coverage: torch.Tensor = torch.zeros(self.num_tags, device=self.device)
        self._num_selected: int = 0

    # ------------------------------------------------------------------
    # Candidate pool
    # ------------------------------------------------------------------

    def _build_excluded_items_by_user(self) -> dict[int, set[int]]:
        """Build per-user exclusion sets from edge-index tensors."""
        excluded: dict[int, set[int]] = {}
        for edge_index in self.exclude_edge_indices:
            if edge_index is None or edge_index.numel() == 0:
                continue
            for u, i in edge_index.T.tolist():
                uid = int(u)
                iid = int(i)
                if uid not in excluded:
                    excluded[uid] = set()
                excluded[uid].add(iid)
        return excluded

    def _precompute_candidate_pools(self):
        """Compute top-M items per user via dot-product score with exclusions."""
        with torch.no_grad():
            scores = torch.matmul(self.user_emb, self.item_emb.T)  # [num_users, num_items]
            for uid, excluded_items in self.excluded_items_by_user.items():
                if not excluded_items:
                    continue
                excluded_idx = torch.tensor(
                    sorted(excluded_items), dtype=torch.long, device=self.device
                )
                scores[uid, excluded_idx] = float('-inf')

            top_scores, self.candidate_pools = torch.topk(scores, k=self.M, dim=1)
            self.candidate_valid_mask = torch.isfinite(top_scores)
            # candidate_pools: [num_users, M]

    # ------------------------------------------------------------------
    # Episode API
    # ------------------------------------------------------------------

    def reset(self, user_id: int) -> torch.Tensor:
        """Start a new episode for *user_id*.  Returns initial state."""
        self.current_user = user_id
        self.step_count = 0
        self._num_selected = 0
        self.selected_mask = (~self.candidate_valid_mask[user_id]).clone()
        self.item_agg_emb = torch.zeros(self.dim, device=self.device)
        self.tag_coverage = torch.zeros(self.num_tags, device=self.device)
        return self._get_state()

    def step(self, action: int):
        """Execute *action* (index 0…M-1 into candidate pool).

        Returns:
            next_state (Tensor | None) : None when episode is done.
            reward     (Tensor [3])    : [r_pref, r_health, r_div].
            done       (bool)          : True when K items selected or pool exhausted.
            item_idx   (int)           : Global item index that was selected.
        """
        assert 0 <= action < self.M, f"Action {action} out of range [0, {self.M})"
        assert not self.selected_mask[action], "Action selects an already-chosen item."

        item_idx = int(self.candidate_pools[self.current_user, action].item())

        r_pref = self._pref_reward(item_idx)
        r_health = self._health_reward(item_idx)
        r_div = self._diversity_reward(item_idx)
        reward = torch.tensor([r_pref, r_health, r_div],
                               dtype=torch.float32, device=self.device)

        # Update internal state.
        self.selected_mask[action] = True
        item_emb = self.item_emb[item_idx]
        self._num_selected += 1
        # Running mean aggregate embedding.
        self.item_agg_emb = (
            (self._num_selected - 1) * self.item_agg_emb + item_emb
        ) / self._num_selected
        # Binary tag union.
        self.tag_coverage = torch.clamp(
            self.tag_coverage + self.food_tags[item_idx], max=1.0
        )
        self.step_count += 1

        done = (self.step_count >= self.K) or bool(self.selected_mask.all())
        next_state = None if done else self._get_state()
        return next_state, reward, done, item_idx

    # ------------------------------------------------------------------
    # State / masking helpers
    # ------------------------------------------------------------------

    def _get_state(self) -> torch.Tensor:
        u = self.user_emb[self.current_user]
        t_norm = torch.tensor(
            [self.step_count / self.K], dtype=torch.float32, device=self.device
        )
        return torch.cat([u, self.item_agg_emb, self.tag_coverage, t_norm])

    def get_action_mask(self) -> torch.Tensor:
        """Return boolean mask; True = already selected (invalid action)."""
        return self.selected_mask.clone()

    @property
    def state_dim(self) -> int:
        return self.dim + self.dim + self.num_tags + 1

    @property
    def action_dim(self) -> int:
        return self.M

    # ------------------------------------------------------------------
    # Anchor-based constrained reranking helpers (Phase 1)
    # ------------------------------------------------------------------

    def get_anchor_list_and_scores(self, user_id: int, K: int) -> tuple[list[int], torch.Tensor]:
        """Return top-K items (global indices) and their preference scores for this user.

        This is the one-shot anchor that constrained mode uses as the reference list.
        Scores are cosine similarities (preference rewards).
        """
        u = self.user_emb[user_id]
        scores = F.cosine_similarity(u.unsqueeze(0), self.item_emb).squeeze(0)  # [num_items]

        # Apply exclusions for this user
        excluded_items = self.excluded_items_by_user.get(user_id, set())
        if excluded_items:
            excluded_idx = torch.tensor(
                sorted(excluded_items), dtype=torch.long, device=self.device
            )
            scores[excluded_idx] = float('-inf')

        # Top-K
        top_k_scores, top_k_global_idx = torch.topk(scores, k=min(K, scores.numel()), dim=0)
        anchor_items = top_k_global_idx[:K].tolist()
        anchor_scores = top_k_scores[:K]

        return anchor_items, anchor_scores

    def is_action_feasible_constrained(
        self,
        user_id: int,
        action: int,
        step_count: int,
        anchor_items: list[int],
        anchor_scores: torch.Tensor,
        anchor_epsilon: float,
        anchor_lock_positions: int,
        max_swaps_allowed: int,
        num_swaps_so_far: int,
    ) -> tuple[bool, str]:
        """Check if an action (pool index) satisfies constrained-mode constraints.

        Returns:
            (feasible: bool, rejection_reason: str)
            If feasible, rejection_reason is empty.
        """
        # Invariant 0: Item at this pool position not already selected
        if self.selected_mask[action]:
            return False, "duplicate_already_selected"
        
        # Get the global item index for this pool position
        candidate_global_idx = int(self.candidate_pools[user_id, action].item())
        
        # Invariant 1: Locked positions (1 to anchor_lock_positions) must pick anchor item
        if step_count < anchor_lock_positions:
            anchor_global_idx = anchor_items[step_count]
            if candidate_global_idx != anchor_global_idx:
                return False, f"position_locked (step={step_count}, anchor_lock={anchor_lock_positions})"
            return True, ""
        
        # Invariant 2: Swap budget exhausted
        if num_swaps_so_far >= max_swaps_allowed:
            return False, f"swap_budget_exhausted (used={num_swaps_so_far}, max={max_swaps_allowed})"

        # Invariant 3: Score margin gate
        candidate_score = self._pref_reward(candidate_global_idx)
        anchor_score = float(anchor_scores[step_count].item())
        if candidate_score < anchor_score - anchor_epsilon:
            return (
                False,
                f"margin_gate (cand_score={candidate_score:.4f}, anchor={anchor_score:.4f}, eps={anchor_epsilon:.4f})",
            )

        return True, ""

    # ------------------------------------------------------------------
    # Reward helpers
    # ------------------------------------------------------------------

    def _pref_reward(self, item_idx: int) -> float:
        u = self.user_emb[self.current_user]
        v = self.item_emb[item_idx]
        return float(F.cosine_similarity(u.unsqueeze(0), v.unsqueeze(0)).item())

    def _health_reward(self, item_idx: int) -> float:
        u_tags = self.user_tags[self.current_user]
        i_tags = self.food_tags[item_idx]
        intersection = torch.min(u_tags, i_tags).sum()
        union = torch.max(u_tags, i_tags).sum() + EPS
        return float((intersection / union).item())

    def _diversity_reward(self, item_idx: int) -> float:
        if self._num_selected == 0:
            return 0.0
        v = self.item_emb[item_idx]
        sim = float(
            F.cosine_similarity(v.unsqueeze(0), self.item_agg_emb.unsqueeze(0)).item()
        )
        return max(0.0, 1.0 - sim)
