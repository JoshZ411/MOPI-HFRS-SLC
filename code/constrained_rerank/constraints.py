"""
constraints.py — Hard constraint enforcement layer for constrained reranking.

Implements the FeasibilityChecker with 4 constraints:
1. Position lock (positions 1..L immutable)
2. Score-margin gate (candidate score >= anchor score - epsilon)
3. Swap budget (max swaps per list)
4. Duplicate prevention (no repeated items)
"""


class FeasibilityChecker:
    """Enforces hard constraints on reranking edits with per-list diagnostics."""

    def __init__(self, lock_positions=6, epsilon=0.05, max_swaps=4):
        """
        Args:
            lock_positions: Number of top positions that are immutable (1-indexed).
                           Positions 0..lock_positions-1 cannot be swapped.
            epsilon: Score margin tolerance. Candidate must satisfy:
                     cand_score >= anchor_score - epsilon
            max_swaps: Maximum number of swaps allowed per list.
        """
        self.lock_positions = lock_positions
        self.epsilon = epsilon
        self.max_swaps = max_swaps

    def check_swap(self, position, anchor_score, candidate_score,
                   current_list, candidate_item, swap_count):
        """
        Check whether a proposed swap is feasible.

        Args:
            position: 0-indexed position in the list being edited
            anchor_score: dot-product score of the current item at this position
            candidate_score: dot-product score of the proposed replacement
            current_list: list/tensor of current item indices in the list
            candidate_item: item index of the proposed replacement
            swap_count: number of swaps already performed for this list

        Returns:
            (is_feasible, rejection_reason): tuple of (bool, str or None)
        """
        # 1. Position lock: reject if position is within locked range
        if position < self.lock_positions:
            return False, 'position_lock'

        # 2. Score-margin gate
        if candidate_score < anchor_score - self.epsilon:
            return False, 'score_margin'

        # 3. Duplicate prevention
        if candidate_item in current_list:
            return False, 'duplicate'

        # 4. Swap budget
        if swap_count >= self.max_swaps:
            return False, 'budget_exceeded'

        return True, None


class ListDiagnostics:
    """Tracks per-list constraint diagnostics."""

    def __init__(self):
        self.attempted_swaps = 0
        self.accepted_swaps = 0
        self.rejected_by_position_lock = 0
        self.rejected_by_score_margin = 0
        self.rejected_by_duplicate = 0
        self.rejected_by_budget = 0
        self.forced_anchor_count = 0

    def record_attempt(self, is_feasible, rejection_reason):
        """Record the result of a swap attempt."""
        self.attempted_swaps += 1
        if is_feasible:
            self.accepted_swaps += 1
        else:
            self.forced_anchor_count += 1
            if rejection_reason == 'position_lock':
                self.rejected_by_position_lock += 1
            elif rejection_reason == 'score_margin':
                self.rejected_by_score_margin += 1
            elif rejection_reason == 'duplicate':
                self.rejected_by_duplicate += 1
            elif rejection_reason == 'budget_exceeded':
                self.rejected_by_budget += 1

    def to_dict(self):
        return {
            'attempted_swaps': self.attempted_swaps,
            'accepted_swaps': self.accepted_swaps,
            'rejected_by_position_lock': self.rejected_by_position_lock,
            'rejected_by_score_margin': self.rejected_by_score_margin,
            'rejected_by_duplicate': self.rejected_by_duplicate,
            'rejected_by_budget': self.rejected_by_budget,
            'forced_anchor_count': self.forced_anchor_count,
        }


def aggregate_diagnostics(all_diagnostics):
    """
    Aggregate per-list diagnostics into summary statistics.

    Args:
        all_diagnostics: list of ListDiagnostics objects

    Returns:
        dict with aggregate stats
    """
    total = len(all_diagnostics)
    if total == 0:
        return {}

    agg = {
        'num_lists': total,
        'total_attempted_swaps': sum(d.attempted_swaps for d in all_diagnostics),
        'total_accepted_swaps': sum(d.accepted_swaps for d in all_diagnostics),
        'total_rejected_by_position_lock': sum(d.rejected_by_position_lock for d in all_diagnostics),
        'total_rejected_by_score_margin': sum(d.rejected_by_score_margin for d in all_diagnostics),
        'total_rejected_by_duplicate': sum(d.rejected_by_duplicate for d in all_diagnostics),
        'total_rejected_by_budget': sum(d.rejected_by_budget for d in all_diagnostics),
        'total_forced_anchor': sum(d.forced_anchor_count for d in all_diagnostics),
    }

    attempted = agg['total_attempted_swaps']
    if attempted > 0:
        agg['swap_rate'] = agg['total_accepted_swaps'] / attempted
        agg['rejection_rate'] = 1.0 - agg['swap_rate']
        agg['forced_anchor_rate'] = agg['total_forced_anchor'] / attempted
    else:
        agg['swap_rate'] = 0.0
        agg['rejection_rate'] = 0.0
        agg['forced_anchor_rate'] = 0.0

    return agg
