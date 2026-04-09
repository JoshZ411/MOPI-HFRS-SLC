# Auto Implementation Log — MOPI-HFRS MORL Module

This file is a running log of all implementation work on the MORL sequential recommendation
augmentation. Each entry corresponds to a completed phase and records decisions, file paths,
and notes for downstream phases.

---

## Phase 1: Embedding Extraction — COMPLETE

**Date:** 2026-03-11

### What was implemented

- **`code/main.py`** — Added a frozen-embedding checkpoint save after the test evaluation block.
  One additional forward pass is run (with `torch.no_grad()`) to obtain the final embeddings in
  inference mode, then saved as:
  ```python
  torch.save(
      {'user_emb': users_emb_final.detach().cpu(),
       'item_emb': items_emb_final.detach().cpu()},
      'embeddings_checkpoint.pt'
  )
  ```
  This is the **only** change to `main.py`. The training loop, `pareto_loss`, and MGDA
  gradient balancing are completely untouched.

### Key decisions

- A fresh `model.forward()` call (not reusing the last training-loop forward pass) ensures
  that the saved embeddings reflect the test-time (eval-mode) computation graph, consistent
  with what the baseline evaluation uses.
- Embeddings are saved to CPU to be device-agnostic for the MORL loading step.

### Files created / modified

| File | Change |
|------|--------|
| `code/main.py` | +8 lines after wandb logging block (post test eval) |

### Next-phase notes

- `morl_main.py` expects the checkpoint at `embeddings_checkpoint.pt` relative to the `code/`
  working directory (same directory as `main.py`).
- Tag tensors (`graph['user'].tags`, `graph['food'].tags`) are loaded directly from the graph
  file in `morl_main.py`; no additional preprocessing needed.

---

## Phase 2: Candidate Pool Construction — COMPLETE

**Date:** 2026-03-11

### What was implemented

- **`code/morl/environment.py`** — `build_candidate_pools()` function:
  - Computes `score[u, i] = user_emb[u] · item_emb[i]` in batches (512 users/batch).
  - Selects top-M items per user via `torch.topk`.
  - Supports optional masking of training/validation positives (used at eval time only).

### Key decisions

- **No exclusion during RL training**: candidate pools during training include all top-M items
  regardless of training-positive status. This is deliberate — the frozen GNN embeddings reflect
  the full interaction distribution, and the RL agent learns to re-rank within this space.
- **Exclusion at evaluation time**: `exclude_per_user` dicts are passed to `build_candidate_pools`
  for validation and test evaluation (consistent with standard RecSys protocol).
- Pool size `M=200` by default; configurable via `--M` CLI flag.

### Files created / modified

| File | Change |
|------|--------|
| `code/morl/environment.py` | New file — `build_candidate_pools()` + `RecommendationEnv` class |
| `code/morl/__init__.py` | New file — package docstring |

---

## Phase 3: MDP Definition — COMPLETE

**Date:** 2026-03-11

### What was implemented

- **`code/morl/environment.py`** — `RecommendationEnv` class:
  - `reset(user_id)` → initialises episode state and returns `s_0`.
  - `step(action)` → transitions state, computes reward vector, returns `(s_{t+1}, r_t, done)`.
  - State: `concat(user_emb, agg_emb, tag_coverage, [t/K])`.
  - Action: integer index into `env.remaining` (the un-selected portion of the candidate pool).
  - Terminal condition: `t == K` or pool exhausted.

### Key decisions

- **Incremental mean aggregation**: `agg_emb` is updated via `(agg * (t-1) + item_emb) / t`
  to avoid storing the full selected-item list on GPU.
- **Tag coverage as binary float**: stored as a float tensor `[0, 1]` rather than bool to allow
  smooth gradient flow if needed in future extensions.
- **State dimensionality**: `2*d + tag_dim + 1` where `d = embedding_dim`, `tag_dim` from graph.

---

## Phase 4: Multi-Objective Reward Structure — COMPLETE

**Date:** 2026-03-11

### What was implemented

- **`code/morl/environment.py`** — `RecommendationEnv.step()` reward computation:
  - `r_pref`: cosine similarity between L2-normalised user and item embeddings.
  - `r_health`: Jaccard similarity between cumulative tag coverage and user health tags.
  - `r_div`: negative mean cosine similarity between the new item and all previously selected
    items (0.0 at the first step when there is nothing to compare against).

### Key decisions

- Pre-computed L2-normalised embeddings (`_user_emb_norm`, `_item_emb_norm`) avoid redundant
  normalisation inside the hot loop.
- `r_health` uses the **updated** tag coverage (after adding the new item) — this rewards
  cumulative tag breadth rather than per-item tag overlap.
- Scalarisation happens **outside** the environment (in the training loop), keeping the reward
  signal multi-objective throughout.

---

## Phase 5: Conditional MORL Policy — COMPLETE

**Date:** 2026-03-11

### What was implemented

- **`code/morl/policy.py`** — `ConditionalPolicy` class:
  - 3-layer MLP: `(state_dim + 3) → 256 → 256 → M`.
  - `forward(state, weight, mask)` → log-probabilities over the full candidate pool.
  - `select_action(state, weight, remaining_indices, num_candidates, greedy)` → samples or
    argmaxes from the restricted remaining pool.
- **`code/morl/policy.py`** — `sample_weight_vector()` helper:
  - Samples `w ~ Dirichlet(α * ones_3)` for use during training episodes.

### Key decisions

- **Masking approach**: the policy outputs logits over the full pool of size `M`. Actions for
  already-selected items are masked to `−∞` before softmax. This avoids the need to re-index
  the output head each step and is efficient for the discrete action space.
- **`select_action` interface**: returns a *local* action index into `remaining_indices` (0-based)
  so it is directly compatible with `env.step(action)`.

---

## Phase 6: Training Loop — COMPLETE

**Date:** 2026-03-11

### What was implemented

- **`code/morl/training.py`** — `train_morl()` function:
  - REINFORCE with per-batch mean-return baseline.
  - `num_epochs=200`, `batch_size=64`, `lr=1e-3` defaults.
  - Checkpoint saved every 10 epochs and at the end of training.
  - Returns the trained `ConditionalPolicy`.
- **`code/morl/training.py`** — `evaluate_morl()` function:
  - Greedy rollout for a given weight vector `w`.
  - Computes NDCG@K, health score, diversity, recall@K.

### Key decisions

- **No gradient into embeddings**: `user_emb` and `item_emb` are never passed to the optimizer
  and are always accessed as read-only tensors inside the environment.
- **Baseline**: simple mean-return per batch (sufficient for discrete action space on A100).
- **Checkpoint strategy**: every 10 epochs + final; allows resuming from an intermediate state.

---

## Phase 7: Trade-Off Selection — COMPLETE

**Date:** 2026-03-11

### What was implemented

- **`code/morl/morl_main.py`** — CLI entry point:
  - Loads embeddings and graph; reproduces 60/20/20 split.
  - Calls `train_morl()` (Phases 3–6).
  - Evaluates 15 weight vectors on validation split.
  - Selects `w*` via: maximise `0.7 * NDCG + 0.3 * Health` subject to `Diversity ≥ median_div`.
  - Final evaluation on test split; saves results to `morl_output/test_results.pt`.

### Key decisions

- **Weight grid**: 3 corner points + uniform + 11 random Dirichlet samples = 15 total.
- **Selection rule** (Option A from plan): `α=0.7`, `β=0.3`; diversity threshold = median across
  the validation grid. Can be overridden with `--val_weight_alpha`.
- **Comparison baseline**: baseline one-shot Pareto ranking metrics are available from the
  `main.py` terminal output; explicit comparison table can be constructed post-hoc.

### Files created / modified

| File | Change |
|------|--------|
| `implementation_plan.md` | New file — comprehensive phase-by-phase plan |
| `auto_logs.md` | New file — this log |
| `code/main.py` | +8 lines (embedding checkpoint save) |
| `code/morl/__init__.py` | New file |
| `code/morl/environment.py` | New file — `RecommendationEnv` + `build_candidate_pools` |
| `code/morl/policy.py` | New file — `ConditionalPolicy` + `sample_weight_vector` |
| `code/morl/training.py` | New file — `train_morl` + `evaluate_morl` |
| `code/morl/morl_main.py` | New file — CLI entry point |

---

## Notes for Future Sessions

- To run the full MORL pipeline after SGSL training:
  ```bash
  cd code
  python main.py             # trains SGSL+MGDA; saves embeddings_checkpoint.pt
  python -m morl.morl_main   # trains MORL policy; evaluates; saves morl_output/
  ```
- GPU recommended (A100); falls back to CPU automatically.
- To adjust hyperparameters: use CLI flags (`--epochs`, `--batch_size`, `--K`, `--M`, `--lr`).
- Tag tensors must exist on `graph['user'].tags` and `graph['food'].tags`; the `benchmark_macro.pt`
  graph file satisfies this requirement (confirmed from `main.py` code).
