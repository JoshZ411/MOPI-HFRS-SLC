# MOPI-HFRS Sequential MORL Implementation Plan

## Overview

This document provides a comprehensive phase-by-phase implementation plan for augmenting the
MOPI-HFRS pipeline with a sequential Multi-Objective Reinforcement Learning (MORL) recommendation
stage. The MORL stage operates on frozen embeddings produced by the SGSL+MGDA training phase,
without modifying that training phase.

### Pipeline Architecture

```
SGSL Training (MGDA multi-obj balancing)
    → Frozen user/item embeddings
    → MORL sequential list construction
    → Evaluation: baseline (one-shot) vs MORL (sequential)
```

### Critical Constraints

- **DO NOT** modify SGSL backbone, graph preprocessing, or health tag enrichment.
- **DO NOT** alter the GNN training loop or `pareto_loss` (MGDA gradient balancing).
- **DO NOT** modify `main.py` training mechanics — only add one checkpoint-save call at the end.
- SGSL + MGDA training completes unchanged; embeddings are frozen for RL.
- MORL replaces the inference-time recommendation mechanism (one-shot Pareto ranking → sequential
  list construction).

---

## Phase 1: Embedding Extraction (Post-Training)

**Goal:** Save frozen embeddings after SGSL training completes; prepare them for MORL.

**File changes:**
- `code/main.py` — Add a single `torch.save(...)` call after test evaluation.

**Steps:**

1. Run `python main.py` normally; verify SGSL/MGDA training completes unchanged.
2. After the test evaluation block (the last `with torch.no_grad()` block), add:
   ```python
   torch.save(
       {'user_emb': users_emb_final.detach().cpu(),
        'item_emb': items_emb_final.detach().cpu()},
       'embeddings_checkpoint.pt'
   )
   ```
3. Create `code/morl/morl_main.py` that loads the checkpoint and proceeds through Phases 2–6.

**Deliverable:** `embeddings_checkpoint.pt` written to `code/` after every training run.

---

## Phase 2: Candidate Pool Construction

**Goal:** Define a fixed discrete action space per user for the sequential MDP.

**File:** `code/morl/morl_main.py` (and helper in `environment.py`)

**Steps:**

1. Load frozen embeddings from `embeddings_checkpoint.pt`.
2. Compute dot-product scores: `score[u, i] = user_emb[u] · item_emb[i]`.
3. Take `top-M` items per user (default `M=200`).
4. During **RL training**, include all top-M items (no exclusion of training positives).
5. During **evaluation/test**, mask out training and validation positive edges before ranking.

**Key design choice:** Candidate pools are computed once and kept fixed throughout RL training.

**Deliverable:** `candidate_pools: dict[int, List[int]]` mapping user index → list of M item indices.

---

## Phase 3: MDP Definition (Sequential List Construction)

**Goal:** Formalise a deterministic K-step episode environment.

**File:** `code/morl/environment.py`

### State `s_t`

| Component | Description | Dimension |
|-----------|-------------|-----------|
| `user_emb` | frozen user embedding | `d` |
| `agg_emb` | mean of selected item embeddings (zero at t=0) | `d` |
| `tag_coverage` | cumulative binary union of selected items' health tags | `tag_dim` |
| `timestep` | normalised `t / K` | `1` |

Total: `2d + tag_dim + 1`

### Action

Select one item from the candidate pool that has not yet been chosen in this episode.
If the remaining pool is exhausted before K steps, the episode terminates early.

### Transition

```
append selected item to list
agg_emb ← mean(agg_emb * t + item_emb) / (t + 1)   [incremental mean]
tag_coverage ← tag_coverage | item_tags               [bitwise OR]
t ← t + 1
```

### Episode end

After `K = 20` selections (or early termination if pool exhausted).

**Deliverable:** `RecommendationEnv` class in `code/morl/environment.py`.

---

## Phase 4: Multi-Objective Reward Structure

**Goal:** Define decomposed, interpretable per-step rewards.

**File:** `code/morl/environment.py` (reward methods)

### Reward components

| Signal | Formula | Meaning |
|--------|---------|---------|
| `r_pref` | `user_emb · item_emb` (dot product, normalised) | Preference alignment |
| `r_health` | Jaccard(`tag_coverage_new`, `user_tags`) | Health tag coverage |
| `r_div` | `−mean_cosine_sim(item_emb, selected_embs)` | Diversity (penalise redundancy) |

The environment returns `r_t = [r_pref_t, r_health_t, r_div_t]` at each step.
Scalarisation (`w · r_t`) is performed inside the training loop, not in the environment.

**Deliverable:** Environment `step()` returns `(next_state, reward_vector, done)`.

---

## Phase 5: Conditional MORL Policy

**Goal:** Learn a single trade-off-aware policy `π(a | s, w)`.

**File:** `code/morl/policy.py`

### Architecture

```
input: concat(s_t, w)   — shape: (2d + tag_dim + 1 + 3,)
→ Linear(input_dim, 256) + ReLU
→ Linear(256, 256) + ReLU
→ Linear(256, M)         — logit for each candidate item
→ mask out already-selected items (set to −inf before softmax)
→ softmax → action probabilities
```

Weight vector `w ∈ ℝ³`, `w_i ≥ 0`, `sum(w_i) = 1` is sampled from `Dirichlet(α=[1,1,1])`.

### Training

- REINFORCE with a simple mean-return baseline per batch.
- No gradients flow into GNN embeddings.

**Deliverable:** `ConditionalPolicy` class in `code/morl/policy.py`.

---

## Phase 6: Training Loop

**Goal:** Stable episodic training across diverse trade-off preferences.

**File:** `code/morl/training.py`

### Configuration

| Parameter | Value |
|-----------|-------|
| `batch_size` | 64 users per gradient step |
| `num_epochs` | 200 |
| `K` | 20 (list length) |
| `M` | 200 (candidate pool size) |
| `lr` | 1e-3 |
| `weight_sampling` | `Dirichlet(α=[1,1,1])` |
| `device` | `cuda` (A100) |

### Algorithm

```
for epoch in range(num_epochs):
    sample batch of B users
    for each user u:
        sample weight vector w ~ Dirichlet([1,1,1])
        run K-step episode conditioned on w:
            collect {s_t, a_t, r_t}
        R_episode = Σ_t  w · r_t
    policy_loss = −Σ_u Σ_t log π(a_t | s_t, w) * (R_episode − baseline)
    optimise policy_loss w.r.t. policy parameters only
checkpoint every 10 epochs: {policy_weights, training_stats}
```

**Deliverable:** `train_morl()` function in `code/morl/training.py`.

---

## Phase 7: Trade-Off Selection via Validation Metrics

**Goal:** Select optimal operating weight vector `w*` empirically.

**File:** `code/morl/morl_main.py` (evaluation section)

**Steps:**

1. Define evaluation grid `W_eval` of 15 weight vectors (corner + uniform + random samples from simplex).
2. For each `w ∈ W_eval`:
   - Generate Top-K lists on validation split using `π(· | s, w)`.
   - Compute NDCG@K, health score, diversity score.
3. Select `w*` via: maximise `0.7 * NDCG + 0.3 * Health` subject to `Diversity ≥ median_diversity`.
4. Evaluate `π(· | s, w*)` on test split.
5. Compare with baseline one-shot Pareto ranking from `main.py`.

**Deliverable:** Final test-set metrics under `w*`; comparison table vs. baseline.

---

## Logging and Documentation

An `auto_logs.md` file is maintained in the repository root and updated after each phase with:

- Phase completion summary (files created/modified).
- Key implementation decisions.
- Git commit references.
- Blockers and resolutions.
- Next-phase notes.

---

## Contribution Statement Alignment

| Component | Role |
|-----------|------|
| **SGSL + MGDA** | Multi-objective gradient balancing in embedding space during GNN training. Produces a single set of compromised embeddings. |
| **MORL** | Takes frozen embeddings; constructs recommendations sequentially. Learns `π(a|s,w)` that adapts to preference weights. Performs horizon-aware allocation of objectives across K recommendation steps. |

**Key difference:**
- MGDA: myopic, parameter-space compromise **during training**.
- MORL: foresighted, policy-space allocation **during recommendation** (inference).

Both operate on the identical frozen embeddings.

---

## End Condition

The system produces:
- SGSL-trained GNN with frozen embeddings (MGDA-balanced).
- Trained conditional MORL policy `π(a | s, w)`.
- Validation-selected trade-off weight `w*`.

Evaluation compares:
- **Baseline:** one-shot Pareto ranking (`main.py` inference) on the same frozen embeddings.
- **MORL:** sequential list construction using trained policy at `w*`.
- **Metrics:** NDCG@K, health score, diversity score, coverage on the test set.
