# MORL Implementation Log

---

## Phase 1: Embedding Extraction (Post-Training)

**Date:** 2026-03-11

### What was implemented
- Modified `code/main.py` to save frozen user and item embeddings after SGSL test evaluation completes.
- Added a forward pass inside the existing `torch.no_grad()` block to obtain `users_emb_final` and `items_emb_final`.
- Saved both tensors (CPU, detached) to `embeddings_checkpoint.pt` via `torch.save`.

### Files modified
- `code/main.py` — minimal addition at the end of the `with torch.no_grad()` block; no changes to training loop or `pareto_loss`.

### Key decisions
- Used the test-time forward pass (with `test_edge_index`, `pos_test_edge_index`, `neg_test_edge_index`) to obtain the final embeddings after full training.
- Saved to `embeddings_checkpoint.pt` in the working directory (same as main.py invocation directory).

### Code snippet (critical addition)
```python
# Save frozen embeddings for downstream MORL training (Phase 1)
torch.save({'user_emb': users_emb_final.detach().cpu(),
            'item_emb': items_emb_final.detach().cpu()},
           'embeddings_checkpoint.pt')
print("Frozen embeddings saved to embeddings_checkpoint.pt")
```

### Next-phase notes
- MORL module loads this checkpoint; no SGSL parameters are involved beyond this point.

---

## Phase 2 & 3: Candidate Pool Construction + MDP Environment

**Date:** 2026-03-11

### What was implemented
- Created `code/morl/environment.py` containing:
  - `build_candidate_pools(user_emb, item_emb, M, exclude_edges)` — computes top-M items per user via dot-product scoring; supports optional exclusion of seen interactions at evaluation time.
  - `RecommendationEnv` — deterministic K-step MDP with the following state/action/transition/reward spec:

#### State (s_t)
```
concat(
  user_emb         (d),
  aggregated_emb   (d),   ← mean pooling of selected items; zero vector at t=0
  tag_coverage     (tag_dim),  ← binary union of selected items' health tags
  normalized_t     (1)    ← t/K
)
```
Total dimension: `2·d + tag_dim + 1`

#### Action
Index into the current remaining candidate pool (masked to prevent re-selection).

#### Transition
- Append item to list.
- Incremental mean update for `aggregated_emb`.
- Bitwise OR (clamped to 1) for `tag_coverage`.
- Increment `t`.

#### Reward vector per step: `[r_pref, r_health, r_div]`
- `r_pref` — dot product user_emb · item_emb.
- `r_health` — Jaccard similarity between (tag_coverage ∪ item_tag) and user_tags.
- `r_div` — negative mean cosine similarity of new item to already-selected items.

### Files created
- `code/morl/__init__.py`
- `code/morl/environment.py`

### Key decisions
- Candidate pool exclusion: **NOT** applied during RL training (pools include training positives); exclusion applied at eval/test time.
- Candidate pool size M defaults to 200; tunable via CLI.
- Episode terminates early if pool is exhausted (`|remaining| = 0`).

---

## Phase 4: Multi-Objective Reward Structure

**Date:** 2026-03-11

### What was implemented
Reward computation is integrated directly in `RecommendationEnv._compute_reward()`.

Components:
- `r_pref_t` — preference signal; simple dot product (proxy for BPR ranking quality).
- `r_health_t` — Jaccard similarity of running tag coverage union vs user profile tags.
- `r_div_t` — negative mean cosine similarity to already-selected items; promotes intra-list diversity.

Environment returns a `(3,)` reward vector per step. Scalarization (`w · r_t`) happens only in the training loop.

---

## Phase 5: Conditional MORL Policy

**Date:** 2026-03-11

### What was implemented
- Created `code/morl/policy.py` containing:
  - `MORLPolicy` — conditional policy π(a | s, w) with architecture:
    - Input: extended state `s'_t = concat(s_t, w)` of dimension `state_dim + 3`.
    - State encoder: 2-layer MLP with ReLU activations and dropout → hidden_dim.
    - Candidate scoring: dot product between encoded state and linearly projected candidate embeddings.
    - Output: log-softmax over candidate pool.
  - `sample_weight_vector(batch_size, alpha, device)` — Dirichlet(α) sampler for preference weight vectors.

### Key decisions
- Dot-product scoring between encoded state and projected candidates scales efficiently to large M.
- Greedy action selection at inference time (argmax); multinomial sampling during training.

---

## Phase 6: Training Loop

**Date:** 2026-03-11

### What was implemented
- Created `code/morl/training.py` containing `train_morl()`:
  - Per gradient step: sample B users, one weight vector w per user, run K-step episode.
  - REINFORCE with running-mean baseline (momentum 0.99) for variance reduction.
  - Gradient clipping (default max norm 1.0).
  - Saves best policy checkpoint (highest mean return).
  - Logs per-objective mean returns at configurable intervals.

### Configuration defaults
| Parameter | Value |
|-----------|-------|
| batch_size | 32 |
| lr | 1e-3 |
| dirichlet_alpha | 1.0 |
| grad_clip | 1.0 |
| num_episodes | 2000 |

### Key decisions
- No gradients flow into GNN embeddings (frozen; only policy parameters updated).
- REINFORCE chosen for simplicity and flexibility with variable-length episodes.
- Baseline = running exponential moving average of scalar returns.

---

## Phase 7: Trade-off Selection via Validation Metrics

**Date:** 2026-03-11

### What was implemented
- Created `code/morl/evaluation.py` containing:
  - `generate_recommendations` — deterministic greedy list construction under fixed w.
  - `ndcg_at_k`, `health_score_at_k`, `diversity_score_at_k` — per-user metric functions.
  - `evaluate_policy_on_split` — compute mean NDCG, Health, Diversity, Recall over a user set.
  - `select_operating_weight` — grid search over ~100 simplex weight vectors to find w*:
    - **Option A**: maximize α·NDCG + β·Health subject to diversity ≥ threshold.
    - Falls back to relaxed constraint if no point meets diversity threshold.

### Key decisions
- Grid size: ~100 points (n_grid=10) balances coverage vs evaluation cost.
- Include corner weights (1,0,0), (0,1,0), (0,0,1) and uniform (1/3,1/3,1/3) in grid.
- Default selection: α=0.7, β=0.3, diversity_threshold=0.3 (tunable via CLI).

---

## Entry Point

**Date:** 2026-03-11

### What was implemented
- Created `code/morl/morl_main.py`:
  - Full pipeline: load embeddings → build pools → construct environment → train policy → select w* → evaluate and compare vs baseline.
  - Accepts all hyperparameters via argparse.
  - Compares MORL (w*) vs one-shot dot-product baseline on the test set.
  - Saves results summary to `morl_results.pt`.

### Run command
```bash
cd code
python -m morl.morl_main \
    --embeddings_path embeddings_checkpoint.pt \
    --graph_path ../processed_data/benchmark_macro.pt \
    --K 20 --M 200 --num_episodes 2000 --batch_size 32 --lr 1e-3
```

---

## File Summary

| File | Role |
|------|------|
| `code/main.py` | SGSL training (unchanged) + embedding checkpoint save (1 addition) |
| `code/morl/__init__.py` | Package marker |
| `code/morl/environment.py` | MDP: state, action, transition, reward; candidate pool builder |
| `code/morl/policy.py` | Conditional policy π(a\|s,w); weight sampler |
| `code/morl/training.py` | REINFORCE training loop with baseline |
| `code/morl/evaluation.py` | Metrics; w* grid search |
| `code/morl/morl_main.py` | Full pipeline entry point |
