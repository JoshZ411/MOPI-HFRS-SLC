# Sequential MORL for MOPI-HFRS

This document describes the **implicit sequential Multi-Objective Reinforcement
Learning (MORL)** stage added to the MOPI-HFRS pipeline.

---

## Architecture

```
SGSL Training (MGDA multi-objective balancing)
    ↓  unchanged
Frozen user / item embeddings  ←  embeddings_checkpoint.pt
    ↓
Implicit Sequential MORL (code/seqmorl/)
    ↓
Single-policy evaluation on val / test splits
    ↓
Comparison: one-shot baseline  vs  sequential policy
```

### Why implicit (no explicit preference vector)?

Prior experiments with weight-conditioned MORL surfaced three failure modes:

1. **Conditioning collapse** — the policy ignored the weight vector; all users
   received near-identical recommendations regardless of `w`.
2. **Operating-point bottleneck** — a grid-search over `w*` introduced an extra
   selector stage that masked true policy learning.
3. **High-variance scalarised REINFORCE** — sampling Dirichlet weights per
   episode amplified gradient variance and destabilised optimisation.

The implicit design eliminates all three by handling objective balancing at
*gradient level* via MGDA (the same algorithm used in the SGSL training stage),
not at inference time.

---

## Key design decisions

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Policy input | `state` only (no `w`) | Removes weight-dominance bottleneck |
| Objective balancing | MGDA min-norm gradient combination | Mirrors existing `pareto_loss` pattern |
| Candidate pool | Top-M (M=200) from frozen embeddings | Avoids very large M degeneracy |
| State | `[user_emb ‖ item_agg_emb ‖ tag_coverage ‖ t/K]` | Captures user preference, current list, health, position |
| Reward | `[r_pref, r_health, r_div]` per step | Explicit objectives, implicit combination at update |
| Baseline | Per-objective EMA | Reduces policy-gradient variance per objective |
| Entropy | Monitored via W&B | Collapse guard: `mean_entropy < 0.05` → pause |

---

## State and reward specification

**State dimension** = `2×dim + num_tags + 1`

- `user_emb [dim]`             — frozen user embedding
- `item_agg_emb [dim]`         — running mean of selected item embeddings (0 initially)
- `tag_coverage [num_tags]`    — binary union of selected item health tags
- `t/K [1]`                    — normalised timestep

**Reward per step**

| Channel | Formula |
|---------|---------|
| `r_pref`   | cosine\_similarity(user\_emb, item\_emb) |
| `r_health` | Jaccard(user\_tags, item\_tags) |
| `r_div`    | 1 − cosine\_similarity(item\_emb, item\_agg\_emb) |

---

## MGDA gradient combination (training)

For each epoch the training loop:

1. Rolls out K-step episodes for all training users.
2. Computes per-objective REINFORCE losses
   `L_obj = -E[log π(a|s) · A_obj]`
   using per-objective advantage `A_obj = R_obj − baseline_obj`.
3. Collects per-parameter gradients for each objective.
4. L2-normalises gradients (same `gradient_normalizers` as `pareto_loss`).
5. Solves the min-norm convex combination via `MinNormSolver.find_min_norm_element_FW`.
6. Forms combined update and steps the optimiser.

No gradients flow into frozen SGSL/GNN parameters.

---

## W&B diagnostics and collapse guardrails

All metrics are logged with prefix `train/` to a W&B offline run.

| Metric | Critical threshold |
|--------|--------------------|
| `mgda_coeff_{obj}` | Any > 0.95 sustained → objective collapse |
| `objective_dominance_ratio` | > 10 for 50+ epochs → PAUSE |
| `mean_entropy` | < 0.05 → mode collapse |
| `probe_first_action_span` | ≈ 0 → prior failure symptom |

Inspect a live offline run with:

```bash
python -m wandb beta leet run <offline-run-path>
```

The exact command is saved to `<output_dir>/wandb_leet_command.txt` after training.

---

## File structure

```
code/seqmorl/
  __init__.py          Package exports
  environment.py       Sequential MDP (state / action / reward)
  policy.py            MLP policy with action masking
  training.py          MGDA-style policy gradient training loop
  evaluation.py        Greedy rollout + metric computation
  logging_utils.py     WandbTracker (offline) + JSONL helpers
  seqmorl_main.py      CLI entry point
  README_SEQMORL.md    This document
```

---

## Run commands

### Development / laptop (CPU)

```bash
cd code
python seqmorl/seqmorl_main.py \
    --device cpu \
    --epochs 500 \
    --M 200 \
    --K 20 \
    --lr 1e-3 \
    --output_dir seqmorl_output
```

### GPU cluster (e.g. A100)

```bash
cd code
python seqmorl/seqmorl_main.py \
    --device cuda \
    --epochs 500 \
    --M 200 \
    --K 20 \
    --lr 1e-3 \
    --output_dir seqmorl_output \
    --use_wandb
```

### Auto-detect (recommended default)

```bash
cd code
python seqmorl/seqmorl_main.py \
    --device auto \
    --epochs 500 \
    --use_wandb \
    --output_dir seqmorl_output
```

### Prerequisites

1. Run `python main.py` first to produce `embeddings_checkpoint.pt`.
2. `embeddings_checkpoint.pt` must be present in the working directory (or
   supply `--checkpoint <path>`).
3. Graph data at `../processed_data/benchmark_macro.pt` (or `--graph <path>`).

---

## Acceptance criteria

The pivot is considered successful when **all** of the following hold:

1. Inference requires no tradeoff vector (`--w` flag does not exist).
2. Sequential policy is trained with implicit MGDA balancing (no Dirichlet sampling).
3. Evaluation uses a single greedy rollout path.
4. W&B offline run contains all required metrics without collapse warnings.
5. `--device cpu` and `--device cuda` (where available) both run without errors.
6. Sequential metrics are at least competitive with one-shot baseline on test split.
