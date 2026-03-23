# Implicit Sequential MORL Pivot Automation Plan

You are an AI coding agent operating inside the MOPI-HFRS repository.

Your objective is to augment the original MOPI-HFRS pipeline with a sequential recommendation stage that remains consistent with original one-shot inference semantics:
- no user-provided tradeoff vector,
- one deployed scoring policy at inference,
- multi-objective balancing handled implicitly during training.

This pivot implements an implicit multi-objective sequential RL design (no explicit weight conditioning) that uses MGDA-style gradient combination at update time.

## CRITICAL CONSTRAINTS

- DO NOT modify SGSL backbone architecture.
- DO NOT modify graph preprocessing or health tag enrichment.
- DO NOT alter core MGDA mechanics in existing GNN training (`pareto_loss`, min-norm solver logic).
- DO NOT require any inference-time preference/tradeoff vector input.
- Build new sequential modules from scratch in a new package path.
- Sequential training/inference MUST support both CPU and CUDA via runtime toggle (`--device cpu|cuda|auto`) without code changes.

---

## MANDATORY PREREQUISITE: Deep Codebase Understanding

**BEFORE starting any implementation, you MUST thoroughly read and understand the following:**

1. **Original MGDA Framework (SGSL Training)**
   - Read `code/main.py` end-to-end: understand SGSL training loop, checkpoint flow, baseline evaluation.
   - Read `code/RCSYS_utils.py` lines 514–551 (pareto_loss pattern): how multi-objective gradients are aggregated, how objectives are defined (BPR, health, diversity).
   - Read `code/min_norm_solvers.py`: MinNormSolver algorithm, gradient normalization utilities, how the solver produces convex combination coefficients.
   - Read `code/RCSYS_models.py`: SGSL model architecture, embedding dimensions, forward pass semantics.
   - Document in your implementation notes: exact line-by-line flow of a multi-objective training step in the original framework.

2. **Existing Evaluation and Metrics Pipeline**
   - Read how baseline SGSL evaluates on val/test splits: metrics computation (NDCG, recall, health, diversity) in `code/main.py`.
   - Document: what metrics are used, how they are computed, where in code they are defined.

3. **W&B Logging Patterns (from prior sequential RL experiments)**
   
   You MUST implement W&B logging that tracks these per-epoch metrics during implicit MORL training:
   
   **Per-objective metrics (pref, health, div):**
   - `train/mean_reward_X` and `train/std_reward_X`: objective performance and variance per epoch
   - `train/objective_grad_norm_X`: gradient magnitude BEFORE L2 normalization (shows raw objective strength)
   - `train/mgda_coeff_X`: MGDA solver coefficients per objective (PRIMARY collapse detector - watch for any > 0.95)
   - `train/cumulative_advantage_X`: per-objective advantage signal (should increase if learning, plateau = collapse)
   
   **Aggregate metrics:**
   - `train/policy_loss`: combined policy gradient loss per epoch
   - `train/grad_norm`: gradient norm AFTER MGDA combination
   - `train/objective_dominance_ratio`: max(mgda_coeffs) / min(mgda_coeffs) - **CRITICAL**: if sustained > 10 for 50+ epochs = prior MORL failure recurrence
   - `train/mean_entropy`: action distribution entropy - **CRITICAL**: if drops below 0.05 = mode collapse (prior failure)
   - `train/mean_action_position`: which positions in candidate pool being selected
   
   **Probing diagnostics (every 25 epochs):**
   - `train/probe_first_action_span`: action variation across users - **CRITICAL**: if near 0 = prior MORL failure symptom
   - `train/probe_pairwise_jaccard`: action set diversity
   
   **Implementation:** Create a WandbTracker class (optional, can be toggled off) that wraps wandb.init() with offline mode and generates leet command:
   
   ```python
   tracker = WandbTracker(enabled=True, project='seqmorl-implicit', mode='offline', 
                         base_dir=output_dir, config=args)
   # Each epoch in training loop:
   tracker.log({f'train/{key}': value for key, value in epoch_stats.items()}, step=epoch)
   # At end of training:
   leet_cmd = tracker.leet_command()  # e.g., "python -m wandb beta leet run ..."
   save_json(os.path.join(output_dir, 'wandb_leet_command.txt'), {'command': leet_cmd})
   ```

**Verification checklist (do NOT skip):**
- [ ] Can you explain the full forward/backward pass of a multi-objective SGSL training step in `code/main.py`?
- [ ] Can you explain why the implicit MORL design avoids the prior failure mode: no explicit weight conditioning means weight-dominance bottleneck cannot exist?
- [ ] Can you implement a WandbTracker class that logs metrics with offline mode support and generates `wandb beta leet run` command?
- [ ] Can you identify the THREE critical W&B diagnostics that catch prior MORL failure modes: objective_dominance_ratio (should stay < 2), action_entropy (should stay > 0.5), probe_first_action_span (should not be 0)?

**If you cannot answer all four with specific design rationale and W&B metrics, DO NOT proceed. The implicit design is fundamentally different from weight-conditioned MORL - must understand why first.**

## Pipeline architecture

SGSL Training (existing MGDA multi-objective balancing)
-> Frozen user/item embeddings
-> Implicit Sequential MORL training (no explicit w)
-> Single-policy evaluation on val/test
-> Baseline (one-shot) vs Sequential (list-wise) comparison

Implementation contract:
- Sequential stage is an inference-time augmentation on top of frozen embeddings, not a replacement for SGSL training.
- SGSL training phase remains unchanged; only a minimal embedding checkpoint handshake is allowed.
- Baseline and sequential policy must be compared on identical splits and identical frozen embeddings.

---

## Why this design is chosen (lessons from prior experiments)

Prior experiments surfaced several issues that this design must avoid:

1. Inference mismatch with product intent.
The explicit framework required tradeoff vectors and post-training weight selection (`w*`), while intended behavior is one-shot per-user inference without preference input.

2. Conditioning collapse risk.
Weight-conditioned policy behavior collapsed in some runs (`first_action_span` near zero), indicating unstable dependence on explicit weight inputs.

3. Coarse operating-point selection.
Validation weight-grid search introduced an additional selector bottleneck that could mask policy learning and produce corner solutions.

4. High-variance scalarized REINFORCE signal.
Sampling weights and scalarizing returns per episode increased variance and made optimization less stable.

This plan removes those failure modes by making objective balancing implicit in optimization (MGDA-style gradient combination across objective-specific policy gradients).

---

## Existing code references to reuse

Primary reference files in the original framework:

- `code/main.py`
  - Existing SGSL training and evaluation orchestration.
  - Keep one-shot baseline path intact for comparison.

- `code/RCSYS_utils.py`
  - `pareto_loss` pattern for multi-objective gradient aggregation.
  - Existing objective definitions and evaluation helpers.

- `code/min_norm_solvers.py`
  - `MinNormSolver.find_min_norm_element_FW`.
  - `gradient_normalizers` for scale balancing.

- `code/RCSYS_models.py`, `code/utils.py`
  - Existing embedding and metric ecosystem; do not redesign unless required.

These references define the intended implicit balancing philosophy and must guide sequential RL update design.

---

## New code organization (clean branch assumption: no `code/morl`)

Create a new package:

- `code/seqmorl/`
  - `__init__.py`
  - `environment.py`
  - `policy.py`
  - `training.py`
  - `evaluation.py`
  - `seqmorl_main.py`
  - `README_SEQMORL.md`

Keep this new package fully decoupled from SGSL training internals except for reading frozen embeddings and graph/tag artifacts.

---

## IMPLEMENTATION PHASES

## Phase 0: Scope lock and baseline contract

Goal: lock behavior and establish W&B instrumentation with failure-mode detection.

Steps:
1. Confirm baseline one-shot SGSL metrics pipeline is unchanged and runnable.
2. Define hard acceptance criteria:
   - no inference-time tradeoff input,
   - one trained sequential policy,
   - improved or competitive list-level utility versus one-shot baseline,
   - stable implicit multi-objective balancing (MGDA coefficients balanced, no single objective > 0.9 consistently).
3. Add cross-hardware execution contract (laptop CPU and cluster GPU):
   - Add CLI flag `--device` to `seqmorl_main.py` with choices: `cpu`, `cuda`, `auto`.
   - Implement `resolve_device()` behavior:
     - `auto`: use CUDA if available, else CPU.
     - `cpu`: force CPU execution.
     - `cuda`: require CUDA; fail with clear message if unavailable.
   - Ensure tensors, model, and rollout buffers are moved consistently to resolved device.
   - Keep checkpoint format portable so CPU can load checkpoints trained on GPU (`map_location` handling).
4. Validate parity by running smoke tests in both modes:
   - local laptop: `--device cpu`
   - cluster/A100: `--device cuda`
   - verify both modes produce train/eval logs and W&B curves without device mismatch errors.
5. Implement W&B infrastructure:
   - Create `code/seqmorl/logging_utils.py` with WandbTracker class (offline mode, leet command generation).
   - Project: `seqmorl-implicit`. All runs log offline for `wandb beta leet` reproducibility.
   - Store leet command in output_dir/wandb_leet_command.txt.
6. Define W&B metrics and collapse detection (from W&B Logging Patterns section above):
   - Per-objective: reward, gradient norm, MGDA coefficients, cumulative advantage.
   - Aggregate: policy_loss, action_entropy, objective_dominance_ratio.
   - Probing (every 25 epochs): first_action_span (MUST NOT be 0), pairwise_jaccard.
   - **Guardrails**: objective_dominance_ratio > 10 for 50+ epochs → PAUSE (prior MORL failure). action_entropy < 0.05 → PAUSE (mode collapse). probe_first_action_span ≈ 0 → PAUSE (prior failure).
7. Log all metrics to local JSONL and W&B.

Deliverable:
- Acceptance gates document.
- WandbTracker implementation with offline mode and leet command support.
- First training run produces valid W&B offline run with all metrics accessible via leet command.
- CPU and CUDA smoke-test run records captured in `auto_logs.md`.

---

## Phase 1: Embedding extraction handshake (minimal SGSL touch)

Goal: ensure frozen embeddings are available to sequential stage.

Steps:
1. Reuse existing SGSL training flow in `code/main.py`.
2. If not already present, add a minimal post-training checkpoint save:
   - `user_emb`, `item_emb` (cpu tensors),
   - optional metadata (`seed`, `hidden_dim`, split identifiers).
3. Save as `code/embeddings_checkpoint.pt`.

Constraints:
- No changes to SGSL training loop mechanics.
- No changes to `pareto_loss` behavior.

Deliverable:
- Deterministic checkpoint output consumed by `seqmorl_main.py`.

---

## Phase 2: Sequential environment and candidate pool

Goal: define deterministic top-K construction MDP with fixed candidate pools.

Steps:
1. Build top-M candidate pools per user from frozen embeddings (dot-product score).
2. Environment state at step t:
   - user embedding,
   - aggregate embedding of selected items,
   - cumulative tag coverage,
   - normalized timestep t/K.
3. Action:
   - select one item from remaining candidate slots.
4. Transition:
   - append item,
   - update aggregate embedding,
   - update coverage,
   - advance timestep.
5. Episode ends at K selections or empty pool.

Design notes:
- Start with M around 200 (avoid very large M degeneracy observed before).
- Candidate pools are computed once at startup and reused across training (no dynamic reranking in baseline implementation).
- Tag coverage state uses binary union (bitwise OR) over selected item tags.
- State dimensionality should be documented explicitly as: user_dim + item_agg_dim + tag_dim + 1.
- Keep reward channels explicit but do not pass tradeoff vectors into policy.

Evaluation masking protocol:
- During evaluation, exclude seen interactions from ranking according to split protocol (train interactions for val/test; plus val interactions when evaluating test if that is the repo baseline rule).
- Keep masking behavior consistent between one-shot baseline and sequential policy evaluation.

Deliverable:
- `code/seqmorl/environment.py` with deterministic step/reset API.

---

## Phase 3: Policy architecture (implicit objective balancing compatible)

Goal: single policy that consumes state only and outputs logits over candidate actions.

Steps:
1. Implement `SequentialPolicy(state) -> logits`.
2. Apply action masking for already-selected items before softmax.
3. Provide both sampling mode (train) and greedy mode (eval).
4. Keep architecture simple for first pass (single shared trunk + action head).

Important carry-over decision:
- Prior objective-head/logit-composition trick is not baseline here.
- Hold it as optional ablation only after implicit baseline is validated.

Deliverable:
- `code/seqmorl/policy.py` with stable masked action selection.

---

## Phase 4: Objective decomposition and trajectory collection

Goal: compute objective-specific returns from shared trajectories.

Steps:
1. For each episode, collect per-step reward vector:
   - preference channel,
   - health channel,
   - diversity channel.
2. Compute objective-specific episode returns:
   - `R_pref`, `R_health`, `R_div`.
3. Build policy-gradient terms per objective from same log-prob trajectory.

Reward semantics clarification:
- Environment should emit full vector reward per step (`[r_pref, r_health, r_div]`).
- Optimization combines objective information at gradient level via MGDA-style aggregation (not via inference-time preference vectors).
- Validation/test must report objective metrics independently (not just a single scalar).

Key distinction from old explicit approach:
- No sampled Dirichlet weights.
- No scalar `w dot r` episode return for primary optimization.

Deliverable:
- Objective-wise rollout buffers in `code/seqmorl/training.py`.

---

## Phase 5: MGDA-style gradient combination inside RL updates

Goal: implicit multi-objective balancing at gradient level with comprehensive W&B instrumentation.

Steps:
1. Construct three objective losses (policy-gradient style), one per objective.
2. Compute per-objective gradients on shared parameters (retain graph as needed).
3. Normalize gradients (start with L2 normalization).
4. Use min-norm solver to obtain convex combination coefficients.
5. Form combined update direction and apply optimizer step.

Training boundary:
- No gradients flow into SGSL/GNN embedding parameters.
- Only sequential policy (and optional sequential value/baseline heads) are trainable in this stage.

Reference pattern:
- Mirror the aggregation workflow used in `code/RCSYS_utils.py` + `code/min_norm_solvers.py`, adapted for policy-gradient objectives.

**W&B Logging (REQUIRED per epoch):**
According to W&B Logging Patterns section, log to W&B with prefix `train/`:
- `objective_grad_norm_pref`, `objective_grad_norm_health`, `objective_grad_norm_div` (norms BEFORE L2 normalization)
- `mgda_coeff_pref`, `mgda_coeff_health`, `mgda_coeff_div` (solver output - PRIMARY collapse detector)
- `cumulative_advantage_pref`, `cumulative_advantage_health`, `cumulative_advantage_div` (per-objective advantage)
- `objective_dominance_ratio`: max(mgda_coeffs) / min(mgda_coeffs); if > 10 for 50+ epochs, pause training
- `grad_norm` (combined norm after MGDA combination)
- `mean_reward_pref`, `mean_reward_health`, `mean_reward_div` + std variants
- Use `tracker.log({...}, step=epoch)` to send all metrics to W&B with step counter for proper alignment on dashboard

**Mid-run diagnostics:**
- Check W&B dashboard every 50 epochs using `wandb beta leet run <offline-run-path>` to visually confirm MGDA coefficients are not dominated (none should stay > 0.95)
- If any coefficient dominates, stop and investigate before continuing training

Required logs (local JSONL + W&B):
- All metrics listed above, per epoch
- Make metrics human-readable for json debugging

Deliverable:
- MGDA-style policy update in `code/seqmorl/training.py` with full W&B instrumentation per epoch.

---

## Phase 6: Variance control and guardrails

Goal: prevent instability and objective collapse.

Steps:
1. Add objective-wise baselines (or moving averages) to form objective advantages.
2. Apply gradient clipping.
3. Add fail-fast diagnostics:
   - objective dominance ratio,
   - collapsed action diversity,
   - near-zero useful learning signal.
4. Implement robust rollout diagnostics for implicit training, incorporating only concepts validated in prior experiments.

Guardrail examples:
- if one objective coefficient stays near 1.0 for long windows, flag imbalance.
- if action distributions collapse across users, flag mode collapse.

**W&B Guardrail Visualization:**
- Plot objective dominance ratio (max coefficient / min coefficient) on W&B; if > 10 for 50+ epochs, trigger alert.
- Plot action entropy per objective on separate W&B lines; if any drops below 0.5 nats, flag as mode collapse risk.
- Plot baseline variance per objective; use W&B annotations to mark high-variance phases.
- Create W&B alerts/triggers for: single objective coefficient > 0.95, action entropy < 0.5 nats, any loss NaN or inf.

Deliverable:
- Stable training loop with early warning logs, rich W&B instrumentation, and mid-run leet viewer inspection capability.

---

## Phase 7: Evaluation design (single-policy)

Goal: evaluate as a single deployed policy.

Steps:
1. Define evaluation protocol with one policy inference path only (no weight-grid search and no deployment-time w* selection).
2. Run one policy inference path on validation and test splits.
3. Compute and log:
   - ndcg, recall/precision,
   - health score,
   - diversity score,
   - optional coverage metrics.
4. Compare directly against one-shot SGSL baseline on same splits.

Deliverable:
- `code/seqmorl/evaluation.py` + evaluation block in `seqmorl_main.py`.

---

## Phase 8: Optional ablations (only after baseline pass)

Goal: improve performance only with isolated, gated changes.

Ablation order:
1. + stronger objective-wise baseline/value estimator.
2. + optional objective-head architecture if objective disentanglement appears weak.
3. + optional terminal shaping if metrics indicate step/final mismatch.

Rules:
- one change at a time,
- fixed seeds and split parity,
- keep or revert strictly by metric and stability gates.

Deliverable:
- Clear keep/revert decisions for each ablation.

---

## Phase 9: Output artifacts and handoff package

Goal: make implementation reproducible and reviewable.

Required outputs:
- trained sequential policy checkpoints,
- run config json,
- train/eval metrics jsonl,
- concise analysis summary markdown,
- baseline vs sequential comparison table.

Documentation:
- update `README_SEQMORL.md` with architecture, equations, and run commands.
- explicitly describe how implicit balancing differs from explicit weight-conditioned MORL.
- include two canonical run commands: one for CPU laptop (`--device cpu`) and one for cluster/A100 (`--device cuda`), plus recommended `--device auto` default.

---

## Logging and implementation trace

Maintain `auto_logs.md` at repository root after each phase.

Each phase log entry must include:
- files added/modified,
- design decisions and rationale,
- metric snapshots (from both local JSONL and W&B),
- blockers and fixes,
- next-phase prerequisites.

**W&B Plotting and Monitoring (MANDATORY):**
- Every training run MUST log to W&B in offline mode (`mode='offline'`).
- Every phase involving training must include:
  - W&B dashboard screenshots showing per-objective losses and MGDA coefficients.
  - W&B leet viewer command (e.g., `python -m wandb beta leet run ...`) saved to auto_logs.md.
  - Three-objective stacked area chart of MGDA coefficients (visual check for dominance).
- Use `wandb beta leet run` during training to interactively monitor stability; do NOT wait until end of run.
- If objective_dominance_ratio > 10 for 50+ consecutive epochs OR action_entropy < 0.05 OR probe_first_action_span ≈ 0:
  - PAUSE training immediately.
  - Document the exact epoch and metric value in auto_logs.md.
  - Do NOT suppress the warning; investigate root cause before resuming.
  - These are signals of prior MORL failure mode recurrence.

---

## Acceptance criteria

The pivot is successful only if all are true:

1. Inference does not require tradeoff vector input.
2. Sequential policy is trained with implicit multi-objective balancing (MGDA-style gradient combination).
3. Evaluation uses a single policy path with no weight-grid selection stage.
4. Metrics are at least competitive with one-shot baseline and show stable objective behavior.
5. Training logs show non-degenerate multi-objective optimization (no persistent single-objective collapse).
6. **W&B instrumentation is complete**: all metrics from W&B Logging Patterns section are logged and visible in offline W&B dashboard. No warnings (objective dominance > 10x, action entropy < 0.05, probe_first_action_span ≈ 0) triggered during training.
7. Device portability is verified: the same codepath runs successfully on CPU (`--device cpu`) and GPU (`--device cuda` when available), and `--device auto` resolves correctly.

---

## End condition

System produces:
- original SGSL+MGDA training path unchanged,
- frozen embedding checkpoint,
- implicit sequential MORL package in `code/seqmorl`,
- single-policy evaluation outputs,
- reproducible implementation and ablation log trail.

Proceed phase-by-phase and do not skip acceptance gates.