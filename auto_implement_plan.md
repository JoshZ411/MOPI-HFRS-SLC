# RL-Constrained Rerank Pivot Automation Plan

You are an AI coding agent operating inside the MOPI-HFRS repository.

Your objective is to build, from scratch, a constrained reranker with a reinforcement-learning control policy that augments original one-shot MOPI-HFRS outputs while preserving ranking quality as a hard constraint.

## Mission

Build an RL-constrained rerank stage with these properties:
- no inference-time tradeoff vector input,
- baseline SGSL ranking semantics preserved,
- RL policy controls only bounded rerank decisions,
- strict relevance guardrails first,
- secondary metric optimization (health/diversity/coverage) only within feasible edits,
- deterministic fallback to baseline anchor list when constraints reject policy actions.

## Why this pivot exists

Prior sequential MORL trials failed ranking-floor requirements despite extensive tuning.
Root cause: attempting to optimize SGSL’s target with a high-variance unconstrained RL loop caused unstable tradeoffs and ranking degradation.

This design keeps RL, but constrains where RL is allowed to act.

## Critical constraints (must not be violated)

- DO NOT modify SGSL architecture in code/RCSYS_models.py.
- DO NOT modify graph preprocessing/tag enrichment pipeline.
- DO NOT modify MGDA logic used by original SGSL training in code/RCSYS_utils.py.
- DO NOT introduce an unconstrained sequential RL policy as primary ranker.
- DO NOT require any inference-time user preference vector.
- All comparisons MUST be apples-to-apples with baseline evaluation protocol.
- RL policy actions MUST pass hard feasibility checks (lock, margin, budget, duplicates).

## MANDATORY PREREQUISITE: Deep Codebase Understanding

BEFORE starting any implementation, you MUST thoroughly read and understand the following:

1. Baseline SGSL flow and artifacts
- Read code/main.py end-to-end.
- Identify where user/item embeddings are produced and saved.
- Identify baseline evaluation flow and output metrics.

2. Existing evaluation and metrics semantics
- Read metric/eval utilities in code/RCSYS_utils.py.
- Document ndcg, recall, precision, health, diversity, coverage semantics.
- Document split/masking behavior used for val and test.

3. Data/split protocol contract
- Confirm split creation and seen-vs-target edge handling.
- Confirm constrained rerank can evaluate on same split protocol with no leakage.

4. Runtime/dependency contract
- Confirm CPU/CUDA behavior and map_location handling.
- Confirm required dependencies for evaluation and logging.

5. W&B logging patterns (required)
- Implement W&B offline logging for constrained RL rerank runs.
- Track baseline-vs-rerank metrics and RL/control diagnostics.
- Persist reproducible wandb beta leet run command in output artifacts.

Implementation notes requirement:
- Add a section in auto_logs.md titled Prerequisite Understanding before Phase 0.
- Summarize baseline flow, metrics, masking, and split semantics.

Verification checklist (do not skip):
- [ ] Can you explain baseline one-shot inference/evaluation flow from code/main.py?
- [ ] Can you point to exact metric functions and explain outputs?
- [ ] Can you state exact val/test masking policy?
- [ ] Can you show constrained RL rerank uses same split/masking protocol?
- [ ] Can you describe run-level W&B metrics and where they are logged?

If any checklist item is unanswered, STOP. Do not begin implementation.

## Pipeline architecture

SGSL Training (existing path, unchanged)
-> Frozen user/item embeddings
-> Anchor top-K generation
-> RL-constrained rerank controller (bounded, feasibility-checked)
-> Single-path evaluation on val/test
-> Baseline (one-shot) vs RL-constrained rerank comparison

Implementation contract:
- Rerank stage is inference-time augmentation on frozen embeddings.
- SGSL training phase remains unchanged.
- RL policy is secondary control only (not a full replacement ranker).
- Baseline and reranked outputs must use identical split, mask, and metric definitions.

## Why this design is chosen (lessons from prior experiments)

1. Objective mismatch in noisy RL loops
- Relearning SGSL target with unconstrained RL degraded ranking.

2. Soft-tradeoff instability
- Multi-objective policy tradeoffs improved secondary metrics while violating hard ranking floor.

3. Acceptance ambiguity
- Without deterministic feasibility/fallback, behavior drifted and debugging was unclear.

This plan keeps RL signal where useful while forcing hard ranking compliance.

## Existing code references to reuse

Primary references in original framework:
- code/main.py: SGSL orchestration and baseline eval path.
- code/RCSYS_utils.py: split/eval and metric semantics.
- code/RCSYS_models.py, code/utils.py: embedding and utility ecosystem.

These define baseline semantics and must guide parity-safe constrained rerank evaluation.

## New code organization

Create package:
- code/constrained_rerank/
  - __init__.py
  - anchor.py                # baseline anchor generation and score extraction
  - constraints.py           # feasibility checks (lock, margin, budget, duplicates)
  - rl_policy.py             # bounded control policy (bandit/policy network)
  - rl_training.py           # offline RL or contextual-bandit training loop
  - reranker.py              # constrained edit executor using policy proposals
  - evaluation.py            # parity-safe eval wrappers + constrained diagnostics
  - main.py                  # CLI entrypoint
  - logging_utils.py         # json + required wandb offline wrapper
  - README_CONSTRAINED_RERANK.md

Keep untouched:
- existing SGSL training path in code/main.py.

---

## Phase 0: Baseline parity lock (do not skip)

Goal: ensure baseline metrics are comparable before any reranker code is written.

Tasks:
1. Read baseline flow in code/main.py and evaluation helpers in code/RCSYS_utils.py.
2. Confirm split protocol and masking behavior used by baseline one-shot evaluation.
3. Freeze baseline reference artifacts:
   - val metrics,
   - test metrics,
   - eval configuration (K, split ids, masking policy, seed).
4. Create parity checklist in code comments/docs.

Hard gate:
- If parity cannot be demonstrated, STOP.

Deliverables:
- Baseline parity record in auto_logs.md.
- Saved baseline metrics JSON under output dir.

---

## Phase 1: Scope reset and RL architecture contract

Goal: codify RL-constrained rerank as post-ranking augmentation.

Architecture:
1. Baseline anchor list: one-shot top-K from frozen embeddings.
2. RL policy: proposes bounded edit actions only.
3. Feasibility layer: validates proposed action against hard constraints.
4. Fallback: if invalid, force anchor item.

Required objective hierarchy (lexicographic):
1. Primary: relevance/ranking-floor compliance.
2. Secondary: health/diversity/coverage improvements.

Forbidden in v1:
- full unconstrained MORL policy,
- exposure-allocation RL,
- simulator-based long-horizon control.

Deliverables:
- Contract section in docs and module docstrings.

---

## Phase 2: Anchor list and candidate construction

Goal: deterministic anchor generation consistent with baseline semantics.

Tasks:
1. Load frozen embeddings from checkpoint.
2. For each user, compute ranked candidates using baseline-compatible score function.
3. Apply same exclusion masking as baseline evaluation.
4. Build:
   - anchor top-K item ids,
   - per-position anchor relevance scores,
   - optional top-M pool for feasible swap candidates.

Rules:
- Deterministic for fixed seed.
- No ranking replacement in this phase.

Deliverables:
- anchor.py with tested API get_anchor_list_and_scores(...).

---

## Phase 3: Hard constraints and bounded action space

Goal: define feasible action space for RL policy.

Required constraints:
1. Position lock: positions 1..L immutable anchor (default L=6).
2. Score-margin gate: cand_score >= anchor_score - epsilon.
3. Swap budget: max swaps per list (default 4).
4. Duplicate prevention: no repeated items.

Fallback contract:
- Any rejected proposal -> force anchor item for that position.

Diagnostics per list:
- attempted policy actions,
- accepted actions,
- rejected by margin/budget/duplicate/lock,
- forced-anchor count.

Deliverables:
- constraints.py and reranker.py with deterministic behavior.

---

## Phase 4: RL controller implementation (constrained role)

Goal: train/use RL policy only within bounded feasible rerank control.

Controller scope options (in priority order):
1. Contextual bandit controller (recommended v1): choose among safe candidate edits.
2. Lightweight policy-gradient controller over feasible action subset.

State (minimum):
- user embedding summary,
- anchor position features,
- current swap budget usage,
- position index and recent rejection context.

Action:
- choose candidate index among feasibility-filtered set.

Reward (lexicographic-compatible):
- hard penalty if floor-violating behavior proxy is triggered,
- positive signal for secondary utility gains within feasible edits,
- optional small penalty for excessive edits.

Training mode:
- offline (static data compatible), no online adaptation required in v1.

Deliverables:
- rl_policy.py and rl_training.py with reproducible training config.

---

## Phase 5: CLI and runtime contract

Goal: reproducible runs on CPU/GPU without code edits.

CLI minimal required surface:
- --device cpu|cuda|auto
- --K
- --M
- --anchor_lock_positions
- --anchor_epsilon
- --max_swaps_per_list
- --output_dir
- --rl_mode bandit|policy

Advanced optional flags (debug/ablation):
- --seed
- --train_user_limit
- --val_user_limit
- --exclude_seen_candidates

Default behavior guidance:
- Default path should use repo-consistent full splits.
- Optional limit flags only for smoke/ablation speed.

Device behavior:
- auto uses cuda if available else cpu.
- cuda fails clearly if unavailable.
- checkpoint loading must be portable via map_location.

Deliverables:
- CLI help output and run_config.json per run.

---

## Phase 6: Evaluation protocol and acceptance gates

Goal: evaluate RL-constrained rerank against baseline with strict parity.

Report metrics (val/test):
- ndcg, recall, precision, health, diversity, coverage, car (if available).

Report constrained RL diagnostics:
- swap rate, rejection rate, forced-anchor rate,
- rejection breakdown by reason,
- policy action entropy,
- mean accepted action position.

Primary hard gate:
- test ndcg drop fraction <= configured floor (default 0.07).

Secondary utility gate:
- non-negative trend in selected secondary metrics with non-zero accepted RL edits.

If hard gate fails:
- run is failed regardless of secondary gains.

Deliverables:
- results.json containing baseline, reranked, RL diagnostics, and gate verdict.

---

## Phase 7: Incremental experiment matrix

Goal: isolate contributions and avoid confounded conclusions.

Run order:
1. Config A: baseline-only (no edits).
2. Config B: heuristic constrained rerank (no RL learning).
3. Config C: constrained rerank + RL bandit controller.
4. Config D: constrained rerank + RL policy controller + budget.

Protocol:
- fixed seeds for comparability,
- identical user subsets for matrix,
- short smoke runs first, then substantial run for best gate-passing config.

Selection rule:
- choose best config that passes hard ranking gate.
- if none pass, tighten constraints before adding complexity.

Deliverables:
- comparison table in output dir and auto_logs.md.

---

## Phase 8: Decision tree for failures

If baseline mismatch detected:
- stop and fix parity first.

If ndcg floor fails:
- tighten one dimension at a time:
  1) narrower editable window,
  2) lower epsilon,
  3) lower swap budget.

If zero accepted RL actions:
- treat as over-constrained or poor policy signal; relax one constraint slightly and rerun smoke.

If diagnostics missing/inconsistent:
- fail run; fix instrumentation before continuing.

If duplicate selection occurs:
- fail-fast and patch duplicate prevention logic.

---

## Phase 9: Logging and reproducibility

Maintain auto_logs.md after each phase with:
- files changed,
- commands run,
- metric snapshots,
- gate pass/fail status,
- blockers/fixes,
- next phase decision.

Artifacts required per run:
- run_config.json,
- results.json,
- rerank_metrics.jsonl,
- wandb offline metadata,
- wandb_leet_command.txt,
- comparison table.

W&B policy (required):
- offline mode only,
- JSON is source of truth; W&B mirrors it,
- required panels:
  - baseline vs rerank deltas,
  - ndcg_drop_fraction,
  - swap/rejection/forced-anchor diagnostics,
  - RL action entropy and acceptance gate status.

---

## Phase 10: Acceptance criteria

Pivot succeeds only if all are true:
1. Baseline SGSL path remains unchanged.
2. No inference-time tradeoff vector is required.
3. RL policy acts only inside hard-feasible bounded edit space.
4. Final list always valid (no duplicates, lock, budget, margin compliant).
5. Test ranking floor passes.
6. Secondary metrics are neutral-to-improved with non-trivial accepted RL edits.
7. Full run is reproducible from saved configs and commands.

---

## Non-goals

- Do not rebuild MGDA MORL sequential training in this branch.
- Do not retrain SGSL to force rerank gains.
- Do not add simulator/planner infrastructure in v1.
- Do not add exposure-allocation RL in v1.

---

## Quick-start commands (template)

From code/ directory:

1. Baseline parity smoke:
- python main.py --seed 42 --K 20

2. Constrained RL rerank smoke:
- python constrained_rerank/main.py --device auto --K 20 --M 200 --anchor_lock_positions 6 --anchor_epsilon 0.05 --max_swaps_per_list 4 --rl_mode bandit --output_dir constrained_rl_rerank_smoke

3. Ablation matrix run:
- run Config A/B/C/D with only constrained/RL flags changed.

4. Substantial run:
- promote only best gate-passing config.

---

## End condition

System produces:
- unchanged original SGSL path,
- standalone constrained-rerank package with bounded RL controller,
- parity-safe baseline vs reranked comparisons,
- reproducible artifacts and phase logs,
- clear pass/fail verdict against ranking floor.

Proceed phase-by-phase and do not skip gates.
