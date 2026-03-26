# Constrained Rerank Pivot Automation Plan

You are an AI coding agent operating inside the MOPI-HFRS repository.

Your objective is to build, from scratch, a constrained reranker that augments the original one-shot MOPI-HFRS outputs while preserving ranking quality as a hard constraint.

## Mission

Build a constrained rerank stage with these properties:
- no inference-time tradeoff vector input,
- baseline SGSL ranking semantics preserved,
- strict relevance guardrails first,
- secondary metric optimization (health/diversity/coverage) only within feasible edits,
- deterministic fallback to baseline anchor list when constraints reject edits.

## Why this pivot exists

Prior sequential MORL trials failed the ranking floor by large margins despite extensive tuning.
Root cause: attempting to optimize the same target as the SGSL ranker in a higher-variance RL loop produced unstable tradeoffs and ranking degradation.

This new design treats ranking as a hard feasibility contract, not a soft objective.

## Critical constraints (must not be violated)

- DO NOT modify SGSL architecture in `code/RCSYS_models.py`.
- DO NOT modify graph preprocessing/tag enrichment pipeline.
- DO NOT modify MGDA logic used by original SGSL training in `code/RCSYS_utils.py`.
- DO NOT introduce a new unconstrained sequential RL policy as primary ranker.
- DO NOT require any inference-time user preference vector.
- All comparisons MUST be apples-to-apples with baseline evaluation protocol.

## Mandatory prerequisite: deep codebase understanding

BEFORE writing any constrained reranker code, you MUST read and document the following:

1. Baseline training and checkpoint flow
- Read `code/main.py` end-to-end.
- Identify where user/item embeddings are produced and how baseline evaluation is run.
- Confirm what checkpoint artifacts already exist and what can be safely reused.

2. Baseline evaluation and metric semantics
- Read metric/eval utilities in `code/RCSYS_utils.py` used by baseline val/test reporting.
- Document exact definitions for ndcg, recall, precision, health-related metrics, diversity, and coverage-like outputs used in repo.
- Document split handling and exclusion masking behavior (train exclusion for val/test, and any additional test-time exclusions used by baseline).

3. Data and split contract
- Confirm split creation path and split identifiers in the repository.
- Confirm which edges are considered seen history versus target positives for each split.
- Confirm that constrained rerank evaluation can be run on the same split protocol with no leakage.

4. Existing dependency/runtime constraints
- Confirm CPU/CUDA behavior for the repo environment and checkpoint portability expectations.
- Confirm required package imports for evaluation path work in the active environment.

5. W&B logging patterns (required for this pivot)
- Implement W&B offline logging for constrained rerank runs (do not skip).
- Track both baseline-vs-rerank metrics and constrained diagnostics per run.
- Persist a reproducible `wandb beta leet run` command in output artifacts.

Implementation notes requirement:
- Add a short section in `auto_logs.md` titled `Prerequisite Understanding` before Phase 0 execution.
- Summarize baseline flow, metrics, masking, and split semantics in concrete bullet points.

Verification checklist (do not skip):
- [ ] Can you explain baseline one-shot inference/evaluation flow from `code/main.py` without ambiguity?
- [ ] Can you point to the exact metric functions and explain their outputs used for comparison?
- [ ] Can you state the exact masking policy used for val and test evaluation?
- [ ] Can you show that constrained rerank will use the same split/masking protocol?
- [ ] Can you describe the run-level W&B metrics and where they are logged in code?

If any checklist item is unanswered, STOP. Do not begin constrained reranker implementation.

## Pipeline architecture

SGSL Training (existing multi-objective GNN path, unchanged)
-> Frozen user/item embeddings
-> Constrained reranker (anchor list + bounded feasible edits)
-> Single-path evaluation on val/test
-> Baseline (one-shot) vs Constrained rerank comparison

Implementation contract:
- Constrained reranker is an inference-time augmentation on top of frozen embeddings.
- SGSL training phase remains unchanged.
- Baseline and reranked outputs must be compared on identical splits, masks, and metric definitions.

## Why this design is chosen (lessons from prior experiments)

Prior experiments surfaced failures this design avoids:

1. Objective mismatch in noisy RL loop
- Trying to relearn the same ranking target as SGSL in higher-variance RL degraded ranking.

2. Soft-tradeoff instability
- Multi-objective policy tradeoffs improved secondary metrics while violating hard ranking floor.

3. Unclear acceptance behavior
- Without hard feasibility gates and deterministic fallback, behavior drifted and was hard to debug.

This plan fixes those issues by enforcing hard constraints around baseline ranking and optimizing only inside feasible edit space.

## Existing code references to reuse

Primary references in original framework:

- `code/main.py`
   - Existing SGSL training/eval orchestration.
   - Keep one-shot baseline path intact.

- `code/RCSYS_utils.py`
   - Existing split/eval and metric ecosystem.
   - Reuse metric semantics and masking behavior.

- `code/RCSYS_models.py`, `code/utils.py`
   - Embedding ecosystem and utilities; do not redesign.

These files define baseline semantics and must guide parity-safe constrained rerank evaluation.

## Phase 0: Baseline parity lock (do not skip)

Goal: ensure baseline metrics are comparable before any reranker code is written.

Tasks:
1. Read baseline flow in `code/main.py` and evaluation helpers in `code/RCSYS_utils.py`.
2. Confirm split protocol and masking behavior used by baseline one-shot evaluation.
3. Freeze baseline reference artifacts:
   - val metrics,
   - test metrics,
   - eval configuration (K, split ids, masking policy, seed).
4. Create parity checklist in code comments/docs:
   - same splits,
   - same K,
   - same exclusion masking,
   - same metric functions.

Hard gate:
- If parity cannot be demonstrated, STOP. Do not implement reranker yet.

Deliverables:
- Baseline parity record in `auto_logs.md`.
- Saved baseline metrics JSON under output dir.

---

## Phase 1: Scope reset and architecture contract

Goal: codify constrained reranker as post-ranking augmentation, not replacement ranker.

Architecture:
1. Baseline anchor list:
   - one-shot top-K from frozen embeddings and existing masking rules.
2. Reranker:
   - proposes bounded edits to positions in anchor list.
3. Feasibility layer:
   - enforces hard constraints.
4. Fallback:
   - if proposal violates constraints, use anchor item.

Required objective hierarchy (lexicographic):
1. Primary: relevance/ranking floor compliance.
2. Secondary: health/diversity/coverage improvements.

Forbidden in v1:
- training a full unconstrained MORL policy,
- exposure-allocation policy learning,
- simulator-based long-horizon RL.

Deliverables:
- Contract section in docs and module docstrings.

---

## Phase 2: New package and file map

Goal: establish clean implementation boundaries.

Create package:
- `code/constrained_rerank/`
  - `__init__.py`
  - `anchor.py`              # baseline anchor generation and score extraction
  - `constraints.py`         # feasibility checks (lock, margin, budget, duplicates)
  - `reranker.py`            # constrained edit executor
  - `evaluation.py`          # parity-safe eval wrappers + diagnostics
  - `main.py`                # CLI entrypoint
  - `logging_utils.py`       # structured json logging + optional wandb wrapper
  - `README_CONSTRAINED_RERANK.md`

Keep untouched:
- existing SGSL training path in `code/main.py`.

Deliverables:
- Package skeleton committed and importable.

---

## Phase 3: Anchor list and candidate construction

Goal: deterministic anchor generation consistent with baseline semantics.

Tasks:
1. Load frozen embeddings from checkpoint.
2. For each user, compute ranked candidates using baseline-compatible score function.
3. Apply same exclusion masking as baseline evaluation.
4. Build:
   - anchor top-K item ids,
   - per-position anchor relevance scores,
   - optional top-M pool for swap candidates.

Rules:
- Anchor generation must be deterministic for fixed seed.
- No learned policy required in v1.

Deliverables:
- `anchor.py` with tested API:
  - `get_anchor_list_and_scores(user_id, K, ...)`.

---

## Phase 4: Hard constraints and edit mechanics

Goal: bounded feasible edits only.

Required constraints:
1. Position lock:
   - positions 1..L are immutable anchor (default L=6).
2. Score-margin gate:
   - candidate score must satisfy `cand_score >= anchor_score - epsilon`.
3. Swap budget:
   - max swaps per list (default 4).
4. Duplicate prevention:
   - no repeated item in final list.

Fallback contract:
- Any rejected edit -> force anchor item for that position.

Diagnostics per list:
- attempted swaps,
- accepted swaps,
- rejected by margin,
- rejected by budget,
- rejected by duplicate,
- forced-anchor count.

Deliverables:
- `constraints.py` and `reranker.py` with deterministic behavior.

---

## Phase 5: CLI and runtime contract

Goal: reproducible runs on CPU/GPU without code edits.

CLI requirements in `code/constrained_rerank/main.py` (minimal required surface):
- `--device cpu|cuda|auto`
- `--K`
- `--M`
- `--anchor_lock_positions`
- `--anchor_epsilon`
- `--max_swaps_per_list`
- `--output_dir`

Advanced optional flags (debug/ablation only, not required for normal runs):
- `--seed`
- `--train_user_limit`
- `--val_user_limit`
- `--exclude_seen_candidates`

Default behavior guidance:
- Production/default path should rely on repo-consistent defaults and full splits.
- Optional limit flags exist only to speed smoke tests and ablations.

Device behavior:
- `auto`: cuda if available else cpu,
- `cuda`: fail clearly if unavailable,
- checkpoint loading must be portable with `map_location`.

Deliverables:
- CLI run help output and config JSON saved per run.

---

## Phase 6: Evaluation protocol and acceptance gates

Goal: evaluate constrained rerank against baseline with strict parity.

Metrics to report (val and test):
- ndcg,
- recall,
- precision,
- health,
- diversity,
- coverage,
- car (if present in baseline ecosystem).

Constrained diagnostics to report:
- aggregate swap rate,
- aggregate rejection rate,
- forced-anchor rate,
- rejection breakdown by reason.

Primary hard gate:
- test ndcg drop fraction must be <= configured floor (default 0.07).

Secondary utility gate:
- non-negative trend in selected secondary metrics with non-zero accepted swaps.

If hard gate fails:
- mark run failed regardless of secondary gains.

Deliverables:
- `results.json` containing baseline, constrained, and diagnostics blocks.

---

## Phase 7: Incremental experiment matrix

Goal: isolate contributions and avoid confounded conclusions.

Run order (small smoke first):
1. Config A: baseline-only reference (no edits).
2. Config B: lock-only (positions 1..L locked, no margin/budget restriction beyond locks).
3. Config C: lock + margin.
4. Config D: lock + margin + budget (target v1).

Protocol:
- fixed seeds,
- identical user subsets for matrix,
- short runs first (smoke), then substantial run for selected config.

Selection rule:
- choose best config that passes hard ranking gate.
- if none pass, stop and tighten constraints (epsilon/window/budget) before adding complexity.

Deliverables:
- comparison table in output dir and `auto_logs.md`.

---

## Phase 8: Decision tree for failures

If baseline mismatch detected:
- Stop, fix evaluation parity first.

If ndcg floor fails:
- tighten one dimension at a time in this order:
  1. reduce editable window,
  2. reduce epsilon,
  3. reduce swap budget.

If zero accepted swaps:
- treat as over-constrained; relax one setting slightly and re-run smoke.

If diagnostics missing/inconsistent:
- fail run and fix instrumentation before further experiments.

If duplicate selection occurs:
- fail-fast with explicit error and patch duplicate-prevention logic.

---

## Phase 9: Logging and reproducibility

Maintain `auto_logs.md` after each phase with:
- files changed,
- commands run,
- metric snapshots,
- gate pass/fail status,
- blockers/fixes,
- next phase decision.

Artifacts required per run:
- `run_config.json`,
- `results.json`,
- `train_metrics.jsonl` or `rerank_metrics.jsonl`,
- wandb offline metadata,
- `wandb_leet_command.txt`,
- concise comparison table.

W&B policy (required):
- offline mode only,
- must mirror JSON metrics; W&B is visualization, JSON is source of truth,
- required run-level panels:
   - baseline vs rerank metric deltas,
   - ndcg_drop_fraction,
   - swap/rejection/forced-anchor diagnostics,
   - acceptance gate status.

---

## Phase 10: Acceptance criteria

The constrained-rerank pivot is successful only if all are true:

1. Baseline SGSL training/eval path remains unchanged.
2. No inference-time tradeoff vector is required.
3. Reranker edits are bounded and always feasibility-checked.
4. Final list always valid (no duplicates, lock compliance, budget compliance).
5. Test ranking floor passes (`ndcg_drop_fraction <= threshold`).
6. Secondary metrics are at least neutral-to-improved under non-trivial edit activity.
7. Full run is reproducible from saved config and commands.

---

## Non-goals (explicit)

- Do not build a new MGDA RL training stage in this branch.
- Do not retrain SGSL to force reranker gains.
- Do not add simulator/planner infrastructure in v1.
- Do not add exposure-allocation policy learning in v1.

---

## Quick-start commands (template)

From `code/` directory:

1. Baseline parity smoke:
- `python main.py --seed 42 --K 20`

2. Constrained rerank smoke:
- `python constrained_rerank/main.py --device auto --K 20 --M 200 --anchor_lock_positions 6 --anchor_epsilon 0.05 --max_swaps_per_list 4 --output_dir constrained_rerank_smoke`

3. Ablation matrix run:
- run Config A/B/C/D with only constraint flags changing, same split and seed.

4. Substantial run:
- promote only best gate-passing config from matrix.

---

## End condition

System produces:
- unchanged original SGSL path,
- standalone constrained-rerank package,
- parity-safe baseline vs constrained comparisons,
- reproducible artifacts and phase logs,
- clear pass/fail verdict against ranking floor.

Proceed phase-by-phase and do not skip gates.
