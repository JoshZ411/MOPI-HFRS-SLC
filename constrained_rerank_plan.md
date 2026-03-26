# Implementation Plan: Constrained Reranker for MOPI-HFRS-SLC

## Context

The MOPI-HFRS pipeline is a multi-objective health-aware food recommendation system using SGSL (graph structure learning + LightGCN) with MGDA-based Pareto optimization across ranking, health, and diversity objectives. Prior attempts to add a sequential MORL stage degraded ranking quality. This plan adds a **constrained reranker** — a post-training inference-time module that improves secondary metrics (health/diversity/coverage) while enforcing a hard ranking floor. No inference-time weight vector is needed.

**Primary spec**: `auto_implement_plan.md` (root-level). The coding agent MUST read this file in full before writing any code.

---

## 1. How the Current Pipeline Works

### Data
- HeteroData graph loaded from `../processed_data/benchmark_macro.pt`
- Node types: `user` (features + health tags) and `food` (features + health tags)
- Edge type: `('user', 'eats', 'food')` with `edge_index` and `edge_label_index`

### Training (`code/main.py`)
1. Split data 60/20/20 via `split_data_new()` (`RCSYS_utils.py:48-74`)
2. Model: `SGSL` (`RCSYS_models.py:285-331`) = feature graph + semantic graph + fusion + LightGCN
3. Per epoch: forward pass → mini-batch sample (2048) → `pareto_loss()` (`RCSYS_utils.py:514-555`)
4. `pareto_loss` computes 3 losses (BPR, diversity, health), collects per-task gradients, runs MGDA `MinNormSolver` (`min_norm_solvers.py:141-185`) to find Pareto-optimal weights, returns weighted loss
5. 500 epochs, Adam optimizer, ExponentialLR decay

### Embeddings
- `model.forward()` returns 4 tensors: `users_emb_final` [num_users, 128], `users_emb_0`, `items_emb_final` [num_items, 128], `items_emb_0`
- **Not currently saved to disk** — no checkpoint code exists

### Evaluation (`RCSYS_utils.py:384-420`, `get_metrics` at line 319-380)
1. `eval()` re-runs `model.forward()` on the **eval split's edge_index** (val or test), meaning embeddings are recomputed with different graph structure than training
2. It then subsets embeddings to `user_indices` from `structured_negative_sampling` on eval edges — but `get_metrics()` at line 319 receives the **full** `users_emb_final` / `items_emb_final` from eval's forward pass (not the subsetted ones used for BPR loss)
3. `rating = torch.matmul(user_embedding, item_embedding.T)` → [num_users, num_items]
4. Exclusion masking: `rating[user, item] = -(1 << 10)` for items in `exclude_edge_indices` (which is `[neg_train_edge_index]`)
5. `top_K_items = torch.topk(rating, k=K)` → [num_users, K]
6. Ground truth users: `edge_index[0].unique()` — only users present in the eval split
7. Metrics: recall@K, precision@K, ndcg@K, health_score, avg_health_tags_ratio, percentage_recommended_foods

### CRITICAL: Embedding Parity Concern
The baseline `eval()` function (line 400-401) calls `model.forward(feature_dict, edge_index, pos_edge_index, neg_edge_index)` with **val/test edge indices**, producing different embeddings than a forward pass with train edges. The reranker will use frozen train-time embeddings instead. This means the reranker's "baseline" anchor list will differ from what `eval()` produces. Two options:
1. **Option A (recommended)**: Save the full model `state_dict` in the checkpoint. In the reranker, load the model and re-run `model.forward()` on the appropriate eval edges to generate embeddings that match baseline eval exactly. This is more faithful to baseline parity.
2. **Option B**: Accept that frozen train-edge embeddings differ from eval-edge embeddings. Document this as a known design difference. The reranker compares its own baseline (from frozen embeddings) against its reranked output — so the comparison is internally consistent even if absolute metrics differ from `main.py` eval.

**Decision**: Use Option A for maximum parity. Save model state_dict + all edge tensors in checkpoint. Reranker loads model, runs forward on val/test edges to get embeddings, then operates on those.

---

## 2. Integration Strategy

### UNTOUCHED (do not modify)
- `code/RCSYS_models.py` — SGSL, LightGCN, SignedGCN, all model classes
- `code/RCSYS_utils.py` — losses, MGDA `pareto_loss`, metric functions, `eval()`, `get_metrics()`, `split_data_new()`
- `code/min_norm_solvers.py` — MinNormSolver
- `code/utils.py` — utility functions
- `code/ablations.py` — ablation runner
- `preprocess/` — all preprocessing

### MINIMAL EDIT (one file)
- `code/main.py` — Add ~15 lines after training loop (line 149) to save frozen embeddings + model checkpoint + graph metadata to disk. This is the only edit to existing code.

### NEW (entire new package)
- `code/constrained_rerank/` — all new files for the constrained reranker

---

## 3. File-Level Plan

### Edit: `code/main.py`
**After line 149** (end of training loop, before test eval), add checkpoint saving:
```python
# Save model and metadata for constrained reranker
import os
checkpoint_dir = '../checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
torch.save({
    'model_state_dict': model.state_dict(),
    'user_tags': user_tags.cpu(),
    'food_tags': food_tags.cpu(),
    'user_features': user_features.cpu(),
    'food_features': food_features.cpu(),
    'num_users': num_users,
    'num_foods': num_foods,
    'feature_dict': {k: v.cpu() for k, v in feature_dict.items()},
    'train_edge_index': train_edge_index.cpu(),
    'val_edge_index': val_edge_index.cpu(),
    'test_edge_index': test_edge_index.cpu(),
    'pos_train_edge_index': pos_train_edge_index.cpu(),
    'neg_train_edge_index': neg_train_edge_index.cpu(),
    'pos_val_edge_index': pos_val_edge_index.cpu(),
    'neg_val_edge_index': neg_val_edge_index.cpu(),
    'pos_test_edge_index': pos_test_edge_index.cpu(),
    'neg_test_edge_index': neg_test_edge_index.cpu(),
    'seed': SEED,
    'K': args.K,
    'hidden_dim': HIDDEN_DIM,
    'layers': LAYERS,
    'feature_threshold': TH,
}, f'{checkpoint_dir}/sgsl_checkpoint.pt')
print(f"Checkpoint saved to {checkpoint_dir}/sgsl_checkpoint.pt")
```
Also add `import os` at top of file.

**Why model_state_dict instead of just embeddings**: The baseline `eval()` function re-runs `model.forward()` on val/test edge indices, producing different embeddings than train-time. To achieve exact parity, the reranker must load the model and re-run forward on the same eval edges. Saving `model_state_dict` + all hyperparams (`HIDDEN_DIM`, `LAYERS`, `TH`) allows reconstructing the SGSL model exactly.

### New: `code/constrained_rerank/__init__.py`
Empty init to make it a package.

### New: `code/constrained_rerank/anchor.py`
**Purpose**: Load model checkpoint, reconstruct SGSL, run forward pass on eval edges to get embeddings (matching baseline eval exactly), generate anchor lists with same masking as `get_metrics()`.

Key functions:
```python
def load_model_and_data(checkpoint_path, graph_path, device):
    """
    Load checkpoint, reconstruct SGSL model from state_dict, load graph for model init.
    Returns: model, checkpoint data dict
    Note: Needs graph for SGSL constructor (uses graph.node_types, num_nodes).
    Load graph from graph_path='../processed_data/benchmark_macro.pt'.
    Reconstruct: model = SGSL(graph, embedding_dim=ckpt['hidden_dim'],
                              feature_threshold=ckpt['feature_threshold'],
                              num_layer=ckpt['layers'])
    model.load_state_dict(ckpt['model_state_dict'])
    """

def get_embeddings_for_split(model, feature_dict, edge_index, pos_edge_index, neg_edge_index, device):
    """
    Run model.forward() on given split's edges (same as eval() does at RCSYS_utils.py:400-401).
    Returns: users_emb_final, items_emb_final
    """

def get_anchor_list_and_scores(users_emb, items_emb, K, M, exclude_edge_indices):
    """
    Replicate masking from get_metrics() (RCSYS_utils.py:337-348), then topk.
    Returns:
        anchor_topk: [num_users, K] — baseline top-K item indices
        anchor_scores: [num_users, K] — dot-product scores for anchor items
        candidate_pool: [num_users, M] — top-M candidate items for swap pool
        candidate_scores: [num_users, M] — scores for candidate pool
        rating_matrix: [num_users, num_items] — full masked rating matrix
    """
```

Must replicate exact masking logic from `get_metrics()` (RCSYS_utils.py:337-348):
- Build `user_pos_items` from `exclude_edge_indices` using `get_user_positive_items()`
- Set `rating[exclude_users, exclude_items] = -(1 << 10)`
- `topk` for anchor list, `topk(M)` for candidate pool

### New: `code/constrained_rerank/constraints.py`
**Purpose**: Hard constraint enforcement layer.

Key class:
```python
class FeasibilityChecker:
    def __init__(self, lock_positions=6, epsilon=0.05, max_swaps=4):
        ...

    def check_swap(self, position, anchor_score, candidate_score,
                   current_list, candidate_item) -> (bool, str):
        """Returns (is_feasible, rejection_reason)"""
        # 1. Position lock: reject if position <= lock_positions
        # 2. Score margin: reject if candidate_score < anchor_score - epsilon
        # 3. Duplicate: reject if candidate_item already in current_list
        # 4. Budget: reject if swap count >= max_swaps
```

Diagnostics tracking per list: attempted/accepted/rejected-by-reason/forced-anchor counts.

### New: `code/constrained_rerank/reranker.py`
**Purpose**: Constrained edit executor.

Key function:
```python
def constrained_rerank(anchor_topk, anchor_scores, candidate_pool,
                       candidate_scores, checker, user_tags, food_tags):
    """
    For each user:
      1. Copy anchor list as starting point
      2. For editable positions (after lock_positions):
         - Find best candidate from pool that improves secondary metric
         - Check feasibility via FeasibilityChecker
         - If feasible: swap; else: keep anchor item (fallback)
      3. Return reranked list + per-user diagnostics

    Secondary metric heuristic (v1): prefer candidates with higher
    health tag overlap (Jaccard similarity with user tags).
    """
```

### New: `code/constrained_rerank/evaluation.py`
**Purpose**: Parity-safe evaluation wrappers.

Must reuse exact metric semantics from `RCSYS_utils.py`:
- Import and call `RecallPrecision_ATk`, `NDCGatK_r`, `calculate_health_score`, `calculate_average_health_tags`, `calculate_percentage_recommended_foods` from `RCSYS_utils`
- OR replicate their logic exactly (to avoid import path issues)

Key functions:
```python
def evaluate_reranked_list(reranked_topk, edge_index, user_tags, food_tags, K):
    """Compute all 6 metrics on reranked list, same as get_metrics()"""

def compute_acceptance_gates(baseline_metrics, reranked_metrics, ndcg_floor=0.07):
    """Check hard gate: ndcg_drop_fraction <= floor. Return pass/fail + details."""

def generate_comparison_table(baseline_metrics, reranked_metrics, diagnostics):
    """Produce formatted comparison for logging."""
```

### New: `code/constrained_rerank/logging_utils.py`
**Purpose**: Structured JSON logging + W&B offline wrapper.

Key functions:
```python
def save_run_config(output_dir, config_dict): ...
def save_results_json(output_dir, baseline, reranked, diagnostics, gates): ...
def init_wandb_offline(run_name, config): ...
def log_wandb_metrics(metrics_dict): ...
def save_wandb_leet_command(output_dir): ...
def append_auto_logs(phase, files_changed, commands, metrics, gate_status, blockers): ...
```

### New: `code/constrained_rerank/main.py`
**Purpose**: CLI entrypoint.

```python
# Required args
--checkpoint_path   (default: '../checkpoints/sgsl_checkpoint.pt')
--device            (cpu|cuda|auto)
--K                 (default: 20)
--M                 (default: 200)
--anchor_lock_positions  (default: 6)
--anchor_epsilon    (default: 0.05)
--max_swaps_per_list (default: 4)
--output_dir        (required)

# Optional args
--seed              (default: 42)
--ndcg_floor        (default: 0.07)
--use_wandb         (default: False)
```

Flow:
1. Load checkpoint
2. Generate anchor lists (baseline top-K + candidate pool top-M)
3. Compute baseline metrics (using anchor top-K directly)
4. Run constrained reranking
5. Compute reranked metrics
6. Check acceptance gates
7. Save results.json, run_config.json, auto_logs.md entry
8. Print comparison table

### New: `code/constrained_rerank/README_CONSTRAINED_RERANK.md`
Brief documentation of the module, its purpose, and usage.

---

## 4. Build Phases (Step-by-Step for Coding Agent)

### PREREQUISITE: Read `auto_implement_plan.md`
**Before writing ANY code**, the coding agent must:
1. Read `auto_implement_plan.md` in the repo root — this is the primary spec
2. Understand all 10 phases, constraints, and acceptance criteria
3. Follow the phase-by-phase approach defined there

### Phase 0: Baseline Parity Lock
1. Read `code/main.py` and `code/RCSYS_utils.py` to confirm understanding
2. Add checkpoint saving to `code/main.py` (the minimal edit described above)
3. Run baseline: `cd code && python main.py --seed 42 --K 20`
4. Record baseline val/test metrics in `auto_logs.md`
5. Save baseline metrics as JSON in output dir
6. **Hard gate**: baseline must run and produce metrics before proceeding

### Phase 1: Package Skeleton
1. Create `code/constrained_rerank/` directory
2. Create all 7 files as empty stubs with docstrings
3. Verify package is importable: `python -c "import constrained_rerank"`

### Phase 2: Anchor Generation (`anchor.py`)
1. Implement `get_anchor_list_and_scores()`
2. Replicate masking logic from `get_metrics()` (RCSYS_utils.py:337-348)
3. Test: anchor top-K must produce identical metrics to baseline eval

### Phase 3: Constraints (`constraints.py`)
1. Implement `FeasibilityChecker` with 4 constraints
2. Implement per-list diagnostics tracking
3. Unit test: verify lock/margin/budget/duplicate rejections work correctly

### Phase 4: Reranker (`reranker.py`)
1. Implement `constrained_rerank()` function
2. Health-based swap heuristic: prefer candidates with higher Jaccard overlap with user tags
3. Deterministic behavior for fixed seed
4. Fallback to anchor on any constraint violation

### Phase 5: Evaluation (`evaluation.py`)
1. Implement parity-safe metric computation (reuse or replicate RCSYS_utils metric functions)
2. Implement acceptance gate checker (ndcg_drop_fraction <= 0.07)
3. Implement comparison table generator

### Phase 6: Logging (`logging_utils.py`)
1. Implement JSON config/results saving
2. Implement W&B offline logging wrapper
3. Implement `auto_logs.md` append function

### Phase 7: CLI Entrypoint (`main.py`)
1. Wire all components together
2. Implement argparse with all required flags
3. Full pipeline: load → anchor → baseline metrics → rerank → reranked metrics → gates → save

### Phase 8: Smoke Test
1. Run: `python constrained_rerank/main.py --device auto --K 20 --M 200 --anchor_lock_positions 6 --anchor_epsilon 0.05 --max_swaps_per_list 4 --output_dir ../constrained_rerank_smoke`
2. Verify baseline metrics match Phase 0 recorded values (parity check)
3. Verify acceptance gates
4. Inspect diagnostics (swap/rejection rates)

### Phase 9: Experiment Matrix (if smoke passes)
1. Config A: baseline only (no edits — anchor_lock_positions=20, i.e., all locked)
2. Config B: lock only (lock=6, epsilon=999, budget=20)
3. Config C: lock + margin (lock=6, epsilon=0.05, budget=20)
4. Config D: lock + margin + budget (lock=6, epsilon=0.05, budget=4) — target v1
5. Compare all configs, select best that passes hard gate

---

## 5. Critical Instruction for Coding Agent

**READ `auto_implement_plan.md` FIRST.** It is the authoritative spec. This plan provides repo-specific implementation details and file-level guidance, but the auto_implement_plan defines the phases, gates, acceptance criteria, and constraints that must be followed. If there is any conflict, `auto_implement_plan.md` takes precedence.

---

## 6. Key Risks and Pitfalls

| Risk | Mitigation |
|------|------------|
| **Masking mismatch**: Reranker uses different exclusion logic than baseline `get_metrics()` | Replicate exact masking from `RCSYS_utils.py:337-348`. The baseline masks `neg_train_edge_index` items. |
| **Embedding drift**: Baseline eval re-runs forward on val/test edges, not train edges | Save full model state_dict. Reranker reconstructs model and runs forward on same eval edges as baseline. Requires loading original graph for SGSL constructor. |
| **Metric function divergence**: Reimplementing metrics introduces subtle bugs | Import from `RCSYS_utils` if possible. If not (path issues), copy exact implementations with attribution. |
| **CUDA/CPU mismatch**: Checkpoint saved on GPU, loaded on CPU | Always use `map_location='cpu'` when loading, then `.to(device)`. |
| **eval() uses edge_index for both embeddings AND metric users**: The eval function re-computes embeddings from eval edges | Reranker now does the same: loads model, runs forward on eval edges. Anchor list should produce identical metrics to baseline eval. |
| **Graph file dependency**: SGSL constructor requires the original HeteroData graph object | Reranker loads `../processed_data/benchmark_macro.pt` in `anchor.py` to reconstruct the model. Add `--graph_path` CLI arg with this default. |
| **`utils.py` star import**: `RCSYS_utils.py` does `from utils import *` which pulls in heavy deps (transformers, etc.) | When importing metric functions from RCSYS_utils in constrained_rerank, the star import chain will trigger. Either copy metric functions to avoid this, or ensure environment has all deps installed. |
| **No checkpoint exists yet**: Current codebase has no `torch.save` | Phase 0 adds this. Must run baseline training first. |
| **Import path issues**: Running from `code/` vs `code/constrained_rerank/` | Use relative imports or `sys.path` manipulation in constrained_rerank/main.py to import from parent `code/` directory. |
| **Over-constrained reranker**: All swaps rejected, no secondary metric improvement | Config A (all-locked) serves as baseline check. If Config D has zero swaps, relax per Phase 8 decision tree. |

---

## 7. Validation Checklist

### Parity Gates
- [ ] Baseline training runs successfully (`python main.py --seed 42 --K 20`)
- [ ] Checkpoint saved to `../checkpoints/sgsl_checkpoint.pt`
- [ ] Anchor list from frozen embeddings produces identical top-K as baseline eval
- [ ] Baseline metrics from reranker module match baseline metrics from main.py

### Functional Gates
- [ ] Constrained reranker CLI runs end-to-end without errors
- [ ] Position lock enforced (positions 1..L never swapped)
- [ ] Score margin gate works (rejected swaps have `cand_score < anchor_score - epsilon`)
- [ ] Swap budget enforced (no list exceeds max_swaps)
- [ ] No duplicate items in any reranked list
- [ ] Fallback to anchor on any constraint violation

### Acceptance Gates (from auto_implement_plan.md Phase 10)
- [ ] Baseline SGSL training/eval path remains unchanged (only checkpoint saving added)
- [ ] No inference-time tradeoff vector required
- [ ] Reranker edits bounded and always feasibility-checked
- [ ] Final lists always valid (no duplicates, lock compliance, budget compliance)
- [ ] Test `ndcg_drop_fraction <= 0.07` (hard gate)
- [ ] Secondary metrics at least neutral-to-improved under non-trivial edit activity
- [ ] Full run reproducible from saved config and commands

### Artifact Gates
- [ ] `run_config.json` saved per run
- [ ] `results.json` with baseline, constrained, and diagnostics blocks
- [ ] `auto_logs.md` updated after each phase
- [ ] W&B offline metadata saved (if enabled)
- [ ] Comparison table printed and saved

---

## Key Files Reference

| File | Status | Purpose |
|------|--------|---------|
| `auto_implement_plan.md` | READ ONLY | Primary implementation spec |
| `code/main.py` | MINIMAL EDIT | Add checkpoint saving (~15 lines + import) |
| `code/RCSYS_utils.py` | DO NOT MODIFY | Reuse: `get_metrics`, `RecallPrecision_ATk`, `NDCGatK_r`, `calculate_health_score`, `calculate_average_health_tags`, `calculate_percentage_recommended_foods`, `get_user_positive_items`, `split_data_new` |
| `code/RCSYS_models.py` | DO NOT MODIFY | SGSL model (produces embeddings) |
| `code/min_norm_solvers.py` | DO NOT MODIFY | MGDA solver |
| `code/constrained_rerank/__init__.py` | NEW | Package init |
| `code/constrained_rerank/anchor.py` | NEW | Frozen embedding loading + anchor list generation |
| `code/constrained_rerank/constraints.py` | NEW | FeasibilityChecker with 4 hard constraints |
| `code/constrained_rerank/reranker.py` | NEW | Constrained edit executor with health heuristic |
| `code/constrained_rerank/evaluation.py` | NEW | Parity-safe metrics + acceptance gates |
| `code/constrained_rerank/logging_utils.py` | NEW | JSON + W&B offline logging |
| `code/constrained_rerank/main.py` | NEW | CLI entrypoint |
| `code/constrained_rerank/README_CONSTRAINED_RERANK.md` | NEW | Module documentation |
