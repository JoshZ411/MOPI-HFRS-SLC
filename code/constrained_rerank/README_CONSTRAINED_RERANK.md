# Constrained Reranker for MOPI-HFRS-SLC

Post-training inference-time module that improves secondary recommendation metrics (health, diversity, coverage) while enforcing a hard ranking floor constraint on the baseline SGSL outputs.

## Design

- Loads frozen SGSL model checkpoint and reconstructs embeddings
- Generates baseline anchor top-K lists using identical masking as baseline eval
- Proposes bounded edits using a health-overlap heuristic
- Enforces 4 hard constraints: position lock, score margin, swap budget, duplicate prevention
- Falls back to anchor item when any constraint is violated
- No inference-time weight vector required

## Usage

```bash
# From code/ directory

# 1. First train baseline and save checkpoint
python main.py --seed 42 --K 20

# 2. Run constrained reranker
python -m constrained_rerank.main \
    --output_dir ../constrained_rerank_results \
    --device auto \
    --K 20 \
    --M 200 \
    --anchor_lock_positions 6 \
    --anchor_epsilon 0.05 \
    --max_swaps_per_list 4

# 3. Run on validation split
python -m constrained_rerank.main \
    --output_dir ../constrained_rerank_val \
    --split val \
    --K 20 --M 200 \
    --anchor_lock_positions 6 \
    --anchor_epsilon 0.05 \
    --max_swaps_per_list 4
```

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--output_dir` | (required) | Output directory for results |
| `--checkpoint_path` | `../checkpoints/sgsl_checkpoint.pt` | Path to SGSL checkpoint |
| `--graph_path` | `../processed_data/benchmark_macro.pt` | Path to graph data |
| `--device` | `auto` | Device: cpu, cuda, or auto |
| `--K` | `20` | Top-K recommendations |
| `--M` | `200` | Candidate pool size |
| `--anchor_lock_positions` | `6` | Immutable top positions |
| `--anchor_epsilon` | `0.05` | Score margin tolerance |
| `--max_swaps_per_list` | `4` | Max swaps per list |
| `--ndcg_floor` | `0.07` | Max allowed NDCG drop |
| `--seed` | `42` | Random seed |
| `--split` | `test` | Eval split (val or test) |
| `--use_wandb` | `False` | Enable W&B offline logging |

## Output Artifacts

- `run_config.json` — full configuration
- `results.json` — baseline, reranked metrics, diagnostics, gate results
- `comparison_table.txt` — formatted comparison
- `auto_logs.md` — phase log (appended at repo root)

## Acceptance Gates

- **Hard gate**: `ndcg_drop_fraction <= 0.07` (configurable)
- **Secondary gate**: health, diversity, coverage metrics non-negative delta
